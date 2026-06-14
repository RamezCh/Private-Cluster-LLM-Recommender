"""Collaborative filtering using SVD matrix factorization."""

import json
import numpy as np
from pathlib import Path
from typing import Optional
from scipy import sparse
from scipy.sparse.linalg import svds


class CollaborativeFilter:
    """SVD-based collaborative filter for rating predictions."""
    
    def __init__(self, feedback_path: Path, n_factors: int = 50, min_ratings: int = 3):
        self.feedback_path = feedback_path
        self.n_factors = n_factors
        self.min_ratings = min_ratings
        
        self.user_to_idx: dict[str, int] = {}
        self.idx_to_user: dict[int, str] = {}
        self.model_to_idx: dict[str, int] = {}
        self.idx_to_model: dict[int, str] = {}
        
        self.ratings_matrix: Optional[sparse.csr_matrix] = None
        self.user_means: Optional[np.ndarray] = None
        self.U: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        self.Vt: Optional[np.ndarray] = None
        
        self._trained = False
        self._feedback_cache: list[dict] = []
    
    def _load_feedback(self) -> list[dict]:
        """Load feedback data from JSONL file."""
        if not self.feedback_path.exists():
            return []
        
        records = []
        with open(self.feedback_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records
    
    def _build_mappings(self, records: list[dict]) -> tuple[dict, dict, dict, dict]:
        """Build user and model index mappings."""
        users = sorted(set(r["user_id"] for r in records))
        models = sorted(set(r["model_id"] for r in records))
        
        user_to_idx = {u: i for i, u in enumerate(users)}
        idx_to_user = {i: u for u, i in user_to_idx.items()}
        model_to_idx = {m: i for i, m in enumerate(models)}
        idx_to_model = {i: m for m, i in model_to_idx.items()}
        
        return user_to_idx, idx_to_user, model_to_idx, idx_to_model
    
    def _build_matrix(self, records: list[dict]) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Build user-item rating matrix."""
        n_users = len(self.user_to_idx)
        n_models = len(self.model_to_idx)
        
        rows, cols, data = [], [], []
        user_totals = np.zeros(n_users)
        user_counts = np.zeros(n_users)
        
        for record in records:
            user_idx = self.user_to_idx.get(record["user_id"])
            model_idx = self.model_to_idx.get(record["model_id"])
            rating = record["rating"]
            
            if user_idx is not None and model_idx is not None:
                rows.append(user_idx)
                cols.append(model_idx)
                data.append(rating)
                user_totals[user_idx] += rating
                user_counts[user_idx] += 1
        
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_models))
        
        user_means = np.zeros(n_users)
        for i in range(n_users):
            if user_counts[i] >= self.min_ratings:
                user_means[i] = user_totals[i] / user_counts[i]
            elif user_counts[i] > 0:
                user_means[i] = user_totals[i] / user_counts[i]
            else:
                user_means[i] = 3.5
        
        return matrix, user_means
    
    def train(self) -> "CollaborativeFilter":
        """Train the collaborative filter using SVD."""
        if self._trained:
            return self
        
        self._feedback_cache = self._load_feedback()
        
        if len(self._feedback_cache) < 10:
            self._trained = True
            return self
        
        (
            self.user_to_idx,
            self.idx_to_user,
            self.model_to_idx,
            self.idx_to_model,
        ) = self._build_mappings(self._feedback_cache)
        
        self.ratings_matrix, self.user_means = self._build_matrix(self._feedback_cache)
        
        n_users, n_models = self.ratings_matrix.shape
        k = min(self.n_factors, min(n_users, n_models) - 1)
        
        matrix_centered = self.ratings_matrix.toarray().astype(np.float64)
        for i in range(n_users):
            mask = matrix_centered[i] != 0
            matrix_centered[i, mask] -= self.user_means[i]
        
        self.U, self.S, self.Vt = svds(sparse.csr_matrix(matrix_centered), k=k)
        
        idx = np.argsort(self.S)[::-1]
        self.S = self.S[idx]
        self.U = self.U[:, idx]
        self.Vt = self.Vt[idx, :]
        
        self._trained = True
        return self
    
    def predict(self, user_id: str, model_id: str) -> tuple[float, float]:
        """
        Predict rating for a user-model pair.
        
        Returns:
            tuple: (predicted_rating, confidence)
            - predicted_rating: 1-5 scale
            - confidence: 0-1 based on data availability
        """
        if not self._trained or self.ratings_matrix is None:
            return 3.5, 0.0
        
        user_idx = self.user_to_idx.get(user_id)
        model_idx = self.model_to_idx.get(model_id)
        
        if user_idx is None or model_idx is None:
            return 3.5, 0.0
        
        n_users = self.ratings_matrix.shape[0]
        n_models = self.ratings_matrix.shape[1]
        
        if user_idx >= n_users or model_idx >= n_models:
            return 3.5, 0.0
        
        predicted = self.user_means[user_idx]
        
        if self.U is not None and len(self.S) > 0:
            cf_component = np.dot(self.U[user_idx, :] * self.S, self.Vt[:, model_idx])
            predicted += cf_component
        
        predicted = max(1.0, min(5.0, predicted))
        
        user_ratings_count = self.ratings_matrix.getrow(user_idx).nnz
        model_ratings_count = self.ratings_matrix.getcol(model_idx).nnz
        confidence = min(1.0, (user_ratings_count + model_ratings_count) / 20.0)
        
        return float(predicted), float(confidence)
    
    def get_user_predictions(
        self, 
        user_id: str, 
        model_ids: list[str],
        exclude_rated: bool = True
    ) -> dict[str, tuple[float, float]]:
        """
        Get predictions for a user across multiple models.
        Uses vectorized SVD computation for O(k) instead of O(n*k).
        
        Returns:
            dict: {model_id: (predicted_rating, confidence)}
        """
        if not self._trained:
            return {m: (3.5, 0.0) for m in model_ids}
        
        user_idx = self.user_to_idx.get(user_id)
        rated_models = set()
        
        if user_idx is not None and self.ratings_matrix is not None:
            user_row = self.ratings_matrix.getrow(user_idx)
            for model_idx in user_row.indices:
                if model_idx < len(self.idx_to_model):
                    rated_models.add(self.idx_to_model[model_idx])
        
        # Vectorized: compute all predictions at once using SVD factors
        predictions = {}
        if user_idx is not None and self.U is not None and len(self.S) > 0:
            # user_vector = U[user_idx, :] * S — shape (k,)
            user_vector = self.U[user_idx, :] * self.S
            user_mean = self.user_means[user_idx]
            user_nnz = self.ratings_matrix.getrow(user_idx).nnz
            
            for model_id in model_ids:
                if exclude_rated and model_id in rated_models:
                    continue
                
                model_idx = self.model_to_idx.get(model_id)
                if model_idx is None:
                    predictions[model_id] = (3.5, 0.0)
                    continue
                
                # predicted = mu_u + U[u,:] * S * Vt[:, m]
                cf_component = np.dot(user_vector, self.Vt[:, model_idx])
                predicted = max(1.0, min(5.0, user_mean + cf_component))
                
                model_nnz = self.ratings_matrix.getcol(model_idx).nnz
                confidence = min(1.0, (user_nnz + model_nnz) / 20.0)
                
                predictions[model_id] = (float(predicted), float(confidence))
        else:
            for model_id in model_ids:
                if exclude_rated and model_id in rated_models:
                    continue
                predictions[model_id] = (3.5, 0.0)
        
        return predictions
    
    def has_user_data(self, user_id: str) -> bool:
        """Check if user has feedback data."""
        if not self._trained or self.ratings_matrix is None:
            return False
        
        user_idx = self.user_to_idx.get(user_id)
        if user_idx is None:
            return False
        
        return self.ratings_matrix.getrow(user_idx).nnz >= self.min_ratings
    
    @property
    def total_users(self) -> int:
        return len(self.user_to_idx)
    
    @property
    def total_models(self) -> int:
        return len(self.model_to_idx)
    
    @property
    def total_ratings(self) -> int:
        if self.ratings_matrix is None:
            return 0
        return self.ratings_matrix.nnz


_cf_instance: Optional[CollaborativeFilter] = None


def get_collaborative_filter(
    feedback_path: Optional[Path] = None,
    n_factors: int = 50
) -> CollaborativeFilter:
    """Get or create the global collaborative filter instance."""
    global _cf_instance
    
    if _cf_instance is None:
        if feedback_path is None:
            feedback_path = Path(__file__).parent.parent.parent / "data_gathering_pipeline" / "data" / "feedback_data.jsonl"
        
        _cf_instance = CollaborativeFilter(feedback_path, n_factors=n_factors)
        _cf_instance.train()
    
    return _cf_instance