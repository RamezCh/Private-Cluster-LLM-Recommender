"""Hybrid recommender combining collaborative filtering and content-based filtering."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from backend.services.recommender import get_recommender, ScoredModel
from backend.services.parser import ParsedHardware
from backend.services.collaborative import get_collaborative_filter


@dataclass
class HybridScoredModel(ScoredModel):
    """Extended model with hybrid scoring information."""
    cf_prediction: Optional[float] = None
    cf_confidence: Optional[float] = None
    hybrid_score: Optional[float] = None
    blend_weight: Optional[float] = None


class HybridRecommender:
    """
    Hybrid recommender that combines:
    - Content-based filtering (benchmark scores, hardware fit, semantic similarity)
    - Collaborative filtering (user feedback patterns via SVD)
    
    The hybrid score is calculated as:
        hybrid_score = alpha * norm_cf + (1 - alpha) * content_score
    
    Where alpha controls the blend between approaches.
    """
    
    def __init__(
        self,
        alpha: float = 0.6,
        min_confidence_for_cf: float = 0.3
    ):
        """
        Initialize hybrid recommender.
        
        Args:
            alpha: Weight for collaborative filtering (0.6 = 60% CF, 40% content)
            min_confidence_for_cf: Minimum CF confidence to consider the prediction
        """
        self.alpha = alpha
        self.min_confidence = min_confidence_for_cf
        self._collaborative_filter = None
    
    @property
    def collaborative_filter(self):
        """Lazy-load collaborative filter."""
        if self._collaborative_filter is None:
            self._collaborative_filter = get_collaborative_filter()
        return self._collaborative_filter
    
    def _normalize_content_scores(
        self, 
        models: list[ScoredModel]
    ) -> dict[str, float]:
        """Normalize content scores to 0-1 range."""
        if not models:
            return {}
        
        scores = [m.final_score for m in models]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score - min_score < 0.001:
            return {m.model_id: 0.5 for m in models}
        
        return {
            m.model_id: (m.final_score - min_score) / (max_score - min_score)
            for m in models
        }
    
    def _normalize_cf_predictions(
        self,
        predictions: dict[str, tuple[float, float]],
        user_has_data: bool
    ) -> dict[str, float]:
        """Normalize CF predictions to 0-1 range (1-5 scale to 0-1)."""
        if not predictions:
            return {}
        
        values = [pred[0] for pred in predictions.values()]
        
        # Even if the user has no data, predictions contains the global 
        # baseline / popularity scores for these models! We should use them.
        return {mid: (v - 1) / 4 for mid, v in zip(predictions.keys(), values)}
    
    def get_hybrid_recommendations(
        self,
        hardware: ParsedHardware,
        use_case_text: str,
        user_query: str,
        user_id: Optional[str] = None,
        top_k: int = 5
    ) -> list[HybridScoredModel]:
        """
        Get hybrid recommendations combining content-based and CF scores.
        
        Args:
            hardware: Parsed hardware configuration
            use_case_text: Use case description
            user_query: Full user query for semantic search
            user_id: Optional user ID for personalized CF scores
            top_k: Number of recommendations to return
        
        Returns:
            List of HybridScoredModel sorted by hybrid_score
        """
        content_recommender = get_recommender()
        content_models = content_recommender.recommend(
            hardware=hardware,
            use_case_text=use_case_text,
            user_query=user_query,
            top_k=top_k * 3
        )
        
        if not content_models:
            return []
        
        model_ids = [m.model_id for m in content_models]
        norm_content = self._normalize_content_scores(content_models)
        
        user_has_data = user_id and self.collaborative_filter.has_user_data(user_id)
        cf_predictions = {}
        
        if user_id and self.collaborative_filter.total_ratings > 0:
            cf_predictions = self.collaborative_filter.get_user_predictions(
                user_id, model_ids, exclude_rated=False
            )
        
        norm_cf = self._normalize_cf_predictions(cf_predictions, user_has_data)
        
        if user_has_data:
            alpha = self.alpha
        elif len(cf_predictions) > 0:
            avg_confidence = np.mean([p[1] for p in cf_predictions.values()])
            alpha = self.alpha * avg_confidence
        else:
            alpha = 0.0
        
        hybrid_scores = {}
        for model_id in model_ids:
            content_score = norm_content.get(model_id, 0.5)
            cf_score = norm_cf.get(model_id, 0.5)
            
            hybrid = alpha * cf_score + (1 - alpha) * content_score
            hybrid_scores[model_id] = hybrid
        
        model_map = {m.model_id: m for m in content_models}
        ranked_ids = sorted(model_ids, key=lambda x: hybrid_scores[x], reverse=True)
        
        results = []
        for model_id in ranked_ids[:top_k]:
            original = model_map[model_id]
            cf_pred, cf_conf = cf_predictions.get(model_id, (None, None))
            
            hybrid_model = HybridScoredModel(
                model_id=original.model_id,
                hf_repo_id=original.hf_repo_id,
                base_model=original.base_model,
                params_billions=original.params_billions,
                model_type=original.model_type,
                architecture=original.architecture,
                coding=original.coding,
                math_score=original.math_score,
                reasoning=original.reasoning,
                intelligence_index=original.intelligence_index,
                safetensors_size_gb=original.safetensors_size_gb,
                vram_fp16=original.vram_fp16,
                vram_int8=original.vram_int8,
                vram_int4=original.vram_int4,
                hosting_strategy=original.hosting_strategy,
                is_moe=original.is_moe,
                semantic_score=original.semantic_score,
                benchmark_score=original.benchmark_score,
                hardware_score=original.hardware_score,
                final_score=original.final_score,
                matched_hardware=original.matched_hardware,
                cf_prediction=cf_pred,
                cf_confidence=cf_conf,
                hybrid_score=hybrid_scores[model_id],
                blend_weight=alpha if user_has_data else None,
            )
            results.append(hybrid_model)
        
        return results
    
    def get_content_only_recommendations(
        self,
        hardware: ParsedHardware,
        use_case_text: str,
        user_query: str,
        top_k: int = 5
    ) -> list[ScoredModel]:
        """Get pure content-based recommendations (no CF influence)."""
        recommender = get_recommender()
        return recommender.recommend(
            hardware=hardware,
            use_case_text=use_case_text,
            user_query=user_query,
            top_k=top_k
        )
    
    def get_user_preference_stats(self, user_id: str) -> dict:
        """Get statistics about a user's preferences from CF data."""
        if not user_id or not self.collaborative_filter.has_user_data(user_id):
            return {"has_data": False, "total_ratings": 0}
        
        cf = self.collaborative_filter
        user_idx = cf.user_to_idx.get(user_id)
        
        if user_idx is None:
            return {"has_data": False, "total_ratings": 0}
        
        user_ratings = cf.ratings_matrix.getrow(user_idx).toarray()[0]
        rated_models = []
        ratings = []
        
        for model_idx, rating in enumerate(user_ratings):
            if rating > 0:
                if model_idx < len(cf.idx_to_model):
                    rated_models.append(cf.idx_to_model[model_idx])
                    ratings.append(rating)
        
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        return {
            "has_data": True,
            "total_ratings": len(ratings),
            "avg_rating": round(avg_rating, 2),
            "ratings_distribution": self._get_rating_distribution(ratings),
        }
    
    def _get_rating_distribution(self, ratings: list[int]) -> dict[str, int]:
        """Get distribution of ratings."""
        dist = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        for r in ratings:
            key = str(int(r))
            if key in dist:
                dist[key] += 1
        return dist


_hybrid_instance: Optional[HybridRecommender] = None


def get_hybrid_recommender(alpha: float = 0.6) -> HybridRecommender:
    """Get or create the global hybrid recommender instance."""
    global _hybrid_instance
    
    if _hybrid_instance is None:
        _hybrid_instance = HybridRecommender(alpha=alpha)
    
    return _hybrid_instance