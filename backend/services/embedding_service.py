import json
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
        data_path: Optional[str] = None
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        
        self.index_path = index_path or "embeddings/faiss_index.bin"
        self.data_path = data_path or "embeddings/model_data.json"
        
        self.index: Optional[faiss.IndexFlatIP] = None
        self.model_ids: list[str] = []
        self.model_texts: list[str] = []
        
    def create_text_representation(
        self,
        model_id: str,
        base_model: Optional[str],
        model_type: str,
        architecture: Optional[str]
    ) -> str:
        parts = [
            model_id.split("/")[-1] if "/" in model_id else model_id,
            base_model.split("/")[-1] if base_model and "/" in base_model else (base_model or ""),
            model_type,
            architecture or ""
        ]
        return " | ".join([p for p in parts if p])
    
    def load_or_build_index(
        self,
        db_path: str = "data_gathering_pipeline/data/master_model_db.jsonl"
    ) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.data_path):
            self.load_index()
        else:
            self.build_index(db_path)
    
    def build_index(self, db_path: str) -> None:
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        
        records = []
        with open(db_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        self.model_ids = []
        self.model_texts = []
        
        for record in records:
            model_id = record.get("model_id", "")
            self.model_ids.append(model_id)
            
            text = self.create_text_representation(
                model_id=model_id,
                base_model=record.get("base_model"),
                model_type=record.get("model_type", ""),
                architecture=record.get("architecture")
            )
            self.model_texts.append(text)
        
        embeddings = self.model.encode(
            self.model_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        
        self.save_index()
    
    def save_index(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_ids": self.model_ids,
                "model_texts": self.model_texts
            }, f, ensure_ascii=False)
    
    def load_index(self) -> None:
        self.index = faiss.read_index(self.index_path)
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.model_ids = data["model_ids"]
            self.model_texts = data["model_texts"]
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        if self.index is None:
            return []
        
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.model_ids):
                results.append((self.model_ids[idx], float(dist)))
        
        return results
    
    def get_similarity_score(self, model_id: str, query: str) -> float:
        if model_id not in self.model_ids:
            return 0.0
        
        idx = self.model_ids.index(model_id)
        text = self.model_texts[idx]
        
        embeddings = self.model.encode(
            [query, text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)


_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
        _embedding_service.load_or_build_index()
    return _embedding_service