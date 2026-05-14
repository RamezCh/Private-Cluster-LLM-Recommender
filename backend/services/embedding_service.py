"""FAISS embedding service with simple module-level singleton."""

import os

from backend.logging import get_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from transformers.utils import logging as tf_logging
tf_logging.disable_progress_bar()

logger = get_logger(__name__)

_instance: Optional["EmbeddingService"] = None
_instance_lock: Optional[object] = None


class EmbeddingService:
    MODEL_NAME = "all-MiniLM-L6-v2"
    DIM = 384

    def __init__(self) -> None:
        self.model = SentenceTransformer(self.MODEL_NAME)
        from config.config import EMBEDDING_INDEX_PATH, EMBEDDING_DATA_PATH
        self.index_path = EMBEDDING_INDEX_PATH
        self.data_path = EMBEDDING_DATA_PATH
        self._temp_dir = tempfile.mkdtemp(prefix="llm_rec_faiss_")
        self._local_index = os.path.join(self._temp_dir, "faiss_index.bin")
        self._local_data = os.path.join(self._temp_dir, "model_data.json")
        self.index: Optional[faiss.IndexFlatIP] = None
        self.model_ids: list[str] = []
        self.model_texts: list[str] = []
        self._emb_cache: Optional[np.ndarray] = None

    def __del__(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        try:
            if os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp dir: {e}")

    def _text_repr(self, model_id: str, base_model: Optional[str], model_type: str,
                   architecture: Optional[str]) -> str:
        parts = [
            (model_id.split("/")[-1] if "/" in model_id else model_id),
            (base_model.split("/")[-1] if base_model and "/" in base_model else (base_model or "")),
            model_type,
            architecture or "",
        ]
        return " | ".join([p for p in parts if p])

    def load_or_build(self, db_path: Optional[str] = None) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.data_path):
            self._load()
        else:
            db_path = db_path or os.environ.get(
                "DB_PATH",
                "data_gathering_pipeline/data/master_model_db.jsonl"
            )
            self._build(db_path)

    def _build(self, db_path: str) -> None:
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        records = []
        with open(db_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        self.model_ids = []
        self.model_texts = []
        for rec in records:
            mid = rec.get("model_id", "")
            self.model_ids.append(mid)
            self.model_texts.append(self._text_repr(
                mid, rec.get("base_model"), rec.get("model_type", ""), rec.get("architecture")
            ))

        logger.info(f"Encoding {len(self.model_texts)} model texts...")
        embeddings = self.model.encode(
            self.model_texts, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        self.index = faiss.IndexFlatIP(self.DIM)
        self.index.add(embeddings)
        self._emb_cache = embeddings
        self._save()

    def _save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, self._local_index)
            shutil.copy2(self._local_index, self.index_path)
        with open(self._local_data, "w", encoding="utf-8") as f:
            json.dump({"model_ids": self.model_ids, "model_texts": self.model_texts}, f)
        shutil.copy2(self._local_data, self.data_path)

    def _load(self) -> None:
        shutil.copy2(self.index_path, self._local_index)
        self.index = faiss.read_index(self._local_index)
        shutil.copy2(self.data_path, self._local_data)
        with open(self._local_data, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.model_ids = data["model_ids"]
            self.model_texts = data["model_texts"]
        logger.info(f"Loaded index with {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.index is None:
            return []
        query = query[:2000] if query else ""
        q_emb = self.model.encode(
            [query], convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        dists, idxs = self.index.search(q_emb, top_k)
        results = []
        for d, i in zip(dists[0], idxs[0]):
            if 0 <= i < len(self.model_ids):
                results.append((self.model_ids[int(i)], float(d)))
        return results


def get_embedding_service() -> EmbeddingService:
    global _instance
    if _instance is None:
        import threading
        global _instance_lock
        if _instance_lock is None:
            _instance_lock = threading.Lock()
        with _instance_lock:
            if _instance is None:
                svc = EmbeddingService()
                svc.load_or_build()
                _instance = svc
    return _instance


def reset_embedding_service() -> None:
    global _instance
    if _instance is not None:
        _instance._cleanup()
    _instance = None