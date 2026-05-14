#!/usr/bin/env python3
"""Build FAISS index. Run: python embeddings/build_index.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.embedding_service import EmbeddingService


def main():
    print("Building FAISS index...")
    print("Model: all-MiniLM-L6-v2 (384 dims)")
    print()

    svc = EmbeddingService()
    db_path = "data_gathering_pipeline/data/master_model_db.jsonl"
    print(f"Loading from {db_path}...")
    svc._build(db_path)

    print(f"\nIndex built: {len(svc.model_ids)} models")
    print(f"  Index: {svc.index_path}")
    print(f"  Data:  {svc.data_path}")

    print("\nSearch test:")
    for q in ["code generation python", "mathematical reasoning", "chat assistant"]:
        results = svc.search(q, top_k=3)
        print(f"\n  Query: '{q}'")
        for mid, score in results:
            print(f"    - {mid.split('/')[-1]} ({score:.3f})")


if __name__ == "__main__":
    main()