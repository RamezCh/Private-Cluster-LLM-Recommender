#!/usr/bin/env python3
"""Build the FAISS embedding index for the LLM Recommender."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.embedding_service import EmbeddingService


def main():
    print("Building FAISS embedding index...")
    print("Model: all-MiniLM-L6-v2 (384 dimensions)")
    print()
    
    service = EmbeddingService()
    
    db_path = "data_gathering_pipeline/data/master_model_db.jsonl"
    
    print(f"Loading models from {db_path}...")
    service.build_index(db_path)
    
    print(f"\nIndex built successfully!")
    print(f"  - Total models: {len(service.model_ids)}")
    print(f"  - Index saved to: {service.index_path}")
    print(f"  - Metadata saved to: {service.data_path}")
    
    print("\nTesting search...")
    test_queries = [
        "code generation python",
        "mathematical reasoning",
        "chat assistant"
    ]
    
    for query in test_queries:
        results = service.search(query, top_k=3)
        print(f"\n  Query: '{query}'")
        for model_id, score in results:
            print(f"    - {model_id.split('/')[-1]} (score: {score:.3f})")


if __name__ == "__main__":
    main()