import os
import sys
import json
import math
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.hybrid_recommender import get_hybrid_recommender
from backend.services.parser import parse_hardware_input

def compute_ndcg(ranked_list, ground_truth, k):
    """Compute NDCG@K for a single user."""
    dcg = 0.0
    idcg = 1.0 # Only 1 relevant item (ground truth)
    
    for i, item in enumerate(ranked_list[:k]):
        if item == ground_truth:
            dcg = 1.0 / math.log2(i + 2) # i=0 -> log2(2)=1
            break
            
    return dcg / idcg

def evaluate_golden():
    print("=" * 60)
    print("EVALUATING HYBRID RECOMMENDER (GOLDEN DATASET)")
    print("=" * 60)
    
    golden_file = Path(__file__).parent / "golden_dataset.json"
    with open(golden_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
        
    hybrid_rec = get_hybrid_recommender(alpha=0.0) # Pure content for new users
    
    k = 5
    hits = 0
    ndcg_sum = 0.0
    
    print(f"\nEvaluating Top-{k} recommendations against Golden Dataset...")
    
    for i, test in enumerate(test_cases):
        gt_model = test['expected_model']
        hw_text = test['hardware']
        use_case = test['use_case']
        user_query = test['query']
        
        hw = parse_hardware_input(hw_text)
        if not hw:
            print(f"Skipping {user_query} due to hardware parse error")
            continue
            
        results = hybrid_rec.get_hybrid_recommendations(
            hardware=hw,
            use_case_text=use_case,
            user_query=user_query,
            user_id="golden_user", # default
            top_k=k
        )
        
        ranked_models = [r.model_id for r in results]
        
        if gt_model in ranked_models:
            hits += 1
            hit_rank = ranked_models.index(gt_model) + 1
            print(f"[HIT] Query: '{user_query}' -> Found at Rank {hit_rank}")
        else:
            print(f"[MISS] Query: '{user_query}' -> Expected: {gt_model}, Got: {ranked_models[0] if ranked_models else 'None'}")
            
        ndcg_sum += compute_ndcg(ranked_models, gt_model, k)
        
    recall_at_k = hits / len(test_cases)
    avg_ndcg_at_k = ndcg_sum / len(test_cases)
    
    print("\n" + "=" * 60)
    print("GOLDEN DATASET EVALUATION RESULTS")
    print("=" * 60)
    print(f"Metrics evaluated on {len(test_cases)} golden queries (Top-{k}):")
    print(f"Recall@{k}: {recall_at_k:.4f} ({hits}/{len(test_cases)} hits)")
    print(f"NDCG@{k}:   {avg_ndcg_at_k:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_golden()
