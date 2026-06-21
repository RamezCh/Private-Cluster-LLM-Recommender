import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from unittest.mock import patch
import math
import tempfile

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.collaborative import CollaborativeFilter
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

def evaluate_recommender():
    print("=" * 60)
    print("EVALUATING HYBRID RECOMMENDER (LEAVE-ONE-OUT)")
    print("=" * 60)
    
    feedback_file = project_root / "data_gathering_pipeline" / "data" / "feedback_data.jsonl"
    if not feedback_file.exists():
        print(f"Error: {feedback_file} not found.")
        return
        
    # Load all feedback
    feedbacks = []
    with open(feedback_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    feedbacks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                    
    # Group by user
    user_feedbacks = defaultdict(list)
    for fb in feedbacks:
        user_feedbacks[fb['user_id']].append(fb)
        
    # Sort by created_at descending
    for uid in user_feedbacks:
        user_feedbacks[uid].sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
    # Identify valid users and ground truth
    # We want users who have at least 2 ratings, and their most recent "like" (rating >= 4)
    # is the hold-out test set.
    test_cases = []
    training_data = []
    
    for uid, fbs in user_feedbacks.items():
        if len(fbs) < 2:
            training_data.extend(fbs)
            continue
            
        hold_out_idx = -1
        for i, fb in enumerate(fbs):
            if fb['rating'] >= 4:
                hold_out_idx = i
                break
                
        if hold_out_idx == -1:
            training_data.extend(fbs)
            continue
            
        # Hold out the latest like
        hold_out = fbs[hold_out_idx]
        test_cases.append({
            'user_id': uid,
            'ground_truth_model': hold_out['model_id'],
            'hardware': hold_out.get('hardware_used', ''),
            'use_case': hold_out.get('use_case', '')
        })
        
        # Add all other feedbacks to training data
        for i, fb in enumerate(fbs):
            if i != hold_out_idx:
                training_data.append(fb)
                
    print(f"Total feedbacks: {len(feedbacks)}")
    print(f"Valid test users (with at least 1 like): {len(test_cases)}")
    print(f"Training interactions: {len(training_data)}")
    
    if not test_cases:
        print("Not enough data to evaluate.")
        return
        
    # We create a temporary jsonl file for training the CF
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl', encoding='utf-8') as tmp:
        for t in training_data:
            tmp.write(json.dumps(t) + "\n")
        temp_path = tmp.name
        
    try:
        cf = CollaborativeFilter(Path(temp_path), n_factors=50)
        cf.train()
        
        # Monkey patch get_collaborative_filter to return our CF
        with patch('backend.services.hybrid_recommender.get_collaborative_filter', return_value=cf):
            hybrid_rec = get_hybrid_recommender(alpha=0.6)
            hybrid_rec._collaborative_filter = cf
            
            k = 5
            hits = 0
            ndcg_sum = 0.0
            
            print(f"\nEvaluating Top-{k} recommendations...")
            for i, test in enumerate(test_cases):
                user_id = test['user_id']
                gt_model = test['ground_truth_model']
                hw_text = test['hardware'] or "MacBook Pro M3 Max"
                use_case = test['use_case'] or "general purpose"
                
                hw = parse_hardware_input(hw_text)
                if not hw:
                    continue
                    
                user_query = f"{use_case} {hw_text}"
                
                results = hybrid_rec.get_hybrid_recommendations(
                    hardware=hw,
                    use_case_text=use_case,
                    user_query=user_query,
                    user_id=user_id,
                    top_k=k
                )
                
                ranked_models = [r.model_id for r in results]
                
                if i == 0:
                    print(f"Debug [User {i}]: Ground Truth: {gt_model}")
                    print(f"Debug [User {i}]: Ranked Models: {ranked_models}")
                
                if gt_model in ranked_models:
                    hits += 1
                    
                ndcg_sum += compute_ndcg(ranked_models, gt_model, k)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(test_cases)} users...")
                    
            recall_at_k = hits / len(test_cases)
            avg_ndcg_at_k = ndcg_sum / len(test_cases)
            
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            print(f"Metrics evaluated on {len(test_cases)} users (Top-{k}):")
            print(f"Recall@{k}: {recall_at_k:.4f} ({hits}/{len(test_cases)} hits)")
            print(f"NDCG@{k}:   {avg_ndcg_at_k:.4f}")
            print("=" * 60)
            
    finally:
        os.remove(temp_path)

if __name__ == "__main__":
    evaluate_recommender()
