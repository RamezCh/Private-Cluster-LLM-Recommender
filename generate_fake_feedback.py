#!/usr/bin/env python3
"""
Generate fake user feedback data for the LLM Recommender system.

Creates realistic feedback records with:
- 100 users with unique profiles
- Each user rates 5-10 recommendations
- Rating distribution follows realistic bell curve (avg ~3.5)
- Higher ratings for top-ranked models (positive correlation with benchmark scores)
- Some noise/randomness to simulate human behavior
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


def load_model_database(db_path: Path) -> list[dict]:
    """Load the master model database."""
    if not db_path.exists():
        print(f"Warning: Model database not found at {db_path}")
        return []
    
    models = []
    with open(db_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    models.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return models


def generate_users(num_users: int) -> list[dict]:
    """Generate fake user profiles."""
    user_templates = [
        "ml_researcher", "data_scientist", "ml_engineer", 
        "startup_founder", "hobbyist", "enterprise_dev",
        "academic_researcher", "student", "freelancer"
    ]
    
    hardware_configs = [
        "1x NVIDIA RTX 4090 24GB",
        "2x NVIDIA A100 80GB",
        "4x NVIDIA RTX 3090 24GB",
        "1x NVIDIA A100 40GB",
        "8x NVIDIA H100 80GB",
        "MacBook Pro M3 Max 128GB",
        "1x NVIDIA RTX 4080 16GB",
        "2x NVIDIA A6000 48GB",
        "4x NVIDIA A100 80GB",
        "1x AMD MI300X 192GB",
    ]
    
    use_cases = [
        "code generation", "math reasoning", "general purpose",
        "text summarization", "data analysis", "creative writing",
        "question answering", "code review", "research assistance"
    ]
    
    users = []
    for i in range(num_users):
        user = {
            "user_id": f"user_{random.randint(100000, 999999)}_{random.randint(1000, 9999)}",
            "profile": random.choice(user_templates),
            "preferred_hardware": random.choice(hardware_configs),
            "primary_use_case": random.choice(use_cases),
        }
        users.append(user)
    
    return users


def calculate_expected_rating(model: dict, rank: int) -> float:
    """
    Calculate expected rating based on model quality and rank.
    
    Higher ratings for:
    - Higher benchmark scores
    - Lower rank (top recommendations)
    - Models with good hardware compatibility
    """
    base_score = 3.0
    
    benchmarks = model.get("benchmarks", {})
    intelligence = benchmarks.get("intelligence_index", 50) or 50
    coding = benchmarks.get("coding", 50) or 50
    
    model_quality = (intelligence + coding) / 2
    
    rank_factor = max(0, 1 - (rank * 0.1))
    
    expected = base_score + (model_quality / 100) * 1.5 + rank_factor * 0.5
    
    return min(5.0, max(1.0, expected))


def generate_rating(expected: float, randomness: float = 0.6) -> int:
    """
    Generate a realistic rating with some randomness.
    
    Uses a distribution that peaks around the expected value
    with configurable spread.
    """
    noise = random.gauss(0, randomness)
    adjusted = expected + noise
    
    rating = round(adjusted)
    return max(1, min(5, rating))


def generate_feedback_for_user(
    user: dict, 
    models: list[dict],
    min_ratings: int = 5,
    max_ratings: int = 10
) -> list[dict]:
    """Generate feedback records for a single user."""
    if not models:
        return []
    
    num_ratings = random.randint(min_ratings, max_ratings)
    
    selected_models = random.sample(models, min(num_ratings, len(models)))
    
    feedbacks = []
    base_time = datetime.utcnow() - timedelta(days=random.randint(1, 90))
    
    for rank, model in enumerate(selected_models, 1):
        expected = calculate_expected_rating(model, rank)
        rating = generate_rating(expected)
        
        time_offset = timedelta(hours=random.randint(0, 72 * rank))
        recommended_at = base_time - time_offset
        created_at = recommended_at + timedelta(minutes=random.randint(1, 30))
        
        feedback = {
            "user_id": user["user_id"],
            "model_id": model.get("model_id") or model.get("base_model", "Unknown"),
            "rating": rating,
            "hardware_used": user["preferred_hardware"],
            "use_case": user["primary_use_case"],
            "recommended_at": recommended_at.isoformat(),
            "created_at": created_at.isoformat(),
        }
        feedbacks.append(feedback)
    
    return feedbacks


def generate_all_feedback(
    users: list[dict],
    models: list[dict],
    output_path: Path
) -> dict:
    """Generate feedback for all users and save to file."""
    all_feedbacks = []
    
    print(f"Generating feedback for {len(users)} users...")
    print(f"Using {len(models)} models from database")
    
    for i, user in enumerate(users):
        feedbacks = generate_feedback_for_user(user, models)
        all_feedbacks.extend(feedbacks)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(users)} users ({len(all_feedbacks)} total feedbacks)")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for feedback in all_feedbacks:
            f.write(json.dumps(feedback, ensure_ascii=False) + "\n")
    
    return {
        "total_users": len(users),
        "total_feedbacks": len(all_feedbacks),
        "output_file": str(output_path),
    }


def analyze_feedback(output_path: Path) -> dict:
    """Analyze generated feedback for statistics."""
    if not output_path.exists():
        return {}
    
    ratings = []
    ratings_by_model = {}
    
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                rating = record.get("rating", 0)
                model_id = record.get("model_id", "Unknown")
                
                ratings.append(rating)
                
                if model_id not in ratings_by_model:
                    ratings_by_model[model_id] = {"count": 0, "total": 0, "ratings": []}
                ratings_by_model[model_id]["count"] += 1
                ratings_by_model[model_id]["total"] += rating
                ratings_by_model[model_id]["ratings"].append(rating)
            except json.JSONDecodeError:
                continue
    
    if not ratings:
        return {}
    
    distribution = {str(i): 0 for i in range(1, 6)}
    for r in ratings:
        distribution[str(r)] = distribution.get(str(r), 0) + 1
    
    avg_rating = sum(ratings) / len(ratings)
    
    top_rated_models = sorted(
        ratings_by_model.items(),
        key=lambda x: x[1]["total"] / x[1]["count"] if x[1]["count"] > 0 else 0,
        reverse=True
    )[:5]
    
    return {
        "total_feedbacks": len(ratings),
        "average_rating": round(avg_rating, 2),
        "distribution": distribution,
        "top_rated_models": [
            {
                "model": model,
                "avg_rating": round(data["total"] / data["count"], 2),
                "count": data["count"]
            }
            for model, data in top_rated_models
        ],
    }


def main():
    print("=" * 60)
    print("LLM RECOMMENDER - FAKE FEEDBACK DATA GENERATOR")
    print("=" * 60)
    print()
    
    project_root = Path(__file__).parent
    
    db_path = project_root / "data_gathering_pipeline" / "data" / "master_model_db.jsonl"
    output_path = project_root / "data_gathering_pipeline" / "data" / "feedback_data.jsonl"
    
    print(f"Loading model database from: {db_path}")
    models = load_model_database(db_path)
    
    if not models:
        print("Warning: No models found. Using fallback model list.")
        models = [
            {
                "model_id": "Qwen2.5-72B-Instruct",
                "benchmarks": {"intelligence_index": 85, "coding": 88}
            },
            {
                "model_id": "Llama-3.1-70B-Instruct",
                "benchmarks": {"intelligence_index": 82, "coding": 85}
            },
            {
                "model_id": "Mistral-Nemo-Instruct",
                "benchmarks": {"intelligence_index": 75, "coding": 78}
            },
            {
                "model_id": "Phi-3.5-mini-instruct",
                "benchmarks": {"intelligence_index": 70, "coding": 72}
            },
        ]
    
    print(f"Loaded {len(models)} models")
    print()
    
    num_users = 100
    print(f"Generating {num_users} users with feedback...")
    print()
    
    users = generate_users(num_users)
    
    result = generate_all_feedback(users, models, output_path)
    
    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total users:     {result['total_users']}")
    print(f"  Total feedbacks: {result['total_feedbacks']}")
    print(f"  Output file:     {result['output_file']}")
    print()
    
    print("Analyzing feedback data...")
    stats = analyze_feedback(output_path)
    
    if stats:
        print()
        print("STATISTICS:")
        print("-" * 40)
        print(f"  Average rating: {stats['average_rating']}")
        print()
        print("  Rating distribution:")
        for rating, count in stats["distribution"].items():
            pct = (count / stats["total_feedbacks"]) * 100 if stats["total_feedbacks"] > 0 else 0
            bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
            print(f"    {rating} stars: [{bar}] {count} ({pct:.1f}%)")
        
        if stats.get("top_rated_models"):
            print()
            print("  Top 5 rated models:")
            for i, model_data in enumerate(stats["top_rated_models"], 1):
                model_name = model_data['model'][:40] if model_data['model'] else 'Unknown'
                print(f"    {i}. {model_name:40s} (avg: {model_data['avg_rating']:.2f}, n={model_data['count']})")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()