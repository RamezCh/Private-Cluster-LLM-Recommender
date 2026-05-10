import json
from pathlib import Path
from collections import defaultdict

data_path = Path("data/master_model_db.jsonl")

if not data_path.exists():
    print(f"Error: {data_path} not found.")
    exit(1)

records = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

total_models = len(records)
print(f"=== Data Overview ===")
print(f"Total Models Gathered: {total_models}\n")

missing_counts = defaultdict(int)
models_with_any_missing = 0

for r in records:
    has_missing = False
    
    # Check top-level keys
    if not r.get("hosting_strategy") or r.get("hosting_strategy") == "Unknown":
        missing_counts["hosting_strategy"] += 1
        has_missing = True
        
    # Check HF Metadata
    hf = r.get("hf_metadata", {})
    if hf.get("metadata_status") != "verified":
        missing_counts["hf_metadata_unverified"] += 1
        has_missing = True
    if not hf.get("repo_id"):
        missing_counts["hf_repo_id"] += 1
        has_missing = True

    # Check Benchmarks
    b = r.get("benchmarks", {})
    bench_keys = ["coding", "math", "reasoning", "elo", "intelligence_index", "throughput_tokens_per_sec"]
    for k in bench_keys:
        if b.get(k) is None:
            missing_counts[f"benchmark_{k}"] += 1
            has_missing = True
            
    # Check Hardware Fit
    hw = r.get("hardware_fit", {})
    if hw.get("status") != "Compatible":
        missing_counts["hardware_incompatible"] += 1
        has_missing = True

    if has_missing:
        models_with_any_missing += 1

print("=== Missing Values Breakdown ===")
for key, count in sorted(missing_counts.items(), key=lambda x: x[1], reverse=True):
    pct = (count / total_models) * 100
    print(f"- {key}: {count} models missing ({pct:.1f}%)")

print("\n=== Overall Missingness ===")
pct_any = (models_with_any_missing / total_models) * 100
print(f"Models with at least one missing field: {models_with_any_missing} ({pct_any:.1f}%)")
print(f"Models fully complete: {total_models - models_with_any_missing} ({100 - pct_any:.1f}%)")
