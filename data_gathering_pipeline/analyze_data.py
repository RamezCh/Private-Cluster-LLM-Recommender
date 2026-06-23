import json
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

data_path = Path("data/master_model_db.jsonl")

if not data_path.exists():
    print(f"Error: {data_path} not found.")
    exit(1)

records = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

total = len(records)
print(f"=== Data Overview ===")
print(f"Total Models: {total}\n")

print("=== Benchmark Coverage ===")
benchmark_keys = ["coding", "math", "reasoning", "intelligence_index", "elo"]
for key in benchmark_keys:
    count = sum(1 for r in records if r.get("benchmarks", {}).get(key) is not None)
    pct = round(count / total * 100, 1) if total else 0
    print(f"  {key}: {count}/{total} ({pct}%)")

print("\n=== Architecture Types ===")
moe = sum(1 for r in records if r.get("is_moe"))
dense = total - moe
print(f"  MoE: {moe} ({round(moe/total*100, 1) if total else 0}%)")
print(f"  Dense: {dense} ({round(dense/total*100, 1) if total else 0}%)")

print("\n=== Hosting Strategies ===")
strategies = defaultdict(int)
for r in records:
    strategies[r.get("hosting_strategy", "Unknown")] += 1
for strategy, count in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
    print(f"  {strategy}: {count} ({round(count/total*100, 1) if total else 0}%)")

print("\n=== Source Status ===")
statuses = defaultdict(int)
for r in records:
    statuses[r.get("source_status", "unknown")] += 1
for status, count in sorted(statuses.items(), key=lambda x: x[1], reverse=True):
    print(f"  {status}: {count} ({round(count/total*100, 1) if total else 0}%)")

print("\n=== Multi-Source Models ===")
multi = sum(1 for r in records if len(r.get("_sources", [])) > 1)
print(f"  Models from 2+ sources: {multi} ({round(multi/total*100, 1) if total else 0}%)")

print("\n=== Extended Benchmarks ===")
ext_keys = defaultdict(int)
for r in records:
    for k in r.get("extended_benchmarks", {}).keys():
        ext_keys[k] += 1
for k, count in sorted(ext_keys.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {k}: {count} ({round(count/total*100, 1) if total else 0}%)")

print("\n=== Missing Values ===")
missing_counts = defaultdict(int)
for r in records:
    has_missing = False
    if not r.get("hf_repo_id"):
        missing_counts["hf_repo_id"] += 1
        has_missing = True
    if r.get("source_status") == "missing_hf_metadata":
        missing_counts["hf_metadata_missing"] += 1
        has_missing = True
    if r.get("params_billions") is None:
        missing_counts["params_billions"] += 1
        has_missing = True
    if not r.get("vram_gb"):
        missing_counts["vram_gb"] += 1
        has_missing = True
    if not r.get("hardware_fit"):
        missing_counts["hardware_fit"] += 1
        has_missing = True
    
    bmarks = r.get("benchmarks", {})
    if not bmarks.get("coding"): missing_counts["benchmark:coding"] += 1
    if not bmarks.get("math"): missing_counts["benchmark:math"] += 1
    if not bmarks.get("reasoning"): missing_counts["benchmark:reasoning"] += 1
    if not bmarks.get("intelligence_index"): missing_counts["benchmark:intelligence_index"] += 1

for key, count in sorted(missing_counts.items(), key=lambda x: x[1], reverse=True):
    pct = round(count / total * 100, 1) if total else 0
    print(f"  {key}: {count} ({pct}%)")

has_any_missing = sum(
    1 for r in records
    if not r.get("hf_repo_id")
    or r.get("source_status") == "missing_hf_metadata"
    or r.get("params_billions") is None
)
complete = total - has_any_missing
print(f"\n=== Completeness ===")
print(f"  Complete: {complete} ({round(complete/total*100, 1) if total else 0}%)")
print(f"  Incomplete: {has_any_missing} ({round(has_any_missing/total*100, 1) if total else 0}%)")

print("\n=== Sample Records ===")
for r in records[:3]:
    print(f"  {r.get('model_id', 'unknown')}")
    print(f"    type: {r.get('model_type', 'unknown')}, params: {r.get('params_billions', 'N/A')}B")
    print(f"    bench: coding={r.get('benchmarks', {}).get('coding')}, "
          f"math={r.get('benchmarks', {}).get('math')}, "
          f"reasoning={r.get('benchmarks', {}).get('reasoning')}")
    print(f"    strategy: {r.get('hosting_strategy')}, vram_fp16: {r.get('vram_gb', {}).get('fp16', 'N/A')}GB")
    print()