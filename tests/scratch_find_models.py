import json
models = []
with open('data_gathering_pipeline/data/master_model_db.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        models.append(json.loads(line))

for cat, kw, bench in [('coding', 'coder', 'coding'), ('math', 'math', 'math'), ('general', '', 'intelligence_index')]:
    filtered = [m for m in models if kw in m['model_id'].lower() or not kw]
    filtered.sort(key=lambda x: x['benchmarks'].get(bench) or 0, reverse=True)
    print(f"\n--- {cat} ---")
    for m in filtered[:10]:
        size = m.get('params_billions', 0)
        print(f"{m['model_id']} (Params: {size}B)")
