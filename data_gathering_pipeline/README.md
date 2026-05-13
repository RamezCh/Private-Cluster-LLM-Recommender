# Open Source LLM Recommender

A local searchable database that maps LLM intelligence (benchmarks) directly to hardware constraints (VRAM/GPU). Built for data scientists and ML engineers who need actionable deployment recommendations for their specific hardware setup.

**Target: Open-weight, locally-hostable LLMs only.**

---

## Quick Start

```bash
# Setup
cd data_gathering_pipeline
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run (set HF_TOKEN env var for authenticated requests)
python main.py                    # Full pipeline (~5-8 min)
python main.py --hf-only          # Fast mode (HF dataset only, no OpenCompass)
python main.py --scrape-only      # Scrape OpenCompass to cache
python main.py --merge-only       # Merge using cached data
python main.py --report           # Report from existing JSONL
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT SOURCES                                                  │
│                                                                  │
│  1. open-llm-leaderboard (HF Dataset)                          │
│     4,576 rows, 36 cols, "Available on the hub" filter         │
│     Benchmarks: IFEval, BBH, MATH Lvl 5, GPQA, MUSR, MMLU-PRO  │
│                                                                  │
│  2. OpenCompass General (rank.opencompass.org.cn)               │
│     Monthly leaderboard, 5 ability dimensions                   │
│     Benchmarks: C-Eval, MMLU, GSM8K, BBH, HumanEval, ...       │
│                                                                  │
│  3. OpenCompass Academic (rank.opencompass.org.cn)              │
│     Real-time evaluations, fresh academic benchmarks            │
└─────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1              PHASE 2              PHASE 3             │
│  Load HF dataset      Scrape OpenCompass    Cross-source dedup  │
│  + open-weight        (Selenium, 2 pages)   by Base Model +     │
│    filter + dedup                            fuzzy matching     │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4-6                                                     │
│  Benchmark merge → HF metadata (20 threads) → VRAM calc        │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: master_model_db.jsonl                                  │
│  One record per locally-hostable open-weight model             │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Phases

| Phase | Action | Time | Parallelized? |
|-------|--------|------|---------------|
| 1 | Load `open-llm-leaderboard` + filter + dedup | ~30s | No |
| 2 | Scrape OpenCompass General + Academic | ~5 min | Sequential (Selenium) |
| 3 | Cross-source deduplication + canonicalization | ~10s | No |
| 4 | Benchmark merging (priority-based) | ~2s | No |
| 5 | HF metadata fetching | ~5s | **Yes (20 threads)** |
| 6 | VRAM calculation + hardware fit | ~5s | No |
| 7 | Save output | <1s | No |

**Total Runtime: ~5-8 minutes** (dominated by Selenium scraping)

---

## Project Structure

```
data_gathering_pipeline/
├── src/
│   ├── config.py             # Config, proprietary filter, open-weight config
│   ├── models.py             # Dataclasses (including OpenWeightModelRecord v2)
│   ├── gpu_catalog.py        # 35+ GPU configs (unchanged)
│   ├── orchestrator.py       # New 7-phase pipeline
│   ├── fetchers/
│   │   ├── __init__.py       # Updated exports
│   │   ├── hf_ollm.py        # NEW: open-llm-leaderboard loader
│   │   ├── opencompass.py    # NEW: OpenCompass scraper (general + academic)
│   │   └── artificial_analysis.py  # ARCHIVED: kept, not used
│   └── services/
│       ├── __init__.py       # Updated exports
│       ├── hf_metadata.py    # Parallelized HF API fetcher (unchanged)
│       ├── fuzzy_matcher.py  # Name alignment (unchanged)
│       ├── deduplicator.py   # NEW: Base Model grouping + variant selection
│       ├── benchmark_merger.py  # NEW: multi-source benchmark merging
│       └── hardware.py       # VRAM calculations (unchanged)
├── main.py                   # CLI entry point (updated flags)
├── analyze_data.py           # Data quality analysis (updated schema)
├── requirements.txt          # Dependencies
└── data/
    └── master_model_db.jsonl # Output (new schema)
```

---

## GPU Catalog

| Tier | GPUs | NVLink | InfiniBand |
|------|------|--------|------------|
| **BHT Cluster** | A100 40GB, A100 80GB, H100 80GB, V100 32GB | Yes | Yes |
| **Data Center** | H200 141GB, B200 192GB, MI300X 192GB | Yes | Yes |
| **Professional** | RTX A6000 48GB, RTX A5000 24GB | Yes | No |
| **Consumer** | RTX 4090 24GB, RTX 3090 Ti 24GB, RTX 4080 Super 16GB | No | No |
| **Laptop** | MacBook Pro M3 Max 128GB, Laptop RTX 4070 8GB | No | No |

---

## Output Schema

```json
{
  "model_id": "Qwen2.5-72B-Instruct",
  "hf_repo_id": "Qwen/Qwen2.5-72B-Instruct",
  "base_model": "Qwen/Qwen2.5-72B",
  "model_type": "💬 chat models (RLHF, DPO, IFT, ...)",
  "architecture": "Qwen2ForCausalLM",
  "precision": "bfloat16",
  "params_billions": 72.0,
  "safetensors_size_gb": 144.0,
  "benchmarks": {
    "coding": 88.2,
    "math": 91.4,
    "reasoning": 76.3,
    "elo": null,
    "intelligence_index": 45.8
  },
  "extended_benchmarks": {
    "humaneval": 88.2,
    "math_level5": 91.4,
    "mmlu_pro": 76.3,
    "big_bench_hard": 73.1
  },
  "is_moe": false,
  "num_experts": null,
  "license": "apache-2.0",
  "hub_likes": 3200,
  "generation": 1,
  "vram_gb": {
    "fp16": 172.8,
    "int8": 86.4,
    "int4": 43.2,
    "model_base_gb": 144.0
  },
  "hardware_fit": {
    "gpu_id": "a100_80gb",
    "gpu_name": "A100 80GB",
    "gpu_count": 4,
    "total_vram_gb": 320,
    "status": "Compatible",
    "is_moe_model": false,
    "hosting_strategy": "TP-Sharded",
    "context_overhead_tier": "extended_128k",
    "tier": "data_center"
  },
  "hosting_strategy": "TP-Sharded",
  "source_status": "verified",
  "all_gpu_compatibility": { ... },
  "_sources": ["open_llm_leaderboard"],
  "_variant_count": 3
}
```

---

## Key Features

1. **Open-Weight Only** — `Available on the hub` filter ensures only locally-hostable models
2. **Smart Deduplication** — Groups by `Base Model`, picks variant with most benchmark scores
3. **Multi-Source Merging** — Cross-source benchmark priority: OpenCompass > open-llm-leaderboard
4. **Parallelized HF API** — 20 concurrent threads for ~10x faster metadata fetching
5. **MoE Detection** — Explicit `MoE` boolean from dataset + name pattern matching
6. **Context-Aware VRAM** — Standard (32k), Extended (128k), Ultra (1M+) multipliers

---

## Benchmark Mapping

| Target Key | Best Source | Benchmark Columns |
|------------|-------------|-------------------|
| `coding` | OpenCompass | HumanEval, MBPP, IFEval |
| `math` | OpenCompass | MATH Lvl 5, GSM8K, MATH |
| `reasoning` | open-llm-leaderboard | MMLU-PRO, BBH, GPQA, C-Eval, MMLU, MUSR, DROP |
| `intelligence_index` | open-llm-leaderboard | Average (0-52 composite) |
| `elo` | OpenCompass | Overall Score |

---

## Deduplication Strategy

```
For each group of variants sharing the same Base Model:
  1. Score each variant: benchmark completeness count
  2. Tie-break: higher generation number (newer model)
  3. Tie-break: type preference (chat > pretrained > fine-tuned > merged)
  4. Select the single best variant per group
```

Result: ~1,500-2,500 clean canonical models (from original 4,576 rows)

---

## VRAM Formulas

```
FP16: Size × 1.2 (Standard), × 1.5 (Extended), × 2.5 (Ultra)
INT8: FP16 / 2
INT4: FP16 / 4
```

| Precision | Formula | Example (144GB model, Extended 128K) |
|-----------|---------|--------------------------------------|
| FP16 | `Size × 1.5` | 216 GB |
| INT8 | `Size / 2 × 1.5` | 108 GB |
| INT4 | `Size / 4 × 1.5` | 54 GB |

---

## Parallelism Strategies

| Model Type | Condition | Strategy | Hardware Benefit |
|------------|-----------|----------|------------------|
| Dense | Fits single GPU | Single-GPU | N/A |
| Dense | >1 GPU needed | Tensor Parallelism (TP) | NVLink optimal / PCIe degraded |
| MoE | Any size | Expert Parallelism (EP) | InfiniBand recommended |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | `` | HuggingFace API token (recommended) |
| `LOGURU_LEVEL` | `INFO` | Log level |
| `LOGURU_ROTATION` | `10 MB` | Log file rotation size |
| `LOGURU_RETENTION` | `7 days` | Log file retention |

---

## CLI Reference

```bash
python main.py                    # Run full pipeline
python main.py --hf-only          # HF dataset only (skip OpenCompass, ~30s)
python main.py --scrape-only      # Only scrape OpenCompass (cache to disk)
python main.py --merge-only       # Merge using cached data (no scraping)
python main.py --report           # Report from existing JSONL
python main.py --visible          # Show browser during OpenCompass scraping
python main.py --output ./custom.jsonl   # Custom output path
python analyze_data.py            # Analyze data quality / NaN rates
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| HF 403/401 | Set `$env:HF_TOKEN = "hf_xxx"` |
| ChromeDriver error | Run `pip install --upgrade webdriver-manager` |
| Empty OpenCompass data | Check `logs/` for Selenium errors; try `--visible` |
| No cached OpenCompass | Run `python main.py --scrape-only` first |
| Empty output | Run `python main.py --hf-only` to test without OpenCompass |

---

## Data Quality

| Metric | Target |
|--------|--------|
| Models with 3+ benchmarks | >90% |
| HF repo metadata verified | >70% |
| Proprietary models (should be 0) | <5 |

Run `python analyze_data.py` after each pipeline run to check quality.

---

## License

Educational and research purposes. Third-party data subject to their respective licenses.

## Credits

- [Open LLM Leaderboard](https://huggingface.co/datasets/open-llm-leaderboard/contents) - HuggingFace benchmark data
- [OpenCompass](https://rank.opencompass.org.cn) - Comprehensive academic LLM evaluation
- [HuggingFace](https://huggingface.co) - Model metadata and datasets