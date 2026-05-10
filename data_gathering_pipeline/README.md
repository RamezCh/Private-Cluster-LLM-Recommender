# Model-Hardware Intelligence Index (MHII) Data Pipeline

A local searchable database that maps LLM intelligence (benchmarks) directly to hardware constraints (VRAM/GPU). Built for data scientists and ML engineers who need actionable deployment recommendations for their specific hardware setup.

---

## Quick Start

```bash
# Setup
cd data_gathering_pipeline
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run (set HF_TOKEN env var for authenticated requests)
python main.py
```

---

## A-Z Workflow Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   INPUTS    │    │  PHASE 1    │    │  PHASE 2    │    │  PHASE 3    │    │  OUTPUTS    │
│             │    │  Web        │    │  HF         │    │  Parallel   │    │             │
│  Sources:   │───→│  Scraping   │───→│  Datasets   │───→│  Metadata   │───→│  master_    │
│             │    │  (Chrome)   │    │  (2 sources)│    │  20 threads │    │  model_     │
│  1. AA      │    │             │    │             │    │             │    │  db.jsonl   │
│  2. OpenEv  │    │   ~1 min    │    │   ~10 sec   │    │   ~5 sec    │    │             │
│  3. LMSYS   │    │             │    │             │    │             │    │  logs/      │
│  4. HF Hub  │    │             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

LEGEND: AA = Artificial Analysis, OpenEv = OpenEvals, HF = HuggingFace
```

### Phase Details

| Phase | Source | Action | Time | Parallelized? |
|-------|--------|--------|------|---------------|
| 1 | Artificial Analysis | Selenium scrapes leaderboard table | ~1 min | No |
| 2 | OpenEvals + LMSYS | HF datasets library downloads | ~10 sec | No |
| 3 | HuggingFace Hub | API metadata for all models | ~5 sec | **Yes (20 threads)** |
| 4 | All sources | Merge, calculate VRAM, generate strategies | <1 sec | No |
| 5 | Output | Save to JSONL | <1 sec | No |

**Total Runtime: ~1.5–2 minutes**

---

## Project Structure

```
data_gathering_pipeline/
├── src/
│   ├── __init__.py
│   ├── config.py             # Config + logging (supports env vars)
│   ├── models.py             # Dataclasses
│   ├── gpu_catalog.py        # 35+ GPU configs
│   ├── orchestrator.py       # 5-phase pipeline
│   ├── fetchers/
│   │   ├── web_scraper.py    # Selenium Chrome
│   │   └── hf_datasets.py    # HF dataset loader
│   └── services/
│       ├── fuzzy_matcher.py  # Name alignment
│       ├── hf_metadata.py    # Parallelized API fetcher
│       └── hardware.py       # VRAM calculations
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

---

## GPU Catalog

| Tier | GPUs | NVLink | InfiniBand |
|------|------|--------|------------|
| **Data Center** | H100 80GB, A100 80GB/40GB, V100 32GB, B200 192GB, H200 141GB | Yes | Yes |
| **Consumer** | RTX 4090 24GB, RTX 3090 Ti 24GB, RTX 4080 Super 16GB | No | No |
| **Professional** | RTX A6000 48GB, RTX A5000 24GB, RTX A4000 16GB | Yes | No |
| **Laptop** | MacBook Pro M3 Max 128GB, Laptop RTX 4070 8GB | No | No |

---

## Output Schema

```json
{
  "model_id": "DeepSeek-V4-Pro",
  "benchmarks": {
    "coding": 91.2,
    "math": 85.4,
    "elo": 1340,
    "intelligence_index": 92.5,
    "throughput_tokens_per_sec": 145.2
  },
  "vram_gb": {
    "fp16": 153.6,
    "int8": 76.8,
    "int4": 38.4,
    "model_base_gb": 128.0
  },
  "hardware_fit": {
    "gpu_id": "a100_80gb",
    "gpu_name": "A100 80GB",
    "gpu_count": 8,
    "total_vram_gb": 640,
    "status": "Compatible",
    "is_moe_model": true,
    "hosting_strategy": "Expert-Distributed",
    "context_overhead_tier": "extended_128k",
    "tier": "data_center"
  },
  "hosting_strategy": "Expert-Distributed",
  "source_status": "verified",
  "all_gpu_compatibility": {
    "all_compatible_gpus": [
      {"name": "H100 80GB SXM5", "count": 4, "vram": 320, "score": 99.0}
    ],
    "best_data_center": {},
    "best_consumer": {}
  },
  "hf_metadata": {
    "repo_id": "deepseek-ai/DeepSeek-V4",
    "safetensors_size_gb": 128.0,
    "is_moe": true,
    "num_experts": 256,
    "metadata_status": "verified"
  }
}
```

---

## Key Features

1. **Comprehensive GPU Support** - BHT cluster, data center, consumer, laptop GPUs
2. **Parallelized HF API** - 20 concurrent threads for ~10x faster metadata fetching
3. **Two-Tier HF Fetching** - Direct repo lookup + search fallback
4. **Fuzzy Name Matching** - Aligns names across sources (85% threshold)
5. **MoE Detection** - Automatic Mixture-of-Experts architecture detection
6. **Context-Aware VRAM** - Standard (32k), Extended (128k), Ultra (1M+) multipliers

---

## VRAM Formulas

```
FP16: Size × 1.2 (Standard), × 1.5 (Extended), × 2.5 (Ultra)
INT8: FP16 / 2
INT4: FP16 / 4
```

| Precision | Formula | Example (100GB model) |
|-----------|---------|----------------------|
| FP16 | `Size × 1.2` | 120 GB |
| INT8 | `Size / 2 × 1.2` | 60 GB |
| INT4 | `Size / 4 × 1.2` | 30 GB |

---

## Parallelism Strategies

| Model Type | Condition | Strategy | Hardware Benefit |
|------------|-----------|----------|------------------|
| Dense | Fits single GPU | Data Parallelism | N/A |
| Dense | >1 GPU needed | Tensor Parallelism | NVLink optimal / PCIe degraded |
| MoE | Any size | Expert Parallelism | InfiniBand recommended |

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
python main.py --scrape-only      # Only scrape Artificial Analysis
python main.py --merge-only       # Merge using cached scrape data
python main.py --report           # Report from existing JSONL
python main.py --visible          # Show browser during scraping
python main.py --output ./custom.jsonl   # Custom output path
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| HF 403/401 | Set `$env:HF_TOKEN = "hf_xxx"` (Windows) or `export HF_TOKEN=hf_xxx` |
| ChromeDriver error | Run `pip install --upgrade webdriver-manager` |
| Empty LMSYS/ELO data | Verify dataset split — pipeline tries `full` then `latest` automatically |
| Empty output | Check `logs/` for errors |

---

## License

Educational and research purposes. Third-party data subject to their respective licenses.

## Credits

- [Artificial Analysis](https://artificialanalysis.ai) - Real-time AI benchmarks
- [OpenEvals](https://huggingface.co/datasets/OpenEvals/leaderboard-data) - Academic benchmarks
- [LMSYS Arena](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset) - Human preference data  
- [HuggingFace](https://huggingface.co) - Model metadata & datasets