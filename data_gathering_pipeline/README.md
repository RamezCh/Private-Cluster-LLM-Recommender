# Model-Hardware Intelligence Index (MHII) Data Pipeline

A local searchable database that maps LLM intelligence (benchmarks) directly to hardware constraints (VRAM/GPU). Built for data scientists and ML engineers who need actionable deployment recommendations for their specific hardware setup.

---

## Quick Start

```bash
# Clone & setup
cd data_gathering_pipeline
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Optional: Add HF token
cp .env.example .env
# Edit .env: HF_TOKEN=hf_xxx

# Run
python main.py
```

---

## A-Z Workflow Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           MHII PIPELINE WORKFLOW                             │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   INPUTS    │    │  PHASE 1    │    │  PHASE 2    │    │  PHASE 3    │    │  OUTPUTS    │
│             │    │  Web        │    │  HF         │    │  Parallel   │    │             │
│  Sources:   │───→│  Scraping   │───→│  Datasets   │───→│  Metadata   │───→│  master_    │
│             │    │  (Sequential│    │  (Sequential│    │  Fetching   │    │  model_     │
│  1. AA      │    │  Chrome)    │    │  2 sources) │    │  20 threads │    │  db.jsonl   │
│  2. OpenEv  │    │             │    │             │    │             │    │             │
│  3. LMSYS   │    │   ~5 min    │    │   ~2 min    │    │   ~1 min    │    │  logs/      │
│  4. HF Hub  │    │             │    │             │    │             │    │  on PVC     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

LEGEND: AA = Artificial Analysis, OpenEv = OpenEvals, HF = HuggingFace
```

### Phase Details

| Phase | Source | Action | Time | Parallelized? |
|-------|--------|--------|------|---------------|
| 1 | Artificial Analysis | Selenium scrapes leaderboard table | ~5 min | No |
| 2 | OpenEvals + LMSYS | HF datasets library downloads | ~2 min | No |
| 3 | HuggingFace Hub | API metadata for all models | ~1 min | **Yes (20 threads)** |
| 4 | All sources | Merge, calculate VRAM, generate strategies | <1 min | No |
| 5 | Output | Save to JSONL | <1 min | No |

**Total Runtime: ~7-9 minutes** (vs ~30 min if sequential)

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
├── k8s/
│   ├── mhii-pvc.yaml         # PVC (10Gi data + 10Gi logs on PVC)
│   └── mhii-job.yaml         # Cluster job manifest
├── Dockerfile                # Container for cluster
├── .dockerignore
├── main.py                   # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## GPU Catalog

| Tier | GPUs | NVLink | InfiniBand |
|------|------|--------|------------|
| **BHT Cluster** | H100 80GB, A100 80GB/40GB, V100 32GB, P100 16GB | Yes | Yes |
| **Data Center** | B200 192GB, H200 141GB, MI300X 192GB | Yes | Yes |
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
    "best_data_center": {...},
    "best_consumer": {...}
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
2. **Parallelized HF API** - 20 concurrent threads for 10x faster metadata fetching
3. **Two-Tier HF Fetching** - Direct lookup + fallback search
4. **Fuzzy Name Matching** - Aligns names across sources (85% threshold)
5. **MoE Detection** - Automatic Mixture-of-Experts architecture detection
6. **Context-Aware VRAM** - Standard (32k), Extended (128k), Ultra (1M+) multipliers
7. **Cluster-Ready** - Docker + K8s with PVC persistence for logs & data

---

## VRAM Formulas

```
FP16: Size × 1.2 (Standard), × 1.5 (Extended), × 2.5 (Ultra)
INT8: Size / 2 × multiplier
INT4: Size / 4 × multiplier
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

## BHT Cluster Deployment

### Complete Workflow

```bash
# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Build & Push Docker Image
# ─────────────────────────────────────────────────────────────────────────────
docker build -t registry.datexis.com/vpdx3758/llm-recommender-data-gathering-pipeline:latest .
docker push registry.datexis.com/vpdx3758/llm-recommender-data-gathering-pipeline:latest

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Create Secret (your HF token)
# ─────────────────────────────────────────────────────────────────────────────
kubectl create secret generic mhii-secrets \
  --from-literal=HF_TOKEN=hf_your_token_here \
  -n vpdx3758

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Deploy Storage & Job
# ─────────────────────────────────────────────────────────────────────────────
kubectl apply -f k8s/mhii-pvc.yaml      # Creates PVCs
kubectl apply -f k8s/mhii-job.yaml      # Submits job

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Monitor Execution
# ─────────────────────────────────────────────────────────────────────────────
kubectl get pods -n vpdx3758 -l app=mhii-pipeline

# Watch live logs (from container stdout)
kubectl logs -n vpdx3758 -l app=mhii-pipeline -f --follow

# Or tail the PVC log file
kubectl exec -n vpdx3758 deploy/mhii-pipeline -- tail -f /app/logs/mhii.log

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Retrieve Results (after job completes)
# ─────────────────────────────────────────────────────────────────────────────

# Option A: Copy from PVC to local
kubectl cp vpdx3758/mhii-pipeline-xxxxx:/data/master_model_db.jsonl ./data/

# Option B: List then copy specific pod
POD=$(kubectl get pod -n vpdx3758 -l app=mhii-pipeline -o name | head -1)
kubectl cp vpdx3758/$POD:/data/master_model_db.jsonl ./data/master_model_db.jsonl

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Review Logs (if something went wrong)
# ─────────────────────────────────────────────────────────────────────────────
# Logs are persisted on PVC at /app/logs/mhii.log
kubectl cp vpdx3758/mhii-pipeline-xxxxx:/app/logs/ ./logs-from-cluster/

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Cleanup
# ─────────────────────────────────────────────────────────────────────────────
kubectl delete -f k8s/mhii-job.yaml
kubectl delete -f k8s/mhii-pvc.yaml

# Or wait for TTL (auto-cleanup 1 hour after completion)
```

### Cluster Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│  BHT Cluster (vpdx3758 namespace)                                         │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ mhii-pvc (15Gi)                                                       │ │
│  │ ├── /data/                                                            │ │
│  │ │   ├── master_model_db.jsonl  ← Final output                        │ │
│  │ │   └── temp/                                                         │ │
│  │ └── /logs/                                                            │ │
│  │     └── mhii.log  ← Persisted logs for debugging                     │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ mhii-pipeline Job                                                     │ │
│  │ └── mhii container                                                    │ │
│  │     ├── Phase 1: Selenium scraping (Chrome headless)                  │ │
│  │     ├── Phase 2: HF datasets loading                                  │ │
│  │     ├── Phase 3: Parallel HF API (20 threads)                        │ │
│  │     └── Phase 4-5: Merge & save                                      │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
          │
          │ kubectl cp
          ▼
    Local ./data/
```

### Persistent Logging

**Logs are written to PVC**, not container ephemeral storage. This means:

1. ✅ Logs survive job completion (until TTL cleanup)
2. ✅ Can review logs after job finishes
3. ✅ Can copy logs locally for detailed debugging
4. ✅ Pipeline report available at end of logs

**Log contents include:**
- Phase timing (how long each phase took)
- Progress indicators (X/Y models processed)
- Error details with full stack traces
- HF metadata verification results
- Final pipeline report

---

## Troubleshooting

### Local Development

```bash
# ChromeDriver issues
pip install --upgrade webdriver-manager

# HF Dataset loading
cp .env.example .env
# Edit .env: HF_TOKEN=hf_your_token_here

# Fuzzy matching issues - lower threshold
python -c "from src.services import FuzzyModelMatcher; print(FuzzyModelMatcher(score_threshold=70))"
```

### Cluster Deployment

```bash
# Check job status
kubectl get jobs -n vpdx3758 -l app=mhii-pipeline

# Check pod logs (if job failed to start)
kubectl describe pod -n vpdx3758 -l app=mhii-pipeline

# Check if PVC is bound
kubectl get pvc -n vpdx3758

# View all logs (stdout from container)
kubectl logs -n vpdx3758 -l app=mhii-pipeline --previous

# Debug PVC contents
kubectl exec -n vpdx3758 deploy/mhii-pipeline -- ls -la /app/data/
kubectl exec -n vpdx3758 deploy/mhii-pipeline -- ls -la /app/logs/

# Recreate secret if needed
kubectl delete secret mhii-secrets -n vpdx3758
kubectl create secret generic mhii-secrets --from-literal=HF_TOKEN=hf_xxx -n vpdx3758
```

### Common Issues

| Issue | Solution |
|-------|----------|
| ImagePullBackOff | Check image exists: `docker images | grep mhii` |
| Secret not found | Create secret: `kubectl create secret generic mhii-secrets ...` |
| PVC pending | Wait for binding or check storage class |
| Job stuck | Check pod events: `kubectl describe pod` |
| HF 403 errors | Token may be invalid or rate limited |
| Empty output | Check logs for errors, likely HF API failure |

---

## Local Usage

### Full Pipeline
```bash
python main.py
```

### Partial Pipeline
```bash
python main.py --scrape-only   # Only scrape web data
python main.py --merge-only    # Merge using cached data
python main.py --report        # Generate report from existing data
```

### Custom Output
```bash
python main.py --output ./data/custom_db.jsonl
python main.py --visible       # Show browser during scraping
```

---

## License

Educational and research purposes. Third-party data subject to their respective licenses.

## Credits

- [Artificial Analysis](https://artificialanalysis.ai) - Real-time AI benchmarks
- [OpenEvals](https://huggingface.co/datasets/OpenEvals/leaderboard-data) - Academic benchmarks
- [LMSYS Arena](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset) - Human preference data
- [HuggingFace](https://huggingface.co) - Model metadata & datasets
- [BHT Cluster](https://docs.cluster.ris.bht-berlin.de/) - Data Science cluster