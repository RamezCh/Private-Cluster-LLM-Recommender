# Data Quality Documentation: OpenCode LLM Recommender

## 1. Project Purpose
The goal of this project is to recommend locally-hostable open-weight Large Language Models (LLMs) based on a user's available hardware (VRAM) and specific use cases (Coding, Math, Reasoning, etc.).

## 2. Data Sources
- **HuggingFace `open-llm-leaderboard`**: Primary source for model records and core benchmarks.
- **OpenCompass (Planned)**: General and Academic leaderboards for additional benchmark enrichment.

## 3. Data Acquisition Pipeline
The pipeline follows a 7-phase flow:
1. **Load OLLM**: Fetch records from the HuggingFace open-llm-leaderboard dataset.
2. **Filter**: Remove proprietary models using a blocklist to ensure only open-weight models remain.
3. **Deduplication**: Group variants by normalized base model name and select the best variant based on benchmark completeness.
4. **Merge**: Combine benchmarks across sources with priority-based rules (e.g., OpenCompass > OLLM for coding/math).
5. **HF Enrich**: Fetch real-time metadata (size, tags) from the HuggingFace Hub API.
6. **VRAM Calculation**: Apply VRAM formulas based on parameter count and architecture.
7. **Save**: Export the final processed data to `master_model_db.jsonl`.

## 4. Data Quality Metrics (as of 2026-06-23)
**Total Records:** 1,996 open-weight models

### Benchmark Fill Rates
| Field | Fill Rate (Before Imputation) | Fill Rate (After Imputation) | Missing Count (After) | Imputation Strategy |
|-------|-------------------------------|------------------------------|-----------------------|---------------------|
| HF Repo ID | ~50% | ~50% | ~975 | N/A (Optional lookup) |
| Coding Benchmark | ~78% | **100%** | **0** | **k-NN (k=5) Imputed** |
| Math Benchmark | ~75% | **100%** | **0** | **k-NN (k=5) Imputed** |
| Reasoning Benchmark| ~85% | **100%** | **0** | **k-NN (k=5) Imputed** |
| Intel. Benchmark | ~80% | **100%** | **0** | **k-NN (k=5) Imputed** |
| Params | >99% | >99% | <5 | Dropped in final build (Hardware VRAM formula requires a parameter count) |

## 5. Automated k-NN Benchmark Imputation

To ensure comprehensive model evaluation without penalizing otherwise-capable models for missing a specific benchmark, MHII v2 employs an **Inverse-Distance Weighted k-Nearest Neighbors (k-NN)** imputation strategy directly within the Data Gathering Pipeline (`Orchestrator._impute_missing_benchmarks`).

* **k=5**: The algorithm finds the 5 most statistically similar models (calculated via Euclidean distance across the *present* benchmarks).
* **Inverse-Distance Weighting**: Closer neighbors exert a proportionally higher mathematical influence on the imputed score.
* **Static Execution**: By applying this during `master_model_db.jsonl` generation rather than at application startup, we completely eliminated the 10-15 second cold start delay.

### HF Metadata Status
| Status | Value | Description |
|--------|-------|-------------|
| Verified | 85.0% | Confirmed via HF Hub API response |
| Rate Limited / Estimated | 15.0% | Size estimated as `params * 2`; no distinct HF repo found |
| Missing | 0% | All records have some form of metadata |

### Model Architecture
- **Dense Models**: 1,768 (97.4%)
- **MoE Models**: 47 (2.6%)

## 5. Size Estimation Logic
For models where the HuggingFace repo could not be resolved (`source_status = "rate_limited_estimated"`), the following estimation is used:
- **Formula**: `safetensors_size_gb = params_billions * 2`
- **Assumption**: Standard FP16 precision (2 bytes per parameter).
- **Accuracy**: Spot-checks indicate this matches expected values for community merges and fine-tunes.

## 6. Deduplication Strategy
The pipeline reduced the raw dataset down to 1,996 records by:
- Normalizing model names to identify base models.
- Grouping all variants of a base model.
- Selecting the "best" variant—defined as the one with the most complete benchmark data.

## 7. Known Gaps & Limitations
- **ELO Scores**: Currently missing due to the removal of LMSYS data and pending OpenCompass scraping.
- **Parameter Counts**: 5 models were missing `params_billions` and required name-parsing fallbacks for VRAM estimation.
- **Estimated Metadata**: ~15% of models rely on parameter-based size estimation rather than actual file sizes from the Hub.

## 8. Maintenance & Refresh
- **Cache**: To perform a full refresh, clear `hf_service.failed_lookups` and `hf_service.cache` in the orchestrator.
- **Rate Limiting**: Parallel workers are limited to 5 to avoid HF Hub 429 errors.
- **Frequency**: Data should be refreshed periodically to capture new model releases on the Open LLM Leaderboard.

## 9. Data Distributions (as of 2026-06-23)
Total Models Analyzed: 1996

| Benchmark | Mean | Median | Mode | Min | Max | Missing Values |
|-----------|------|--------|------|-----|-----|----------------|
| **Coding** | 45.71 | 44.92 | 59.61 | 0.39 | 89.98 | 0 |
| **Math** | 15.59 | 9.74 | 1.36 | 0.08 | 71.45 | 0 |
| **Reasoning** | 28.41 | 29.47 | 48.67 | 0.87 | 76.70 | 0 |
| **Intelligence Index** | 21.98 | 21.43 | 22.74 | 0.74 | 51.23 | 0 |