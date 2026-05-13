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

## 4. Data Quality Metrics (as of 2026-05-14)
**Total Records:** 1,815 open-weight models

### Benchmark Fill Rates
| Benchmark | Coverage | Note |
|-----------|----------|------|
| Coding | 99.7% | High coverage |
| Math | 97.7% | High coverage |
| Reasoning | 100.0% | Complete |
| Intelligence Index | 100.0% | Complete |
| ELO | 0% | LMSYS removed; OpenCompass not yet scraped |

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
The pipeline reduced the initial dataset from 2,254 to 1,815 records by:
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
