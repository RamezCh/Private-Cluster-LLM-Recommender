# LLM Recommender System

A modern web interface for recommending locally-hostable open-weight LLMs based on hardware resources and use case.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python -m uvicorn backend.api:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser.

The system loads with 3 showcase picks immediately. Select your GPU, choose number of results (1-10), enter a use case, and click **Get Recommendations**.

### 3. Run Tests

```bash
pytest tests/
```

## Architecture

```
llm_recommender/
├── backend/
│   ├── api.py                    # FastAPI server (serves frontend + API)
│   ├── logging.py                # Loguru logging setup
│   ├── models.py                 # Pydantic data models
│   └── services/
│       ├── recommender.py        # Hybrid scoring engine
│       ├── embedding_service.py  # FAISS + sentence-transformers
│       ├── parser.py             # Hardware parsing & use case detection
│       └── wandb_logger.py       # Weights & Biases integration
├── frontend/
│   ├── index.html                # Main page (dark theme, carousel UI)
│   ├── styles.css                # Dark theme styles & carousel CSS
│   └── app.js                    # API calls, carousel controls, form handling
├── data_gathering_pipeline/      # Data collection & processing
│   ├── src/
│   │   ├── orchestrator.py       # Pipeline orchestration
│   │   ├── gpu_catalog.py        # GPU specifications
│   │   ├── models.py             # Data models
│   │   ├── config.py             # Configuration
│   │   ├── fetchers/             # Web scrapers & API clients
│   │   └── services/             # Data processing services
│   └── requirements.txt
├── embeddings/
│   └── build_index.py            # Pre-compute FAISS index
├── config/
│   └── config.py                 # GPU catalog, weights, keywords
├── tests/
│   ├── test_cases.py             # Validation test cases
│   ├── test_hardware_parser.py   # Hardware parser tests
│   └── test_api.py               # API endpoint tests
├── data/
│   └── master_model_db.jsonl     # Open-weight model database (~1815 models)
└── logs/                         # Application logs
```

## How It Works

1. **Hardware Input**: Parse "8x A100" into GPU configuration with VRAM calculation
2. **Use Case Detection**: Keyword matching for coding/math/reasoning
3. **Hardware Filter**: Strict filter - only models that fit user's GPUs
4. **Hybrid Scoring**:
   - 30% Semantic similarity (vector embeddings)
   - 50% Use-case weighted benchmarks
   - 20% Hardware efficiency
5. **Recommendation**: Top-K sorted by final score

## API Reference

### `GET /api/showcase`
Returns 3 showcase picks (one per hardware tier) with 1-hour caching.
```json
{
  "success": true,
  "showcase": [
    {
      "category": "laptop",
      "label": "Works on your laptop",
      "hardware": { "gpu_id": "...", "gpu_name": "...", "total_vram_gb": 128, ... },
      "model": { "name": "...", "provider": "...", "quantization": "...", ... }
    },
    ...
  ]
}
```

### `POST /recommend`
Get personalized recommendations based on user hardware and use case.
```json
// Request
{ "hardware_text": "8 A100s", "use_case": "code generation", "top_k": 10 }

// Response
{ "success": true, "hardware": {...}, "recommendations": [...] }
```

### `GET /models/count`
Returns total model count for footer display.

## Test Cases

| Test | Hardware | Use Case | Expected |
|------|----------|----------|----------|
| TC-01 | 8x A100 80GB | Coding | Llama-3.3-70B |
| TC-02 | 1x H200 141GB | Math | Large math models |
| TC-03 | 4x RTX 4090 | Creative writing | 24B-30B models |
| TC-04 | MacBook M3 Max | On-device inference | 7B-14B models |
| TC-05 | 1x A100 40GB | Memory-constrained coding | 7B-30B INT4 |

## Configuration

- **GPU Catalog**: Edit `config/config.py` to add/modify GPU specifications
- **Use Case Keywords**: Modify `USE_CASE_KEYWORDS` in `config/config.py`
- **Benchmark Weights**: Adjust `USE_CASE_BENCHMARK_WEIGHTS` in `config/config.py`
- **Frontend Styles**: Edit `frontend/styles.css` for visual customization
- **Theme**: Dark mode by default (`#0d1117` background, `#58a6ff` accent)

## Development

### Frontend Structure
- `index.html` - Page structure with hardware selector, showcase carousel, results carousel
- `styles.css` - Dark theme CSS variables, carousel animations, mobile responsiveness
- `app.js` - Carousel class, API fetch handlers, form logic, touch/keyboard support