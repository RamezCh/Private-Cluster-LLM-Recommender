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
│       ├── recommender.py        # Content-based scoring engine
│       ├── collaborative.py      # SVD-based collaborative filtering
│       ├── hybrid_recommender.py # Combines CF + content-based
│       ├── embedding_service.py  # FAISS + sentence-transformers
│       ├── parser.py             # Hardware parsing & use case detection
│       └── wandb_logger.py       # Weights & Biases integration
├── frontend/
│   ├── index.html                # Main page (dark theme, carousel UI, tabs)
│   ├── styles.css                # Dark theme styles, carousel CSS, tab styles
│   └── app.js                    # API calls, carousel controls, form handling, tab switching
├── data_gathering_pipeline/      # Data collection & processing
│   ├── data/
│   │   ├── master_model_db.jsonl # Open-weight model database (~1815 models)
│   │   └── feedback_data.jsonl   # User feedback ratings (~400 records)
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
├── generate_fake_feedback.py     # Generate synthetic feedback data
└── logs/                         # Application logs
```

## How It Works

### Content-Based Recommendations

1. **Hardware Input**: Parse "8x A100" into GPU configuration with VRAM calculation
2. **Use Case Detection**: Keyword matching for coding/math/reasoning
3. **Hardware Filter**: Strict filter - only models that fit user's GPUs
4. **Hybrid Scoring**:
   - 30% Semantic similarity (vector embeddings)
   - 50% Use-case weighted benchmarks
   - 20% Hardware efficiency
5. **Recommendation**: Top-K sorted by final score

### Hybrid Recommender (v2.0)

The system combines content-based filtering with collaborative filtering:

- **Content-Based**: Benchmark scores, hardware fit, semantic similarity (40% weight)
- **Collaborative Filtering**: SVD-based matrix factorization on user feedback (60% weight)
- **Two Views**:
  - "Recommended" - Pure content-based (default, for new users)
  - "For You" - Hybrid personalized recommendations (requires feedback history)

### User Feedback System

Users can rate recommendations (1-5) which:
- Improves personalized recommendations over time
- Trains the collaborative filter
- Generates data for the SVD model

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
Get recommendations with optional hybrid mode.
```json
// Request
{
  "hardware_text": "8 A100s",
  "use_case": "code generation",
  "top_k": 10,
  "mode": "hybrid",     // "pure" or "hybrid"
  "user_id": "user_abc" // optional, for hybrid mode
}

// Response (pure mode)
{
  "success": true,
  "mode": "pure",
  "hardware": {...},
  "use_case": "code generation",
  "recommendations": [...],
  "user_has_feedback": false
}

// Response (hybrid mode)
{
  "success": true,
  "mode": "hybrid",
  "hardware": {...},
  "use_case": "code generation",
  "recommendations": [
    {
      ...,
      "cf_prediction": 4.5,      // SVD-predicted rating
      "cf_confidence": 0.85,     // Confidence based on data density
      "hybrid_score": 0.82,      // Combined score
      "blend_weight": 0.6        // CF weight used
    },
    ...
  ],
  "user_has_feedback": true
}
```

### `POST /feedback`
Submit feedback for a recommendation.
```json
// Request
{
  "user_id": "user_abc123",
  "model_id": "Qwen2.5-72B-Instruct",
  "rating": 4,
  "hardware_used": "8x NVIDIA A100 80GB",
  "use_case": "coding"
}

// Response
{ "success": true, "message": "Thank you for your feedback!" }
```

### `GET /feedback/stats`
Get aggregate feedback statistics.
```json
{
  "success": true,
  "total_feedbacks": 400,
  "avg_rating": 4.47,
  "ratings_distribution": { "1": 0, "2": 0, "3": 16, "4": 180, "5": 204 },
  "ratings_per_model": {
    "Qwen2.5-72B-Instruct": { "count": 100, "total": 457, "avg": 4.57 }
  }
}
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
- **Hybrid Weight (alpha)**: Edit `backend/services/hybrid_recommender.py` (default: 0.6 = 60% CF)
- **SVD Factors**: Edit `backend/services/collaborative.py` `n_factors` (default: 50)
- **Frontend Styles**: Edit `frontend/styles.css` for visual customization
- **Theme**: Dark mode by default (`#0a0a12` background, `#00f0ff` cyan accent)

## Development

### Frontend Structure
- `index.html` - Page structure with hardware selector, showcase carousel, results carousel, tab navigation
- `styles.css` - Dark theme CSS variables, carousel animations, tab styles, mobile responsiveness
- `app.js` - Carousel class, API fetch handlers, form logic, tab switching, feedback submission

### Collaborative Filtering
- `backend/services/collaborative.py` - SVD matrix factorization
  - Loads feedback data on startup
  - Builds user-item rating matrix
  - Predicts ratings for user-model pairs
  - Returns confidence based on data density

### Hybrid Recommender
- `backend/services/hybrid_recommender.py` - Combines CF + content scores
  - Normalizes both score types to [0,1]
  - Blends with configurable alpha (0.6 default)
  - Handles cold start (new users get content-only)

### Generating Fake Feedback Data
```bash
python generate_fake_feedback.py
```
This generates ~100 users with ~400 feedback ratings in `data_gathering_pipeline/data/feedback_data.jsonl`.

## Disclaimer

> [!CAUTION]
> **Scraping Disclaimer**: The OpenCompass Selenium scrapers located in this repository were developed and executed strictly once for **educational purposes** to demonstrate headless browser automation. The scraped OpenCompass records were ultimately deprecated and dropped from the final `master_model_db.jsonl` due to sparse benchmark fill rates. No proprietary data from OpenCompass powers the final recommendation engine.