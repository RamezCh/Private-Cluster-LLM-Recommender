# LLM Recommender System

A chat-based interface for recommending locally-hostable open-weight LLMs based on hardware resources and use case.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build Embeddings (Optional - done automatically on first run)

```bash
python embeddings/build_index.py
```

### 3. Run the Application

**Streamlit Frontend:**
```bash
cd frontend
streamlit run main.py
```

**API Server:**
```bash
cd backend
uvicorn api:app --reload --port 8000
```

### 4. Run Tests

```bash
pytest tests/
```

## Architecture

```
llm_recommender/
├── backend/
│   ├── api.py                    # FastAPI server
│   ├── logging.py                # Loguru logging setup
│   ├── models.py                 # Pydantic data models
│   └── services/
│       ├── recommender.py        # Hybrid scoring engine
│       ├── embedding_service.py  # FAISS + sentence-transformers
│       ├── parser.py             # Hardware parsing & use case detection
│       └── wandb_logger.py       # Weights & Biases integration
├── frontend/
│   └── main.py                   # Streamlit chat interface
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
├── .streamlit/
│   └── config.toml               # Streamlit configuration
├── data/
│   └── master_model_db.jsonl     # Open-weight model database
├── logs/                         # Application logs
└── wandb/                        # Weights & Biases runs
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

## Test Cases

| Test | Hardware | Use Case | Expected |
|------|----------|----------|----------|
| TC-01 | 8x A100 80GB | Coding | Llama-3.3-70B |
| TC-02 | 1x H200 141GB | Math | Large math models |
| TC-03 | 4x RTX 4090 | Creative writing | 24B-30B models |
| TC-04 | MacBook M3 Max | On-device inference | 7B-14B models |
| TC-05 | 1x A100 40GB | Memory-constrained coding | 7B-30B INT4 |

## W&B Integration

Set `WANDB_API_KEY` environment variable to enable experiment tracking.

```bash
export WANDB_API_KEY=your_key
streamlit run frontend/main.py
```

## Configuration

- **GPU Catalog**: Edit `config/config.py` to add/modify GPU specifications
- **Use Case Keywords**: Modify `USE_CASE_KEYWORDS` in `config/config.py`
- **Benchmark Weights**: Adjust `USE_CASE_BENCHMARK_WEIGHTS` in `config/config.py`
- **Streamlit Settings**: Edit `.streamlit/config.toml`