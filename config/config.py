"""Central config for LLM Recommender. Single source of truth."""

from dataclasses import dataclass
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DB_PATH = str(ROOT / "data_gathering_pipeline" / "data" / "master_model_db.jsonl")
EMBEDDINGS_DIR = str(ROOT / "embeddings")
EMBEDDING_INDEX_PATH = str(ROOT / "embeddings" / "faiss_index.bin")
EMBEDDING_DATA_PATH = str(ROOT / "embeddings" / "model_data.json")

# ─── VRAM ─────────────────────────────────────────────────────────────────────
VRAM_MULTIPLIERS = {
    "fp16": 1.2,
    "int8": 0.6,
    "int4": 0.3,
}

# ─── VRAM Efficiency Scoring ──────────────────────────────────────────────────
KV_CACHE_RESERVE = 0.20           # Reserve 20% of VRAM for KV-cache
OPTIMAL_VRAM_UTILIZATION = 0.60   # Penalize models utilizing less than 60% of total VRAM
VRAM_UTILIZATION_SIGMA = 0.50     # Broad Gaussian spread for a small penalty on underutilization
QUANT_BONUSES = {"fp16": 1.0, "int8": 0.90, "int4": 0.75}

# ─── Benchmark Imputation ─────────────────────────────────────────────────────
IMPUTATION_K = 5  # Nearest neighbors for missing benchmark imputation

# ─── Scoring Weights ──────────────────────────────────────────────────────────
SEMANTIC_WEIGHT = 0.30
BENCHMARK_WEIGHT = 0.50
HARDWARE_WEIGHT = 0.20
TOP_K_RECOMMENDATIONS = 5
DEFAULT_USE_CASE = "general"


USE_CASE_KEYWORDS = {
    "coding": ["code", "coding", "programming", "developer", "software", "debug", "debugging",
               "python", "javascript", "java", "script", "algorithm", "function", "class",
               "code generation", "code review", "refactor", "git", "repository"],
    "math": ["math", "mathematics", "maths", "calculus", "algebra", "equation", "numerical",
             "calculate", "physics", "statistics", "probability", "linear algebra", "matrix",
             "vector", "derivative", "integral", "optimization", "theorem"],
    "reasoning": ["reason", "reasoning", "logic", "logical", "think", "thinking", "analyze",
                  "analysis", "solve", "problem solving", "puzzle", "deduction", "inference",
                  "critical thinking", "decision", "strategy", "planning", "goal"],
    "intelligence_index": ["knowledge", "intelligence", "trivia", "facts", "information", 
                           "general knowledge", "q&a", "question", "answer", "creative", 
                           "writing", "brainstorm"],
    "general": [],
}

# ─── GPU Catalog ──────────────────────────────────────────────────────────────
GPU_TIERS = {
    "data_center": {"nvlink_support": True, "multi_gpu_optimized": True, "infiniband_support": True,
                    "examples": ["A100", "H100", "H200", "B200", "V100", "P100"]},
    "professional": {"nvlink_support": True, "multi_gpu_optimized": True, "infiniband_support": False,
                     "examples": ["RTX A6000", "RTX A5000", "RTX A4000"]},
    "consumer": {"nvlink_support": False, "multi_gpu_optimized": False, "infiniband_support": False,
                 "examples": ["RTX 4090", "RTX 3090", "RTX 4080"]},
    "laptop": {"nvlink_support": False, "multi_gpu_optimized": False, "infiniband_support": False,
               "examples": ["Laptop RTX 4070", "MacBook Pro M3 Max"]},
}


@dataclass
class GPUConfig:
    name: str
    vram_gb: float
    tier: str
    tensor_cores: bool
    nvlink: bool
    max_power_watts: int
    tflops_fp16: float
    infiniband: bool
    cluster: str = "general"


GPU_CATALOG: dict[str, GPUConfig] = {
    "a100_40gb": GPUConfig("A100 40GB", 40, "data_center", True, True, 400, 312, True, "bht"),
    "a100_80gb": GPUConfig("A100 80GB", 80, "data_center", True, True, 400, 624, True, "bht"),
    "h100_80gb": GPUConfig("H100 80GB", 80, "data_center", True, True, 700, 989, True, "bht"),
    "h100_sxm5_80gb": GPUConfig("H100 SXM5 80GB", 80, "data_center", True, True, 700, 989, True, "bht"),
    "h200_141gb": GPUConfig("H200 141GB", 141, "data_center", True, True, 700, 989, True, "general"),
    "b200_192gb": GPUConfig("B200 192GB", 192, "data_center", True, True, 1000, 1728, True, "general"),
    "b100_192gb": GPUConfig("B100 192GB", 192, "data_center", True, True, 1000, 1440, True, "general"),
    "v100_16gb": GPUConfig("V100 16GB", 16, "data_center", True, True, 300, 125, True, "bht"),
    "v100_32gb": GPUConfig("V100 32GB", 32, "data_center", True, True, 300, 125, True, "bht"),
    "p100_16gb": GPUConfig("Tesla P100 16GB", 16, "data_center", False, False, 250, 80, False, "bht"),
    "mi300x_192gb": GPUConfig("MI300X 192GB", 192, "data_center", True, True, 750, 1307, True, "general"),
    "mi250_128gb": GPUConfig("MI250X 128GB", 128, "data_center", True, True, 500, 761, True, "general"),
    "a6000_48gb": GPUConfig("RTX A6000 48GB", 48, "professional", False, True, 300, 310, False, "general"),
    "a5000_24gb": GPUConfig("RTX A5000 24GB", 24, "professional", False, True, 230, 160, False, "general"),
    "a4000_16gb": GPUConfig("RTX A4000 16GB", 16, "professional", False, False, 140, 150, False, "general"),
    "rtx_4090": GPUConfig("RTX 4090 24GB", 24, "consumer", False, False, 450, 330, False, "general"),
    "rtx_3090": GPUConfig("RTX 3090 24GB", 24, "consumer", False, False, 350, 320, False, "general"),
    "rtx_4080": GPUConfig("RTX 4080 16GB", 16, "consumer", False, False, 320, 200, False, "general"),
    "rtx_4070": GPUConfig("RTX 4070 12GB", 12, "consumer", False, False, 200, 150, False, "general"),
    "rtx_3090_ti": GPUConfig("RTX 3090 Ti 24GB", 24, "consumer", False, False, 450, 355, False, "general"),
    "rtx_4080_s": GPUConfig("RTX 4080 Super 16GB", 16, "consumer", False, False, 320, 210, False, "general"),
    "rtx_4070_ti": GPUConfig("RTX 4070 Ti Super 16GB", 16, "consumer", False, False, 285, 180, False, "general"),
    "laptop_rtx_4060": GPUConfig("Laptop RTX 4060 8GB", 8, "laptop", False, False, 140, 120, False, "general"),
    "laptop_rtx_4070": GPUConfig("Laptop RTX 4070 8GB", 8, "laptop", False, False, 140, 130, False, "general"),
    "laptop_rtx_4080": GPUConfig("Laptop RTX 4080 12GB", 12, "laptop", False, False, 175, 190, False, "general"),
    "laptop_3090": GPUConfig("Laptop RTX 3090 16GB", 16, "laptop", False, False, 200, 250, False, "general"),
    "macbook_pro_m3_max": GPUConfig("MacBook Pro M3 Max 128GB", 128, "laptop", False, False, 92, 300, False, "general"),
    "macbook_pro_m2_max": GPUConfig("MacBook Pro M2 Max 96GB", 96, "laptop", False, False, 86, 270, False, "general"),
    "macbook_pro_m3_pro": GPUConfig("MacBook Pro M3 Pro 36GB", 36, "laptop", False, False, 70, 180, False, "general"),
    "macbook_pro_m2_pro": GPUConfig("MacBook Pro M2 Pro 36GB", 36, "laptop", False, False, 65, 150, False, "general"),
    "macbook_pro_m3": GPUConfig("MacBook Pro M3 24GB", 24, "laptop", False, False, 60, 120, False, "general"),
    "laptop_integrated": GPUConfig("Integrated GPU 4GB", 4, "laptop", False, False, 30, 20, False, "general"),
}

GPU_NAME_MAPPINGS: dict[str, str] = {
    "a100": "a100_80gb", "a100 40gb": "a100_40gb", "a100 80gb": "a100_80gb",
    "h100": "h100_80gb", "h100 80gb": "h100_80gb",
    "h200": "h200_141gb", "h200 141gb": "h200_141gb",
    "b200": "b200_192gb", "b100": "b100_192gb",
    "v100": "v100_32gb", "v100 32gb": "v100_32gb", "v100 16gb": "v100_16gb",
    "p100": "p100_16gb",
    "mi300x": "mi300x_192gb", "mi250": "mi250_128gb",
    "a6000": "a6000_48gb", "a5000": "a5000_24gb", "a4000": "a4000_16gb",
    "rtx 4090": "rtx_4090", "rtx4090": "rtx_4090",
    "rtx 3090": "rtx_3090", "rtx3090": "rtx_3090",
    "rtx 4080": "rtx_4080", "rtx4080": "rtx_4080",
    "rtx 4070": "rtx_4070", "rtx4070": "rtx_4070",
    "3090 ti": "rtx_3090_ti", "3090ti": "rtx_3090_ti",
    "4080 super": "rtx_4080_s", "4080s": "rtx_4080_s",
    "4070 ti": "rtx_4070_ti", "4070ti": "rtx_4070_ti",
    "m3 max": "macbook_pro_m3_max", "m2 max": "macbook_pro_m2_max",
    "m3 pro": "macbook_pro_m3_pro", "m2 pro": "macbook_pro_m2_pro",
    "m3": "macbook_pro_m3",
    "macbook pro m2 pro": "macbook_pro_m2_pro",
    "laptop rtx 4060": "laptop_rtx_4060",
    "laptop rtx 4070": "laptop_rtx_4070",
    "laptop rtx 4080": "laptop_rtx_4080",
    "laptop 3090": "laptop_3090",
}

GPU_DISPLAY_NAMES = [
    ("A100 40GB", "a100_40gb"), ("A100 80GB", "a100_80gb"), ("H100 80GB", "h100_80gb"),
    ("H100 SXM5 80GB", "h100_sxm5_80gb"), ("H200 141GB", "h200_141gb"), ("B200 192GB", "b200_192gb"),
    ("B100 192GB", "b100_192gb"), ("V100 32GB", "v100_32gb"), ("V100 16GB", "v100_16gb"),
    ("P100 16GB", "p100_16gb"), ("MI300X 192GB", "mi300x_192gb"), ("MI250X 128GB", "mi250_128gb"),
    ("RTX A6000 48GB", "a6000_48gb"), ("RTX A5000 24GB", "a5000_24gb"), ("RTX A4000 16GB", "a4000_16gb"),
    ("RTX 4090 24GB", "rtx_4090"), ("RTX 3090 24GB", "rtx_3090"), ("RTX 4080 16GB", "rtx_4080"),
    ("RTX 4070 12GB", "rtx_4070"), ("MacBook Pro M3 Max 128GB", "macbook_pro_m3_max"),
    ("MacBook Pro M2 Max 96GB", "macbook_pro_m2_max"), ("MacBook Pro M3 Pro 36GB", "macbook_pro_m3_pro"),
    ("MacBook Pro M3 24GB", "macbook_pro_m3"),
]