"""Configuration constants for the MHII data gathering pipeline."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
TEMP_DATA_DIR = DATA_DIR / "temp"

DATA_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DATA_DIR.mkdir(exist_ok=True, parents=True)

HF_TOKEN = os.getenv("HF_TOKEN", "")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


@dataclass
class SeleniumConfig:
    url: str = "https://artificialanalysis.ai/leaderboards/models"
    implicit_wait: int = 10
    page_load_timeout: int = 60
    headless: bool = True


@dataclass
class FuzzyConfig:
    score_threshold: int = 85
    scorer: str = "token_set_ratio"


@dataclass
class VRAMConfig:
    multipliers: Dict[str, float] = None
    kv_cache_multipliers: Dict[str, float] = None

    def __post_init__(self):
        self.multipliers = {
            "fp16": 1.2,
            "int8": 0.6,
            "int4": 0.3,
        }
        self.kv_cache_multipliers = {
            "standard_32k": 1.2,
            "extended_128k": 1.5,
            "ultra_1m": 2.5,
        }


SELENIUM_CONFIG = SeleniumConfig()
FUZZY_CONFIG = FuzzyConfig()
VRAM_CONFIG = VRAMConfig()

HF_DATASETS = {
    "open_evals": "OpenEvals/leaderboard-data",
    "lmsys_arena": "lmarena-ai/leaderboard-dataset",
}

MODEL_TYPE_PATTERNS = {
    "moe": [
        "moe",
        "mixture",
        "deepseek-v3",
        "deepseek-v4",
        "qwen-moe",
        "mixtral",
        "dbrx",
    ],
    "dense": [
        "llama",
        "gpt",
        "claude",
        "gemini",
        "gemma",
        "qwen",
        "mistral",
        "yi",
        "command",
        "command-r",
    ],
}

OUTPUT_FILE = DATA_DIR / "master_model_db.jsonl"
TEMP_PERFORMANCE_FILE = TEMP_DATA_DIR / "temp_performance.jsonl"

logging_config = {
    "rotation": os.getenv("LOGURU_ROTATION", "10 MB"),
    "retention": os.getenv("LOGURU_RETENTION", "7 days"),
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    "level": os.getenv("LOGURU_LEVEL", "INFO"),
}


def get_gpu_config(gpu_id: str) -> "GPUConfig":
    from src.gpu_catalog import GPU_CATALOG

    return GPU_CATALOG.get(gpu_id.lower(), GPU_CATALOG["a100_80gb"])


def get_all_gpus_by_tier(tier: str) -> Dict[str, "GPUConfig"]:
    from src.gpu_catalog import GPU_CATALOG

    return {k: v for k, v in GPU_CATALOG.items() if v.tier == tier}


def get_total_vram(gpu_id: str, count: int = 1) -> float:
    gpu = get_gpu_config(gpu_id)
    return gpu.vram_gb * count
