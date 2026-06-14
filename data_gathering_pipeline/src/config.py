"""Configuration constants for the Open Source LLM Recommender pipeline v2."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
_root = BASE_DIR.parent
if (_root / ".env").exists():
    load_dotenv(_root / ".env")
else:
    load_dotenv()
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
TEMP_DATA_DIR = DATA_DIR / "temp"

DATA_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DATA_DIR.mkdir(exist_ok=True, parents=True)

HF_TOKEN = os.getenv("HF_TOKEN", "")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


HF_DATASETS = {
    "open_llm_leaderboard": "open-llm-leaderboard/contents",
}


OPEN_WEIGHT_FILTER = {
    "min_benchmarks": 3,
    "preferred_types": ["💬", "🟢"],
    "exclude_generation_threshold": None,
}


PROPRIETARY_FILTER = {
    "model_name_prefixes": [
        "gpt-", "claude-", "gemini-", "command-",
        "grok-", "nova-", "o1-", "o3-", "claude",
        "gpt4", "claude3",
    ],
    "provider_excludes": [
        "OpenAI", "Anthropic", "Google AI", "Google DeepMind",
        "Cohere", "Mistral AI", "xAI",
    ],
}


OPEN_WEIGHT_ORGS = {
    "meta-llama", "mistralai", "Qwen", "deepseek-ai", "01-ai",
    "google", "microsoft", "databricks", "nvidia", "apple",
    "allenai", "bigcode", "bigscience", "EleutherAI", "tiiuae",
    "falcon", "xverse", "sambanova", "ai21", "cohere",
    "stability-ai", " Wizardmath", "openchat", "xwin-lm",
    "lmsys", "NousResearch", "huggingfaceh4", "open-llm-leaderboard",
}


DEDUP_STRATEGY = {
    "group_by": "Base Model",
    "score_variant_by": "benchmark_completeness",
    "tie_breakers": ["generation", "type_preference"],
    "type_preference_order": ["💬", "🟢", "🔶", "🤝", "💻", "🔢", "⚡"],
}


BENCHMARK_CONFIG = {
    "source_columns": {
        "open_llm_leaderboard": [
            "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO", "Average"
        ],
    },
    "target_keys": {
        "coding": ["IFEval", "HumanEval", "MBPP"],
        "math": ["MATH Lvl 5", "GSM8K", "MATH"],
        "reasoning": ["BBH", "MMLU-PRO", "GPQA", "DROP", "MUSR", "C-Eval", "MMLU"],
        "intelligence_index": ["Average", "overall", "Overall Score"],
        "elo": [],
    },
}


OPENCOMPASS_CONFIG = {
    "general": {
        "url": "https://rank.opencompass.org.cn/leaderboard-llm/",
        "default_month": "26-04",
        "month_format": "%y-%m",
    },
    "academic": {
        "url": "https://rank.opencompass.org.cn/leaderboard-llm-academic/",
        "default_month": "REALTIME",
    },
}


SELENIUM_CONFIG = {
    "implicit_wait": 10,
    "page_load_timeout": 60,
    "headless": True,
}


OUTPUT_FILE = DATA_DIR / "master_model_db.jsonl"
TEMP_OPENCOMPASS_GENERAL = TEMP_DATA_DIR / "temp_oc_general.jsonl"
TEMP_OPENCOMPASS_ACADEMIC = TEMP_DATA_DIR / "temp_oc_academic.jsonl"


logging_config = {
    "rotation": os.getenv("LOGURU_ROTATION", "10 MB"),
    "retention": os.getenv("LOGURU_RETENTION", "7 days"),
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    "level": os.getenv("LOGURU_LEVEL", "INFO"),
}


VRAM_CONFIG = {
    "multipliers": {
        "fp16": 1.2,
        "int8": 0.6,
        "int4": 0.3,
    },
    "kv_cache_multipliers": {
        "standard_32k": 1.2,
        "extended_128k": 1.5,
        "ultra_1m": 2.5,
    },
}


MODEL_TYPE_PATTERNS = {
    "moe": [
        "moe", "mixture", "deepseek-v3", "deepseek-v4",
        "qwen-moe", "mixtral", "dbrx",
    ],
    "dense": [
        "llama", "gpt", "gemma", "qwen", "mistral",
        "yi", "command", "command-r",
    ],
}


from src.gpu_catalog import get_gpu_config, get_all_gpus_by_tier, get_total_vram


def is_proprietary_model(name: str, provider: str = "") -> bool:
    """Check if a model name or provider indicates proprietary/closed-source."""
    combined = f"{name} {provider}".lower()

    for prefix in PROPRIETARY_FILTER["model_name_prefixes"]:
        if prefix.lower() in combined:
            return True

    for provider_excl in PROPRIETARY_FILTER["provider_excludes"]:
        if provider_excl.lower() in combined.lower():
            return True

    return False