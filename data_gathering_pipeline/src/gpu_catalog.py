"""GPU Catalog - re-exports from central config for backward compatibility.

This module re-exports GPU catalog items from config.gpu_catalog for
backward compatibility with existing imports in the data pipeline.
"""

from pathlib import Path
import sys

from config import (
    GPU_CATALOG,
    GPU_NAME_MAPPINGS,
    GPUConfig,
)

def get_gpu_config(gpu_id: str) -> GPUConfig:
    return GPU_CATALOG.get(gpu_id.lower(), GPU_CATALOG["a100_80gb"])

def get_all_gpus_by_tier(tier: str) -> dict[str, GPUConfig]:
    return {k: v for k, v in GPU_CATALOG.items() if v.tier == tier}

def get_total_vram(gpu_id: str, count: int = 1) -> float:
    gpu = get_gpu_config(gpu_id)
    return gpu.vram_gb * count

__all__ = [
    "GPU_CATALOG",
    "GPU_NAME_MAPPINGS",
    "GPUConfig",
    "GPU_TIERS",
    "get_gpu_config",
    "get_all_gpus_by_tier",
    "get_total_vram",
]


GPU_TIERS: dict = {
    "data_center": {
        "nvlink_support": True,
        "multi_gpu_optimized": True,
        "infiniband_support": True,
        "examples": ["A100", "H100", "H200", "B200", "V100", "P100"],
    },
    "legacy": {
        "nvlink_support": False,
        "multi_gpu_optimized": False,
        "infiniband_support": False,
        "examples": ["Tesla K80"],
    },
    "consumer": {
        "nvlink_support": False,
        "multi_gpu_optimized": False,
        "infiniband_support": False,
        "examples": ["RTX 4090", "RTX 3090", "RTX 4080"],
    },
    "professional": {
        "nvlink_support": True,
        "multi_gpu_optimized": True,
        "infiniband_support": False,
        "examples": ["RTX A6000", "RTX A5000", "RTX A4000"],
    },
    "laptop": {
        "nvlink_support": False,
        "multi_gpu_optimized": False,
        "infiniband_support": False,
        "examples": ["Laptop RTX 4070", "MacBook Pro M3 Max"],
    },
}