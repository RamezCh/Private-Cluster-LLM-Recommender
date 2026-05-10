"""GPU Catalog - Complete list of supported hardware configurations."""

from typing import Dict
from dataclasses import dataclass

from src.models import GPUConfig

GPU_CATALOG: Dict[str, GPUConfig] = {
    # BHT Cluster GPUs
    "a100_40gb": GPUConfig(
        "A100 40GB", 40, "data_center", True, True, 400, 312, True, "bht"
    ),
    "a100_80gb": GPUConfig(
        "A100 80GB", 80, "data_center", True, True, 400, 624, True, "bht"
    ),
    "h100_80gb": GPUConfig(
        "H100 80GB", 80, "data_center", True, True, 700, 989, True, "bht"
    ),
    "h100_sxm5_80gb": GPUConfig(
        "H100 SXM5 80GB", 80, "data_center", True, True, 700, 989, True, "bht"
    ),
    "v100_16gb": GPUConfig(
        "V100 16GB", 16, "data_center", True, True, 300, 125, True, "bht"
    ),
    "v100_32gb": GPUConfig(
        "V100 32GB", 32, "data_center", True, True, 300, 125, True, "bht"
    ),
    "p100_16gb": GPUConfig(
        "Tesla P100 16GB", 16, "data_center", False, False, 250, 80, False, "bht"
    ),
    "k80_12gb": GPUConfig(
        "Tesla K80 12GB", 12, "legacy", False, False, 300, 40, False, "bht"
    ),
    # Data Center GPUs (General)
    "h200_141gb": GPUConfig(
        "H200 141GB", 141, "data_center", True, True, 700, 989, True, "general"
    ),
    "b200_192gb": GPUConfig(
        "B200 192GB", 192, "data_center", True, True, 1000, 1728, True, "general"
    ),
    "b100_192gb": GPUConfig(
        "B100 192GB", 192, "data_center", True, True, 1000, 1440, True, "general"
    ),
    "mi300x_192gb": GPUConfig(
        "MI300X 192GB", 192, "data_center", True, True, 750, 1307, True, "general"
    ),
    "mi250_128gb": GPUConfig(
        "MI250X 128GB", 128, "data_center", True, True, 500, 761, True, "general"
    ),
    # Professional GPUs
    "a6000_48gb": GPUConfig(
        "RTX A6000 48GB", 48, "professional", False, True, 300, 310, False, "general"
    ),
    "a5000_24gb": GPUConfig(
        "RTX A5000 24GB", 24, "professional", False, True, 230, 160, False, "general"
    ),
    "a4000_16gb": GPUConfig(
        "RTX A4000 16GB", 16, "professional", False, False, 140, 150, False, "general"
    ),
    # Consumer GPUs
    "rtx_4090": GPUConfig(
        "RTX 4090 24GB", 24, "consumer", False, False, 450, 330, False, "general"
    ),
    "rtx_3090": GPUConfig(
        "RTX 3090 24GB", 24, "consumer", False, False, 350, 320, False, "general"
    ),
    "rtx_4080": GPUConfig(
        "RTX 4080 16GB", 16, "consumer", False, False, 320, 200, False, "general"
    ),
    "rtx_4070": GPUConfig(
        "RTX 4070 12GB", 12, "consumer", False, False, 200, 150, False, "general"
    ),
    "rtx_3090_ti": GPUConfig(
        "RTX 3090 Ti 24GB", 24, "consumer", False, False, 450, 355, False, "general"
    ),
    "rtx_4080_s": GPUConfig(
        "RTX 4080 Super 16GB", 16, "consumer", False, False, 320, 210, False, "general"
    ),
    "rtx_4070_ti": GPUConfig(
        "RTX 4070 Ti Super 16GB",
        16,
        "consumer",
        False,
        False,
        285,
        180,
        False,
        "general",
    ),
    # Laptop GPUs
    "laptop_rtx_4060": GPUConfig(
        "Laptop RTX 4060 8GB", 8, "laptop", False, False, 140, 120, False, "general"
    ),
    "laptop_rtx_4070": GPUConfig(
        "Laptop RTX 4070 8GB", 8, "laptop", False, False, 140, 130, False, "general"
    ),
    "laptop_rtx_4080": GPUConfig(
        "Laptop RTX 4080 12GB", 12, "laptop", False, False, 175, 190, False, "general"
    ),
    "laptop_3090": GPUConfig(
        "Laptop RTX 3090 16GB", 16, "laptop", False, False, 200, 250, False, "general"
    ),
    "macbook_pro_m3_max": GPUConfig(
        "MacBook Pro M3 Max", 128, "laptop", False, False, 92, 300, False, "general"
    ),
    "macbook_pro_m2_max": GPUConfig(
        "MacBook Pro M2 Max", 96, "laptop", False, False, 86, 270, False, "general"
    ),
    "macbook_pro_m3_pro": GPUConfig(
        "MacBook Pro M3 Pro", 36, "laptop", False, False, 70, 180, False, "general"
    ),
    "macbook_pro_m2_pro": GPUConfig(
        "MacBook Pro M2 Pro", 36, "laptop", False, False, 65, 150, False, "general"
    ),
    "macbook_pro_m3": GPUConfig(
        "MacBook Pro M3", 24, "laptop", False, False, 60, 120, False, "general"
    ),
    "laptop_integrated": GPUConfig(
        "Integrated GPU", 4, "laptop", False, False, 30, 20, False, "general"
    ),
}


GPU_TIERS: Dict[str, Dict] = {
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


BHT_CLUSTER_GPUS: Dict[str, Dict] = {
    "a100_40gb": {
        "label": "a100",
        "vram": "40GB",
        "best_for": "Large models, long contexts",
    },
    "a100_80gb": {"label": "a100", "vram": "80GB", "best_for": "Very large models"},
    "h100_80gb": {
        "label": "h100",
        "vram": "80GB",
        "best_for": "Largest models, longest contexts",
    },
    "v100_32gb": {"label": "v100", "vram": "32GB", "best_for": "Medium models"},
    "p100_16gb": {"label": "p100", "vram": "16GB", "best_for": "Smaller models"},
    "k80_12gb": {"label": "k80", "vram": "12GB", "best_for": "Legacy (avoid)"},
}


def get_all_gpus_by_cluster(cluster: str) -> Dict[str, GPUConfig]:
    return {k: v for k, v in GPU_CATALOG.items() if v.cluster == cluster}


def get_bht_cluster_gpus() -> Dict[str, GPUConfig]:
    return get_all_gpus_by_cluster("bht")
