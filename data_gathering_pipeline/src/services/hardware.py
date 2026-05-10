"""Hardware utility functions for VRAM calculations and parallelism recommendations."""

import re
from typing import Dict, Optional, Tuple, List

from loguru import logger

from src.config import VRAM_CONFIG
from src.gpu_catalog import GPU_CATALOG, GPU_TIERS
from src.models import VRAMRequirements, HardwareRecommendation, MultiHardwareFit


def normalize_model_name(name: str) -> str:
    """Normalize model names for consistent matching."""
    name = name.lower()
    name = re.sub(r"[_\-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"instruct|chat|preview|beta", "", name)
    return name.strip()


def is_moe_model(
    model_name: str, config_tags: Optional[List[str]] = None
) -> Tuple[bool, Optional[int]]:
    """Detect if a model is Mixture-of-Experts."""
    model_lower = model_name.lower()

    moe_indicators = [
        "moe",
        "mixture",
        "deepseek-v3",
        "deepseek-v4",
        "qwen-moe",
        "mixtral",
        "mixtral-8x7b",
        "dbrx",
        "switch-transformer",
    ]

    if any(ind in model_lower for ind in moe_indicators):
        return True, None

    if config_tags and any(tag in ["moe", "mixture-of-experts"] for tag in config_tags):
        return True, None

    return False, None


def parse_model_size(name: str) -> Optional[float]:
    """Extract parameter count from model name (in billions)."""
    patterns = [
        r"(\d+(?:\.\d+)?)\s*[Bb]",
        r"(\d+)B",
        r"-(\d+(?:\.\d+)?)[Bb]",
    ]

    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            size = float(match.group(1))
            if 0.5 <= size <= 400:
                return size
    return None


def estimate_size_from_params(
    param_count: Optional[int] = None, size_bytes: Optional[int] = None
) -> float:
    """Estimate model size in GB (FP16 = 2 bytes per parameter)."""
    if size_bytes:
        return size_bytes / (1024**3)
    if param_count:
        return param_count * 2 / (1024**3)
    return 0.0


class VRAMCalculator:
    """Calculate VRAM requirements for different quantization levels."""

    def __init__(self, context_tier: str = "standard_32k"):
        self.context_tier = context_tier
        self.multiplier = VRAM_CONFIG.kv_cache_multipliers.get(context_tier, 1.2)

    def calculate(self, model_size_gb: float) -> VRAMRequirements:
        """Calculate VRAM requirements with context overhead."""
        fp16 = model_size_gb * self.multiplier

        return VRAMRequirements(
            fp16_gb=round(fp16, 2),
            int8_gb=round(fp16 * 0.5, 2),
            int4_gb=round(fp16 * 0.25, 2),
            model_size_gb=model_size_gb,
            context_overhead_tier=self.context_tier,
        )


def get_recommended_context_tier(
    intelligence_index: Optional[float] = None, model_name: Optional[str] = None
) -> str:
    """Recommend context tier based on model characteristics."""
    if intelligence_index and intelligence_index >= 90:
        return "extended_128k"

    if model_name:
        lower = model_name.lower()
        if any(ind in lower for ind in ["1m", "200k"]):
            return "ultra_1m"
        if any(ind in lower for ind in ["128k", "long", "context"]):
            return "extended_128k"

    return "standard_32k"


class HardwareService:
    """Determine optimal hardware deployment strategy for models."""

    def __init__(self, gpu_id: str = "a100_80gb", gpu_count: int = 8):
        self.gpu_id = gpu_id
        self.gpu_count = gpu_count
        self.gpu = GPU_CATALOG.get(gpu_id, GPU_CATALOG["a100_80gb"])
        self.tier_info = GPU_TIERS.get(self.gpu.tier, {})

    def determine_strategy(
        self,
        vram_req: VRAMRequirements,
        is_moe: bool = False,
        num_experts: Optional[int] = None,
    ) -> HardwareRecommendation:
        """Determine optimal deployment strategy for a given GPU configuration."""
        total_vram = self.gpu.vram_gb * self.gpu_count

        fits_single = vram_req.int4_gb <= self.gpu.vram_gb
        fits_multi = vram_req.int4_gb <= total_vram or vram_req.fp16_gb <= total_vram

        if is_moe:
            ep_note = "" if self.tier_info.get("nvlink_support") else " (degraded)"
            return HardwareRecommendation(
                fits_single_gpu=False,
                fits_multi_gpu=fits_multi,
                recommended_parallelism=f"Expert Parallelism (EP={self.gpu_count}){ep_note}",
                hosting_strategy="Expert-Distributed",
                requires_sharding=True,
                tensor_parallel_size=self.gpu_count if fits_multi else None,
                moe_experts=num_experts,
                is_moe=True,
                context_overhead_tier=vram_req.context_overhead_tier,
            )

        if fits_single:
            return HardwareRecommendation(
                fits_single_gpu=True,
                fits_multi_gpu=True,
                recommended_parallelism="Single-GPU",
                hosting_strategy="Single-GPU",
                requires_sharding=False,
                context_overhead_tier=vram_req.context_overhead_tier,
            )

        if fits_multi:
            tp_opt = " (Optimal)" if self.tier_info.get("nvlink_support") else " (PCIe)"

            tp_size = self.gpu_count
            if vram_req.fp16_gb <= self.gpu.vram_gb:
                tp_size = 1
            elif vram_req.fp16_gb <= self.gpu.vram_gb * 2:
                tp_size = min(2, self.gpu_count)
            elif vram_req.fp16_gb <= self.gpu.vram_gb * 4:
                tp_size = min(4, self.gpu_count)

            return HardwareRecommendation(
                fits_single_gpu=False,
                fits_multi_gpu=True,
                recommended_parallelism=f"Tensor Parallelism (TP={tp_size}){tp_opt}",
                hosting_strategy="TP-Sharded",
                requires_sharding=True,
                tensor_parallel_size=tp_size,
                is_moe=False,
                context_overhead_tier=vram_req.context_overhead_tier,
            )

        return HardwareRecommendation(
            fits_single_gpu=False,
            fits_multi_gpu=False,
            recommended_parallelism=f"Exceeds capacity ({total_vram}GB)",
            hosting_strategy="Unsupported",
            requires_sharding=True,
            is_moe=is_moe,
            context_overhead_tier=vram_req.context_overhead_tier,
        )


def check_all_gpu_compatibility(
    vram_req: VRAMRequirements, is_moe: bool = False, limit: int = 20
) -> List[MultiHardwareFit]:
    """Check model compatibility across all GPU types."""
    results = []

    for gpu_id, gpu in GPU_CATALOG.items():
        for count in [1, 2, 4, 8]:
            if gpu.tier == "laptop" and count > 2:
                continue

            total_vram = gpu.vram_gb * count

            if vram_req.int4_gb > total_vram and vram_req.fp16_gb > total_vram:
                continue

            fits_fp16 = vram_req.fp16_gb <= total_vram
            quant = "FP16" if fits_fp16 else "INT4"

            tier_info = GPU_TIERS.get(gpu.tier, {})

            if is_moe:
                nvlink_bonus = 30 if tier_info.get("nvlink_support") else 0
                score = 100 + nvlink_bonus
                strategy = "Expert-Distributed"
                parallelism = f"Expert Parallelism (EP={count})"
            elif fits_fp16 and vram_req.fp16_gb <= gpu.vram_gb:
                score = 110 if gpu.tier == "data_center" else 90
                strategy = "Single-GPU"
                parallelism = (
                    "Single-GPU" if count == 1 else f"Data Parallelism (DP={count})"
                )
            else:
                score = 90 if tier_info.get("nvlink_support") else 60
                strategy = "TP-Sharded"
                parallelism = f"Tensor Parallelism (TP={count})"

            if gpu.tier == "data_center":
                score *= 1.1
            elif gpu.tier == "laptop":
                score *= 0.7

            results.append(
                MultiHardwareFit(
                    gpu_id=gpu_id,
                    gpu_name=gpu.name,
                    vram_gb=gpu.vram_gb,
                    count=count,
                    total_vram=total_vram,
                    fits_fp16=fits_fp16,
                    fits_int8=vram_req.int8_gb <= total_vram,
                    fits_int4=vram_req.int4_gb <= total_vram,
                    recommended_quantization=quant,
                    recommended_parallelism=parallelism,
                    hosting_strategy=strategy,
                    compatibility_score=min(100, round(score, 1)),
                    tier=gpu.tier,
                )
            )

    results.sort(key=lambda x: x.compatibility_score, reverse=True)
    return results[:limit]


def format_hardware_summary(
    vram_req: VRAMRequirements, is_moe: bool = False, num_experts: Optional[int] = None
) -> Dict:
    """Format hardware fit information for a specific GPU."""
    service = HardwareService()
    strategy = service.determine_strategy(vram_req, is_moe, num_experts)

    return {
        "gpu_id": service.gpu_id,
        "gpu_name": service.gpu.name,
        "gpu_count": service.gpu_count,
        "total_vram_gb": service.gpu.vram_gb * service.gpu_count,
        "status": "Compatible" if strategy.fits_multi_gpu else "Incompatible",
        "recommended_parallelism": strategy.recommended_parallelism,
        "multi_gpu_scaling": strategy.tensor_parallel_size is not None or is_moe,
        "tensor_parallel_size": strategy.tensor_parallel_size,
        "moe_experts": num_experts,
        "is_moe_model": is_moe,
        "hosting_strategy": strategy.hosting_strategy,
        "context_overhead_tier": strategy.context_overhead_tier,
        "tier": service.gpu.tier,
    }


def format_all_fits(vram_req: VRAMRequirements, is_moe: bool = False) -> Dict:
    """Format hardware fit for all GPU configurations."""
    compatible = check_all_gpu_compatibility(vram_req, is_moe)

    by_tier = {
        tier: []
        for tier in ["data_center", "professional", "consumer", "laptop", "legacy"]
    }

    for fit in compatible:
        by_tier[fit.tier].append(
            {
                "gpu_name": fit.gpu_name,
                "count": fit.count,
                "total_vram": fit.total_vram,
                "quantization": fit.recommended_quantization,
                "parallelism": fit.recommended_parallelism,
                "strategy": fit.hosting_strategy,
                "score": fit.compatibility_score,
            }
        )

    return {
        "all_compatible_gpus": [
            {
                "name": f.gpu_name,
                "count": f.count,
                "vram": f.total_vram,
                "score": f.compatibility_score,
            }
            for f in compatible[:10]
        ],
        "by_tier": by_tier,
        "best_data_center": (
            by_tier["data_center"][0] if by_tier["data_center"] else None
        ),
        "best_consumer": by_tier["consumer"][0] if by_tier["consumer"] else None,
        "best_laptop": by_tier["laptop"][0] if by_tier["laptop"] else None,
    }
