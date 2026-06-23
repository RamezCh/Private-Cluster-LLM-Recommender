"""Hardware parsing, use case detection, and scoring. Single service for both."""

import math
import re
from dataclasses import dataclass
from typing import Optional

from config.config import (
    GPU_CATALOG,
    GPU_NAME_MAPPINGS,
    GPU_DISPLAY_NAMES,
    USE_CASE_KEYWORDS,
    DEFAULT_USE_CASE,
    VRAM_MULTIPLIERS,
    KV_CACHE_RESERVE,
    OPTIMAL_VRAM_UTILIZATION,
    VRAM_UTILIZATION_SIGMA,
    QUANT_BONUSES,
)


@dataclass
class ParsedHardware:
    gpu_id: str
    gpu_name: str
    vram_gb: float
    count: int
    total_vram_gb: float
    tier: str


def parse_hardware_input(text: str) -> Optional[ParsedHardware]:
    if not text or not text.strip():
        return None

    original_text = text.lower().strip()
    text = original_text

    count_match = re.match(r'^(\d+)\s*(?:x\s*)?', text)
    count = 1
    if count_match:
        count = int(count_match.group(1))
        text = text[count_match.end():].strip()

    text = re.sub(r'^(?:x|×)\s*', '', text).strip()
    text_normalized = text.replace(" ", "").replace("-", "")

    gpu_id: Optional[str] = None

    for pattern, gid in GPU_NAME_MAPPINGS.items():
        pattern_norm = pattern.lower().replace(" ", "").replace("-", "")
        if pattern_norm in text_normalized or text_normalized in pattern_norm:
            gpu_id = gid
            break

    if not gpu_id:
        for gid, cfg in GPU_CATALOG.items():
            name_norm = cfg.name.lower().replace(" ", "").replace("-", "")
            if name_norm in text_normalized:
                gpu_id = gid
                break

    if not gpu_id:
        keywords = ["a100", "h100", "h200", "b200", "v100", "p100", "rtx", "macbook", "m3", "m2"]
        for kw in keywords:
            if kw in text_normalized:
                for pattern, gid in GPU_NAME_MAPPINGS.items():
                    if kw in pattern.lower():
                        gpu_id = gid
                        break
                if gpu_id:
                    break

    if not gpu_id:
        return None

    config = GPU_CATALOG[gpu_id]

    if "40gb" in original_text and gpu_id == "a100_80gb" and "a100_40gb" in GPU_CATALOG:
        config = GPU_CATALOG["a100_40gb"]
        gpu_id = "a100_40gb"

    return ParsedHardware(
        gpu_id=gpu_id,
        gpu_name=config.name,
        vram_gb=config.vram_gb,
        count=count,
        total_vram_gb=config.vram_gb * count,
        tier=config.tier,
    )


def get_available_gpu_options():
    return GPU_DISPLAY_NAMES


def detect_use_case(text: str) -> tuple[str, dict[str, float]]:
    """Multi-label use case detection with proportional keyword weights.

    Scans the query for ALL matching keywords across every use-case category
    and returns proportional weights based on keyword hit density.

    Returns:
        tuple: (primary_use_case, proportions_dict)
        proportions_dict maps each matched use_case to its proportion (sums to 1.0).
    """
    text_lower = text.lower()
    hits: dict[str, int] = {}

    for use_case, keywords in USE_CASE_KEYWORDS.items():
        if use_case == "general":
            continue
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            hits[use_case] = count

    if not hits:
        return DEFAULT_USE_CASE, {"general": 1.0}

    total_hits = sum(hits.values())
    proportions = {uc: count / total_hits for uc, count in hits.items()}
    primary = max(hits, key=hits.get)

    return primary, proportions


def blend_benchmark_weights(proportions: dict[str, float]) -> dict[str, float]:
    """Blend per-use-case benchmark weights dynamically based on keyword count.

    Directly uses the proportions of matched keywords. If only one category
    is matched, it gets 100% of the weight. General queries get an even split.

    Args:
        proportions: {use_case: proportion} from detect_use_case(), sums to 1.0.

    Returns:
        Blended weights dict with keys: coding, math, reasoning, intelligence_index.
    """
    blended = {"coding": 0.0, "math": 0.0, "reasoning": 0.0, "intelligence_index": 0.0}

    # If general, split evenly
    if "general" in proportions and len(proportions) == 1:
        return {"coding": 0.25, "math": 0.25, "reasoning": 0.25, "intelligence_index": 0.25}

    for uc, proportion in proportions.items():
        if uc in blended:
            blended[uc] = proportion

    # Ensure it always sums to 1.0 (in case of intelligence_index / missing keys)
    total = sum(blended.values())
    if total > 0:
        for k in blended:
            blended[k] /= total
    else:
        return {"coding": 0.25, "math": 0.25, "reasoning": 0.25, "intelligence_index": 0.25}

    return blended


def get_primary_use_case(text: str) -> str:
    uc, _ = detect_use_case(text)
    return uc


def determine_quantization(model_vram: float, total_vram: float) -> str:
    if total_vram >= model_vram * VRAM_MULTIPLIERS["fp16"]:
        return "FP16"
    if total_vram >= model_vram * VRAM_MULTIPLIERS["int8"]:
        return "INT8"
    if total_vram >= model_vram * VRAM_MULTIPLIERS["int4"]:
        return "INT4"
    return "Insufficient"





def calculate_hardware_score(model_vram_fp16: float, user_total_vram: float) -> float:
    """Continuous VRAM-aware hardware score with KV-cache reservation.

    Scores models on a smooth Gaussian curve based on how well they utilize
    the available VRAM after reserving 20% for KV-cache. The peak score is
    at ~85% of usable VRAM. Models that are too small for the hardware are
    penalized (low utilization), and models requiring aggressive quantization
    receive a quality penalty.

    Args:
        model_vram_fp16: Model's VRAM requirement at FP16 precision (GB).
        user_total_vram: User's total available VRAM across all GPUs (GB).

    Returns:
        Score in [0.0, 1.0] where higher means better VRAM utilization.
    """
    if user_total_vram <= 0 or model_vram_fp16 <= 0:
        return 0.0

    usable_vram = user_total_vram * (1 - KV_CACHE_RESERVE)

    # Determine best quantization that fits and effective VRAM usage
    # INT8 ≈ 50% of FP16, INT4 ≈ 25% of FP16
    if model_vram_fp16 <= usable_vram:
        effective_vram = model_vram_fp16
        quant_bonus = QUANT_BONUSES["fp16"]
    elif model_vram_fp16 * 0.5 <= usable_vram:
        effective_vram = model_vram_fp16 * 0.5
        quant_bonus = QUANT_BONUSES["int8"]
    elif model_vram_fp16 * 0.25 <= usable_vram:
        effective_vram = model_vram_fp16 * 0.25
        quant_bonus = QUANT_BONUSES["int4"]
    else:
        return 0.0

    # Utilization ratio: how much of usable VRAM the model consumes
    utilization = effective_vram / usable_vram

    if utilization <= OPTIMAL_VRAM_UTILIZATION:
        util_score = 1.0
    else:
        sigma = VRAM_UTILIZATION_SIGMA  # 0.35 by default -> stricter penalty for overutilization
        util_score = math.exp(
            -((utilization - OPTIMAL_VRAM_UTILIZATION) ** 2)
            / (2 * sigma ** 2)
        )

    return util_score * quant_bonus