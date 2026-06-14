"""Hardware parsing and use case detection. Single service for both."""

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


def detect_use_case(text: str) -> tuple[str, list[str]]:
    text_lower = text.lower()
    matched = []
    for use_case, keywords in USE_CASE_KEYWORDS.items():
        if use_case == "general":
            continue
        for kw in keywords:
            if kw in text_lower:
                if use_case not in matched:
                    matched.append(use_case)
                break
    if not matched:
        return DEFAULT_USE_CASE, []
    return matched[0], matched


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


def calculate_benchmark_score(coding: float, math: float, reasoning: float,
                               intelligence_index: float, use_case: str) -> float:
    from config.config import USE_CASE_BENCHMARK_WEIGHTS
    w = USE_CASE_BENCHMARK_WEIGHTS.get(use_case, USE_CASE_BENCHMARK_WEIGHTS["general"])
    score = (w["coding"] * (coding or 0) + w["math"] * (math or 0) +
             w["reasoning"] * (reasoning or 0) + w["intelligence_index"] * (intelligence_index or 0))
    return score / 100.0


def calculate_hardware_score(model_vram: float, user_total_vram: float) -> float:
    if user_total_vram >= model_vram * VRAM_MULTIPLIERS["fp16"]:
        return 1.0
    if user_total_vram >= model_vram * VRAM_MULTIPLIERS["int8"]:
        return 0.8
    if user_total_vram >= model_vram * VRAM_MULTIPLIERS["int4"]:
        return 0.6
    return 0.0