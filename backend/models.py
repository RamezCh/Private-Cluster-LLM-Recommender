from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUConfig:
    name: str
    vram_gb: float
    tier: str
    tensor_cores: bool
    nvlink: bool


@dataclass
class HardwareInput:
    gpu_config: GPUConfig
    count: int
    total_vram_gb: float
    fits_fp16: bool
    fits_int8: bool
    fits_int4: bool


@dataclass
class ModelRecord:
    model_id: str
    hf_repo_id: Optional[str]
    base_model: Optional[str]
    params_billions: Optional[float]
    model_type: str
    architecture: Optional[str]

    coding: Optional[float]
    math: Optional[float]
    reasoning: Optional[float]
    intelligence_index: Optional[float]

    safetensors_size_gb: float
    vram_fp16: float
    vram_int8: float
    vram_int4: float

    hosting_strategy: str
    is_moe: bool

    semantic_score: float = 0.0
    benchmark_score: float = 0.0
    hardware_score: float = 0.0
    final_score: float = 0.0


@dataclass
class Recommendation:
    model: ModelRecord
    use_case: str
    matched_use_cases: list[str]
    hosting_recommendation: dict
    huggingface_url: str