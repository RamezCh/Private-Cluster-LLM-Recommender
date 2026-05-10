"""Data models for MHII pipeline - Single source of truth for all dataclasses."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class GPUConfig:
    """Configuration for a GPU type."""
    name: str
    vram_gb: float
    tier: str
    tensor_cores: bool
    nvlink: bool
    max_power_watts: int
    tflops_fp16: float
    infiniband: bool = False
    cluster: str = "general"


@dataclass
class VRAMRequirements:
    """VRAM requirements for different quantization levels."""
    fp16_gb: float
    int8_gb: float
    int4_gb: float
    model_size_gb: float
    context_overhead_tier: str


@dataclass
class HardwareRecommendation:
    """Hardware deployment recommendation for a GPU configuration."""
    fits_single_gpu: bool
    fits_multi_gpu: bool
    recommended_parallelism: str
    hosting_strategy: str
    requires_sharding: bool
    tensor_parallel_size: Optional[int] = None
    data_parallel_replicas: Optional[int] = None
    pipeline_parallel_size: Optional[int] = None
    moe_experts: Optional[int] = None
    is_moe: bool = False
    context_overhead_tier: str = "standard_32k"


@dataclass
class MultiHardwareFit:
    """Multi-GPU compatibility result."""
    gpu_id: str
    gpu_name: str
    vram_gb: float
    count: int
    total_vram: float
    fits_fp16: bool
    fits_int8: bool
    fits_int4: bool
    recommended_quantization: str
    recommended_parallelism: str
    hosting_strategy: str
    compatibility_score: float
    tier: str


@dataclass
class BenchmarkData:
    """Standardized benchmark scores from all sources."""
    coding: Optional[float] = None
    math: Optional[float] = None
    reasoning: Optional[float] = None
    elo: Optional[float] = None
    intelligence_index: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    vibes_score: Optional[float] = None


@dataclass
class HFModelMetadata:
    """Complete metadata for a model from HuggingFace."""
    model_id: str
    repo_id: Optional[str]
    safetensors_size_gb: float
    parameter_count: Optional[int]
    is_moe: bool
    num_experts: Optional[int]
    model_type: Optional[str]
    library_name: Optional[str]
    tags: List[str]
    metadata_status: str


@dataclass
class ModelMapping:
    """Result of fuzzy matching between model names across sources."""
    canonical_name: str
    sources: Dict[str, str]
    match_score: int
    is_confident: bool


@dataclass
class PerformanceData:
    """Model performance data from Artificial Analysis."""
    model_name: str
    intelligence_index: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    source: str = "artificial_analysis"


@dataclass
class FinalModelRecord:
    """Complete model record for master_model_db.jsonl."""
    model_id: str
    benchmarks: Dict[str, Any]
    vram_gb: Dict[str, float]
    hardware_fit: Dict[str, Any]
    hosting_strategy: str
    source_status: str
    all_gpu_compatibility: Dict[str, Any]
    hf_metadata: Optional[Dict] = None
    source_variants: Optional[Dict] = None
    match_confidence: Optional[int] = None
    raw_data: Optional[Dict] = None


@dataclass
class PipelineReport:
    """Pipeline execution report."""
    total_models: int
    source_status: Dict[str, int]
    architecture_types: Dict[str, int]
    hosting_strategies: Dict[str, int]
    hf_metadata_stats: Dict[str, Any]
    pipeline_errors: List[Dict]
    output_file: str