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
class OpenWeightModelRecord:
    """The new master record schema — one per locally-hostable model."""

    model_id: str
    hf_repo_id: Optional[str] = None
    base_model: Optional[str] = None
    model_type: str = "unknown"
    architecture: Optional[str] = None
    precision: Optional[str] = None

    params_billions: Optional[float] = None
    safetensors_size_gb: float = 0.0

    benchmarks: BenchmarkData = field(default_factory=BenchmarkData)

    extended_benchmarks: Dict[str, Any] = field(default_factory=dict)

    is_moe: bool = False
    num_experts: Optional[int] = None

    license: Optional[str] = None
    hub_likes: int = 0
    generation: int = 0

    vram_gb: Dict[str, float] = field(default_factory=dict)
    hardware_fit: Dict[str, Any] = field(default_factory=dict)
    hosting_strategy: str = "unknown"

    source_status: str = "unknown"
    all_gpu_compatibility: Dict[str, Any] = field(default_factory=dict)
    hf_metadata: Optional[Dict] = None
    match_confidence: Optional[int] = None

    raw_data: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "hf_repo_id": self.hf_repo_id,
            "base_model": self.base_model,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "precision": self.precision,
            "params_billions": self.params_billions,
            "safetensors_size_gb": self.safetensors_size_gb,
            "benchmarks": {
                "coding": self.benchmarks.coding,
                "math": self.benchmarks.math,
                "reasoning": self.benchmarks.reasoning,
                "elo": self.benchmarks.elo,
                "intelligence_index": self.benchmarks.intelligence_index,
            },
            "extended_benchmarks": self.extended_benchmarks,
            "is_moe": self.is_moe,
            "num_experts": self.num_experts,
            "license": self.license,
            "hub_likes": self.hub_likes,
            "generation": self.generation,
            "vram_gb": self.vram_gb,
            "hardware_fit": self.hardware_fit,
            "hosting_strategy": self.hosting_strategy,
            "source_status": self.source_status,
            "all_gpu_compatibility": self.all_gpu_compatibility,
            "hf_metadata": self.hf_metadata,
            "match_confidence": self.match_confidence,
        }


@dataclass
class OpenLLMLeaderboardRow:
    """Normalized row from open-llm-leaderboard dataset."""

    model: str
    fullname: str
    base_model: Optional[str]
    params_billions: float
    average: float
    is_moe: bool
    architecture: str
    precision: str
    model_type: str
    license: str
    hub_likes: int
    generation: int
    available_on_hub: bool
    flagged: bool
    chat_template: bool
    merged: bool
    official_providers: bool
    benchmarks: Dict[str, float] = field(default_factory=dict)


@dataclass
class OpenCompassRow:
    """Normalized row from OpenCompass leaderboard scraper."""

    model_name: str
    provider: Optional[str]
    overall_score: Optional[float]
    benchmarks: Dict[str, float] = field(default_factory=dict)
    rank: Optional[int] = None
    submission_date: Optional[str] = None
    source: str = "general"


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
    dedup_stats: Dict[str, int] = field(default_factory=dict)
    benchmark_coverage: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelMapping:
    """Result of fuzzy matching between model names across sources."""

    canonical_name: str
    sources: Dict[str, str] = field(default_factory=dict)
    match_score: int = 100
    is_confident: bool = True