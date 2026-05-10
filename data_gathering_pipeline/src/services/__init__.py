"""Services module - Business logic for MHII pipeline."""

from src.services.fuzzy_matcher import FuzzyModelMatcher, ModelMapping
from src.services.hf_metadata import HFMetadataService, HFModelMetadata
from src.services.hardware import (
    HardwareService,
    VRAMCalculator,
    check_all_gpu_compatibility,
    format_hardware_summary,
    format_all_fits,
    normalize_model_name,
    is_moe_model,
    parse_model_size,
    estimate_size_from_params,
    get_recommended_context_tier,
)

__all__ = [
    "FuzzyModelMatcher",
    "ModelMapping",
    "HFMetadataService",
    "HFModelMetadata",
    "HardwareService",
    "VRAMCalculator",
    "check_all_gpu_compatibility",
    "format_hardware_summary",
    "format_all_fits",
    "normalize_model_name",
    "is_moe_model",
    "parse_model_size",
    "estimate_size_from_params",
    "get_recommended_context_tier",
]
