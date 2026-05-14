import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.weights import (
    USE_CASE_BENCHMARK_WEIGHTS,
    SEMANTIC_WEIGHT,
    BENCHMARK_WEIGHT,
    HARDWARE_WEIGHT,
    TOP_K_RECOMMENDATIONS,
)
from .embedding_service import get_embedding_service, EmbeddingService
from .hardware_parser import ParsedHardware
from .wandb_logger import WandbLogger


@dataclass
class ScoredModel:
    model_id: str
    hf_repo_id: Optional[str]
    base_model: Optional[str]
    params_billions: float
    model_type: str
    architecture: Optional[str]
    coding: float
    math_score: float
    reasoning: float
    intelligence_index: float
    safetensors_size_gb: float
    vram_fp16: float
    vram_int8: float
    vram_int4: float
    hosting_strategy: str
    is_moe: bool
    semantic_score: float
    benchmark_score: float
    hardware_score: float
    final_score: float
    matched_hardware: dict


def calculate_benchmark_score(
    coding: float,
    math: float,
    reasoning: float,
    intelligence_index: float,
    use_case: str
) -> float:
    weights = USE_CASE_BENCHMARK_WEIGHTS.get(
        use_case,
        USE_CASE_BENCHMARK_WEIGHTS["general"]
    )
    
    score = (
        weights["coding"] * (coding or 0) +
        weights["math"] * (math or 0) +
        weights["reasoning"] * (reasoning or 0) +
        weights["intelligence_index"] * (intelligence_index or 0)
    )
    
    return score / 100.0


def calculate_hardware_score(
    model_vram: float,
    user_total_vram: float
) -> float:
    if user_total_vram >= model_vram:
        return 1.0
    
    needed_for_int8 = model_vram * 0.5
    needed_for_int4 = model_vram * 0.25
    
    if user_total_vram >= needed_for_int8:
        return 0.8
    elif user_total_vram >= needed_for_int4:
        return 0.6
    else:
        return 0.0


class LLMRecommender:
    def __init__(
        self,
        db_path: Optional[str] = None,
        wandb_logger: Optional[WandbLogger] = None
    ):
        self.db_path = db_path or "data_gathering_pipeline/data/master_model_db.jsonl"
        self.models: list[dict] = []
        self.embedding_service: Optional[EmbeddingService] = None
        self.wandb_logger = wandb_logger
        
        self._load_models()
    
    def _load_models(self) -> None:
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Model database not found: {self.db_path}")
        
        with open(self.db_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    self.models.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    def _init_embeddings(self) -> None:
        if self.embedding_service is None:
            self.embedding_service = get_embedding_service()
    
    def find_compatible_models(
        self,
        hardware: ParsedHardware
    ) -> list[dict]:
        compatible = []
        total_vram = hardware.total_vram_gb
        
        for model in self.models:
            vram_fp16 = model.get("vram_gb", {}).get("fp16", 0)
            
            if total_vram >= vram_fp16:
                compatible.append((model, "FP16"))
            elif total_vram >= vram_fp16 * 0.5:
                compatible.append((model, "INT8"))
            elif total_vram >= vram_fp16 * 0.25:
                compatible.append((model, "INT4"))
            else:
                compatible.append((model, "Insufficient"))
        
        return [m for m, q in compatible if q != "Insufficient"]
    
    def recommend(
        self,
        hardware: ParsedHardware,
        use_case_text: str,
        user_query: str,
        top_k: int = TOP_K_RECOMMENDATIONS
    ) -> list[ScoredModel]:
        start_time = time.time()
        
        self._init_embeddings()
        
        compatible_models = self.find_compatible_models(hardware)
        
        if not compatible_models:
            return []
        
        use_case = self._detect_use_case_from_text(use_case_text)
        
        semantic_results = {}
        if self.embedding_service and self.embedding_service.index is not None:
            search_results = self.embedding_service.search(user_query, top_k=100)
            semantic_results = {mid: score for mid, score in search_results}
        
        scored_models = []
        
        for model in compatible_models:
            model_id = model.get("model_id", "")
            
            semantic_score = semantic_results.get(model_id, 0.5)
            
            benchmarks = model.get("benchmarks", {})
            benchmark_score = calculate_benchmark_score(
                coding=benchmarks.get("coding"),
                math=benchmarks.get("math"),
                reasoning=benchmarks.get("reasoning"),
                intelligence_index=benchmarks.get("intelligence_index"),
                use_case=use_case
            )
            
            vram_fp16 = model.get("vram_gb", {}).get("fp16", 0)
            hardware_score = calculate_hardware_score(
                model_vram=vram_fp16,
                user_total_vram=hardware.total_vram_gb
            )
            
            final_score = (
                SEMANTIC_WEIGHT * semantic_score +
                BENCHMARK_WEIGHT * benchmark_score +
                HARDWARE_WEIGHT * hardware_score
            )
            
            matched_hardware = self._get_matched_hardware_info(model, hardware)
            
            scored_models.append(ScoredModel(
                model_id=model_id,
                hf_repo_id=model.get("hf_repo_id"),
                base_model=model.get("base_model"),
                params_billions=model.get("params_billions") or 0,
                model_type=model.get("model_type", "unknown"),
                architecture=model.get("architecture"),
                coding=benchmarks.get("coding") or 0,
                math_score=benchmarks.get("math") or 0,
                reasoning=benchmarks.get("reasoning") or 0,
                intelligence_index=benchmarks.get("intelligence_index") or 0,
                safetensors_size_gb=model.get("safetensors_size_gb") or 0,
                vram_fp16=vram_fp16,
                vram_int8=model.get("vram_gb", {}).get("int8", 0),
                vram_int4=model.get("vram_gb", {}).get("int4", 0),
                hosting_strategy=model.get("hosting_strategy", "unknown"),
                is_moe=model.get("is_moe", False),
                semantic_score=semantic_score,
                benchmark_score=benchmark_score,
                hardware_score=hardware_score,
                final_score=final_score,
                matched_hardware=matched_hardware
            ))
        
        scored_models.sort(key=lambda x: x.final_score, reverse=True)
        
        if self.wandb_logger:
            self.wandb_logger.log_recommendation(
                query=user_query,
                hardware=f"{hardware.count}x {hardware.gpu_name}",
                use_case=use_case,
                num_results=len(scored_models[:top_k]),
                top_model=scored_models[0].model_id if scored_models else "N/A"
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return scored_models[:top_k]
    
    def _detect_use_case_from_text(self, text: str) -> str:
        text_lower = text.lower()
        
        scores = {}
        for use_case, keywords in {
            "coding": ["code", "programming", "developer", "software", "debug", "python", "javascript"],
            "math": ["math", "calculus", "equation", "physics", "calculate", "numerical"],
            "reasoning": ["reason", "logic", "analyze", "solve", "think", "strategy"],
        }.items():
            scores[use_case] = sum(1 for kw in keywords if kw in text_lower)
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return "general"
    
    def _get_matched_hardware_info(
        self,
        model: dict,
        hardware: ParsedHardware
    ) -> dict:
        gpu_id = hardware.gpu_id
        gpu_count = hardware.count
        total_vram = hardware.total_vram_gb
        
        all_compat = model.get("all_gpu_compatibility", {})
        by_tier = all_compat.get("by_tier", {})
        
        for tier_key in ["data_center", "professional", "consumer", "laptop"]:
            tier_data = by_tier.get(tier_key, [])
            for config in tier_data:
                if hardware.gpu_name in config.get("gpu_name", ""):
                    count = config.get("count", 0)
                    if count == gpu_count:
                        return {
                            "status": "Compatible",
                            "quantization": config.get("quantization", "Unknown"),
                            "parallelism": config.get("parallelism", "Unknown"),
                            "strategy": config.get("strategy", "Unknown"),
                            "score": config.get("score", 0)
                        }
        
        vram_fp16 = model.get("vram_gb", {}).get("fp16", 0)
        if total_vram >= vram_fp16:
            return {
                "status": "Compatible (FP16)",
                "quantization": "FP16",
                "parallelism": "Single-GPU" if gpu_count == 1 else f"TP-{gpu_count}",
                "strategy": "TP-Sharded" if gpu_count > 1 else "Single-GPU",
                "score": 100
            }
        elif total_vram >= vram_fp16 * 0.5:
            return {
                "status": "Compatible (INT8)",
                "quantization": "INT8",
                "parallelism": "Quantized",
                "strategy": "Single-GPU" if gpu_count == 1 else f"DP-{gpu_count}",
                "score": 80
            }
        elif total_vram >= vram_fp16 * 0.25:
            return {
                "status": "Compatible (INT4)",
                "quantization": "INT4",
                "parallelism": "Quantized",
                "strategy": "Single-GPU" if gpu_count == 1 else f"DP-{gpu_count}",
                "score": 60
            }
        
        return {
            "status": "Insufficient VRAM",
            "quantization": "N/A",
            "parallelism": "N/A",
            "strategy": "N/A",
            "score": 0
        }


_recommender: Optional[LLMRecommender] = None


def get_recommender() -> LLMRecommender:
    global _recommender
    if _recommender is None:
        _recommender = LLMRecommender()
    return _recommender