"""Hybrid LLM recommender engine. Simple, efficient, works."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.logger import get_logger
from backend.services.parser import (
    ParsedHardware, determine_quantization, calculate_benchmark_score,
    calculate_hardware_score, detect_use_case, parse_hardware_input
)
from backend.services.embedding_service import get_embedding_service
from backend.services.wandb_logger import get_wandb_logger
from config.config import (
    DB_PATH, DEFAULT_USE_CASE, SEMANTIC_WEIGHT, BENCHMARK_WEIGHT, HARDWARE_WEIGHT,
    TOP_K_RECOMMENDATIONS, GPU_CATALOG, USE_CASE_BENCHMARK_WEIGHTS,
)

logger = get_logger(__name__)


def _determine_quant(model_vram: float, total_vram: float) -> str:
    from config.config import VRAM_MULTIPLIERS as VM
    if total_vram >= model_vram * VM["fp16"]:
        return "FP16"
    if total_vram >= model_vram * VM["int8"]:
        return "INT8"
    if total_vram >= model_vram * VM["int4"]:
        return "INT4"
    return "Insufficient"


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
    semantic_score: float = 0.0
    benchmark_score: float = 0.0
    hardware_score: float = 0.0
    final_score: float = 0.0
    matched_hardware: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "hf_repo_id": self.hf_repo_id,
            "base_model": self.base_model,
            "params_billions": self.params_billions,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "benchmarks": {
                "coding": round(self.coding, 2),
                "math": round(self.math_score, 2),
                "reasoning": round(self.reasoning, 2),
                "intelligence_index": round(self.intelligence_index, 2),
            },
            "vram_fp16_gb": round(self.vram_fp16, 2),
            "vram_int8_gb": round(self.vram_int8, 2),
            "vram_int4_gb": round(self.vram_int4, 2),
            "hosting_strategy": self.hosting_strategy,
            "is_moe": self.is_moe,
            "scores": {
                "semantic": round(self.semantic_score, 3),
                "benchmark": round(self.benchmark_score, 3),
                "hardware": round(self.hardware_score, 3),
                "final": round(self.final_score, 3),
            },
            "matched_hardware": self.matched_hardware,
        }


class LLMRecommender:
    def __init__(
        self,
        db_path: Optional[str] = None,
        semantic_weight: float = SEMANTIC_WEIGHT,
        benchmark_weight: float = BENCHMARK_WEIGHT,
        hardware_weight: float = HARDWARE_WEIGHT,
    ):
        self.db_path = db_path or DB_PATH
        self.models: list[dict] = []
        self.semantic_weight = semantic_weight
        self.benchmark_weight = benchmark_weight
        self.hardware_weight = hardware_weight
        self._wandb = get_wandb_logger()
        self._load_models()

    def _load_models(self) -> None:
        if not Path(self.db_path).exists():
            logger.warning(f"Model database not found: {self.db_path}")
            return
        with open(self.db_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    self.models.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        logger.info(f"Loaded {len(self.models)} models")

    def _hw_info(self, model: dict, hw: ParsedHardware) -> dict:
        total_vram = hw.total_vram_gb
        vram_fp16 = model.get("vram_gb", {}).get("fp16", 0)
        quant = _determine_quant(vram_fp16, total_vram)

        if quant == "FP16":
            return {
                "status": "Compatible (FP16)",
                "quantization": "FP16",
                "parallelism": "Single-GPU" if hw.count == 1 else f"TP-{hw.count}",
                "strategy": "TP-Sharded" if hw.count > 1 else "Single-GPU",
                "score": 100,
            }
        if quant == "INT8":
            return {
                "status": "Compatible (INT8)",
                "quantization": "INT8",
                "parallelism": "Quantized",
                "strategy": "Single-GPU" if hw.count == 1 else f"DP-{hw.count}",
                "score": 80,
            }
        if quant == "INT4":
            return {
                "status": "Compatible (INT4)",
                "quantization": "INT4",
                "parallelism": "Quantized",
                "strategy": "Single-GPU" if hw.count == 1 else f"DP-{hw.count}",
                "score": 60,
            }
        return {"status": "Insufficient VRAM", "quantization": "N/A",
                "parallelism": "N/A", "strategy": "N/A", "score": 0}

    def recommend(
        self,
        hardware: ParsedHardware,
        use_case_text: str,
        user_query: str,
        top_k: int = TOP_K_RECOMMENDATIONS,
    ) -> list[ScoredModel]:
        start = time.time()

        if len(use_case_text) > 5000:
            use_case_text = use_case_text[:5000]
        if len(user_query) > 5000:
            user_query = user_query[:5000]

        use_case, _ = detect_use_case(use_case_text)
        if not use_case or use_case == DEFAULT_USE_CASE:
            if use_case_text.strip():
                use_case = DEFAULT_USE_CASE

        total_vram = hardware.total_vram_gb
        compatible = [
            m for m in self.models
            if _determine_quant(m.get("vram_gb", {}).get("fp16", 0), total_vram) != "Insufficient"
        ]

        if not compatible:
            logger.warning(f"No models fit {hardware.gpu_name} ({total_vram}GB)")
            return []

        emb_svc = get_embedding_service()
        sem_results = {}
        if emb_svc.index is not None:
            for mid, score in emb_svc.search(user_query, top_k=100):
                sem_results[mid] = score

        weights = USE_CASE_BENCHMARK_WEIGHTS.get(use_case, USE_CASE_BENCHMARK_WEIGHTS["general"])
        scored = []

        for model in compatible:
            mid = model.get("model_id", "")
            sem_score = sem_results.get(mid, 0.5)

            bmarks = model.get("benchmarks", {})
            c = bmarks.get("coding", 0) or 0
            m = bmarks.get("math", 0) or 0
            r = bmarks.get("reasoning", 0) or 0
            ii = bmarks.get("intelligence_index", 0) or 0
            bm_score = (
                weights["coding"] * c + weights["math"] * m +
                weights["reasoning"] * r + weights["intelligence_index"] * ii
            ) / 100.0

            vram_fp16 = model.get("vram_gb", {}).get("fp16", 0)
            hw_score = calculate_hardware_score(vram_fp16, total_vram)

            final = (
                self.semantic_weight * sem_score +
                self.benchmark_weight * bm_score +
                self.hardware_weight * hw_score
            )

            scored.append(ScoredModel(
                model_id=mid,
                hf_repo_id=model.get("hf_repo_id"),
                base_model=model.get("base_model"),
                params_billions=model.get("params_billions") or 0,
                model_type=model.get("model_type", "unknown"),
                architecture=model.get("architecture"),
                coding=c,
                math_score=m,
                reasoning=r,
                intelligence_index=ii,
                safetensors_size_gb=model.get("safetensors_size_gb") or 0,
                vram_fp16=vram_fp16,
                vram_int8=model.get("vram_gb", {}).get("int8", 0),
                vram_int4=model.get("vram_gb", {}).get("int4", 0),
                hosting_strategy=model.get("hosting_strategy", "unknown"),
                is_moe=model.get("is_moe", False),
                semantic_score=sem_score,
                benchmark_score=bm_score,
                hardware_score=hw_score,
                final_score=final,
                matched_hardware=self._hw_info(model, hardware),
            ))

        scored.sort(key=lambda x: x.final_score, reverse=True)

        latency_ms = (time.time() - start) * 1000
        try:
            if self._wandb.enabled:
                top = scored[0] if scored else None
                self._wandb.log_recommendation(
                    query=user_query[:1000],
                    hardware=f"{hardware.count}x {hardware.gpu_name}",
                    use_case=use_case,
                    num_compatible=len(compatible),
                    num_returned=len(scored[:top_k]),
                    top_model=top.model_id if top else "N/A",
                    top_model_score=top.final_score if top else 0.0,
                    latency_ms=latency_ms,
                )
        except Exception as e:
            logger.warning(f"W&B logging failed: {e}")

        logger.info(f"Done in {latency_ms:.1f}ms, {len(scored[:top_k])} returned")
        return scored[:top_k]

    @property
    def model_count(self) -> int:
        return len(self.models)

    def get_showcase_picks(self) -> list[dict]:
        """Get top model per hardware tier for showcase display."""
        showcase_configs = [
            {
                "category": "laptop",
                "label": "Works on your laptop",
                "hardware_text": "MacBook Pro M3 Max",
            },
            {
                "category": "consumer",
                "label": "Budget workstation",
                "hardware_text": "2x RTX 4090",
            },
            {
                "category": "data_center",
                "label": "Data center powerhouse",
                "hardware_text": "2x A100 40GB",
            },
        ]

        picks = []
        for cfg in showcase_configs:
            hw = parse_hardware_input(cfg["hardware_text"])
            if hw is None:
                continue

            results = self.recommend(
                hardware=hw,
                use_case_text="general purpose assistant",
                user_query="best general purpose open weight LLM",
                top_k=1,
            )

            if results:
                model_dict = results[0].to_dict()
                picks.append({
                    "category": cfg["category"],
                    "label": cfg["label"],
                    "hardware": {
                        "gpu_id": hw.gpu_id,
                        "gpu_name": hw.gpu_name,
                        "vram_gb": hw.vram_gb,
                        "count": hw.count,
                        "total_vram_gb": hw.total_vram_gb,
                        "tier": hw.tier,
                    },
                    "model": model_dict,
                })

        return picks


_recommender: Optional[LLMRecommender] = None
_recommender_lock = None


def get_recommender() -> LLMRecommender:
    global _recommender
    if _recommender is None:
        global _recommender_lock
        if _recommender_lock is None:
            import threading
            _recommender_lock = threading.Lock()
        with _recommender_lock:
            if _recommender is None:
                _recommender = LLMRecommender()
    return _recommender


def reset_recommender() -> None:
    global _recommender
    _recommender = None