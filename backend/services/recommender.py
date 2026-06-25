"""Hybrid LLM recommender engine. Simple, efficient, works."""

import bisect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.logger import get_logger
from backend.services.parser import (
    ParsedHardware, determine_quantization,
    calculate_hardware_score, detect_use_case, parse_hardware_input,
    blend_benchmark_weights,
)
from backend.services.embedding_service import get_embedding_service
from backend.services.wandb_logger import get_wandb_logger
from config.config import (
    DB_PATH, DEFAULT_USE_CASE, SEMANTIC_WEIGHT, BENCHMARK_WEIGHT, HARDWARE_WEIGHT,
    TOP_K_RECOMMENDATIONS, GPU_CATALOG,
    KV_CACHE_RESERVE, IMPUTATION_K,
)

logger = get_logger(__name__)


def _determine_quant(model_base_gb: float, total_vram: float) -> str:
    from config.config import VRAM_MULTIPLIERS as VM, KV_CACHE_RESERVE
    usable = total_vram * (1 - KV_CACHE_RESERVE)
    if usable >= model_base_gb * VM["fp16"]:
        return "FP16"
    if usable >= model_base_gb * VM["int8"]:
        return "INT8"
    if usable >= model_base_gb * VM["int4"]:
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
        # Percentile rank lookup tables: {field_name: sorted_values_list}
        self._benchmark_sorted: dict[str, list[float]] = {}
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

        # Post-load processing: build percentile tables
        self._precompute_benchmark_stats()

    # ── Percentile-Rank Normalization ─────────────────────────────────────────

    def _precompute_benchmark_stats(self) -> None:
        """Build sorted value arrays for O(log n) percentile rank lookups."""
        fields = ["coding", "math", "reasoning", "intelligence_index"]
        for fld in fields:
            values = sorted([
                m.get("benchmarks", {}).get(fld, 0) or 0
                for m in self.models
                if (m.get("benchmarks", {}).get(fld, 0) or 0) > 0
            ])
            self._benchmark_sorted[fld] = values
        logger.info(
            f"Benchmark percentile tables built: "
            + ", ".join(f"{f}={len(self._benchmark_sorted[f])}" for f in fields)
        )

    def _get_percentile_rank(self, field: str, value: float) -> float:
        """Return the percentile rank of a value within its benchmark dimension.

        Uses binary search on pre-sorted arrays for O(log n) performance.
        Returns 0.0 for zero/missing values, otherwise a value in (0.0, 1.0].
        """
        if value <= 0:
            return 0.0
        values = self._benchmark_sorted.get(field, [])
        if not values:
            return 0.0
        pos = bisect.bisect_left(values, value)
        return pos / len(values)



    # ── Hardware Display Info ─────────────────────────────────────────────────

    def _hw_info(self, model: dict, hw: ParsedHardware) -> dict:
        total_vram = hw.total_vram_gb
        vram_gb = model.get("vram_gb", {})
        vram_fp16 = vram_gb.get("fp16", 0)
        model_base_gb = vram_gb.get("model_base_gb", vram_fp16 / 1.2 if vram_fp16 else 0)
        quant = _determine_quant(model_base_gb, total_vram)

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

    # ── Main Recommendation Pipeline ──────────────────────────────────────────

    def recommend(
        self,
        hardware: ParsedHardware,
        use_case_text: str,
        user_query: str,
        top_k: int = TOP_K_RECOMMENDATIONS,
    ) -> list[ScoredModel]:
        start = time.time()

        use_case_text = use_case_text[:5000]
        user_query = user_query[:5000]

        # Multi-label use case detection: get proportional weights
        use_case, proportions = detect_use_case(use_case_text)
        if not use_case or use_case == DEFAULT_USE_CASE:
            if use_case_text.strip():
                use_case = DEFAULT_USE_CASE

        # Blend benchmark weights across all matched use cases
        blended = blend_benchmark_weights(proportions)

        total_vram = hardware.total_vram_gb

        # Pre-filter: only models that fit at INT4 within total VRAM (100% limit)
        # The penalty for eating into KV cache (>80%) is handled by calculate_hardware_score
        usable_vram = total_vram 
        # Model fits at INT4 if vram_fp16 * 0.25 <= usable_vram
        max_fp16_for_int4 = usable_vram / 0.25

        compatible = [
            m for m in self.models
            if (m.get("vram_gb", {}).get("fp16", 0) or 0) <= max_fp16_for_int4
        ]

        if not compatible:
            logger.warning(f"No models fit {hardware.gpu_name} ({total_vram}GB)")
            return []

        # Semantic search: build dict once, O(1) lookups per model
        emb_svc = get_embedding_service()
        sem_results: dict[str, float] = {}
        if emb_svc.index is not None:
            for mid, score in emb_svc.search(user_query, top_k=100):
                sem_results[mid] = score

        w_coding = blended["coding"]
        w_math = blended["math"]
        w_reasoning = blended["reasoning"]
        w_ii = blended["intelligence_index"]
        sw, bw, hw_w = self.semantic_weight, self.benchmark_weight, self.hardware_weight
        
        # Dynamic Weight Shifting:
        # If the query maps to a specific benchmark (math, coding, reasoning), the benchmark 
        # score is an objective measure of intent. Semantic search (name matching) becomes harmful.
        # We shift 100% of the semantic weight into the benchmark weight.
        # If use_case is "general" (e.g. "medical", "finance"), semantic search remains crucial.
        if use_case != "general":
            shift_amount = sw * 1.0
            sw -= shift_amount
            bw += shift_amount

        scored = []
        for model in compatible:
            mid = model.get("model_id", "")
            sem_score = sem_results.get(mid, 0.0)

            bmarks = model.get("benchmarks", {})
            c = bmarks.get("coding", 0) or 0
            m = bmarks.get("math", 0) or 0
            r = bmarks.get("reasoning", 0) or 0
            ii = bmarks.get("intelligence_index", 0) or 0

            # Percentile-rank normalization: convert raw scores to their
            # position within the population distribution (0.0 to 1.0)
            c_pct = self._get_percentile_rank("coding", c)
            m_pct = self._get_percentile_rank("math", m)
            r_pct = self._get_percentile_rank("reasoning", r)
            ii_pct = self._get_percentile_rank("intelligence_index", ii)

            # Weighted benchmark score using blended multi-label weights
            bm_score = w_coding * c_pct + w_math * m_pct + w_reasoning * r_pct + w_ii * ii_pct

            # Continuous hardware score with KV-cache-aware VRAM utilization
            vram_fp16 = model.get("vram_gb", {}).get("fp16", 0) or 0
            hw_score = calculate_hardware_score(vram_fp16, total_vram)

            final = sw * sem_score + bw * bm_score + hw_w * hw_score

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
        result = scored[:top_k]

        latency_ms = (time.time() - start) * 1000
        try:
            if self._wandb.enabled:
                top = result[0] if result else None
                self._wandb.log_recommendation(
                    query=user_query[:1000],
                    gpu_name=hardware.gpu_name,
                    num_gpus=hardware.count,
                    use_case=use_case,
                    num_compatible=len(compatible),
                    num_returned=len(result),
                    top_model=top.model_id if top else "N/A",
                    top_model_score=top.final_score if top else 0.0,
                    output_models=[m.model_id for m in result],
                    latency_ms=latency_ms,
                )
        except Exception as e:
            logger.warning(f"W&B logging failed: {e}")

        logger.info(f"Done in {latency_ms:.1f}ms, {len(result)} returned")
        return result

    @property
    def model_count(self) -> int:
        return len(self.models)

    def get_showcase_picks(self) -> list[dict]:
        """Get top model per hardware tier for showcase display."""
        showcase_configs = [
            {
                "category": "laptop",
                "label": "Works on your laptop",
                "hardware_text": "Laptop RTX 4070",
            },
            {
                "category": "consumer",
                "label": "Budget workstation",
                "hardware_text": "2x RTX 4090",
            },
            {
                "category": "data_center",
                "label": "Data center powerhouse",
                "hardware_text": "2x B200",
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