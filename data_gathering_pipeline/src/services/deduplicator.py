"""Model deduplication: groups variants by base model, selects best per group."""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

from loguru import logger

from src.config import PROPRIETARY_FILTER, OPEN_WEIGHT_ORGS


@dataclass
class VariantScore:
    """Scoring result for a single variant."""

    benchmark_count: int
    benchmark_total: float
    has_hub_url: bool
    generation: int
    type_score: int
    size_info: bool
    raw_score: Tuple


class ModelDeduplicator:
    """Groups and deduplicates model variants across all data sources.

    Strategy:
    1. Collect all model names from all sources
    2. Group by normalized base name (using Base Model field + fuzzy)
    3. Score each variant by benchmark completeness and other signals
    4. Select the best variant per group
    5. Merge benchmark data from all variants/sources for the selected model
    """

    TYPE_PREFERENCE = {
        "💬": 0,   # Chat models (most useful)
        "🟢": 1,   # Pretrained base (authoritative)
        "🔶": 2,   # Fine-tuned/domain-specific
        "🤝": 3,   # Merged models (may have issues)
        "💻": 4,   # Coding specialized
        "🔢": 5,   # Math specialized
        "⚡": 6,   # Other specialized
    }

    def __init__(
        self,
        ollm_models: List[Dict],
        oc_general: List[Dict],
        oc_academic: List[Dict],
    ):
        self.ollm_models = ollm_models
        self.oc_general = oc_general
        self.oc_academic = oc_academic

        self.all_models: List[Dict] = []
        self.groups: Dict[str, List[Dict]] = {}
        self.canonical_models: List[Dict] = []
        self.errors: List[Dict] = []

    def run(self) -> List[Dict]:
        """Execute the full deduplication pipeline."""
        self._merge_all_sources()
        self._group_by_base_name()
        self._select_best_per_group()
        return self.canonical_models

    def _merge_all_sources(self) -> None:
        """Collect and tag all models from all sources."""
        for item in self.ollm_models:
            item["_source"] = "open_llm_leaderboard"
            item["_dedup_key"] = self._normalize_key(item.get("base_model") or item.get("fullname") or item.get("model_id", ""))
            self.all_models.append(item)

        for item in self.oc_general:
            item["_source"] = "opencompass_general"
            item["_dedup_key"] = self._normalize_key(item.get("model_name", ""))
            self.all_models.append(item)

        for item in self.oc_academic:
            item["_source"] = "opencompass_academic"
            item["_dedup_key"] = self._normalize_key(item.get("model_name", ""))
            self.all_models.append(item)

        logger.info(f"Merged {len(self.all_models)} model entries from {len(self.all_models)} sources")

    def _normalize_key(self, name: str) -> str:
        """Normalize a model name to a deduplication key."""
        name = name.lower().strip()
        name = re.sub(r"[\-_]", " ", name)
        name = re.sub(r"\s+", " ", name)

        patterns_to_strip = [
            r"\s*(chat|instruct|sft|dpo|rlhf|merged|merge|base|original|preferred)\s*",
            r"\s*(v\d+(?:\.\d+)?)\s*",
            r"\s*(bfloat16|float16|float32|int4|int8|fp16|q4|q8|q16)\s*",
        ]
        for pattern in patterns_to_strip:
            name = re.sub(pattern, " ", name, flags=re.IGNORECASE)

        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _group_by_base_name(self) -> None:
        """Group all model entries by normalized base name."""
        self.groups = defaultdict(list)

        for model in self.all_models:
            key = model["_dedup_key"]
            if not key:
                continue
            self.groups[key].append(model)

        logger.info(f"Grouped into {len(self.groups)} base model clusters")

        small_groups = {k: v for k, v in self.groups.items() if len(v) == 1}
        if small_groups:
            logger.info(f"  {len(small_groups)} singletons, {len(self.groups) - len(small_groups)} multi-variant groups")

    def _select_best_per_group(self) -> None:
        """Select the best canonical model for each group."""
        for group_key, variants in self.groups.items():
            if not variants:
                continue

            try:
                canonical = self._select_canonical(variants)
                self.canonical_models.append(canonical)
            except Exception as e:
                self.errors.append({"group": group_key, "error": str(e)})

        logger.success(f"Selected {len(self.canonical_models)} canonical models")
        if self.errors:
            logger.warning(f"  {len(self.errors)} groups had errors")

    def _select_canonical(self, variants: List[Dict]) -> Dict:
        """Select the best variant from a group, merging data from all sources."""
        if len(variants) == 1:
            return self._enrich_single(variants[0])

        scored = []
        for v in variants:
            score = self._score_variant(v)
            scored.append((score, v))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_variant = scored[0][1]

        return self._enrich_with_group(best_variant, variants)

    def _enrich_single(self, variant: Dict) -> Dict:
        """Enrich a single variant with metadata."""
        result = dict(variant)
        result["canonical_name"] = variant.get("fullname") or variant.get("model_name") or variant.get("model_id", "unknown")
        result["is_moe"] = variant.get("is_moe", False)
        result["params_billions"] = variant.get("params_billions") or variant.get("params")
        result["source_status"] = "verified" if variant.get("_source") == "open_llm_leaderboard" else "scraped"
        result["merged_benchmarks"] = dict(variant.get("benchmarks", {}))
        return result

    def _enrich_with_group(self, best: Dict, variants: List[Dict]) -> Dict:
        """Merge data from all variants into the best one."""
        result = dict(best)
        result["canonical_name"] = best.get("fullname") or best.get("model_name") or best.get("model_id", "unknown")

        merged_benchmarks = {}
        benchmark_sources = defaultdict(list)

        for v in variants:
            source = v.get("_source", "unknown")
            for bench_name, bench_val in v.get("benchmarks", {}).items():
                if bench_val is not None and bench_val > 0:
                    benchmark_sources[bench_name].append((bench_val, source))
                    if bench_name not in merged_benchmarks:
                        merged_benchmarks[bench_name] = bench_val

        for bench_name, sources in benchmark_sources.items():
            sources.sort(key=lambda x: x[0], reverse=True)
            merged_benchmarks[bench_name] = sources[0][0]

        best_avg = best.get("average")
        if best_avg is not None and best_avg > 0:
            merged_benchmarks["Average \u2b06\ufe0f"] = best_avg

        result["merged_benchmarks"] = merged_benchmarks
        result["benchmark_sources"] = {
            bench: [s for _, s in srcs]
            for bench, srcs in benchmark_sources.items()
        }

        result["is_moe"] = any(v.get("is_moe", False) for v in variants)
        result["params_billions"] = (
            best.get("params_billions")
            or next((v.get("params_billions") for v in variants if v.get("params_billions")), None)
        )
        result["generation"] = max((v.get("generation", 0) for v in variants), default=0)
        result["model_type"] = best.get("model_type", "")
        result["architecture"] = best.get("architecture") or next((v.get("architecture") for v in variants if v.get("architecture")), None)
        result["license"] = best.get("license") or next((v.get("license") for v in variants if v.get("license")), None)

        result["source_status"] = "merged"
        result["_variant_count"] = len(variants)
        result["_sources"] = list(set(v.get("_source", "unknown") for v in variants))

        return result

    def _score_variant(self, variant: Dict) -> Tuple:
        """Score a variant for selection priority.

        Returns tuple for sorting (higher is better):
        (has_open_llm, benchmark_count, avg_benchmark, type_score, generation, size_available)
        """
        source = variant.get("_source", "")
        has_open_llm = 1 if source == "open_llm_leaderboard" else 0

        bench_dict = variant.get("benchmarks", {})
        benchmark_vals = [v for v in bench_dict.values() if v is not None and v > 0]
        benchmark_count = len(benchmark_vals)
        avg_benchmark = sum(benchmark_vals) / len(benchmark_vals) if benchmark_vals else 0

        type_str = variant.get("model_type", "")
        type_score = self.TYPE_PREFERENCE.get(type_str, 99)

        generation = variant.get("generation", 0)

        size_available = 1 if variant.get("params_billions") or variant.get("params") else 0

        return (
            has_open_llm,
            benchmark_count,
            avg_benchmark,
            -type_score,
            generation,
            size_available,
        )

    def get_coverage_report(self) -> Dict:
        """Generate a benchmark coverage report across all canonical models."""
        total = len(self.canonical_models)
        if total == 0:
            return {}

        bench_counts = defaultdict(int)
        for model in self.canonical_models:
            for bench in model.get("merged_benchmarks", {}).keys():
                bench_counts[bench] += 1

        return {
            "total_models": total,
            "benchmark_coverage": {
                bench: round(count / total * 100, 1)
                for bench, count in sorted(bench_counts.items(), key=lambda x: x[1], reverse=True)
            },
            "avg_benchmarks_per_model": round(
                sum(len(m.get("merged_benchmarks", {})) for m in self.canonical_models) / total, 1
            ),
            "multi_source_models": sum(
                1 for m in self.canonical_models
                if len(m.get("_sources", [])) > 1
            ),
        }