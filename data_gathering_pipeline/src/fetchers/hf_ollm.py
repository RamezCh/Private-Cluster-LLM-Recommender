"""HuggingFace open-llm-leaderboard dataset loader with open-weight filtering."""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import asdict

from datasets import load_dataset
from loguru import logger

from src.config import HF_DATASETS, HF_TOKEN, OPEN_WEIGHT_FILTER, DEDUP_STRATEGY
from src.models import OpenLLMLeaderboardRow


class HFOpenLLMLeaderboardLoader:
    """Loads and processes the open-llm-leaderboard dataset.

    This is the primary data source. It provides ~4,576 models with
    benchmark scores, parameter counts, architecture info, and an
    explicit "Available on the hub" boolean for open-weight filtering.
    """

    BENCHMARK_COLUMNS = [
        "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO", "Average"
    ]

    AVERAGE_COLUMN = "Average \u2b06\ufe0f"  # "Average ⬆️"

    ALL_COLUMNS = [
        "Model", "fullname", "Base Model", "#Params (B)", "Average",
        "MoE", "Architecture", "Precision", "Type", "Hub License",
        "Hub ❤️", "Generation", "Available on the hub",
        "Flagged", "Chat Template", "Merged", "Official Providers",
        "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO",
        "IFEval Raw", "BBH Raw", "MATH Lvl 5 Raw", "GPQA Raw",
        "MUSR Raw", "MMLU-PRO Raw",
    ]

    def __init__(self, token: Optional[str] = None):
        self.token = token or HF_TOKEN
        self.raw_rows: List[OpenLLMLeaderboardRow] = []
        self.filtered_rows: List[OpenLLMLeaderboardRow] = []
        self.groups: Dict[str, List[OpenLLMLeaderboardRow]] = {}
        self.deduped_rows: List[OpenLLMLeaderboardRow] = []

    def load(self) -> List[OpenLLMLeaderboardRow]:
        """Load the full dataset and normalize to OpenLLMLeaderboardRow."""
        logger.info(f"Loading {HF_DATASETS['open_llm_leaderboard']}")

        try:
            dataset = load_dataset(
                HF_DATASETS["open_llm_leaderboard"],
                split="train",
                token=self.token or None,
            )

            self.raw_rows = []
            for item in dataset:
                row = self._normalize_row(item)
                if row:
                    self.raw_rows.append(row)

            logger.info(f"Loaded {len(self.raw_rows)} raw records")
            return self.raw_rows

        except Exception as e:
            logger.error(f"Failed to load: {e}")
            return []

    def _normalize_row(self, item) -> Optional[OpenLLMLeaderboardRow]:
        """Convert a dataset row to OpenLLMLeaderboardRow."""
        try:
            benchmarks = {}
            for col in self.BENCHMARK_COLUMNS:
                if col in item and item[col] is not None:
                    val = float(item[col])
                    if val >= 0:
                        benchmarks[col] = val

            return OpenLLMLeaderboardRow(
                model=str(item.get("Model", "")),
                fullname=str(item.get("fullname", "")),
                base_model=str(item["Base Model"]) if item.get("Base Model") else None,
                params_billions=float(item["#Params (B)"]) if item.get("#Params (B)") and item["#Params (B)"] > 0 else None,
                average=float(item[self.AVERAGE_COLUMN]) if item.get(self.AVERAGE_COLUMN) is not None else None,
                is_moe=bool(item.get("MoE", False)),
                architecture=str(item.get("Architecture", "")),
                precision=str(item.get("Precision", "")),
                model_type=str(item.get("Type", "")),
                license=str(item.get("Hub License", "")),
                hub_likes=int(item["Hub ❤️"]) if item.get("Hub ❤️") else 0,
                generation=int(item["Generation"]) if item.get("Generation") is not None else 0,
                available_on_hub=bool(item.get("Available on the hub", False)),
                flagged=bool(item.get("Flagged", False)),
                chat_template=bool(item.get("Chat Template", False)),
                merged=bool(item.get("Merged", False)),
                official_providers=bool(item.get("Official Providers", False)),
                benchmarks=benchmarks,
            )
        except Exception as e:
            return None

    def filter_open_weight(self) -> List[OpenLLMLeaderboardRow]:
        """Apply open-weight filters to keep only locally-hostable models."""
        logger.info("Applying open-weight filters")

        min_benchmarks = OPEN_WEIGHT_FILTER.get("min_benchmarks", 3)

        self.filtered_rows = []
        for row in self.raw_rows:
            if not row.available_on_hub:
                continue

            benchmark_count = self._count_valid_benchmarks(row)
            if benchmark_count < min_benchmarks:
                continue

            self.filtered_rows.append(row)

        logger.success(f"Filtered to {len(self.filtered_rows)} open-weight models")
        return self.filtered_rows

    def _count_valid_benchmarks(self, row: OpenLLMLeaderboardRow) -> int:
        """Count how many benchmark columns have valid (non-zero) values."""
        count = 0
        for col in self.BENCHMARK_COLUMNS:
            if col in row.benchmarks and row.benchmarks[col] is not None and row.benchmarks[col] > 0:
                count += 1
        return count

    def group_by_base_model(self) -> Dict[str, List[OpenLLMLeaderboardRow]]:
        """Group filtered rows by their Base Model field."""
        self.groups = defaultdict(list)

        for row in self.filtered_rows:
            key = row.base_model or row.fullname or row.model
            self.groups[key].append(row)

        logger.info(f"Grouped into {len(self.groups)} base model clusters")
        return self.groups

    def select_best_variants(self) -> List[OpenLLMLeaderboardRow]:
        """Select the best variant from each base model group.

        Selection strategy:
        1. Highest benchmark completeness (most non-zero scores)
        2. Tie-break: higher generation number (newer model)
        3. Tie-break: type preference (chat > pretrained > fine-tuned > merged)
        """
        self.deduped_rows = []

        for base_model, variants in self.groups.items():
            best = self._select_best_variant(variants)
            self.deduped_rows.append(best)

        self.deduped_rows.sort(key=lambda r: r.average or 0, reverse=True)
        logger.success(f"Selected {len(self.deduped_rows)} best variants")
        return self.deduped_rows

    def _select_best_variant(
        self, variants: List[OpenLLMLeaderboardRow]
    ) -> OpenLLMLeaderboardRow:
        """Select the single best variant from a list of variants."""
        if len(variants) == 1:
            return variants[0]

        scored = []
        for v in variants:
            score = self._variant_score(v)
            scored.append((score, v))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _variant_score(self, variant: OpenLLMLeaderboardRow) -> Tuple[int, int, int]:
        """Score a variant for selection purposes.

        Returns (benchmark_count, generation, type_preference).
        Higher is better for all three.
        """
        bench_count = self._count_valid_benchmarks(variant)

        gen = variant.generation

        type_order = DEDUP_STRATEGY.get("type_preference_order", ["💬", "🟢", "🔶", "🤝", "💻", "🔢", "⚡"])
        type_pref = len(type_order)
        for i, t in enumerate(type_order):
            if t in variant.model_type:
                type_pref = i
                break

        return (bench_count, gen, -type_pref)

    def get_deduped_model_names(self) -> List[str]:
        """Get list of canonical model names from deduplicated rows."""
        return [row.fullname or row.model for row in self.deduped_rows]

    def get_model_by_name(self, name: str) -> Optional[OpenLLMLeaderboardRow]:
        """Find a deduplicated row by model name."""
        for row in self.deduped_rows:
            if row.fullname == name or row.model == name:
                return row
        return None

    def load_all(self) -> List[OpenLLMLeaderboardRow]:
        """Convenience: run the full pipeline (load → filter → dedup)."""
        self.load()
        self.filter_open_weight()
        self.group_by_base_model()
        self.select_best_variants()
        return self.deduped_rows

    def to_dicts(self) -> List[Dict]:
        """Export deduplicated rows as plain dicts for the pipeline."""
        result = []
        for row in self.deduped_rows:
            result.append({
                "model_id": row.fullname or row.model,
                "model": row.model,
                "fullname": row.fullname,
                "base_model": row.base_model,
                "params_billions": row.params_billions,
                "average": row.average,
                "is_moe": row.is_moe,
                "architecture": row.architecture,
                "precision": row.precision,
                "model_type": row.model_type,
                "license": row.license,
                "hub_likes": row.hub_likes,
                "generation": row.generation,
                "benchmarks": row.benchmarks,
                "flagged": row.flagged,
                "chat_template": row.chat_template,
                "merged": row.merged,
                "official_providers": row.official_providers,
            })
        return result

    def get_coverage_stats(self) -> Dict:
        """Compute benchmark coverage statistics."""
        total = len(self.deduped_rows)
        if total == 0:
            return {}

        coverage = {}
        for col in self.BENCHMARK_COLUMNS:
            if col == "Average":
                continue
            count = sum(1 for r in self.deduped_rows if r.benchmarks.get(col, 0) > 0)
            coverage[col] = round(count / total * 100, 1)

        return coverage