"""Benchmark merger — maps and merges benchmark scores from all sources into a standardized schema."""

from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class StandardBenchmarkData:
    """Our standardized benchmark schema after merging all sources."""

    coding: Optional[float] = None
    math: Optional[float] = None
    reasoning: Optional[float] = None
    elo: Optional[float] = None
    intelligence_index: Optional[float] = None
    extended: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "coding": self.coding,
            "math": self.math,
            "reasoning": self.reasoning,
            "elo": self.elo,
            "intelligence_index": self.intelligence_index,
        }


class BenchmarkMerger:
    """Merges benchmark data from open-llm-leaderboard, OpenCompass General, and OpenCompass Academic.

    Source column coverage:
    - open_llm_leaderboard: IFEval, BBH, MATH Lvl 5, GPQA, MUSR, MMLU-PRO, Average
    - opencompass_general:  C-Eval, MMLU, GSM8K, MATH, BBH, HumanEval, MBPP, ARC, CMMLU, DROP, HellaSwag, ...
    - opencompass_academic: similar to general, potentially different versions

    Standard schema:
    - coding:     coding ability (HumanEval, MBPP, IFEval)
    - math:       math problem solving (MATH, GSM8K, MATH Lvl 5)
    - reasoning:  general reasoning (BBH, MMLU-PRO, GPQA, C-Eval, MMLU, DROP, MUSR)
    - elo:        ELO-style overall rating (OpenCompass Overall, if available)
    - intelligence_index: composite score (Average from open-llm-leaderboard)
    """

    BENCHMARK_PRIORITIES = {
        "coding": {
            "sources": ["opencompass_general", "opencompass_academic", "open_llm_leaderboard"],
            "columns": {
                "HumanEval": 1.0,
                "MBPP": 0.9,
                "IFEval": 0.8,
            },
        },
        "math": {
            "sources": ["opencompass_general", "opencompass_academic", "open_llm_leaderboard"],
            "columns": {
                "MATH Lvl 5": 1.0,
                "MATH": 0.95,
                "GSM8K": 0.9,
                "Math": 0.95,
            },
        },
        "reasoning": {
            "sources": ["open_llm_leaderboard", "opencompass_general", "opencompass_academic"],
            "columns": {
                "MMLU-PRO": 1.0,
                "BBH": 0.9,
                "GPQA": 0.85,
                "C-Eval": 0.8,
                "MMLU": 0.75,
                "MUSR": 0.7,
                "DROP": 0.7,
                "ARC": 0.6,
                "CMMLU": 0.6,
                "HellaSwag": 0.5,
            },
        },
    }

    AVERAGE_COLUMN = "Average \u2b06\ufe0f"  # "Average ⬆️"

    EXTENDED_BENCHMARK_MAP = {
        "IFEval": "instruction_following",
        "BBH": "big_bench_hard",
        "MMLU-PRO": "mmlu_pro",
        "GPQA": "graduate_reasoning",
        "MUSR": "multi_domain_reasoning",
        "MATH Lvl 5": "math_level5",
        "HumanEval": "humaneval",
        "MBPP": "mbpp",
        "GSM8K": "gsm8k",
        "MATH": "math",
        "C-Eval": "c_eval",
        "MMLU": "mmlu",
        "CMMLU": "cmmliu",
        "DROP": "drop",
        "ARC": "arc",
        "HellaSwag": "hellaswag",
        "PIQA": "piqa",
        "COPA": "copa",
        "BoolQ": "boolq",
        "TriviaQA": "triviaqa",
        "CommonSenseQA": "common_sense_qa",
        "WiC": "word_in_context",
        "AFQMC": "ant金融",
        "Flores": "flores_translation",
        "TyDiQA": "tydi_qa",
    }

    def merge(
        self,
        merged_benchmarks: Dict[str, float],
        benchmark_sources: Optional[Dict[str, List[str]]] = None,
        intelligence_index: Optional[float] = None,
        overall_score: Optional[float] = None,
    ) -> StandardBenchmarkData:
        """Merge all benchmark data into the standardized schema.

        Args:
            merged_benchmarks: Dict of {benchmark_name: score} from all sources
            benchmark_sources: Dict of {benchmark_name: [source_names]} for priority
            intelligence_index: Composite score from open-llm-leaderboard Average
            overall_score: OpenCompass overall/ELO score
        """
        result = StandardBenchmarkData()

        result.coding = self._resolve_best("coding", merged_benchmarks, benchmark_sources)
        result.math = self._resolve_best("math", merged_benchmarks, benchmark_sources)
        result.reasoning = self._resolve_best("reasoning", merged_benchmarks, benchmark_sources)

        result.intelligence_index = intelligence_index
        if intelligence_index is None and overall_score is not None:
            result.intelligence_index = overall_score

        result.elo = overall_score if overall_score and overall_score > 100 else None

        result.extended = self._build_extended(merged_benchmarks)

        return result

    def merge_from_canonical(self, canonical: Dict) -> StandardBenchmarkData:
        """Merge benchmarks from a canonical model dict produced by ModelDeduplicator."""
        merged_bench = canonical.get("merged_benchmarks", {})
        sources = canonical.get("benchmark_sources", {})
        int_index = canonical.get("average") or canonical.get("intelligence_index")
        overall = canonical.get("overall_score")

        return self.merge(merged_bench, sources, int_index, overall)

    def _resolve_best(
        self,
        target_key: str,
        benchmarks: Dict[str, float],
        sources: Optional[Dict[str, List[str]]] = None,
    ) -> Optional[float]:
        """Resolve the best available benchmark value for a target key.

        Strategy:
        1. Collect all matching benchmark columns for the target key
        2. Filter to available values (> 0)
        3. Score each by column priority within each source
        4. Score each by source priority
        5. Return the highest-scoring value
        """
        config = self.BENCHMARK_PRIORITIES.get(target_key, {})
        column_map = config.get("columns", {})
        source_priority = config.get("sources", [])

        candidates = []

        for col_name, col_priority in column_map.items():
            if col_name in benchmarks and benchmarks[col_name] is not None and benchmarks[col_name] > 0:
                src_priority = 0
                if sources and col_name in sources:
                    src_list = sources[col_name]
                    for i, src in enumerate(source_priority):
                        if src in src_list:
                            src_priority = i
                            break

                score = (1.0 / (src_priority + 1)) * col_priority * benchmarks[col_name]
                candidates.append((score, benchmarks[col_name], col_name))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _build_extended(self, benchmarks: Dict[str, float]) -> Dict:
        """Build the extended benchmark map with normalized names."""
        result = {}
        for orig_name, mapped_name in self.EXTENDED_BENCHMARK_MAP.items():
            if orig_name in benchmarks and benchmarks[orig_name] is not None and benchmarks[orig_name] > 0:
                result[mapped_name] = round(benchmarks[orig_name], 4)
        return result

    def get_schema_stats(self) -> Dict:
        """Return benchmark coverage for the standard schema keys."""
        return {
            key: list(config["columns"].keys())
            for key, config in self.BENCHMARK_PRIORITIES.items()
        }


class BenchmarkMergerSimple:
    """Simplified benchmark merger for direct (non-deduplicated) data.

    Used when building records directly from a single source
    (e.g., open-llm-leaderboard alone).
    """

    COLUMN_MAP = {
        "IFEval": ("coding", "IFEval"),
        "HumanEval": ("coding", "HumanEval"),
        "MBPP": ("coding", "MBPP"),
        "MATH Lvl 5": ("math", "MATH Lvl 5"),
        "GSM8K": ("math", "GSM8K"),
        "MATH": ("math", "MATH"),
        "BBH": ("reasoning", "BBH"),
        "MMLU-PRO": ("reasoning", "MMLU-PRO"),
        "GPQA": ("reasoning", "GPQA"),
        "C-Eval": ("reasoning", "C-Eval"),
        "MMLU": ("reasoning", "MMLU"),
        "MUSR": ("reasoning", "MUSR"),
        "DROP": ("reasoning", "DROP"),
        "ARC": ("reasoning", "ARC"),
        "CMMLU": ("reasoning", "CMMLU"),
    }

    def merge(self, benchmarks: Dict[str, float], average: Optional[float] = None) -> Dict:
        """Convert a raw benchmark dict to our standard schema."""
        result = {}
        extended = {}

        for col_name, (target_key, normalized_name) in self.COLUMN_MAP.items():
            if col_name in benchmarks and benchmarks[col_name] > 0:
                if target_key not in result or result[target_key] is None:
                    result[target_key] = benchmarks[col_name]
                extended[normalized_name] = round(benchmarks[col_name], 4)

        if average is not None and average > 0:
            result["intelligence_index"] = average

        result["extended"] = extended
        return result