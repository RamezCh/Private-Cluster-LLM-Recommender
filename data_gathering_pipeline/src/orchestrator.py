"""Orchestrator — main data gathering and merging pipeline for v2.

This pipeline targets OPEN-WEIGHT MODELS ONLY (locally hostable LLMs).

Data sources:
  1. open-llm-leaderboard (HF Dataset) — primary, ~4.5K rows, 6 benchmarks
  2. OpenCompass General (/leaderboard-llm/?m=26-04) — academic, needs Selenium
  3. OpenCompass Academic (/leaderboard-llm-academic/?m=REALTIME) — real-time

Pipeline phases:
  Phase 1: Load HF dataset (open-llm-leaderboard) — open-weight filter + dedup
  Phase 2: Scrape both OpenCompass pages (Selenium)
  Phase 3: Cross-source deduplication + canonicalization
  Phase 4: Benchmark merging — smart priority-based resolution
  Phase 5: HF metadata enrichment (parallel, 20 threads)
  Phase 6: VRAM calculation + hardware fit
  Phase 7: Save output

Total runtime: ~5-8 minutes (dominated by Selenium scraping)
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from loguru import logger

from src.config import (
    OUTPUT_FILE,
    TEMP_OPENCOMPASS_GENERAL,
    TEMP_OPENCOMPASS_ACADEMIC,
    OPENCOMPASS_CONFIG,
)
from src.fetchers import HFOpenLLMLeaderboardLoader, OpenCompassScraper
from src.services import (
    HFMetadataService,
    HFModelMetadata,
    ModelDeduplicator,
    BenchmarkMerger,
    VRAMCalculator,
    is_moe_model,
    parse_model_size,
    estimate_size_from_params,
    get_recommended_context_tier,
    format_all_fits,
)


class Orchestrator:
    """Main orchestrator for the MHII v2 data gathering pipeline."""

    def __init__(self, output_path: Optional[Path] = None):
        self.output_path = output_path or OUTPUT_FILE

        self.ollm_loader = HFOpenLLMLeaderboardLoader()
        self.opencompass_scraper: Optional[OpenCompassScraper] = None
        self.hf_service = HFMetadataService()
        self.hf_service.failed_lookups = []
        self.hf_service.cache = {}  # Clear to re-fetch all (retry previously failed)
        self.merger = BenchmarkMerger()

        self.raw_data: Dict[str, List[Dict]] = {
            "open_llm_leaderboard": [],
            "opencompass_general": [],
            "opencompass_academic": [],
        }

        self.canonical_models: List[Dict] = []
        self.records: List[Dict] = []
        self.errors: List[Dict] = []

    def _load_open_llm_leaderboard(self) -> List[Dict]:
        """Phase 1: Load open-llm-leaderboard, apply filters, deduplicate."""
        logger.info("=" * 60)
        logger.info("PHASE 1: Loading open-llm-leaderboard dataset")
        logger.info("=" * 60)
        t0 = time.time()

        self.ollm_loader.load()
        self.ollm_loader.filter_open_weight()
        self.ollm_loader.group_by_base_model()
        self.ollm_loader.select_best_variants()

        self.raw_data["open_llm_leaderboard"] = self.ollm_loader.to_dicts()

        logger.info(f"Phase 1 complete in {time.time() - t0:.1f}s — "
                    f"{len(self.raw_data['open_llm_leaderboard'])} open-weight models")

        coverage = self.ollm_loader.get_coverage_stats()
        if coverage:
            logger.info("Benchmark coverage (% of models with score > 0):")
            for bench, pct in coverage.items():
                logger.info(f"  {bench}: {pct}%")

        return self.raw_data["open_llm_leaderboard"]

    def _scrape_opencompass(
        self,
        skip_scrape: bool = False,
        use_cache: bool = True,
    ) -> tuple:
        """Phase 2: Scrape both OpenCompass leaderboards with Selenium."""
        logger.info("=" * 60)
        logger.info("PHASE 2: Scraping OpenCompass leaderboards")
        logger.info("=" * 60)
        t0 = time.time()

        general_data = []
        academic_data = []

        if use_cache:
            if TEMP_OPENCOMPASS_GENERAL.exists():
                try:
                    general_data = json.loads(TEMP_OPENCOMPASS_GENERAL.read_text(encoding="utf-8"))
                    logger.info(f"Loaded cached general leaderboard: {len(general_data)} models")
                except Exception:
                    pass

            if TEMP_OPENCOMPASS_ACADEMIC.exists():
                try:
                    academic_data = json.loads(TEMP_OPENCOMPASS_ACADEMIC.read_text(encoding="utf-8"))
                    logger.info(f"Loaded cached academic leaderboard: {len(academic_data)} models")
                except Exception:
                    pass

        if not general_data and not academic_data and not skip_scrape:
            self.opencompass_scraper = OpenCompassScraper(headless=True)

            try:
                general_data, academic_data = self.opencompass_scraper.scrape_both(
                    general_month=OPENCOMPASS_CONFIG["general"]["default_month"],
                    academic_month=OPENCOMPASS_CONFIG["academic"]["default_month"],
                )
            finally:
                self.opencompass_scraper.close()

            if general_data:
                self.opencompass_scraper.save_temp(general_data, TEMP_OPENCOMPASS_GENERAL)
            if academic_data:
                self.opencompass_scraper.save_temp(academic_data, TEMP_OPENCOMPASS_ACADEMIC)
        elif not general_data and not academic_data and skip_scrape:
            logger.warning("Skipping OpenCompass scrape — no cached data found")

        self.raw_data["opencompass_general"] = general_data
        self.raw_data["opencompass_academic"] = academic_data

        total = len(general_data) + len(academic_data)
        logger.info(f"Phase 2 complete in {time.time() - t0:.1f}s — {total} models scraped")

        return general_data, academic_data

    def _deduplicate(self) -> List[Dict]:
        """Phase 3: Cross-source deduplication and canonicalization."""
        logger.info("=" * 60)
        logger.info("PHASE 3: Cross-source deduplication & canonicalization")
        logger.info("=" * 60)
        t0 = time.time()

        dedup = ModelDeduplicator(
            ollm_models=self.raw_data["open_llm_leaderboard"],
            oc_general=self.raw_data["opencompass_general"],
            oc_academic=self.raw_data["opencompass_academic"],
        )

        self.canonical_models = dedup.run()
        logger.info(f"Phase 3 complete in {time.time() - t0:.1f}s — "
                    f"{len(self.canonical_models)} unique models")

        coverage = dedup.get_coverage_report()
        if coverage:
            logger.info(f"Average benchmarks per model: {coverage.get('avg_benchmarks_per_model', 'N/A')}")
            logger.info(f"Multi-source models: {coverage.get('multi_source_models', 0)}")

        return self.canonical_models

    def _merge_benchmarks(self) -> None:
        """Phase 4: Resolve benchmark values using priority-based merging."""
        logger.info("=" * 60)
        logger.info("PHASE 4: Merging benchmarks across sources")
        logger.info("=" * 60)
        t0 = time.time()

        for model in self.canonical_models:
            bench_data = self.merger.merge_from_canonical(model)

            model["benchmarks"] = bench_data.to_dict()
            model["extended_benchmarks"] = bench_data.extended

        logger.info(f"Phase 4 complete in {time.time() - t0:.1f}s")

    def _enrich_hf_metadata(self, max_workers: int = 5) -> None:
        """Phase 5: Fetch HuggingFace metadata in parallel."""
        logger.info("=" * 60)
        logger.info(f"PHASE 5: HF metadata enrichment ({max_workers} threads)")
        logger.info("=" * 60)
        t0 = time.time()

        model_names = []
        params_map = {}
        for m in self.canonical_models:
            key = m.get("canonical_name") or m.get("fullname") or m.get("model_name", "")
            if key:
                model_names.append(key)
                if m.get("params_billions"):
                    params_map[key] = m["params_billions"]

        metadata_results = self.hf_service.parallel_batch_fetch(
            model_names, max_workers=max_workers, params_map=params_map
        )

        for model in self.canonical_models:
            key = model.get("canonical_name") or model.get("fullname") or model.get("model_name", "")
            meta = metadata_results.get(key)

            if meta:
                model["hf_repo_id"] = meta.repo_id
                model["safetensors_size_gb"] = meta.safetensors_size_gb
                model["is_moe"] = meta.is_moe or model.get("is_moe", False)
                model["num_experts"] = meta.num_experts
                model["architecture"] = model.get("architecture") or meta.model_type
                model["source_status"] = meta.metadata_status
            else:
                model["hf_repo_id"] = None
                model["safetensors_size_gb"] = 0.0
                model["source_status"] = "missing_hf_metadata"

        logger.info(f"Phase 5 complete in {time.time() - t0:.1f}s")
        logger.info(f"HF metadata cache stats: {self.hf_service.get_cache_stats()}")

    def _calculate_vram(self) -> None:
        """Phase 6: Calculate VRAM requirements and hardware fit."""
        logger.info("=" * 60)
        logger.info("PHASE 6: VRAM calculation & hardware fit")
        logger.info("=" * 60)
        t0 = time.time()

        for model in self.canonical_models:
            self._calculate_model_vram(model)

        logger.info(f"Phase 6 complete in {time.time() - t0:.1f}s")

    def _calculate_model_vram(self, model: Dict) -> None:
        """Calculate VRAM for a single model."""
        canonical = model.get("canonical_name") or model.get("fullname", "")

        size_gb = model.get("safetensors_size_gb", 0.0)
        if size_gb == 0.0:
            params = model.get("params_billions")
            if params:
                size_gb = params * 2
            else:
                parsed = parse_model_size(canonical)
                if parsed:
                    size_gb = parsed * 2
                else:
                    size_gb = 10.0

        int_index = model.get("benchmarks", {}).get("intelligence_index")
        context_tier = get_recommended_context_tier(int_index, canonical)

        vram_calc = VRAMCalculator(context_tier)
        vram_req = vram_calc.calculate(size_gb)

        is_moe = model.get("is_moe", False) or bool(is_moe_model(canonical)[0])

        hw_fit = {
            "gpu_id": "a100_80gb",
            "gpu_name": "A100 80GB",
            "gpu_count": 8,
            "total_vram_gb": 640,
            "status": "Compatible",
            "is_moe_model": is_moe,
            "hosting_strategy": "Expert-Distributed" if is_moe else "TP-Sharded",
            "context_overhead_tier": context_tier,
            "tier": "data_center",
        }

        all_fits = format_all_fits(vram_req, is_moe)

        model["vram_gb"] = {
            "fp16": vram_req.fp16_gb,
            "int8": vram_req.int8_gb,
            "int4": vram_req.int4_gb,
            "model_base_gb": vram_req.model_size_gb,
        }
        model["hardware_fit"] = hw_fit
        model["all_gpu_compatibility"] = all_fits
        model["hosting_strategy"] = "Expert-Distributed" if is_moe else "TP-Sharded"

    def _build_final_records(self) -> None:
        """Phase 7 (combined): Build final JSONL records from canonical models."""
        self.records = []

        for model in self.canonical_models:
            try:
                record = {
                    "model_id": model.get("canonical_name") or model.get("fullname", ""),
                    "hf_repo_id": model.get("hf_repo_id"),
                    "base_model": model.get("base_model"),
                    "model_type": model.get("model_type", ""),
                    "architecture": model.get("architecture"),
                    "precision": model.get("precision"),

                    "params_billions": model.get("params_billions"),
                    "safetensors_size_gb": model.get("safetensors_size_gb", 0.0),

                    "benchmarks": model.get("benchmarks", {}),
                    "extended_benchmarks": model.get("extended_benchmarks", {}),

                    "is_moe": model.get("is_moe", False),
                    "num_experts": model.get("num_experts"),

                    "license": model.get("license"),
                    "hub_likes": model.get("hub_likes", 0),
                    "generation": model.get("generation", 0),

                    "vram_gb": model.get("vram_gb", {}),
                    "hardware_fit": model.get("hardware_fit", {}),
                    "hosting_strategy": model.get("hosting_strategy", "unknown"),

                    "source_status": model.get("source_status", "unknown"),
                    "all_gpu_compatibility": model.get("all_gpu_compatibility", {}),
                    "match_confidence": model.get("match_confidence"),
                    "canonical_name": model.get("canonical_name"),
                    "_sources": model.get("_sources", []),
                    "_variant_count": model.get("_variant_count", 1),
                }

                self.records.append(record)

            except Exception as e:
                self.errors.append({"model": model.get("canonical_name", "unknown"), "error": str(e)})

    def _save_output(self) -> Path:
        """Save final records to JSONL output file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.success(f"Saved {len(self.records)} records to {self.output_path}")
        return self.output_path

    def run(self, skip_opencompass: bool = False) -> Path:
        """Execute the complete v2 pipeline."""
        phase1_t0 = time.time()

        logger.info("=" * 60)
        logger.info("STARTING MHII v2 PIPELINE — OPEN SOURCE LLM RECOMMENDER")
        logger.info("Target: Open-weight, locally-hostable models only")
        logger.info("=" * 60)

        self._load_open_llm_leaderboard()
        self._scrape_opencompass(skip_scrape=skip_opencompass)
        self._deduplicate()
        self._merge_benchmarks()
        self._enrich_hf_metadata(max_workers=5)
        self._calculate_vram()
        self._build_final_records()
        path = self._save_output()

        total_time = time.time() - phase1_t0
        logger.success(f"Pipeline complete in {total_time:.1f}s — {len(self.records)} models")

        return path

    def get_report(self) -> Dict:
        """Generate pipeline execution report."""
        verified = sum(1 for r in self.records if r["source_status"] == "verified")
        missing = sum(1 for r in self.records if r["source_status"] == "missing_hf_metadata")
        merged = sum(1 for r in self.records if r["source_status"] == "merged")

        benchmarks_filled = {
            key: sum(1 for r in self.records if r["benchmarks"].get(key) is not None)
            for key in ["coding", "math", "reasoning", "intelligence_index"]
        }

        return {
            "total_models": len(self.records),
            "source_status": {
                "verified": verified,
                "missing_hf_metadata": missing,
                "merged": merged,
            },
            "architecture_types": {
                "moe": sum(1 for r in self.records if r["is_moe"]),
                "dense": sum(1 for r in self.records if not r["is_moe"]),
            },
            "hosting_strategies": {
                "tp_sharded": sum(
                    1 for r in self.records if r["hosting_strategy"] == "TP-Sharded"
                ),
                "expert_distributed": sum(
                    1 for r in self.records if r["hosting_strategy"] == "Expert-Distributed"
                ),
            },
            "benchmark_coverage": {
                key: round(count / len(self.records) * 100, 1)
                for key, count in benchmarks_filled.items()
            },
            "multi_source_models": sum(
                1 for r in self.records if len(r.get("_sources", [])) > 1
            ),
            "hf_cache_stats": self.hf_service.get_cache_stats(),
            "errors": self.errors,
            "output_file": str(self.output_path),
        }