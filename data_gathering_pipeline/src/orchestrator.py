"""MHII Orchestrator - Main data gathering and merging pipeline."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from loguru import logger

from src.config import OUTPUT_FILE, TEMP_PERFORMANCE_FILE
from src.fetchers import WebScraper, HFDatasetLoader
from src.services import (
    FuzzyModelMatcher,
    HFMetadataService,
    HardwareService,
    VRAMCalculator,
    check_all_gpu_compatibility,
    format_all_fits,
    is_moe_model,
    parse_model_size,
    estimate_size_from_params,
    get_recommended_context_tier,
)


class BenchmarkMerger:
    """Merges benchmark data from multiple sources."""

    @staticmethod
    def merge(
        aa_data: Optional[Dict], oe_data: Optional[Dict], lmsys_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """Merge benchmark data into standardized format."""
        return {
            "coding": (oe_data.get("sweVerified_score") if oe_data else None)
            or (oe_data.get("swePro_score") if oe_data else None)
            or (oe_data.get("terminalBench_score") if oe_data else None),
            "math": (oe_data.get("gsm8k_score") if oe_data else None)
            or (oe_data.get("aime2026_score") if oe_data else None)
            or (oe_data.get("hmmt2026_score") if oe_data else None),
            "reasoning": (oe_data.get("mmluPro_score") if oe_data else None)
            or (oe_data.get("hle_score") if oe_data else None)
            or (oe_data.get("gpqa_score") if oe_data else None),
            "elo": (lmsys_data.get("elo") if lmsys_data else None)
            or (lmsys_data.get("rating") if lmsys_data else None)
            or (lmsys_data.get("score") if lmsys_data else None),
            "intelligence_index": (
                aa_data.get("intelligence_index") if aa_data else None
            ),
            "throughput_tokens_per_sec": (
                aa_data.get("throughput_tokens_per_sec") if aa_data else None
            ),
            "vibes_score": lmsys_data.get("vibes") if lmsys_data else None,
        }


class Orchestrator:
    """Main orchestrator for the MHII data gathering pipeline."""

    def __init__(self, output_path: Optional[Path] = None):
        self.output_path = output_path or OUTPUT_FILE
        self.scraper = WebScraper(headless=True)
        self.dataset_loader = HFDatasetLoader()
        self.matcher = FuzzyModelMatcher(score_threshold=85)
        self.hf_service = HFMetadataService()
        self.merger = BenchmarkMerger()

        self.raw_data: Dict[str, List[Dict]] = {
            "artificial_analysis": [],
            "open_evals": [],
            "lmsys_arena": [],
        }

        self.records: List[Dict] = []
        self.errors: List[Dict] = []

    def _load_cached_performance(self) -> List[Dict]:
        """Load cached performance data if available."""
        if TEMP_PERFORMANCE_FILE.exists():
            with open(TEMP_PERFORMANCE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} cached records")
            return data
        return []

    def _run_scraper(self) -> List[Dict]:
        """Run the Selenium scraper."""
        logger.info("Starting performance scraper")

        try:
            data = self.scraper.scrape(save_temp=True)
            self.raw_data["artificial_analysis"] = data
            return data
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            self.errors.append({"source": "scraper", "error": str(e)})
            return self._load_cached_performance()

    def _load_datasets(self) -> None:
        """Load benchmark datasets from HuggingFace."""
        self.raw_data["open_evals"] = self.dataset_loader.load_open_evals()
        self.raw_data["lmsys_arena"] = self.dataset_loader.load_lmsys()

    def _get_model_data(self, canonical: str) -> tuple:
        """Get all data for a canonical model name."""
        aa_data = None
        oe_data = None
        lmsys_data = None

        for item in self.raw_data["artificial_analysis"]:
            if item.get("model_name") == canonical:
                aa_data = item
                break

        for item in self.raw_data["open_evals"]:
            if item.get("model_name") == canonical or item.get("name") == canonical:
                oe_data = item
                break

        for item in self.raw_data["lmsys_arena"]:
            if (
                item.get("model_name") == canonical
                or item.get("name") == canonical
                or item.get("title") == canonical
            ):
                lmsys_data = item
                break

        return aa_data, oe_data, lmsys_data

    def _calculate_hardware_fit(
        self, model_id: str, hf_meta: Dict, context_tier: str
    ) -> tuple:
        """Calculate VRAM requirements and hardware fit."""
        size_gb = hf_meta.get("safetensors_size_gb", 0.0)

        if size_gb == 0.0:
            parsed = parse_model_size(model_id)
            size_gb = (
                estimate_size_from_params(param_count=int(parsed * 1e9))
                if parsed
                else 10.0
            )

        vram_req = VRAMCalculator(context_tier).calculate(size_gb)

        is_moe = hf_meta.get("is_moe", False) or is_moe_model(model_id)[0]

        hw_fit = {
            "gpu_id": "a100_80gb",
            "gpu_name": "A100 80GB",
            "gpu_count": 8,
            "total_vram_gb": 640,
            "status": "Compatible",
            "recommended_parallelism": (
                "Expert Parallelism (EP=8)"
                if is_moe
                else "Tensor Parallelism (TP=8) (Optimal)"
            ),
            "multi_gpu_scaling": True,
            "is_moe_model": is_moe,
            "hosting_strategy": "Expert-Distributed" if is_moe else "TP-Sharded",
            "context_overhead_tier": context_tier,
            "tier": "data_center",
        }

        all_fits = format_all_fits(vram_req, is_moe)

        return hw_fit, all_fits, "Expert-Distributed" if is_moe else "TP-Sharded"

    def run(self) -> Path:
        """Execute the complete MHII data gathering pipeline with parallel HF fetching."""
        import time
        from loguru import logger

        logger.info("=" * 60)
        logger.info("STARTING MHII DATA GATHERING PIPELINE")
        logger.info("=" * 60)

        # Phase 1: Sequential web scraping
        logger.info("Phase 1: Web scraping...")
        phase1_start = time.time()
        performance_data = self._run_scraper()
        logger.info(f"Phase 1 complete in {time.time() - phase1_start:.1f}s")

        # Phase 2: Sequential dataset loading (can run in parallel with phase 1 if needed)
        logger.info("Phase 2: Loading HF datasets...")
        phase2_start = time.time()
        self._load_datasets()
        logger.info(f"Phase 2 complete in {time.time() - phase2_start:.1f}s")

        # Build name mappings
        all_names = (
            [
                item.get("model_name")
                for item in performance_data
                if item.get("model_name")
            ]
            + self.dataset_loader.get_model_names("open_evals")
            + self.dataset_loader.get_model_names("lmsys_arena")
        )

        unique_names = list(set(all_names))
        logger.info(f"Found {len(unique_names)} unique models")

        self.matcher.build_mappings(
            [
                item.get("model_name")
                for item in performance_data
                if item.get("model_name")
            ],
            self.dataset_loader.get_model_names("open_evals"),
            self.dataset_loader.get_model_names("lmsys_arena"),
        )

        # Phase 3: PARALLEL HF metadata fetching (the main optimization!)
        logger.info("Phase 3: Parallel HF metadata fetching (20 threads)...")
        phase3_start = time.time()
        hf_metadata_results = self.hf_service.parallel_batch_fetch(
            unique_names, max_workers=20
        )
        logger.info(f"Phase 3 complete in {time.time() - phase3_start:.1f}s")

        # Phase 4: Process and merge results (fast, sequential)
        logger.info("Phase 4: Processing and merging results...")
        for model_name in unique_names:
            try:
                canonical = self.matcher.get_canonical(model_name) or model_name

                aa_data, oe_data, lmsys_data = self._get_model_data(canonical)

                benchmarks = self.merger.merge(aa_data, oe_data, lmsys_data)
                context_tier = get_recommended_context_tier(
                    benchmarks.get("intelligence_index"), canonical
                )

                # Use pre-fetched metadata from parallel phase
                hf_meta = hf_metadata_results.get(canonical)
                if hf_meta is None:
                    hf_meta = self.hf_service.fetch(canonical)  # fallback

                hf_dict = {
                    "model_id": hf_meta.model_id,
                    "repo_id": hf_meta.repo_id,
                    "safetensors_size_gb": hf_meta.safetensors_size_gb,
                    "is_moe": hf_meta.is_moe,
                    "num_experts": hf_meta.num_experts,
                    "metadata_status": hf_meta.metadata_status,
                }

                hw_fit, all_fits, strategy = self._calculate_hardware_fit(
                    canonical, hf_dict, context_tier
                )

                parsed_size = parse_model_size(canonical)
                size_gb = hf_meta.safetensors_size_gb or (
                    parsed_size * 2 if parsed_size else 10.0
                )
                vram_calc = VRAMCalculator(context_tier)
                vram_req = vram_calc.calculate(size_gb)

                record = {
                    "model_id": canonical,
                    "benchmarks": benchmarks,
                    "vram_gb": {
                        "fp16": vram_req.fp16_gb,
                        "int8": vram_req.int8_gb,
                        "int4": vram_req.int4_gb,
                        "model_base_gb": vram_req.model_size_gb,
                    },
                    "hardware_fit": hw_fit,
                    "hosting_strategy": strategy,
                    "source_status": hf_meta.metadata_status,
                    "all_gpu_compatibility": all_fits,
                    "hf_metadata": hf_dict,
                    "match_confidence": (
                        self.matcher.mappings[canonical].match_score
                        if canonical in self.matcher.mappings
                        else None
                    ),
                }

                self.records.append(record)

            except Exception as e:
                logger.error(f"Error processing {model_name}: {e}")
                self.errors.append({"model": model_name, "error": str(e)})

        # Phase 5: Save output
        logger.info("Phase 5: Saving output...")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_time = time.time() - phase1_start
        logger.success(
            f"Pipeline complete! {len(self.records)} records in {total_time:.1f}s"
        )
        logger.success(f"Saved to {self.output_path}")

        return self.output_path

    def get_report(self) -> Dict:
        """Generate pipeline execution report."""
        return {
            "total_models": len(self.records),
            "source_status": {
                "verified": sum(
                    1 for r in self.records if r["source_status"] == "verified"
                ),
                "missing_hf_metadata": sum(
                    1
                    for r in self.records
                    if r["source_status"] == "missing_hf_metadata"
                ),
            },
            "hosting_strategies": {
                "single_gpu": sum(
                    1 for r in self.records if r["hosting_strategy"] == "Single-GPU"
                ),
                "tp_sharded": sum(
                    1 for r in self.records if r["hosting_strategy"] == "TP-Sharded"
                ),
                "expert_distributed": sum(
                    1
                    for r in self.records
                    if r["hosting_strategy"] == "Expert-Distributed"
                ),
            },
            "hf_cache_stats": self.hf_service.get_cache_stats(),
            "errors": self.errors,
            "output_file": str(self.output_path),
        }
