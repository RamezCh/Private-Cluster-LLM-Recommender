#!/usr/bin/env python3
"""
MHII Data Gathering Pipeline - Main Entry Point

Model-Hardware Intelligence Index (MHII)
Bridges the gap between "how smart a model is" and "can I actually host it?"

Usage:
    python main.py                    # Run full pipeline
    python main.py --scrape-only      # Only run web scraper
    python main.py --merge-only       # Only merge using cached data
    python main.py --report           # Generate report from existing data
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from loguru import logger

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.config import LOGS_DIR, OUTPUT_FILE, logging_config, HF_TOKEN
from src.fetchers import WebScraper, HFDatasetLoader
from src.orchestrator import Orchestrator


def setup_logging():
    """Configure Loguru logging."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=logging_config["level"],
        format=logging_config["format"],
    )

    logger.add(
        LOGS_DIR / f"mhii_{datetime.now():%Y%m%d_%H%M%S}.log",
        rotation=logging_config["rotation"],
        retention=logging_config["retention"],
        format=logging_config["format"],
        level=logging_config["level"],
    )


def run_full(args):
    """Execute the complete MHII pipeline."""
    logger.info("Running full MHII pipeline")

    orchestrator = Orchestrator(output_path=args.output or OUTPUT_FILE)
    output_path = orchestrator.run()

    report = orchestrator.get_report()

    print("\n" + "=" * 60)
    print("PIPELINE REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))

    return output_path


def run_scrape_only(args):
    """Run only the web scraper."""
    logger.info("Running scrape-only mode")

    scraper = WebScraper(headless=not args.visible)
    data = scraper.scrape(save_temp=True)

    print(f"\nScraped {len(data)} models")

    return data


def run_merge_only(args):
    """Run only merge using cached data."""
    logger.info("Running merge-only mode")

    from src.config import TEMP_PERFORMANCE_FILE

    if not TEMP_PERFORMANCE_FILE.exists():
        logger.error("No cached data. Run full pipeline first.")
        return None

    orchestrator = Orchestrator(output_path=args.output or OUTPUT_FILE)
    orchestrator.raw_data["artificial_analysis"] = json.loads(
        TEMP_PERFORMANCE_FILE.read_text()
    )

    dataset_loader = HFDatasetLoader()
    orchestrator.raw_data["open_evals"] = dataset_loader.load_open_evals()
    orchestrator.raw_data["lmsys_arena"] = dataset_loader.load_lmsys()

    orchestrator.output_path = args.output or OUTPUT_FILE

    all_names = (
        [
            item.get("model_name")
            for item in orchestrator.raw_data["artificial_analysis"]
        ]
        + dataset_loader.get_model_names("open_evals")
        + dataset_loader.get_model_names("lmsys_arena")
    )

    unique_names = list(set(all_names))
    logger.info(f"Processing {len(unique_names)} unique models")

    orchestrator.matcher.build_mappings(
        [
            item.get("model_name")
            for item in orchestrator.raw_data["artificial_analysis"]
            if item.get("model_name")
        ],
        dataset_loader.get_model_names("open_evals"),
        dataset_loader.get_model_names("lmsys_arena"),
    )

    from src.services import (
        VRAMCalculator,
        is_moe_model,
        parse_model_size,
        estimate_size_from_params,
        get_recommended_context_tier,
        format_all_fits,
    )
    from src.orchestrator import BenchmarkMerger

    for model_name in unique_names:
        try:

            canonical = orchestrator.matcher.get_canonical(model_name) or model_name

            aa_item = next(
                (
                    i
                    for i in orchestrator.raw_data["artificial_analysis"]
                    if i.get("model_name") == canonical
                ),
                None,
            )
            oe_item = next(
                (
                    i
                    for i in orchestrator.raw_data["open_evals"]
                    if i.get("model_name") == canonical or i.get("name") == canonical
                ),
                None,
            )
            lmsys_item = next(
                (
                    i
                    for i in orchestrator.raw_data["lmsys_arena"]
                    if i.get("name") == canonical or i.get("title") == canonical
                ),
                None,
            )

            benchmarks = BenchmarkMerger.merge(aa_item, oe_item, lmsys_item)
            context_tier = get_recommended_context_tier(
                benchmarks.get("intelligence_index"), canonical
            )

            hf_meta = orchestrator.hf_service.fetch(canonical)

            size_gb = hf_meta.safetensors_size_gb
            if size_gb == 0.0:
                parsed = parse_model_size(canonical)
                size_gb = (
                    estimate_size_from_params(param_count=int(parsed * 1e9))
                    if parsed
                    else 10.0
                )

            vram_req = VRAMCalculator(context_tier).calculate(size_gb)

            is_moe = hf_meta.is_moe or is_moe_model(canonical)[0]
            all_fits = format_all_fits(vram_req, is_moe)

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

            orchestrator.records.append(
                {
                    "model_id": canonical,
                    "benchmarks": benchmarks,
                    "vram_gb": {
                        "fp16": vram_req.fp16_gb,
                        "int8": vram_req.int8_gb,
                        "int4": vram_req.int4_gb,
                        "model_base_gb": vram_req.model_size_gb,
                    },
                    "hardware_fit": hw_fit,
                    "hosting_strategy": (
                        "Expert-Distributed" if is_moe else "TP-Sharded"
                    ),
                    "source_status": hf_meta.metadata_status,
                    "all_gpu_compatibility": all_fits,
                    "hf_metadata": {
                        "repo_id": hf_meta.repo_id,
                        "safetensors_size_gb": hf_meta.safetensors_size_gb,
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error: {model_name}: {e}")

    orchestrator.output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(orchestrator.output_path, "w", encoding="utf-8") as f:
        for record in orchestrator.records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.success(f"Saved {len(orchestrator.records)} records")

    return orchestrator.output_path


def run_report(args):
    """Generate report from existing data."""
    input_path = args.input or OUTPUT_FILE

    if not Path(input_path).exists():
        logger.error(f"No data file found: {input_path}")
        return None

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    strategies: defaultdict = defaultdict(int)
    moe_count = 0
    avg_elo = []

    for r in records:
        strategies[r.get("hosting_strategy", "Unknown")] += 1
        if r.get("hardware_fit", {}).get("is_moe_model"):
            moe_count += 1
        if r.get("benchmarks", {}).get("elo"):
            avg_elo.append(r["benchmarks"]["elo"])

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_models": len(records),
        "architecture_types": {"moe": moe_count, "dense": len(records) - moe_count},
        "hosting_strategies": strategies,
        "benchmark_averages": {
            "avg_elo": round(sum(avg_elo) / len(avg_elo), 2) if avg_elo else None,
        },
    }

    print("\n" + "=" * 60)
    print("MHII DATABASE REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MHII Data Gathering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py               Run full pipeline
  python main.py --scrape-only  Only scrape web data
  python main.py --merge-only   Only merge (use cached)
  python main.py --report       Generate report from data
        """,
    )

    parser.add_argument(
        "--scrape-only", action="store_true", help="Run only web scraper"
    )
    parser.add_argument(
        "--merge-only", action="store_true", help="Merge using cached data"
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate report from data"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_FILE, help="Output file path"
    )
    parser.add_argument(
        "--input", type=Path, default=OUTPUT_FILE, help="Input file for --report"
    )
    parser.add_argument(
        "--visible", action="store_true", help="Show browser during scraping"
    )

    args = parser.parse_args()
    setup_logging()

    try:
        if args.scrape_only:
            return run_scrape_only(args)
        elif args.merge_only:
            return run_merge_only(args)
        elif args.report:
            return run_report(args)
        else:
            return run_full(args)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main() or 0)
