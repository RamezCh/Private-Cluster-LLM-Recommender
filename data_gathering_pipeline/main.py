#!/usr/bin/env python3
"""
MHII Data Gathering Pipeline v2 — Main Entry Point

Model-Hardware Intelligence Index (MHII) v2
Bridges the gap between "how smart a model is" and "can I actually host it?"

TARGET: Open-weight, locally-hostable LLMs only
Data sources:
  1. open-llm-leaderboard (HF Dataset) — primary, ~4.5K rows
  2. OpenCompass General — academic benchmarks
  3. OpenCompass Academic — real-time academic benchmarks

Usage:
    python main.py                    # Run full pipeline
    python main.py --hf-only          # Run with HF dataset only (no OpenCompass)
    python main.py --scrape-only      # Only scrape OpenCompass (save to cache)
    python main.py --merge-only       # Only merge using cached OpenCompass data
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
sys.path.insert(0, str(BASE_DIR.parent))

from src.config import LOGS_DIR, OUTPUT_FILE, logging_config
from src.fetchers import OpenCompassScraper
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
    """Execute the complete MHII v2 pipeline."""
    logger.info("Running full MHII v2 pipeline")

    orchestrator = Orchestrator(output_path=args.output or OUTPUT_FILE)
    skip_opencompass = getattr(args, "hf_only", False)

    output_path = orchestrator.run(skip_opencompass=skip_opencompass)

    report = orchestrator.get_report()

    print("\n" + "=" * 60)
    print("PIPELINE REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))

    return output_path


def run_scrape_only(args):
    """Run only the OpenCompass scraper (save to cache)."""
    logger.info("Running scrape-only mode for OpenCompass")

    scraper = OpenCompassScraper(headless=not args.visible)

    try:
        general, academic = scraper.scrape_both(
            general_month="26-04",
            academic_month="REALTIME",
        )
    finally:
        scraper.close()

    if general:
        scraper.save_temp(general, Path("data/temp/temp_oc_general.jsonl"))
        print(f"\nGeneral leaderboard: {len(general)} models")

    if academic:
        scraper.save_temp(academic, Path("data/temp/temp_oc_academic.jsonl"))
        print(f"Academic leaderboard: {len(academic)} models")

    return general, academic


def run_merge_only(args):
    """Run only merge using cached OpenCompass data (no scraping)."""
    logger.info("Running merge-only mode (using cached OpenCompass data)")

    orchestrator = Orchestrator(output_path=args.output or OUTPUT_FILE)

    orchestrator._load_open_llm_leaderboard()
    orchestrator._scrape_opencompass(skip_scrape=True, use_cache=True)
    orchestrator._deduplicate()
    orchestrator._merge_benchmarks()
    orchestrator._impute_missing_benchmarks()
    orchestrator._enrich_hf_metadata(max_workers=20)
    orchestrator._calculate_vram()
    orchestrator._build_final_records()
    path = orchestrator._save_output()

    report = orchestrator.get_report()

    print("\n" + "=" * 60)
    print("MERGE-ONLY REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))

    return path


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

    total = len(records)

    benchmarks = defaultdict(int)
    for r in records:
        for key in ["coding", "math", "reasoning", "intelligence_index"]:
            if r.get("benchmarks", {}).get(key) is not None:
                benchmarks[key] += 1

    strategies = defaultdict(int)
    for r in records:
        strategies[r.get("hosting_strategy", "Unknown")] += 1

    moe_count = sum(1 for r in records if r.get("is_moe"))

    multi_source = sum(
        1 for r in records if len(r.get("_sources", [])) > 1
    )

    avg_intel = []
    for r in records:
        idx = r.get("benchmarks", {}).get("intelligence_index")
        if idx is not None:
            avg_intel.append(idx)

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_models": total,
        "architecture_types": {"moe": moe_count, "dense": total - moe_count},
        "hosting_strategies": dict(strategies),
        "benchmark_coverage": {
            key: f"{count} models ({round(count/total*100, 1) if total else 0}%)"
            for key, count in sorted(benchmarks.items(), key=lambda x: x[1], reverse=True)
        },
        "multi_source_models": multi_source,
        "avg_intelligence_index": round(sum(avg_intel) / len(avg_intel), 2) if avg_intel else None,
        "source_status": {
            "verified": sum(1 for r in records if r.get("source_status") == "verified"),
            "missing_hf": sum(1 for r in records if r.get("source_status") == "missing_hf_metadata"),
            "merged": sum(1 for r in records if r.get("source_status") == "merged"),
        },
    }

    print("\n" + "=" * 60)
    print("MHII v2 DATABASE REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MHII v2 Data Gathering Pipeline — Open Source LLM Recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py               Run full pipeline (~5-8 min)
  python main.py --hf-only     Run with HF dataset only (skip OpenCompass scrape)
  python main.py --scrape-only Only scrape OpenCompass (saves to cache)
  python main.py --merge-only  Merge using cached data (no scraping)
  python main.py --report      Generate report from existing JSONL
  python main.py --visible     Show browser during OpenCompass scraping
  python main.py --output ./custom.jsonl   Custom output path
        """,
    )

    parser.add_argument(
        "--hf-only", action="store_true",
        help="Run with open-llm-leaderboard only (skip OpenCompass)"
    )
    parser.add_argument(
        "--scrape-only", action="store_true",
        help="Only scrape OpenCompass leaderboards (save to cache)"
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Merge using cached OpenCompass data (no scraping)"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate report from existing data"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_FILE,
        help="Output file path"
    )
    parser.add_argument(
        "--input", type=Path, default=OUTPUT_FILE,
        help="Input file for --report"
    )
    parser.add_argument(
        "--visible", action="store_true",
        help="Show browser during OpenCompass scraping"
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
    main()
    sys.exit(0)