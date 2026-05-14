import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.hardware_parser import parse_hardware_input, ParsedHardware
from backend.services.recommender import get_recommender, ScoredModel


@dataclass
class TestCase:
    name: str
    hardware_text: str
    use_case: str
    user_query: str
    expected_model_pattern: Optional[str] = None
    expected_min_coding: float = 0.0
    expected_params_range: tuple[float, float] = (0, 1000)
    description: str = ""


TEST_CASES = [
    TestCase(
        name="TC-01: 8x A100 for Coding",
        hardware_text="8 A100s",
        use_case="code generation and debugging",
        user_query="code generation programming python",
        expected_model_pattern="llama",
        expected_min_coding=80.0,
        expected_params_range=(60, 100),
        description="8x A100 80GB should recommend top coding models like Llama-3.3-70B"
    ),
    TestCase(
        name="TC-02: 1x H200 for Math",
        hardware_text="1 H200",
        use_case="mathematical reasoning and calculations",
        user_query="math calculus equation physics",
        expected_model_pattern=None,
        expected_min_coding=60.0,
        expected_params_range=(30, 100),
        description="H200 141GB should handle large math models"
    ),
    TestCase(
        name="TC-03: 4x RTX 4090 for Creative Writing",
        hardware_text="4 RTX 4090s",
        use_case="creative writing and storytelling",
        user_query="creative writing storytelling",
        expected_model_pattern=None,
        expected_min_coding=50.0,
        expected_params_range=(7, 50),
        description="4x RTX 4090 (96GB) can handle ~30-40B models"
    ),
    TestCase(
        name="TC-04: MacBook M3 Max for On-Device Inference",
        hardware_text="MacBook M3 Max",
        use_case="on-device inference, portable AI assistant",
        user_query="portable chat assistant inference",
        expected_model_pattern=None,
        expected_min_coding=40.0,
        expected_params_range=(7, 50),
        description="M3 Max 128GB can handle larger models than consumer GPUs"
    ),
    TestCase(
        name="TC-05: 1x A100 40GB Memory-Constrained Coding",
        hardware_text="1 A100 40GB",
        use_case="memory-constrained code generation",
        user_query="code generation compact model",
        expected_model_pattern=None,
        expected_min_coding=60.0,
        expected_params_range=(7, 30),
        description="A100 40GB requires smaller models or INT4 quantization"
    ),
]


def run_test_case(test_case: TestCase, verbose: bool = True) -> dict:
    result = {
        "name": test_case.name,
        "passed": False,
        "error": None,
        "recommendations": [],
        "latency_ms": 0,
        "checks": {}
    }
    
    try:
        start_time = time.time()
        
        hardware = parse_hardware_input(test_case.hardware_text)
        
        if hardware is None:
            result["error"] = f"Failed to parse hardware: '{test_case.hardware_text}'"
            return result
        
        recommender = get_recommender()
        recommendations = recommender.recommend(
            hardware=hardware,
            use_case_text=test_case.use_case,
            user_query=test_case.user_query,
            top_k=5
        )
        
        result["latency_ms"] = (time.time() - start_time) * 1000
        result["recommendations"] = recommendations
        
        if not recommendations:
            result["error"] = "No recommendations returned"
            return result
        
        top_model = recommendations[0]
        
        result["checks"]["has_recommendations"] = len(recommendations) > 0
        result["checks"]["top_model_params"] = (
            test_case.expected_params_range[0] <= top_model.params_billions <= test_case.expected_params_range[1]
        )
        result["checks"]["top_model_coding"] = top_model.coding >= test_case.expected_min_coding
        
        if test_case.expected_model_pattern:
            result["checks"]["matches_pattern"] = (
                test_case.expected_model_pattern.lower() in top_model.model_id.lower()
            )
        
        result["passed"] = all(result["checks"].values())
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Test: {test_case.name}")
            print(f"{'='*60}")
            print(f"Hardware: {test_case.hardware_text}")
            print(f"  -> {hardware.count}x {hardware.gpu_name} ({hardware.total_vram_gb} GB)")
            print(f"Use Case: {test_case.use_case}")
            print(f"Latency: {result['latency_ms']:.1f} ms")
            print(f"\nTop 3 Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec.model_id}")
                print(f"     Params: {rec.params_billions:.1f}B | Coding: {rec.coding:.1f}")
                print(f"     Score: {rec.final_score:.3f}")
            print(f"\nChecks:")
            for check, passed in result["checks"].items():
                status = "[PASS]" if passed else "[FAIL]"
                print(f"  {status} {check}: {passed}")
            print(f"\nResult: {'PASSED' if result['passed'] else 'FAILED'}")
    
    except Exception as e:
        result["error"] = str(e)
    
    return result


def run_all_tests(verbose: bool = True) -> dict:
    summary = {
        "total": len(TEST_CASES),
        "passed": 0,
        "failed": 0,
        "results": []
    }
    
    print("\n" + "="*70)
    print("RUNNING ALL TEST CASES")
    print("="*70)
    
    for test_case in TEST_CASES:
        result = run_test_case(test_case, verbose=verbose)
        summary["results"].append(result)
        
        if result["passed"]:
            summary["passed"] += 1
        else:
            summary["failed"] += 1
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total:  {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['passed']/summary['total']*100:.0f}%")
    
    return summary


if __name__ == "__main__":
    run_all_tests(verbose=True)