"""Pytest-based validation test cases for the LLM recommender."""

import pytest
from dataclasses import dataclass
from typing import Optional

from backend.services.parser import parse_hardware_input, ParsedHardware
from backend.services.recommender import get_recommender, reset_recommender
from backend.services.wandb_logger import reset_wandb_logger


@dataclass
class TestCase:
    """Test case definition."""

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
        expected_model_pattern="llama|qwen|mistral",
        expected_min_coding=80.0,
        expected_params_range=(60, 150),
        description="8x A100 80GB should recommend top coding models like Mistral-Large, Llama-3.3-70B or Qwen2.5-72B",
    ),
    TestCase(
        name="TC-02: 1x H200 for Math",
        hardware_text="1 H200",
        use_case="mathematical reasoning and calculations",
        user_query="math calculus equation physics",
        expected_model_pattern=None,
        expected_min_coding=60.0,
        expected_params_range=(30, 100),
        description="H200 141GB should handle large math models",
    ),
    TestCase(
        name="TC-03: 4x RTX 4090 for Creative Writing",
        hardware_text="4 RTX 4090s",
        use_case="creative writing and storytelling",
        user_query="creative writing storytelling",
        expected_model_pattern=None,
        expected_min_coding=50.0,
        expected_params_range=(7, 75),
        description="4x RTX 4090 (96GB) can handle 30-75B models with quantization",
    ),
    TestCase(
        name="TC-04: MacBook M3 Max for On-Device Inference",
        hardware_text="MacBook M3 Max",
        use_case="on-device inference, portable AI assistant",
        user_query="portable chat assistant inference",
        expected_model_pattern=None,
        expected_min_coding=40.0,
        expected_params_range=(7, 50),
        description="M3 Max 128GB can handle larger models than consumer GPUs",
    ),
    TestCase(
        name="TC-05: 1x A100 40GB Memory-Constrained Coding",
        hardware_text="1 A100 40GB",
        use_case="memory-constrained code generation",
        user_query="code generation compact model",
        expected_model_pattern=None,
        expected_min_coding=60.0,
        expected_params_range=(7, 30),
        description="A100 40GB requires smaller models or INT4 quantization",
    ),
]


@pytest.fixture(scope="module")
def recommender():
    """Get recommender instance for all tests."""
    reset_recommender()
    reset_wandb_logger()
    return get_recommender()


@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.name)
def test_recommender_parsing(recommender, test_case: TestCase):
    """Test that hardware parsing works correctly."""
    hardware = parse_hardware_input(test_case.hardware_text)
    assert hardware is not None, f"Failed to parse hardware: '{test_case.hardware_text}'"
    assert hardware.gpu_name is not None
    assert hardware.total_vram_gb > 0


@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.name)
def test_recommender_returns_results(recommender, test_case: TestCase):
    """Test that recommender returns results."""
    hardware = parse_hardware_input(test_case.hardware_text)
    assert hardware is not None

    recommendations = recommender.recommend(
        hardware=hardware,
        use_case_text=test_case.use_case,
        user_query=test_case.user_query,
        top_k=5,
    )

    assert len(recommendations) > 0, f"No recommendations returned for {test_case.name}"


@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.name)
def test_recommender_param_range(recommender, test_case: TestCase):
    """Test that recommended models fall within expected parameter range."""
    hardware = parse_hardware_input(test_case.hardware_text)
    assert hardware is not None

    recommendations = recommender.recommend(
        hardware=hardware,
        use_case_text=test_case.use_case,
        user_query=test_case.user_query,
        top_k=5,
    )

    assert len(recommendations) > 0

    top_model = recommendations[0]
    min_params, max_params = test_case.expected_params_range
    assert min_params <= top_model.params_billions <= max_params, (
        f"Top model param count {top_model.params_billions}B is outside "
        f"expected range [{min_params}, {max_params}]B"
    )


@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.name)
def test_recommender_coding_score(recommender, test_case: TestCase):
    """Test that top model's coding score meets minimum threshold."""
    hardware = parse_hardware_input(test_case.hardware_text)
    assert hardware is not None

    recommendations = recommender.recommend(
        hardware=hardware,
        use_case_text=test_case.use_case,
        user_query=test_case.user_query,
        top_k=5,
    )

    assert len(recommendations) > 0

    top_model = recommendations[0]
    assert top_model.coding >= test_case.expected_min_coding, (
        f"Top model coding score {top_model.coding} is below "
        f"expected minimum {test_case.expected_min_coding}"
    )


@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.name)
def test_recommender_model_pattern(recommender, test_case: TestCase):
    """Test that top model matches expected pattern (if specified)."""
    if not test_case.expected_model_pattern:
        pytest.skip("No expected model pattern specified")

    hardware = parse_hardware_input(test_case.hardware_text)
    assert hardware is not None

    recommendations = recommender.recommend(
        hardware=hardware,
        use_case_text=test_case.use_case,
        user_query=test_case.user_query,
        top_k=5,
    )

    assert len(recommendations) > 0

    top_model = recommendations[0]
    patterns = [p.strip().lower() for p in test_case.expected_model_pattern.split("|")]
    assert any(pat in top_model.model_id.lower() for pat in patterns), (
        f"Top model {top_model.model_id} does not match "
        f"expected pattern {test_case.expected_model_pattern}"
    )


@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.name)
def test_recommender_all_scores_valid(recommender, test_case: TestCase):
    """Test that all scoring fields are valid numbers."""
    hardware = parse_hardware_input(test_case.hardware_text)
    assert hardware is not None

    recommendations = recommender.recommend(
        hardware=hardware,
        use_case_text=test_case.use_case,
        user_query=test_case.user_query,
        top_k=5,
    )

    assert len(recommendations) > 0

    for model in recommendations:
        assert 0 <= model.final_score <= 1, f"Invalid final_score: {model.final_score}"
        assert 0 <= model.semantic_score <= 1, f"Invalid semantic_score: {model.semantic_score}"
        assert 0 <= model.benchmark_score <= 1, f"Invalid benchmark_score: {model.benchmark_score}"
        assert 0 <= model.hardware_score <= 1, f"Invalid hardware_score: {model.hardware_score}"


@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.name)
def test_recommender_has_huggingface_url(recommender, test_case: TestCase):
    """Test that recommended models have HuggingFace repo IDs."""
    hardware = parse_hardware_input(test_case.hardware_text)
    assert hardware is not None

    recommendations = recommender.recommend(
        hardware=hardware,
        use_case_text=test_case.use_case,
        user_query=test_case.user_query,
        top_k=5,
    )

    assert len(recommendations) > 0

    for model in recommendations:
        if model.hf_repo_id:
            assert "huggingface.co" not in model.hf_repo_id


def test_recommender_handles_empty_use_case():
    """Test that recommender handles empty use case text."""
    recommender = get_recommender()
    hardware = parse_hardware_input("8 A100s")

    recommendations = recommender.recommend(
        hardware=hardware,
        use_case_text="",
        user_query="general chat",
        top_k=5,
    )

    assert len(recommendations) > 0


def test_recommender_model_count():
    """Test that recommender reports correct model count."""
    recommender = get_recommender()
    assert recommender.model_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])