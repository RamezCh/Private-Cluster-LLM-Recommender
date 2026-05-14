"""Test configuration and shared fixtures."""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_path(project_root):
    """Get path to model database."""
    return project_root / "data_gathering_pipeline" / "data" / "master_model_db.jsonl"


@pytest.fixture(scope="session")
def db_exists(data_path):
    """Check if model database exists."""
    return data_path.exists()


@pytest.fixture
def sample_hardware_8x_a100():
    """Sample hardware: 8x A100 80GB."""
    return {
        "gpu_id": "a100_80gb",
        "gpu_name": "A100 80GB",
        "vram_gb": 80,
        "count": 8,
        "total_vram_gb": 640,
        "tier": "data_center",
    }


@pytest.fixture
def sample_hardware_4x_rtx4090():
    """Sample hardware: 4x RTX 4090 24GB."""
    return {
        "gpu_id": "rtx_4090",
        "gpu_name": "RTX 4090 24GB",
        "vram_gb": 24,
        "count": 4,
        "total_vram_gb": 96,
        "tier": "consumer",
    }


@pytest.fixture
def sample_hardware_macbook_m3_max():
    """Sample hardware: MacBook M3 Max."""
    return {
        "gpu_id": "macbook_pro_m3_max",
        "gpu_name": "MacBook Pro M3 Max 128GB",
        "vram_gb": 128,
        "count": 1,
        "total_vram_gb": 128,
        "tier": "laptop",
    }


@pytest.fixture
def sample_use_cases():
    """Sample use case texts."""
    return {
        "coding": "code generation and debugging python",
        "math": "mathematical reasoning and calculations",
        "reasoning": "logical reasoning and problem solving",
        "general": "general chat and assistant tasks",
    }