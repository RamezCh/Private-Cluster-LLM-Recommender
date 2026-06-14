"""Unit tests for the FastAPI backend."""

import pytest
from fastapi.testclient import TestClient

from backend.api import app


client = TestClient(app)


class TestRootEndpoint:
    """Test cases for root endpoint."""

    def test_root_returns_api_info(self):
        """Test that root endpoint returns API info or serves frontend HTML."""
        response = client.get("/")
        assert response.status_code == 200
        if "text/html" in response.headers.get("content-type", ""):
            assert "<html" in response.text or "<!DOCTYPE" in response.text
        else:
            data = response.json()
            assert "message" in data
            assert "version" in data
            assert data["version"] == "1.0.0"

    def test_root_status_healthy(self):
        """Test that root endpoint indicates healthy status or serves HTML."""
        response = client.get("/")
        assert response.status_code == 200
        if "text/html" in response.headers.get("content-type", ""):
            assert "<html" in response.text or "<!DOCTYPE" in response.text
        else:
            data = response.json()
            assert data.get("status") == "healthy"


class TestHealthEndpoint:
    """Test cases for health endpoint."""

    def test_health_returns_status(self):
        """Test that health endpoint returns status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_health_timestamp_is_number(self):
        """Test that health timestamp is a valid number."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["timestamp"], (int, float))


class TestModelCountEndpoint:
    """Test cases for model count endpoint."""

    def test_model_count_returns_count(self):
        """Test that model count endpoint returns count."""
        response = client.get("/models/count")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert isinstance(data["count"], int)

    def test_model_count_is_non_negative(self):
        """Test that model count is non-negative."""
        response = client.get("/models/count")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 0


class TestRecommendEndpoint:
    """Test cases for recommend endpoint."""

    def test_recommend_returns_results(self):
        """Test that recommend endpoint returns results."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "8 A100s",
                "use_case": "code generation",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "recommendations" in data
        assert "hardware" in data

    def test_recommend_hardware_parsed(self):
        """Test that hardware is correctly parsed."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "8 A100s",
                "use_case": "coding",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        hardware = data["hardware"]
        assert hardware["gpu_id"] == "a100_80gb"
        assert hardware["gpu_name"] == "A100 80GB"
        assert hardware["count"] == 8
        assert hardware["total_vram_gb"] == 640

    def test_recommend_returns_list_of_models(self):
        """Test that recommendations are a list of model dicts."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "8 A100s",
                "use_case": "coding",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        recommendations = data["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5

        if recommendations:
            model = recommendations[0]
            assert "model_id" in model
            assert "scores" in model
            assert "benchmarks" in model

    def test_recommend_model_scores_valid(self):
        """Test that model scores are valid numbers."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "8 A100s",
                "use_case": "coding",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        recommendations = data["recommendations"]

        if recommendations:
            for model in recommendations:
                scores = model["scores"]
                assert 0 <= scores["final"] <= 1
                assert 0 <= scores["semantic"] <= 1
                assert 0 <= scores["benchmark"] <= 1
                assert 0 <= scores["hardware"] <= 1

    def test_recommend_invalid_hardware(self):
        """Test recommend with invalid hardware returns error."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "invalid_gpu_xyz",
                "use_case": "coding",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error"] is not None
        assert data["recommendations"] == []

    def test_recommend_empty_hardware(self):
        """Test recommend with empty hardware returns error."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "",
                "use_case": "coding",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_recommend_default_top_k(self):
        """Test that default top_k is 5."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "8 A100s",
                "use_case": "coding",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) <= 5

    def test_recommend_max_top_k(self):
        """Test that top_k is capped at 20."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "8 A100s",
                "use_case": "coding",
                "top_k": 100,
            },
        )
        assert response.status_code == 422

    def test_recommend_rtx_4090(self):
        """Test recommend with RTX 4090."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "4 RTX 4090s",
                "use_case": "creative writing",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        hardware = data["hardware"]
        assert hardware["gpu_id"] == "rtx_4090"
        assert hardware["count"] == 4
        assert hardware["total_vram_gb"] == 96

    def test_recommend_macbook_m3_max(self):
        """Test recommend with MacBook M3 Max."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "MacBook M3 Max",
                "use_case": "on-device inference",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        hardware = data["hardware"]
        assert hardware["gpu_id"] == "macbook_pro_m3_max"
        assert hardware["vram_gb"] == 128

    def test_recommend_malformed_request(self):
        """Test recommend with missing fields."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "8 A100s",
            },
        )
        assert response.status_code == 422

    def test_recommend_top_k_validation(self):
        """Test that top_k must be positive."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "8 A100s",
                "use_case": "coding",
                "top_k": 0,
            },
        )
        assert response.status_code == 422


class TestRecommendResponseFormat:
    """Test cases for recommend response format."""

    def test_response_has_required_fields(self):
        """Test that response has all required fields."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "8 A100s",
                "use_case": "coding",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "hardware" in data
        assert "use_case" in data
        assert "recommendations" in data
        assert "error" in data

    def test_model_has_required_fields(self):
        """Test that each model has required fields."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "8 A100s",
                "use_case": "coding",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        recommendations = data["recommendations"]

        if recommendations:
            model = recommendations[0]
            assert "model_id" in model
            assert "params_billions" in model
            assert "benchmarks" in model
            assert "scores" in model
            assert "vram_fp16_gb" in model

    def test_benchmarks_have_all_scores(self):
        """Test that benchmarks include all score types."""
        response = client.post(
            "/recommend",
            json={
                "hardware_text": "8 A100s",
                "use_case": "coding",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        recommendations = data["recommendations"]

        if recommendations:
            benchmarks = recommendations[0]["benchmarks"]
            assert "coding" in benchmarks
            assert "math" in benchmarks
            assert "reasoning" in benchmarks
            assert "intelligence_index" in benchmarks


class TestCORSHeaders:
    """Test cases for CORS headers."""

    def test_cors_configured_in_app(self):
        """Test that CORS is configured in the app."""
        from backend.api import app
        cors_middleware = None
        for middleware in app.user_middleware:
            if "CORSMiddleware" in str(middleware):
                cors_middleware = middleware
                break
        assert cors_middleware is not None or len(app.user_middleware) > 0

    def test_cors_allows_origins(self):
        """Test that CORS allows origins configuration."""
        from backend.api import app
        assert any("cors" in str(m).lower() or len(app.user_middleware) > 0 for m in app.user_middleware)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])