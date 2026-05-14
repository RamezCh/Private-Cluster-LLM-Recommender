"""Unit tests for hardware parser."""

import pytest
from backend.services.parser import parse_hardware_input, get_available_gpu_options, ParsedHardware


class TestParseHardwareInput:
    """Test cases for hardware input parsing."""

    def test_parse_single_a100(self):
        """Test parsing single A100."""
        result = parse_hardware_input("A100")
        assert result is not None
        assert result.gpu_id == "a100_80gb"
        assert result.gpu_name == "A100 80GB"
        assert result.count == 1
        assert result.vram_gb == 80
        assert result.total_vram_gb == 80

    def test_parse_8x_a100(self):
        """Test parsing 8x A100."""
        result = parse_hardware_input("8 A100s")
        assert result is not None
        assert result.gpu_id == "a100_80gb"
        assert result.count == 8
        assert result.total_vram_gb == 640

    def test_parse_8x_a100_with_x(self):
        """Test parsing 8x A100 with 'x' separator."""
        result = parse_hardware_input("8x A100s")
        assert result is not None
        assert result.count == 8
        assert result.total_vram_gb == 640

    def test_parse_a100_80gb(self):
        """Test parsing A100 80GB."""
        result = parse_hardware_input("A100 80GB")
        assert result is not None
        assert result.gpu_id == "a100_80gb"
        assert result.vram_gb == 80

    def test_parse_a100_40gb(self):
        """Test parsing A100 40GB."""
        result = parse_hardware_input("A100 40GB")
        assert result is not None
        assert result.gpu_id == "a100_40gb"
        assert result.vram_gb == 40

    def test_parse_h100(self):
        """Test parsing H100."""
        result = parse_hardware_input("H100")
        assert result is not None
        assert result.gpu_id == "h100_80gb"
        assert result.vram_gb == 80

    def test_parse_h200(self):
        """Test parsing H200."""
        result = parse_hardware_input("H200 141GB")
        assert result is not None
        assert result.gpu_id == "h200_141gb"
        assert result.vram_gb == 141

    def test_parse_rtx_4090(self):
        """Test parsing RTX 4090."""
        result = parse_hardware_input("RTX 4090")
        assert result is not None
        assert result.gpu_id == "rtx_4090"
        assert result.vram_gb == 24

    def test_parse_rtx4090_no_space(self):
        """Test parsing RTX4090 without space."""
        result = parse_hardware_input("RTX4090")
        assert result is not None
        assert result.gpu_id == "rtx_4090"

    def test_parse_4x_rtx_4090(self):
        """Test parsing 4x RTX 4090."""
        result = parse_hardware_input("4 RTX 4090s")
        assert result is not None
        assert result.gpu_id == "rtx_4090"
        assert result.count == 4
        assert result.total_vram_gb == 96

    def test_parse_macbook_m3_max(self):
        """Test parsing MacBook M3 Max."""
        result = parse_hardware_input("MacBook M3 Max")
        assert result is not None
        assert result.gpu_id == "macbook_pro_m3_max"
        assert result.vram_gb == 128
        assert result.tier == "laptop"

    def test_parse_macbook_m3_pro(self):
        """Test parsing MacBook M3 Pro."""
        result = parse_hardware_input("M3 Pro")
        assert result is not None
        assert result.gpu_id == "macbook_pro_m3_pro"

    def test_parse_b200(self):
        """Test parsing B200."""
        result = parse_hardware_input("B200")
        assert result is not None
        assert result.gpu_id == "b200_192gb"
        assert result.vram_gb == 192

    def test_parse_mi300x(self):
        """Test parsing MI300X."""
        result = parse_hardware_input("MI300X")
        assert result is not None
        assert result.gpu_id == "mi300x_192gb"

    def test_parse_invalid_input(self):
        """Test parsing invalid input returns None."""
        result = parse_hardware_input("invalid_gpu_xyz_123")
        assert result is None

    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        result = parse_hardware_input("")
        assert result is None

    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only string returns None."""
        result = parse_hardware_input("   ")
        assert result is None

    def test_parse_case_insensitive(self):
        """Test parsing is case insensitive."""
        result1 = parse_hardware_input("A100")
        result2 = parse_hardware_input("a100")
        assert result1 is not None
        assert result2 is not None
        assert result1.gpu_id == result2.gpu_id

    def test_parse_tier_data_center(self):
        """Test that data center GPUs have correct tier."""
        for gpu_name in ["A100", "H100", "H200", "B200", "V100"]:
            result = parse_hardware_input(gpu_name)
            if result:
                assert result.tier == "data_center", f"{gpu_name} should be data_center tier"

    def test_parse_tier_consumer(self):
        """Test that consumer GPUs have correct tier."""
        for gpu_name in ["RTX 4090", "RTX 3090", "RTX 4080"]:
            result = parse_hardware_input(gpu_name)
            if result:
                assert result.tier == "consumer", f"{gpu_name} should be consumer tier"

    def test_parse_tier_laptop(self):
        """Test that laptop GPUs have correct tier."""
        for gpu_name in ["MacBook M3 Max", "M3 Max"]:
            result = parse_hardware_input(gpu_name)
            if result:
                assert result.tier == "laptop", f"{gpu_name} should be laptop tier"


class TestGetAvailableGpuOptions:
    """Test cases for GPU options retrieval."""

    def test_returns_list_of_tuples(self):
        """Test that function returns list of (name, id) tuples."""
        result = get_available_gpu_options()
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], str)

    def test_all_gpus_have_valid_ids(self):
        """Test that all GPU options have valid IDs."""
        result = get_available_gpu_options()
        valid_ids = {
            "a100_40gb", "a100_80gb", "h100_80gb", "h100_sxm5_80gb",
            "h200_141gb", "b200_192gb", "b100_192gb", "v100_16gb",
            "v100_32gb", "p100_16gb", "mi300x_192gb", "mi250_128gb",
            "a6000_48gb", "a5000_24gb", "a4000_16gb", "rtx_4090",
            "rtx_3090", "rtx_4080", "rtx_4070", "rtx_3090_ti",
            "rtx_4080_s", "rtx_4070_ti", "laptop_rtx_4060",
            "laptop_rtx_4070", "laptop_rtx_4080", "laptop_3090",
            "macbook_pro_m3_max", "macbook_pro_m2_max", "macbook_pro_m3_pro",
            "macbook_pro_m2_pro", "macbook_pro_m3", "laptop_integrated",
        }
        for name, gpu_id in result:
            assert gpu_id in valid_ids, f"Invalid GPU ID: {gpu_id}"

    def test_no_duplicate_ids(self):
        """Test that no GPU IDs are duplicated."""
        result = get_available_gpu_options()
        ids = [gpu_id for _, gpu_id in result]
        assert len(ids) == len(set(ids)), "Duplicate GPU IDs found"


class TestParsedHardware:
    """Test cases for ParsedHardware dataclass."""

    def test_parsed_hardware_fields(self):
        """Test that ParsedHardware has all required fields."""
        result = parse_hardware_input("8 A100s")
        assert result is not None
        assert hasattr(result, "gpu_id")
        assert hasattr(result, "gpu_name")
        assert hasattr(result, "vram_gb")
        assert hasattr(result, "count")
        assert hasattr(result, "total_vram_gb")
        assert hasattr(result, "tier")

    def test_total_vram_calculation(self):
        """Test that total VRAM is calculated correctly."""
        result = parse_hardware_input("4 RTX 4090")
        assert result is not None
        assert result.total_vram_gb == result.vram_gb * result.count

    def test_default_count_is_one(self):
        """Test that default count is 1 when not specified."""
        result = parse_hardware_input("A100")
        assert result is not None
        assert result.count == 1