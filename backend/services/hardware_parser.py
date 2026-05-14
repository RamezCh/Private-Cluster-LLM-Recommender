import re
from dataclasses import dataclass
from typing import Optional

from .gpu_catalog import GPU_CATALOG, GPU_NAME_MAPPINGS


@dataclass
class ParsedHardware:
    gpu_id: str
    gpu_name: str
    vram_gb: float
    count: int
    total_vram_gb: float
    tier: str


def parse_hardware_input(text: str) -> Optional[ParsedHardware]:
    original_text = text.lower().strip()
    text = original_text
    
    count_pattern = r'^(\d+)\s*(?:x\s*)?'
    
    count_match = re.match(count_pattern, text)
    count = 1
    if count_match:
        count = int(count_match.group(1))
        text = text[count_match.end():].strip()
    
    text = re.sub(r'^(?:x|×)\s*', '', text).strip()
    
    gpu_id = None
    matched_pattern = None
    
    text_normalized = text.replace(" ", "").replace("-", "")
    
    for pattern, gid in GPU_NAME_MAPPINGS.items():
        pattern_normalized = pattern.lower().replace(" ", "").replace("-", "")
        if pattern_normalized in text_normalized or text_normalized in pattern_normalized:
            gpu_id = gid
            matched_pattern = pattern
            break
    
    if not gpu_id:
        for gid, config in GPU_CATALOG.items():
            name_normalized = config["name"].lower().replace(" ", "").replace("-", "")
            if name_normalized in text_normalized:
                gpu_id = gid
                break
    
    if not gpu_id:
        keywords = ["a100", "h100", "h200", "b200", "v100", "p100", "rtx", "macbook", "m3", "m2"]
        for kw in keywords:
            if kw in text_normalized:
                for pattern, gid in GPU_NAME_MAPPINGS.items():
                    if kw in pattern.lower():
                        gpu_id = gid
                        matched_pattern = pattern
                        break
                break
    
    if not gpu_id:
        return None
    
    config = GPU_CATALOG[gpu_id]
    
    if "40gb" in original_text.lower() and gpu_id == "a100_80gb":
        if "a100_40gb" in GPU_CATALOG:
            config = GPU_CATALOG["a100_40gb"]
            gpu_id = "a100_40gb"
    
    total_vram = config["vram_gb"] * count
    
    return ParsedHardware(
        gpu_id=gpu_id,
        gpu_name=config["name"],
        vram_gb=config["vram_gb"],
        count=count,
        total_vram_gb=total_vram,
        tier=config["tier"]
    )


def get_available_gpu_options() -> list[tuple[str, str]]:
    return [(config["name"], gid) for gid, config in GPU_CATALOG.items()]