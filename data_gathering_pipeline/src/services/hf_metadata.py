"""HuggingFace Hub metadata fetching with fallback search strategy."""

import re
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, model_info, list_models
from huggingface_hub.hf_api import ModelInfo
from loguru import logger

from src.config import HF_TOKEN


@dataclass
class HFModelMetadata:
    """Complete metadata for a model from HuggingFace."""
    model_id: str
    repo_id: Optional[str] = None
    safetensors_size_gb: float = 0.0
    parameter_count: Optional[int] = None
    is_moe: bool = False
    num_experts: Optional[int] = None
    model_type: Optional[str] = None
    library_name: Optional[str] = None
    tags: List[str] = None
    metadata_status: str = "unknown"

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class HFMetadataService:
    """Fetches model metadata from HuggingFace Hub with two-tier fallback strategy."""

    def __init__(self, token: Optional[str] = None):
        self.api = HfApi(token=token or HF_TOKEN or None)
        self.cache: Dict[str, HFModelMetadata] = {}
        self.failed_lookups: List[str] = []

    def _extract_size(self, info: ModelInfo) -> float:
        """Extract total safetensors file size from model info."""
        total_bytes = 0

        try:
            if hasattr(info, "siblings") and info.siblings:
                for sibling in info.siblings:
                    if sibling.rfilename and sibling.size:
                        if "safetensors" in sibling.rfilename.lower() or "model" in sibling.rfilename.lower():
                            total_bytes += sibling.size
        except Exception:
            pass

        if total_bytes == 0 and hasattr(info, "model_size") and info.model_size:
            total_bytes = info.model_size

        return total_bytes / (1024**3) if total_bytes > 0 else 0.0

    def _detect_moe(self, info: ModelInfo) -> tuple:
        """Detect MoE architecture from model config."""
        try:
            if hasattr(info, "config") and isinstance(info.config, dict):
                num_experts = (
                    info.config.get("num_local_experts")
                    or info.config.get("num_experts")
                    or info.config.get("n_routed_experts")
                )
                if num_experts:
                    return True, num_experts

                if info.config.get("model_type") in ["moe", "mixture"]:
                    return True, None
        except Exception:
            pass

        return False, None

    def _create_candidates(self, model_name: str) -> List[str]:
        """Generate possible HuggingFace repo_id candidates."""
        candidates = []

        if "/" in model_name:
            candidates.append(model_name)
            org, _, rest = model_name.partition("/")
            candidates.append(f"{org}/{rest.replace(' ', '-').replace('_', '-')}")

        norm = model_name.lower()

        org_map = {
            "gpt": "openai",
            "gpt-4": "openai",
            "gpt-3": "openai",
            "claude": "anthropic",
            "llama": "meta-llama",
            "llama-3": "meta-llama",
            "gemini": "google",
            "gemma": "google",
            "mistral": "mistralai",
            "mixtral": "mistralai",
            "qwen": "qwen",
            "deepseek": "deepseek-ai",
            "yi": "01-ai",
            "dbrx": "databricks",
            "phi": "microsoft",
        }

        org = next((v for k, v in org_map.items() if norm.startswith(k)), None)

        if org:
            size_match = re.search(r"(\d+\.?\d*)b", model_name, re.I)
            if size_match:
                candidates.append(
                    f"{org}/{model_name.replace(' ', '-').replace('_', '-')}"
                )

        candidates.append(model_name.replace(" ", "-").replace("_", "-"))
        candidates.append(model_name.lower().replace(" ", "-"))

        return list(dict.fromkeys(candidates))

    def fetch(self, model_name: str, params_billions: Optional[float] = None) -> HFModelMetadata:
        """Two-tier metadata fetching: direct lookup then fallback search.

        Args:
            model_name: The model name to look up
            params_billions: Optional param count from dataset for size estimation fallback
        """
        if model_name in self.cache:
            return self.cache[model_name]

        if model_name in self.failed_lookups:
            return self._make_missing(model_name, params_billions)

        logger.info(f"Fetching metadata for: {model_name}")

        for attempt in range(3):
            for candidate in self._create_candidates(model_name):
                try:
                    info = model_info(candidate, timeout=15)
                    repo_id = info.id if hasattr(info, "id") else candidate

                    is_moe, num_experts = self._detect_moe(info)

                    metadata = HFModelMetadata(
                        model_id=model_name,
                        repo_id=repo_id,
                        safetensors_size_gb=round(self._extract_size(info), 3),
                        parameter_count=(
                            getattr(info, "config", {}).get("num_parameters")
                            if hasattr(info, "config")
                            else None
                        ),
                        is_moe=is_moe,
                        num_experts=num_experts,
                        model_type=getattr(info, "model_type", None)
                        or (
                            info.config.get("model_type")
                            if hasattr(info, "config")
                            else None
                        ),
                        library_name=getattr(info, "library_name", None),
                        tags=[str(t) for t in getattr(info, "tags", [])],
                        metadata_status="verified",
                    )

                    self.cache[model_name] = metadata
                    logger.success(
                        f"Found: {repo_id} ({metadata.safetensors_size_gb:.2f}GB)"
                    )
                    return metadata

                except Exception as e:
                    err_str = str(e).lower()
                    if "403" in err_str or "404" in err_str or "repo.*not.*found" in err_str:
                        break
                    if "429" in err_str or "rate limit" in err_str or "forbidden" in err_str:
                        if attempt < 2:
                            import time
                            wait = (attempt + 1) * 5
                            logger.debug(f"Rate limited, retrying in {wait}s...")
                            time.sleep(wait)
                            break
                    continue
            else:
                continue
            break

        # Tier 2: search fallback
        logger.info(f"Tier 2: Searching for '{model_name}'")

        try:
            results = list(
                list_models(search=model_name, sort="downloads", direction=-1, limit=5)
            )

            for candidate in results:
                try:
                    repo_id = (
                        candidate.id if hasattr(candidate, "id") else str(candidate)
                    )
                    info = model_info(repo_id)

                    metadata = HFModelMetadata(
                        model_id=model_name,
                        repo_id=repo_id,
                        safetensors_size_gb=round(self._extract_size(info), 3),
                        is_moe=self._detect_moe(info)[0],
                        tags=[str(t) for t in getattr(info, "tags", [])],
                        metadata_status="verified",
                    )

                    self.cache[model_name] = metadata
                    logger.success(f"Fallback found: {repo_id}")
                    return metadata

                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"Fallback search failed: {e}")

        logger.warning(f"Failed to find metadata for: {model_name}")
        self.failed_lookups.append(model_name)

        return self._make_missing(model_name, params_billions)

    def _make_missing(
        self, model_name: str, params_billions: Optional[float] = None
    ) -> HFModelMetadata:
        """Create a missing metadata record with estimated size from params."""
        if params_billions and params_billions > 0:
            estimated_gb = round(params_billions * 2, 2)
            return HFModelMetadata(
                model_id=model_name,
                repo_id=None,
                safetensors_size_gb=estimated_gb,
                parameter_count=int(params_billions * 1e9),
                is_moe=False,
                num_experts=None,
                model_type=None,
                library_name=None,
                tags=[],
                metadata_status="rate_limited_estimated",
            )
        return HFModelMetadata(
            model_id=model_name,
            repo_id=None,
            safetensors_size_gb=0.0,
            parameter_count=None,
            is_moe=False,
            num_experts=None,
            model_type=None,
            library_name=None,
            tags=[],
            metadata_status="missing_hf_metadata",
        )

    def batch_fetch(self, model_names: List[str]) -> Dict[str, HFModelMetadata]:
        """Fetch metadata for multiple models (sequential)."""
        return {name: self.fetch(name) for name in model_names}

    def parallel_batch_fetch(
        self,
        model_names: List[str],
        max_workers: int = 20,
        timeout: float = 30.0,
        params_map: Optional[Dict[str, float]] = None,
    ) -> Dict[str, HFModelMetadata]:
        """
        Fetch metadata for multiple models IN PARALLEL using ThreadPoolExecutor.

        HF API calls are I/O bound, so threading provides ~10-20x speedup.

        Args:
            model_names: List of model names to fetch
            max_workers: Number of concurrent threads (default: 20)
            timeout: Timeout per model in seconds (default: 30)
            params_map: Optional dict of {model_name: params_billions} for size estimation fallback

        """
        results: Dict[str, HFModelMetadata] = {}
        start_time = time.time()
        params_map = params_map or {}

        logger.info(
            f"Starting parallel fetch for {len(model_names)} models ({max_workers} workers)"
        )

        def fetch_with_timeout(name: str):
            """Fetch with individual error handling."""
            try:
                metadata = self.fetch(name, params_map.get(name))
                return (name, metadata)
            except Exception as e:
                logger.warning(f"Error fetching {name}: {e}")
                return (name, self._make_missing(name, params_map.get(name)))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_with_timeout, name): name
                for name in model_names
            }

            completed = 0
            for future in as_completed(
                futures,
                timeout=timeout * len(model_names) / max_workers,
            ):
                try:
                    name, metadata = future.result(timeout=timeout)
                    results[name] = metadata
                    completed += 1

                    if completed % 20 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"Progress: {completed}/{len(model_names)} ({rate:.1f} models/sec)"
                        )

                except Exception as e:
                    name = futures[future]
                    logger.error(f"Future error for {name}: {e}")
                    results[name] = HFModelMetadata(
                        model_id=name, metadata_status="timeout"
                    )

        elapsed = time.time() - start_time
        rate = len(results) / elapsed if elapsed > 0 else 0

        logger.success(
            f"Parallel fetch complete: {len(results)}/{len(model_names)} models "
            f"in {elapsed:.1f}s ({rate:.1f} models/sec)"
        )

        return results

    def get_cache_stats(self) -> Dict:
        """Get metadata cache statistics."""
        verified = sum(
            1 for m in self.cache.values() if m.metadata_status == "verified"
        )
        total = len(self.cache) + len(self.failed_lookups)

        return {
            "total_lookups": total,
            "verified": verified,
            "missing": len(self.failed_lookups),
            "verification_rate": round(verified / total * 100, 2) if total > 0 else 0,
        }
