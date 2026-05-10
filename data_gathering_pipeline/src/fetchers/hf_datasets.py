"""HuggingFace dataset loader for benchmark data."""

from typing import List, Dict, Optional
from datasets import load_dataset
from loguru import logger

from src.config import HF_DATASETS, HF_TOKEN


class HFDatasetLoader:
    """Loads benchmark datasets from HuggingFace."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or HF_TOKEN
        self.datasets: Dict[str, List[Dict]] = {}

    def load_open_evals(self) -> List[Dict]:
        """Load OpenEvals/leaderboard-data dataset."""
        logger.info(f"Loading {HF_DATASETS['open_evals']}")

        try:
            dataset = load_dataset(
                HF_DATASETS["open_evals"],
                split="train",
                token=self.token or None,
            )
            data = [dict(item) for item in dataset]
            self.datasets["open_evals"] = data
            logger.success(f"Loaded {len(data)} records")
            return data
        except Exception as e:
            logger.error(f"Failed to load: {e}")
            return []

    def load_lmsys(self) -> List[Dict]:
        """Load lmarena-ai/leaderboard-dataset dataset (config: text, split: full)."""
        logger.info(f"Loading {HF_DATASETS['lmsys_arena']}")

        for split_name in ("full", "latest"):
            try:
                dataset = load_dataset(
                    HF_DATASETS["lmsys_arena"],
                    "text",
                    split=split_name,
                    token=self.token or None,
                )
                data = [dict(item) for item in dataset]
                self.datasets["lmsys_arena"] = data
                logger.success(f"Loaded {len(data)} records (split='{split_name}')")
                return data
            except Exception as e:
                logger.warning(f"Split '{split_name}' failed: {e}")

        logger.error("All LMSYS splits failed — ELO data will be unavailable")
        return []

    def load_all(self) -> Dict[str, List[Dict]]:
        """Load all benchmark datasets."""
        return {
            "open_evals": self.load_open_evals(),
            "lmsys_arena": self.load_lmsys(),
        }

    def get_model_names(self, source: str) -> List[str]:
        """Extract model names from a dataset."""
        data = self.datasets.get(source, [])

        name_keys = ["model_name", "name", "model", "model_id", "title"]

        return [
            str(item[key])
            for item in data
            for key in name_keys
            if key in item and item[key]
        ]
