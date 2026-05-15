"""Weights & Biases integration. Non-critical - fails silently."""

import os
import threading
from typing import Optional

import wandb
from dotenv import load_dotenv

load_dotenv()

from backend.logger import get_logger

logger = get_logger(__name__)

WANDB_ENABLED = os.getenv("WANDB_ENABLED", "false").lower() == "true"


class WandbLogger:
    def __init__(
        self,
        project_name: str = "llm-recommender",
        entity: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.project_name = project_name
        self.entity = entity
        self.run = None

        if enabled is None:
            enabled = WANDB_ENABLED

        try:
            if enabled and self._check_enabled():
                self.enabled = True
                self._init_run()
            else:
                self.enabled = False
        except Exception as e:
            logger.warning(f"W&B init skipped: {e}")
            self.enabled = False

    def _check_enabled(self) -> bool:
        if not os.environ.get("WANDB_API_KEY"):
            return False
        if os.environ.get("WANDB_MODE") == "disabled":
            return False
        try:
            wandb.login(timeout=5)
            return True
        except Exception as e:
            logger.warning(f"W&B login failed: {e}")
            return False

    def _init_run(self) -> None:
        try:
            wandb.init(
                project=self.project_name,
                entity=self.entity,
                mode="online" if os.environ.get("WANDB_API_KEY") else "offline",
            )
            self.run = wandb.run
            if self.run:
                logger.info(f"W&B run started: {self.run.name}")
            else:
                self.enabled = False
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            self.enabled = False
            self.run = None

    def log_config(self, semantic_weight: float, benchmark_weight: float,
                   hardware_weight: float, embedding_model: str, use_case_detection: str) -> None:
        if not self.enabled or self.run is None:
            return
        self.run.config.update({
            "semantic_weight": semantic_weight,
            "benchmark_weight": benchmark_weight,
            "hardware_weight": hardware_weight,
            "embedding_model": embedding_model,
            "use_case_detection": use_case_detection,
        })

    def log_recommendation(self, query: str, hardware: str, use_case: str,
                           num_compatible: int, num_returned: int, top_model: str,
                           top_model_score: float, latency_ms: float) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.log({
                "query": (query[:1000] + "...") if len(query) > 1000 else query,
                "hardware": hardware,
                "use_case": use_case,
                "num_compatible": num_compatible,
                "num_returned": num_returned,
                "top_model": top_model,
                "top_model_score": top_model_score,
                "latency_ms": latency_ms,
            })
        except Exception as e:
            logger.warning(f"W&B log failed: {str(e)[:200]}")
            self.enabled = False

    def log_test_result(self, test_name: str, expected: str, actual: str,
                        passed: bool, latency_ms: float) -> None:
        if not self.enabled or self.run is None:
            return
        self.run.log({
            "test_name": test_name,
            "expected_top": expected,
            "actual_top": actual,
            "passed": passed,
            "latency_ms": latency_ms,
        })

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
                self.run = None
            except Exception as e:
                logger.warning(f"W&B finish error: {e}")


_logger: Optional[WandbLogger] = None
_logger_lock = threading.Lock()


def get_wandb_logger() -> WandbLogger:
    global _logger
    if _logger is None:
        with _logger_lock:
            if _logger is None:
                _logger = WandbLogger()
    return _logger


def reset_wandb_logger() -> None:
    global _logger
    with _logger_lock:
        if _logger:
            _logger.finish()
        _logger = None