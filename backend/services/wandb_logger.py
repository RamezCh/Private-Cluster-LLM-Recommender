import os
from typing import Optional

import wandb
from loguru import logger


class WandbLogger:
    def __init__(
        self,
        project_name: str = "llm-recommender",
        entity: Optional[str] = None,
        enabled: bool = True
    ):
        self.project_name = project_name
        self.entity = entity
        self.enabled = enabled and self._check_wandb_enabled()
        self.run = None
        
        if self.enabled:
            self._init_run()
    
    def _check_wandb_enabled(self) -> bool:
        if "WANDB_API_KEY" in os.environ or "WANDB_MODE" in os.environ:
            return True
        
        try:
            wandb.login(timeout=5)
            return True
        except Exception:
            logger.warning("W&B not configured. Set WANDB_API_KEY to enable experiment tracking.")
            return False
    
    def _init_run(self) -> None:
        try:
            wandb.init(
                project=self.project_name,
                entity=self.entity,
                mode="online" if os.environ.get("WANDB_API_KEY") else "offline"
            )
            self.run = wandb.run
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.enabled = False
    
    def log_config(
        self,
        semantic_weight: float,
        benchmark_weight: float,
        hardware_weight: float,
        embedding_model: str,
        use_case_detection: str
    ) -> None:
        if not self.enabled or self.run is None:
            return
        
        config = {
            "semantic_weight": semantic_weight,
            "benchmark_weight": benchmark_weight,
            "hardware_weight": hardware_weight,
            "embedding_model": embedding_model,
            "use_case_detection": use_case_detection,
        }
        
        self.run.config.update(config)
    
    def log_recommendation(
        self,
        query: str,
        hardware: str,
        use_case: str,
        num_results: int,
        top_model: str
    ) -> None:
        if not self.enabled or self.run is None:
            return
        
        self.run.log({
            "query": query,
            "hardware": hardware,
            "use_case": use_case,
            "num_results": num_results,
            "top_model": top_model
        })
    
    def log_test_result(
        self,
        test_name: str,
        expected: str,
        actual: str,
        passed: bool,
        latency_ms: float
    ) -> None:
        if not self.enabled or self.run is None:
            return
        
        self.run.log({
            "test_name": test_name,
            "expected_top": expected,
            "actual_top": actual,
            "passed": passed,
            "latency_ms": latency_ms
        })
    
    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


_wandb_logger: Optional[WandbLogger] = None


def get_wandb_logger() -> WandbLogger:
    global _wandb_logger
    if _wandb_logger is None:
        _wandb_logger = WandbLogger()
    return _wandb_logger