"""Shared logging configuration for the backend.

This module configures Loguru for consistent logging across all backend services.
"""

import os
import platform
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from loguru import logger


def setup_logging(
    level: Optional[str] = None,
    log_dir: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure Loguru logging for the backend application.

    Args:
        level: Log level (default: from LOGURU_LEVEL env var or INFO)
        log_dir: Directory for log files (default: logs/)
        rotation: Log rotation policy
        retention: Log retention policy
    """
    logger.remove()

    log_level = level or os.getenv("LOGURU_LEVEL", "INFO")
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        level=log_level,
        format=log_format,
        colorize=True,
    )

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        compression = "zip" if platform.system() != "Windows" else None
        logger.add(
            log_dir / "backend_{time:YYYYMMDD}.log",
            rotation=rotation,
            retention=retention,
            format=log_format,
            level=log_level,
            compression=compression,
        )


def get_logger(name: str):
    """Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)


DEFAULT_LOG_DIR = Path(__file__).parent.parent / "logs"

setup_logging(log_dir=DEFAULT_LOG_DIR)