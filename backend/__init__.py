"""Backend package initialization.

Loads environment variables and configures logging.
"""

from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).parent.parent
_env_file = _root / ".env"

if _env_file.exists():
    load_dotenv(_env_file, override=True)
else:
    load_dotenv()

from backend.logging import setup_logging, get_logger  # noqa: E402

__all__ = ["setup_logging", "get_logger"]