"""Data models for backend. Re-exports from recommender.py for compat."""

from backend.services.recommender import ScoredModel

__all__ = ["ScoredModel"]