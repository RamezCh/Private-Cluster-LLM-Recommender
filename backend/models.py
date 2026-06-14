"""Data models for backend. Re-exports from recommender.py for compat."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from backend.services.recommender import ScoredModel


@dataclass
class RecommendationFeedback:
    user_id: str
    model_id: str
    rating: int
    hardware_used: str
    use_case: str
    recommended_at: datetime
    created_at: datetime

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "model_id": self.model_id,
            "rating": self.rating,
            "hardware_used": self.hardware_used,
            "use_case": self.use_case,
            "recommended_at": self.recommended_at.isoformat(),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecommendationFeedback":
        return cls(
            user_id=data["user_id"],
            model_id=data["model_id"],
            rating=data["rating"],
            hardware_used=data["hardware_used"],
            use_case=data["use_case"],
            recommended_at=datetime.fromisoformat(data["recommended_at"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class FeedbackStats:
    total_feedbacks: int
    avg_rating: float
    ratings_distribution: dict
    ratings_per_model: dict


__all__ = ["ScoredModel", "RecommendationFeedback", "FeedbackStats"]