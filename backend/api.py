"""FastAPI server with static file serving for HTML/JS/CSS frontend."""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from backend.services.parser import parse_hardware_input, ParsedHardware
from backend.services.recommender import get_recommender, LLMRecommender
from backend.services.hybrid_recommender import get_hybrid_recommender
from backend.services.collaborative import get_collaborative_filter
from backend.logger import get_logger

logger = get_logger(__name__)

FEEDBACK_FILE = Path(__file__).parent.parent / "data_gathering_pipeline" / "data" / "feedback_data.jsonl"
FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(
    title="LLM Recommender API",
    description="Recommend locally-hostable open-weight LLMs",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    logger.info("Initializing recommender and pre-populating showcase cache at startup...")
    try:
        recommender = get_recommender()
        picks = recommender.get_showcase_picks()
        global _SHOWCASE_CACHE, _SHOWCASE_CACHE_TIME
        _SHOWCASE_CACHE = picks
        _SHOWCASE_CACHE_TIME = time.time()
        logger.info(f"Showcase cache pre-populated with {len(picks)} picks at startup successfully!")
    except Exception as e:
        logger.error(f"Failed to pre-populate showcase cache at startup: {e}")


class RecommendRequest(BaseModel):
    hardware_text: str = Field(..., description="e.g. '8 A100s', '4 RTX 4090s', 'MacBook M3 Max'")
    use_case: str = Field(..., description="e.g. 'code generation', 'math reasoning'")
    top_k: Optional[int] = Field(5, ge=1, le=20)
    mode: Literal["hybrid", "pure"] = Field("pure", description="'hybrid' combines CF + content, 'pure' is content-only")
    user_id: Optional[str] = Field(None, description="User ID for personalized hybrid recommendations")


class RecommendResponse(BaseModel):
    success: bool
    mode: str
    hardware: Optional[dict]
    use_case: str
    recommendations: list[dict]
    user_has_feedback: bool = False
    error: Optional[str] = None


class ShowcaseResponse(BaseModel):
    success: bool
    showcase: list[dict]
    error: Optional[str] = None


class FeedbackRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    model_id: str = Field(..., description="Model ID that was recommended")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    hardware_used: Optional[str] = Field(None, description="Hardware configuration used")
    use_case: Optional[str] = Field(None, description="Use case for the recommendation")


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: Optional[str] = None


class FeedbackStatsResponse(BaseModel):
    success: bool
    total_feedbacks: int = 0
    avg_rating: float = 0.0
    ratings_distribution: dict = {}
    ratings_per_model: dict = {}
    error: Optional[str] = None


_SHOWCASE_CACHE: Optional[list[dict]] = None
_SHOWCASE_CACHE_TIME: float = 0
_SHOWCASE_CACHE_TTL: float = 3600


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response: {request.method} {request.url.path} - {response.status_code} ({(time.time() - start) * 1000:.1f}ms)")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": None,
        },
    )


@app.get("/")
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "LLM Recommender API", "version": "1.0.0", "status": "healthy"}


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/models/count")
def model_count():
    try:
        return {"count": get_recommender().model_count}
    except Exception as e:
        logger.error(f"Error getting model count: {e}")
        raise


@app.get("/api/showcase", response_model=ShowcaseResponse)
def showcase():
    """Get top model per hardware tier for showcase display. Cached for 1 hour."""
    global _SHOWCASE_CACHE, _SHOWCASE_CACHE_TIME

    current_time = time.time()
    if _SHOWCASE_CACHE is not None and (current_time - _SHOWCASE_CACHE_TIME) < _SHOWCASE_CACHE_TTL:
        logger.info("Returning cached showcase picks")
        return ShowcaseResponse(success=True, showcase=_SHOWCASE_CACHE)

    try:
        recommender = get_recommender()
        picks = recommender.get_showcase_picks()
        _SHOWCASE_CACHE = picks
        _SHOWCASE_CACHE_TIME = current_time
        logger.info(f"Cached {len(picks)} showcase picks")
        return ShowcaseResponse(success=True, showcase=picks)
    except Exception as e:
        logger.error(f"Error generating showcase: {e}")
        return ShowcaseResponse(success=False, showcase=[], error="Failed to load showcase models")


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    logger.info(f"Request: hardware='{req.hardware_text}', use_case='{req.use_case}', mode='{req.mode}'")

    hw = parse_hardware_input(req.hardware_text)
    if hw is None:
        return RecommendResponse(
            success=False,
            mode=req.mode,
            hardware=None,
            use_case=req.use_case,
            recommendations=[],
            error=f"Could not parse hardware: '{req.hardware_text}'. Try '8 A100s', '4 RTX 4090s', 'MacBook M3 Max'",
        )

    hw_dict = {
        "gpu_id": hw.gpu_id,
        "gpu_name": hw.gpu_name,
        "vram_gb": hw.vram_gb,
        "count": hw.count,
        "total_vram_gb": hw.total_vram_gb,
        "tier": hw.tier,
    }

    try:
        if req.mode == "hybrid":
            hybrid_rec = get_hybrid_recommender(alpha=0.6)
            user_has_data = req.user_id and hybrid_rec.collaborative_filter.has_user_data(req.user_id)
            
            results = hybrid_rec.get_hybrid_recommendations(
                hardware=hw,
                use_case_text=req.use_case,
                user_query=f"{req.use_case} {req.hardware_text}",
                user_id=req.user_id,
                top_k=req.top_k or 5,
            )
            
            rec_dicts = []
            for r in results:
                d = r.to_dict()
                d["cf_prediction"] = r.cf_prediction
                d["cf_confidence"] = r.cf_confidence
                d["hybrid_score"] = r.hybrid_score
                d["blend_weight"] = r.blend_weight
                rec_dicts.append(d)
            
            return RecommendResponse(
                success=True,
                mode="hybrid",
                hardware=hw_dict,
                use_case=req.use_case,
                recommendations=rec_dicts,
                user_has_feedback=user_has_data,
            )
        else:
            results = get_recommender().recommend(
                hardware=hw,
                use_case_text=req.use_case,
                user_query=f"{req.use_case} {req.hardware_text}",
                top_k=req.top_k or 5,
            )
            return RecommendResponse(
                success=True,
                mode="pure",
                hardware=hw_dict,
                use_case=req.use_case,
                recommendations=[r.to_dict() for r in results],
                user_has_feedback=False,
            )
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return RecommendResponse(
            success=False,
            mode=req.mode,
            hardware=hw_dict,
            use_case=req.use_case,
            recommendations=[],
            error="Failed to generate recommendations. Please try again.",
        )


def _load_feedback_data() -> list[dict]:
    """Load all feedback records from the JSONL file."""
    if not FEEDBACK_FILE.exists():
        return []
    records = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _save_feedback_record(record: dict) -> None:
    """Append a single feedback record to the JSONL file."""
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    """Submit feedback for a recommendation."""
    logger.info(f"Feedback received: user={req.user_id}, model={req.model_id}, rating={req.rating}")
    
    try:
        now = datetime.utcnow()
        feedback_record = {
            "user_id": req.user_id,
            "model_id": req.model_id,
            "rating": req.rating,
            "hardware_used": req.hardware_used or "",
            "use_case": req.use_case or "",
            "recommended_at": now.isoformat(),
            "created_at": now.isoformat(),
        }
        _save_feedback_record(feedback_record)
        
        # Retrain the collaborative filter with the new feedback in real-time
        try:
            from backend.services.collaborative import get_collaborative_filter
            cf = get_collaborative_filter()
            cf.train(force=True)
            logger.info("Collaborative filter retrained successfully with new feedback")
        except Exception as ex:
            logger.error(f"Failed to retrain collaborative filter: {ex}")

        return FeedbackResponse(
            success=True,
            message="Thank you for your feedback!",
            feedback_id=feedback_record["recommended_at"],
        )
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return FeedbackResponse(success=False, message="Failed to save feedback")


@app.get("/feedback/stats", response_model=FeedbackStatsResponse)
def get_feedback_stats():
    """Get aggregate feedback statistics."""
    try:
        records = _load_feedback_data()
        if not records:
            return FeedbackStatsResponse(success=True, total_feedbacks=0)
        
        total = len(records)
        ratings = [r["rating"] for r in records]
        avg_rating = sum(ratings) / total if total > 0 else 0.0
        
        ratings_dist = {str(i): 0 for i in range(1, 6)}
        for r in ratings:
            key = str(r)
            ratings_dist[key] = ratings_dist.get(key, 0) + 1
        
        ratings_per_model: dict = {}
        for r in records:
            model = r["model_id"]
            if model not in ratings_per_model:
                ratings_per_model[model] = {"count": 0, "total": 0, "avg": 0.0}
            ratings_per_model[model]["count"] += 1
            ratings_per_model[model]["total"] += r["rating"]
        
        for model in ratings_per_model:
            count = ratings_per_model[model]["count"]
            total_score = ratings_per_model[model]["total"]
            ratings_per_model[model]["avg"] = round(total_score / count, 2) if count > 0 else 0.0
        
        return FeedbackStatsResponse(
            success=True,
            total_feedbacks=total,
            avg_rating=round(avg_rating, 2),
            ratings_distribution=ratings_dist,
            ratings_per_model=ratings_per_model,
        )
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        return FeedbackStatsResponse(success=False, error="Failed to load feedback statistics")


@app.get("/{file_path:path}")
async def serve_static(file_path: str):
    """Serve static files (CSS, JS) from frontend directory."""
    file_path_lower = file_path.lower()
    if file_path_lower.endswith(".css"):
        media_type = "text/css"
    elif file_path_lower.endswith(".js"):
        media_type = "application/javascript"
    elif file_path_lower.endswith(".json"):
        media_type = "application/json"
    elif file_path_lower.endswith(".png"):
        media_type = "image/png"
    elif file_path_lower.endswith(".svg"):
        media_type = "image/svg+xml"
    else:
        media_type = "text/plain"

    static_path = FRONTEND_DIR / file_path
    if static_path.exists() and static_path.is_file():
        return FileResponse(str(static_path), media_type=media_type)
    raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)