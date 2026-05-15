"""FastAPI server with static file serving for HTML/JS/CSS frontend."""

import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from backend.services.parser import parse_hardware_input, ParsedHardware
from backend.services.recommender import get_recommender, LLMRecommender
from backend.logger import get_logger

logger = get_logger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(
    title="LLM Recommender API",
    description="Recommend locally-hostable open-weight LLMs",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    hardware_text: str = Field(..., description="e.g. '8 A100s', '4 RTX 4090s', 'MacBook M3 Max'")
    use_case: str = Field(..., description="e.g. 'code generation', 'math reasoning'")
    top_k: Optional[int] = Field(5, ge=1, le=20)


class RecommendResponse(BaseModel):
    success: bool
    hardware: Optional[dict]
    use_case: str
    recommendations: list[dict]
    error: Optional[str] = None


class ShowcaseResponse(BaseModel):
    success: bool
    showcase: list[dict]
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
    logger.info(f"Request: hardware='{req.hardware_text}', use_case='{req.use_case}'")

    hw = parse_hardware_input(req.hardware_text)
    if hw is None:
        return RecommendResponse(
            success=False,
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
        results = get_recommender().recommend(
            hardware=hw,
            use_case_text=req.use_case,
            user_query=f"{req.use_case} {req.hardware_text}",
            top_k=req.top_k or 5,
        )
        return RecommendResponse(
            success=True,
            hardware=hw_dict,
            use_case=req.use_case,
            recommendations=[r.to_dict() for r in results],
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return RecommendResponse(
            success=False,
            hardware=hw_dict,
            use_case=req.use_case,
            recommendations=[],
            error="Failed to generate recommendations. Please try again.",
        )


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