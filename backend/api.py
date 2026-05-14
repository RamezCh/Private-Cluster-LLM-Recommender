"""FastAPI server. Simple, no fuss."""

import time
import os
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.services.parser import parse_hardware_input, ParsedHardware
from backend.services.recommender import get_recommender, ScoredModel
from backend.logging import get_logger

logger = get_logger(__name__)

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
            "detail": str(exc) if os.getenv("APP_ENV") == "development" else None,
        },
    )


@app.get("/")
def root():
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
            error=f"Error generating recommendations: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)