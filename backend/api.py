from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from backend.services.recommender import get_recommender, ScoredModel
from backend.services.hardware_parser import parse_hardware_input


app = FastAPI(title="LLM Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    hardware_text: str
    use_case: str
    top_k: Optional[int] = 5


class RecommendResponse(BaseModel):
    success: bool
    hardware: Optional[dict]
    use_case: str
    recommendations: list[dict]
    error: Optional[str] = None


def model_to_dict(model: ScoredModel) -> dict:
    return {
        "model_id": model.model_id,
        "hf_repo_id": model.hf_repo_id,
        "base_model": model.base_model,
        "params_billions": model.params_billions,
        "model_type": model.model_type,
        "architecture": model.architecture,
        "benchmarks": {
            "coding": round(model.coding, 2),
            "math": round(model.math_score, 2),
            "reasoning": round(model.reasoning, 2),
            "intelligence_index": round(model.intelligence_index, 2)
        },
        "vram_fp16_gb": round(model.vram_fp16, 2),
        "vram_int8_gb": round(model.vram_int8, 2),
        "vram_int4_gb": round(model.vram_int4, 2),
        "hosting_strategy": model.hosting_strategy,
        "is_moe": model.is_moe,
        "scores": {
            "semantic": round(model.semantic_score, 3),
            "benchmark": round(model.benchmark_score, 3),
            "hardware": round(model.hardware_score, 3),
            "final": round(model.final_score, 3)
        },
        "matched_hardware": model.matched_hardware
    }


@app.get("/")
def root():
    return {"message": "LLM Recommender API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    hardware = parse_hardware_input(request.hardware_text)
    
    if hardware is None:
        return RecommendResponse(
            success=False,
            hardware=None,
            use_case=request.use_case,
            recommendations=[],
            error=f"Could not parse hardware input: '{request.hardware_text}'"
        )
    
    hardware_dict = {
        "gpu_id": hardware.gpu_id,
        "gpu_name": hardware.gpu_name,
        "vram_gb": hardware.vram_gb,
        "count": hardware.count,
        "total_vram_gb": hardware.total_vram_gb,
        "tier": hardware.tier
    }
    
    try:
        recommender = get_recommender()
        results = recommender.recommend(
            hardware=hardware,
            use_case_text=request.use_case,
            user_query=f"{request.use_case} {request.hardware_text}",
            top_k=request.top_k or 5
        )
        
        return RecommendResponse(
            success=True,
            hardware=hardware_dict,
            use_case=request.use_case,
            recommendations=[model_to_dict(r) for r in results]
        )
    
    except Exception as e:
        return RecommendResponse(
            success=False,
            hardware=hardware_dict,
            use_case=request.use_case,
            recommendations=[],
            error=str(e)
        )


@app.get("/models/count")
def model_count():
    try:
        recommender = get_recommender()
        return {"count": len(recommender.models)}
    except Exception as e:
        return {"count": 0, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)