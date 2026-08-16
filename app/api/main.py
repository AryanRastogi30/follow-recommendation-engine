"""
FastAPI service. Loads the trained model + user directory once at startup
(module-level singleton), then serves recommendations per request.

Run: uvicorn app.api.main:app --reload
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.data.schema import UserRecord
from app.pipeline_service import RecommendationService

app = FastAPI(title="Two-Stage Follow Recommendation API", version="1.0.0")
service: Optional[RecommendationService] = None


@app.on_event("startup")
def _load_service():
    global service
    service = RecommendationService(artifacts_dir="artifacts")


class NewUserRequest(BaseModel):
    name: str
    gender: str
    age: float
    interests: List[str]
    city: str
    country: str
    latitude: float
    longitude: float
    top_k: int = 10


class RecommendationItem(BaseModel):
    user_id: int
    name: str
    city: str
    country: str
    shared_interests: List[str]
    affinity_score: float


@app.get("/health")
def health():
    return {"status": "ok", "users_loaded": len(service.users) if service else 0}


@app.get("/recommend/{user_id}", response_model=List[RecommendationItem])
def recommend_by_id(user_id: int, top_k: int = 10):
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    target = next((u for u in service.users if u.user_id == user_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not found")
    return service.recommend(target, top_k=top_k)


@app.post("/recommend", response_model=List[RecommendationItem])
def recommend_for_new_user(payload: NewUserRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    target = UserRecord(
        user_id=-1, name=payload.name, gender=payload.gender, age=payload.age,
        interests=payload.interests, city=payload.city, country=payload.country,
        latitude=payload.latitude, longitude=payload.longitude,
    )
    return service.recommend(target, top_k=payload.top_k)
