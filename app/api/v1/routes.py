from fastapi import APIRouter
from app.api.v1.endpoints.evaluations import router as evaluations_router

def build_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1")
    # Include evaluations endpoints directly under /api/v1
    router.include_router(evaluations_router, tags=["Evaluations"])
    return router

v1_router = build_router()
