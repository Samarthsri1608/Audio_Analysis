import logging
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app import logger as app_logger
from app.settings import settings
from app.api.v1 import v1_router

logger = logging.getLogger("app")

app = FastAPI(
    title="AI Speaking Assessment Engine",
    version="1.0.0",
    root_path="/voice-assessment",
    docs_url="/docs",
    redoc_url=None,
)

# Middleware setup (order: CORSMiddleware -> Custom middlewares)
_origins = settings.ALLOW_ORIGINS.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False if "*" in _origins else True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path in ("/health", "/") or request.method == "OPTIONS":
        return await call_next(request)
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {duration_ms:.2f}ms",
        extra={
            "http.method": request.method,
            "http.url": str(request.url),
            "http.status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response


# Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail, "success": False},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"message": "Validation error", "success": False, "error": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "success": False},
    )


# Health Endpoint (must be at the root of the app, not under router)
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "success", "success": True, "data": {"service": "AI Speaking Assessment Engine"}}


# Include API Router
app.include_router(v1_router)
