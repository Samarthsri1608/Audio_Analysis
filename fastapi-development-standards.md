# FastAPI Development Standards — Spider Monorepo

> These guidelines define the standard patterns for building, deploying, and maintaining FastAPI microservices in the Spider monorepo. They are designed to be both human-readable and LLM-actionable — an AI agent should be able to follow these rules to produce production-ready code.

---

## 1. Service Directory Structure

Every FastAPI service MUST follow this layout:

```
services/{domain}/{service-name}/
├── main.py                        # FastAPI app entrypoint
├── Dockerfile                     # Multi-stage build with OTEL
├── requirements.txt               # Pinned dependencies
├── .env                           # Local-only env config (git-ignored)
└── app/
    ├── __init__.py                # Logger setup, shared instances (Celery, clients)
    ├── settings.py                # Pydantic BaseSettings config
    ├── models.py                  # Domain/Pydantic models (optional, can be in schemas/)
    ├── api/
    │   └── v1/
    │       ├── __init__.py        # Exports v1_router
    │       ├── routes.py          # Router aggregation (build_router pattern)
    │       ├── endpoints/         # One file per resource/domain
    │       ├── schemas/           # Request/response Pydantic models
    │       └── utils/             # Endpoint-specific helpers
    ├── services/                  # Business logic layer
    ├── core/                      # Domain logic, repository classes
    ├── adapters/                  # External system integrations (DB, APIs, queues)
    ├── utils/                     # Shared utilities
    └── tasks/                     # Celery task definitions (if applicable)
```

### Rules

- **DO NOT** place business logic in endpoint handlers. Endpoints call services; services call adapters.
- **DO NOT** create new top-level directories outside this structure without team discussion.
- One endpoint file per resource domain (e.g., `endpoints/interviews.py`, `endpoints/agents.py`).
- Schemas live in `api/v1/schemas/`, not inline in endpoints.

---

## 2. FastAPI App Setup (`main.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import v1_router

app = FastAPI(
    title="Service Name",
    root_path="/service-name",      # MUST match ingress path prefix
    docs_url="/docs",
    redoc_url=None,
)

# Middleware order matters — add in this sequence
app.add_middleware(CORSMiddleware, ...)
# Custom middleware (request logging, security headers)

app.include_router(v1_router)

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Rules

- `root_path` MUST match the Kubernetes ingress path prefix for the service.
- Every service MUST expose a `GET /health` endpoint at the app root (not under `/api/v1`).
- `redoc_url` set to `None` — we use Swagger UI only.
- Health endpoint returns `{"status": "ok"}` — no database checks, no external calls.

---

## 3. API Versioning & Routing

### URL Pattern

```
https://{domain}/{service-name}/api/v{version}/{resource}
```

Example: `https://interview-api.zeko.ai/agents/api/v1/interviews`

### Router Setup

```python
# app/api/v1/routes.py
from fastapi import APIRouter

def build_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1")
    router.include_router(interviews_router, prefix="/interviews", tags=["Interviews"])
    router.include_router(agents_router, prefix="/agents", tags=["Agents"])
    return router

v1_router = build_router()
```

```python
# app/api/v1/__init__.py
from .routes import v1_router
```

### Rules

- Version prefix is `/api/v1`. When breaking changes are unavoidable, create `/api/v2` as a new module.
- Use `tags` on every `include_router` call for Swagger grouping.
- Endpoint functions use `async def` for I/O-bound operations (which is almost everything).

---

## 4. Configuration (`settings.py`)

```python
from pydantic_settings import BaseSettings
from pydantic import Field
import os

class Settings(BaseSettings):
    APP_NAME: str = Field(default=os.getenv("APP_NAME", "service-name"))
    ENVIRONMENT: str = Field(default=os.getenv("ENVIRONMENT", "local"))
    MONGODB_CONNECTION_URL: str = Field(default=os.getenv("MONGODB_CONNECTION_URL", ""))
    REDIS_HOST: str = Field(default=os.getenv("REDIS_HOST", "localhost"))
    REDIS_PORT: int = Field(default=int(os.getenv("REDIS_PORT", "6379")))
    REDIS_DB: int = Field(default=int(os.getenv("REDIS_DB", "0")))

settings = Settings()
```

### Rules

- Use `pydantic_settings.BaseSettings` — never raw `os.getenv` scattered in code.
- All env vars MUST have defaults that work for local development.
- Sensitive values (API keys, connection strings) come from K8s Secrets, never hardcoded.
- Export a singleton `settings` instance at module level.
- Settings fields use `os.getenv()` as defaults (pattern inherited across services — maintain consistency).

---

## 5. Request/Response Patterns

### Response Format

All API responses MUST follow this structure:

```python
# Success
{"message": "Description of result", "success": True, "data": {...}}

# Error
{"message": "What went wrong", "success": False, "error": "error_detail"}
```

### Schemas

```python
# app/api/v1/schemas/interviews.py
from pydantic import BaseModel, Field
from typing import Optional

class CreateInterviewRequest(BaseModel):
    candidate_email: str = Field(..., description="Candidate's email")
    job_role: str = Field(..., description="Target job role")

class InterviewResponse(BaseModel):
    message: str
    success: bool
    data: Optional[dict] = None
```

### Rules

- All request bodies MUST be Pydantic models — never raw `dict` or `Request.json()`.
- Response models are optional but encouraged for documented endpoints.
- Use `Field(...)` for required fields, `Field(default=...)` for optional.
- Endpoint return type: `dict` is acceptable; formal response models when the API is external-facing.

---

## 6. Error Handling

### Exception Handlers (in `main.py`)

```python
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

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
```

### Rules

- Register exception handlers for `HTTPException`, `RequestValidationError`, and generic `Exception`.
- Always log unhandled exceptions with `exc_info=True`.
- Never expose stack traces or internal details in API responses.
- Use `HTTPException` with appropriate status codes — not generic 500s for known error conditions.

---

## 7. Middleware

### Request Logging Middleware (standard across all services)

```python
import time
import logging

logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path == "/health" or request.method == "OPTIONS":
        return await call_next(request)

    start_time = time.time()
    body = await request.body()
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
```

### Security Headers Middleware

```python
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response
```

### Rules

- Request logging middleware is **mandatory** on every service.
- Exclude `/health` and `OPTIONS` from request logging.
- Use the `extra` dict in log calls for structured OTEL attribute injection.
- Security headers middleware is **mandatory** on every service.

---

## 8. Logging

### Setup Pattern (`app/__init__.py`)

```python
import logging
import sys
from functools import partial

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] "
    "[%(levelname)s] %(name)s: %(message)s"
)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Auto-attach exc_info to error/exception calls
logger.error = partial(logger.error, exc_info=True)
logger.exception = partial(logger.exception, exc_info=True)
```

### Rules

- Log to `stdout` only — never to files. Kubernetes handles log collection.
- Use `logging.getLogger(__name__)` in modules, `logging.getLogger("app")` for the root app logger.
- Error-level logs MUST include `exc_info=True` (handled by the partial pattern above).
- Use structured `extra` dicts for machine-parseable attributes.
- NEVER use `print()` for logging.
- Log format includes process and thread info for debugging concurrent workers.

### OTEL Integration

When the service runs with `opentelemetry-instrument`, trace/span IDs are automatically injected into logs. No manual instrumentation needed for log correlation — the `OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true` env var handles this.

---

## 9. Database Patterns

### MongoDB — Async (Motor) — Preferred for Endpoints

```python
# app/adapters/mongo.py
from motor.motor_asyncio import AsyncIOMotorClient
from app.settings import settings

class AsyncMongoHandler:
    def __init__(self):
        self.client = AsyncIOMotorClient(settings.MONGODB_CONNECTION_URL)
        self.db = self.client[settings.MONGO_DATABASE]

    async def read_document(self, collection_name: str, filter_: dict, projection: dict = None):
        return await self.db[collection_name].find_one(filter_, projection)

    async def write_document(self, collection_name: str, document: dict):
        return await self.db[collection_name].insert_one(document)

    async def update_document(self, collection_name: str, filter_: dict, update: dict):
        return await self.db[collection_name].update_one(filter_, {"$set": update})

ASYNC_MONGO_CLIENT = AsyncMongoHandler()
```

### MongoDB — Sync (PyMongo) — For Celery Tasks Only

```python
# app/adapters/mongo.py
from pymongo import MongoClient
from app.settings import settings

class MongoHandler:
    def __init__(self):
        self.client = MongoClient(settings.MONGODB_CONNECTION_URL)
        self.db = self.client[settings.MONGO_DATABASE]

    def read_document(self, collection_name: str, filter_: dict, projection: dict = None):
        return self.db[collection_name].find_one(filter_, projection)

MONGO_CLIENT = MongoHandler()
```

### Redis

```python
# app/adapters/redis.py
from redis.asyncio import ConnectionPool, Redis
from app.settings import settings

redis_pool = ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    max_connections=50,
    decode_responses=True,
)

async def get_redis() -> Redis:
    return Redis(connection_pool=redis_pool)
```

### Rules

- Use **Motor (async)** for all FastAPI endpoint database operations.
- Use **PyMongo (sync)** only in Celery tasks and synchronous contexts.
- Database clients are **singleton instances** at module level — never create per-request.
- Connection pooling is mandatory for Redis (`max_connections=50` default).
- Collection names are constants or referenced from a central location — no string literals scattered in business logic.

---

## 10. Background Tasks (Celery)

### Setup

```python
# app/__init__.py or app/celery.py
from celery import Celery
from app.settings import settings

celery_app = Celery(
    "service_name",
    broker=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.CELERY_BROKER_DB}",
    backend=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.CELERY_RESULT_DB}",
)

celery_app.conf.update(
    worker_prefetch_multiplier=1,
    worker_concurrency=5,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

celery_app.autodiscover_tasks(["app.tasks"])
```

### Rules

- Celery broker and backend: Redis (same instance, different DB numbers).
- `task_acks_late=True` — tasks acknowledged only after completion.
- `worker_prefetch_multiplier=1` — one task at a time per worker process.
- Tasks go in `app/tasks/` directory with autodiscovery.
- Celery workers run as separate K8s Deployments (not in the API pod).

---

## 11. Dockerfile

```dockerfile
FROM python:3.13-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --user --no-cache-dir -r requirements.txt
ENV PATH=/root/.local/bin:$PATH
RUN opentelemetry-bootstrap -a install
COPY . .

FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8080
ENTRYPOINT ["opentelemetry-instrument", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Rules

- Multi-stage build is **mandatory** — builder stage for dependencies, final stage for runtime.
- Base image: `python:3.13-slim` (or latest supported slim variant).
- `--user` flag on pip install for clean layer separation.
- `opentelemetry-bootstrap -a install` in builder stage — installs OTEL auto-instrumentation packages.
- Entrypoint wraps uvicorn with `opentelemetry-instrument`.
- Port is always `8080`.
- No `CMD` — use `ENTRYPOINT` exclusively.
- Celery workers override the entrypoint in K8s deployment manifests.

---

## 12. Kubernetes Manifests

### Structure (Kustomize)

```
infra/kube-manifests/{domain}/{service}/
├── base/
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   └── background_workers.yaml    # If Celery workers exist
├── testing/
│   ├── kustomization.yaml
│   └── deployment-patch.yaml
└── production/
    ├── kustomization.yaml
    ├── hpa.yaml                   # HorizontalPodAutoscaler
    └── pdb.yaml                   # PodDisruptionBudget
```

### Deployment Requirements

Every deployment MUST include:

| Feature | Specification |
|---------|--------------|
| Update strategy | `RollingUpdate`, `maxUnavailable: 0`, `maxSurge: 1` |
| Startup probe | `httpGet /health`, `failureThreshold: 30`, `periodSeconds: 10` |
| Liveness probe | `httpGet /health`, `failureThreshold: 5`, `periodSeconds: 30` |
| Readiness probe | `httpGet /health`, `failureThreshold: 3`, `periodSeconds: 10` |
| PreStop hook | `sleep 30` (graceful drain) |
| Resource requests | Always set (CPU + memory) |
| Resource limits | Always set (CPU + memory) |
| Pod anti-affinity | `preferredDuringSchedulingIgnoredDuringExecution` on hostname |
| Node selector | `workload-pool: application-workloads` |
| Image pull secrets | `ecr-reg-secret` |
| Environment | `envFrom` ConfigMap + Secret refs |

### OTEL Resource Attributes (per deployment)

```yaml
env:
  - name: OTEL_RESOURCE_ATTRIBUTES
    value: "service.name={service-name},deployment.environment={env},k8s.pod.ip=$(K8S_POD_IP),k8s.cluster.name={cluster}"
```

### Production Extras

- **HPA**: `minReplicas: 2`, `maxReplicas: 20`, CPU target 70%, scaleDown stabilization 300s.
- **PDB**: `minAvailable: 2`.

---

## 13. CI/CD Pipeline

### Flow

```
Code push → GitHub Actions → Detect changes → Build Docker → Push ECR → Update manifests → Commit → ArgoCD sync → Kubernetes rollout
```

### Rules

- Image tags use git SHA: `{service-name}-{git-sha}` — never `latest`.
- Only changed services are built (path-filter detection).
- Manifest updates are atomic — single commit for all changed services.
- ArgoCD handles deployment — CI never runs `kubectl apply`.
- Production deploys from `main` branch, testing from `testing` branch.

### ECR Repository Naming

Services map to ECR repos. The mapping is defined in the CI workflow. When adding a new service, add its mapping to the workflow matrix.

---

## 14. Environment Management

### Config Hierarchy (highest precedence first)

1. Deployment-specific env vars (inline in deployment.yaml)
2. Environment-specific ConfigMap patch (testing/production)
3. Base ConfigMap (`secrets-and-configs/base/configmap.yaml`)
4. K8s Secrets (SealedSecrets)

### Shared Config Location

```
infra/kube-manifests/{domain}/secrets-and-configs/
├── base/
│   ├── configmap.yaml          # Shared env vars
│   ├── sealed-secret.yaml      # Encrypted secrets
│   └── kustomization.yaml
├── testing/
│   ├── configmap-patch.yaml
│   └── sealed-secrets-patch.yaml
└── production/
    ├── configmap-patch.yaml
    └── sealed-secrets-patch.yaml
```

### Rules

- Secrets are **SealedSecrets** — never plain-text in the repo.
- Config shared across services goes in the shared configmap.
- Service-specific config goes in the service's deployment env vars.
- `ENVIRONMENT` var is always set: `local`, `testing`, or `prod`.

---

## 15. Observability

### Stack

| Layer | Tool | Protocol |
|-------|------|----------|
| Traces | SigNoz | OTLP gRPC |
| Logs | SigNoz | OTLP (via logging bridge) |
| Metrics | SigNoz | OTLP gRPC |
| LLM Observability | Langfuse | HTTP |

### Rules

- Tracing is automatic via `opentelemetry-instrument` entrypoint — no manual span creation needed for HTTP endpoints.
- Manual spans only for non-HTTP operations (background jobs, custom pipelines).
- Log correlation is automatic — `OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true`.
- Use structured logging `extra` dicts for custom attributes that should appear in SigNoz.
- Service name in OTEL MUST match the K8s deployment name.

---

## 16. Adding a New Service — Checklist

When creating a new FastAPI service, complete every item:

- [ ] Create service directory under `services/{domain}/{service-name}/` following the standard structure
- [ ] `main.py` with `root_path`, health endpoint, middleware, exception handlers
- [ ] `app/settings.py` with Pydantic BaseSettings
- [ ] `app/__init__.py` with logger setup
- [ ] `app/api/v1/` with routes and endpoints
- [ ] `Dockerfile` with multi-stage build and OTEL entrypoint
- [ ] `requirements.txt` with pinned versions
- [ ] K8s base manifests: deployment, service, ingress, kustomization
- [ ] K8s testing overlay with patches
- [ ] K8s production overlay with HPA and PDB
- [ ] OTEL_RESOURCE_ATTRIBUTES in deployment env
- [ ] Add service to CI workflow path filters and ECR mapping
- [ ] Add ArgoCD Application manifest in `infra/argo-pipelines/applications/`
- [ ] Request logging middleware
- [ ] Security headers middleware
- [ ] Exception handlers (HTTPException, ValidationError, generic)

---

## 17. Code Style & Conventions

- **Async by default**: All endpoint handlers use `async def`. Sync only in Celery tasks.
- **No inline comments** unless logic is genuinely non-obvious.
- **No docstrings on internal functions** — code should be self-documenting.
- **Imports**: stdlib → third-party → local, separated by blank lines.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- **Type hints**: On function signatures for public APIs and service layer methods. Not required on every internal helper.
- **F-strings**: Preferred over `.format()` or `%` formatting.

---

## 18. Security

- No secrets in code or environment files committed to git.
- CORS origins are explicitly listed — never `allow_origins=["*"]` in production.
- JWT validation on protected endpoints.
- Input validation via Pydantic models — never trust raw input.
- Security headers (HSTS, X-Content-Type-Options, X-Frame-Options) on every response.
- Environment-gated debug endpoints (e.g., chaos endpoints only on `ENVIRONMENT=testing`).

---

## 19. LLM / AI Agent Instructions

When an LLM is generating or modifying code in this repository:

1. **Read existing code first** — match the patterns in the service you're modifying.
2. **Follow the directory structure** exactly as defined in Section 1.
3. **Never skip middleware** — request logging and security headers are mandatory.
4. **Never skip exception handlers** — all three handlers must be registered.
5. **Use async Motor** for database operations in endpoints, sync PyMongo only in Celery.
6. **Settings via Pydantic** — never raw `os.getenv` in business logic.
7. **Health endpoint at root** — `GET /health` returning `{"status": "ok"}`.
8. **Port 8080** — hardcoded in Dockerfile and deployment manifests.
9. **OTEL is automatic** — don't add manual instrumentation unless it's a non-HTTP code path.
10. **Image tags are SHA-based** — never use `latest`.
11. **When adding a new service**, complete every item in the checklist (Section 16).
12. **Error responses** follow the standard format: `{"message": ..., "success": ..., "error": ...}`.
13. **Log to stdout only**, using the standard logger setup with `extra` dicts for structured data.
