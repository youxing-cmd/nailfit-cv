import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.measure import router as measure_router
from routers.measure_quality import router as measure_quality_router
from routers.preview import router as preview_router

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Optional Sentry initialization
sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        sentry_sdk.init(
            dsn=sentry_dsn,
            traces_sample_rate=0.1,
            profiles_sample_rate=0.1,
            integrations=[
                StarletteIntegration(),
                FastApiIntegration(),
            ],
        )
        logging.getLogger(__name__).info("Sentry initialized")
    except Exception as e:
        logging.getLogger(__name__).warning("Sentry initialization failed: %s", e)

app = FastAPI(title="NailFit CV API", version="0.2.0")

# CORS: read allowed origins from env, fallback to wildcard for dev
_origins = os.getenv("ALLOWED_ORIGINS", "")
allow_origins = [o.strip() for o in _origins.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(measure_router)
app.include_router(measure_quality_router)
app.include_router(preview_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
