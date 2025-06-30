import time
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.simple_pii_middleware import SimplePIIMiddleware
from app.services.pii import get_background_processor
from app.db.session import create_db_and_tables


logger = configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for FastAPI app. This function will be executed at startup 
    and shutdown of the application.
    """
    # Startup
    logger.info("Starting up IRCC AI Platform")
    await create_db_and_tables()
    logger.info("Database initialization completed")
    
    # Start background PII processor for two-tier architecture
    if settings.ENABLE_PII_FILTERING:
        background_processor = get_background_processor()
        await background_processor.start()
        logger.info("Background PII processor started for two-tier architecture")
    else:
        logger.info("Background PII processor disabled (PII filtering disabled)")
    
    yield
    
    # Shutdown
    logger.info("Shutting down IRCC AI Platform")
    
    # Stop background PII processor
    if settings.ENABLE_PII_FILTERING:
        background_processor = get_background_processor()
        await background_processor.stop()
        logger.info("Background PII processor stopped")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="IRCC AI Platform API",
    version="0.1.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)


# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS if settings.ALLOWED_HOSTS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PII filtering is now handled by EnterpriseContentFilter in the orchestrator
# No middleware needed - filtering happens at the service layer where it belongs
logger.info("PII filtering handled by EnterpriseContentFilter in orchestrator service")


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Get client IP safely
    try:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            # Avoid hanging on client.host access
            client_ip = getattr(request.client, 'host', 'unknown') if request.client else "unknown"
    except Exception:
        client_ip = "unknown"
    
    logger.info(
        f"Request started: {request.method} {request.url.path} from {client_ip}"
    )
    
    try:
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"status={response.status_code} duration={process_time:.3f}s"
        )
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"error={str(e)} duration={process_time:.3f}s"
        )
        raise


# Re-enable API router (without enhanced_moderation)
app.include_router(api_router, prefix=settings.API_V1_STR)
print("API router enabled (enhanced_moderation disabled)")
print("Registered routes:")
for route in app.routes:
    print(f"{route.path} [{route.name}]")

@app.get("/", tags=["Status"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "version": "0.1.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
