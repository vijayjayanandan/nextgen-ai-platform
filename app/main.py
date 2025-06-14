import time
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.pii_middleware import PIIFilteringMiddleware
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
    logger.info("Database tables created")
    
    yield
    
    # Shutdown
    logger.info("Shutting down IRCC AI Platform")


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

# Add Enhanced PII Filtering Middleware
if settings.ENABLE_PII_FILTERING:
    app.add_middleware(
        PIIFilteringMiddleware,
        enable_filtering=settings.ENABLE_PII_FILTERING
    )
    logger.info("Enhanced PII filtering middleware enabled")
else:
    logger.info("Enhanced PII filtering middleware disabled")


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Get client IP
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0]
    else:
        client_ip = request.client.host if request.client else "unknown"
    
    logger.info(
        f"Request started: {request.method} {request.url.path} from {client_ip}"
    )
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"Request completed: {request.method} {request.url.path} "
        f"status={response.status_code} duration={process_time:.3f}s"
    )
    
    return response


# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)
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
