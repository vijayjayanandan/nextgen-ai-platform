from fastapi import APIRouter

from app.api.v1.endpoints import completions, chat, documents, retrieval, moderation, models, users, auth, enhanced_moderation

# Main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, tags=["Authentication"])
api_router.include_router(completions.router, prefix="/completions", tags=["Completions"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(retrieval.router, prefix="/retrieval", tags=["Retrieval"])
api_router.include_router(moderation.router, prefix="/moderation", tags=["Moderation"])
api_router.include_router(enhanced_moderation.router, prefix="/moderation", tags=["Enhanced Moderation"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
