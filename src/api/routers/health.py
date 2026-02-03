import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

health_router = APIRouter()

@health_router.get("/health", tags=["Health"])
async def health_check():
    """Проверка состояния сервиса."""
    logger.debug("Health check вызван")
    return {
        "status": "OK",
    }