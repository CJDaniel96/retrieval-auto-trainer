"""
FastAPI Application Factory - 應用程式工廠
"""

import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Import controllers
from .controllers import (
    get_training_router,
    get_orientation_router,
    get_download_router,
    get_config_router
)

# Import middleware
from .middleware import (
    setup_cors,
    setup_error_handling,
    setup_logging
)

# Configure logging
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    創建並配置FastAPI應用程式
    """
    # Create FastAPI app
    app = FastAPI(
        title="Retrieval Auto Trainer API",
        description="自動化影像檢索訓練系統 API",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Setup middleware (order matters!)
    setup_cors(app)           # CORS should be first
    setup_logging(app)        # Logging middleware
    setup_error_handling(app) # Error handling should be last

    # Register API routers
    app.include_router(get_training_router())
    app.include_router(get_orientation_router())
    app.include_router(get_download_router())
    app.include_router(get_config_router())

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """健康檢查端點"""
        return {"status": "healthy", "service": "Retrieval Auto Trainer API"}

    # Root endpoint
    @app.get("/")
    async def root():
        """根端點"""
        return {
            "message": "Retrieval Auto Trainer API",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/health"
        }

    # Setup static files (if needed)
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    logger.info("FastAPI 應用程式已成功初始化")
    return app


# Create the app instance
app = create_app()