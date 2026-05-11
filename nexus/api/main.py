"""NEXUS API Main Module."""

from fastapi import FastAPI
from loguru import logger

from config import settings
from api.routes import router as nexus_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=settings.app_name,
        description="Agentic RAG System for Financial Document Intelligence",
        version="1.0.0",
        debug=settings.debug
    )
    
    # Include routers
    app.include_router(nexus_router, prefix="/api/v1")
    
    # Startup event
    @app.on_event("startup")
    async def startup():
        logger.info(f"Starting {settings.app_name}")
        logger.info(f"Debug mode: {settings.debug}")
        logger.info(f"Log level: {settings.log_level}")
        
        # Initialize database connection
        from db.neon import init_db
        await init_db()
        logger.info("Database connection initialized")
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Shutting down...")
        
        # Close database connection
        from db.neon import close_db
        await close_db()
        logger.info("Database connection closed")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": settings.app_name
        }
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
