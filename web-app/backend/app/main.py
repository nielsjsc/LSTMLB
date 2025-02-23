from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import logging
import time
import sys
import os
from sqlalchemy import text
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

# Change relative imports to absolute
from app.routes import players, trades, projections, prospects
from app.database import SessionLocal, engine
# Add security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

# Configure logging with more detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LongBall API",
    description="API for baseball projections and trade analysis",
    version="1.0.0",
    # Remove docs_url prefix
    docs_url="/docs",
    redoc_url="/redoc"
)

# Environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
PORT = int(os.getenv("PORT", 8000))
WORKERS = int(os.getenv("WEB_CONCURRENCY", 4))
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
RAILWAY_STATIC_URL = os.getenv("RAILWAY_STATIC_URL", "")
RAILWAY_URL = os.getenv("RAILWAY_URL", "")

# CORS configuration
origins = [
    FRONTEND_URL,
    "https://longball-production.up.railway.app",
    "https://longball-analytics.com",
    "https://longball-api.onrender.com"  # Add Render URL
]

if RAILWAY_STATIC_URL:
    origins.append(f"https://{RAILWAY_STATIC_URL}")

# Add middlewares
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)



@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    try:
        start_time = datetime.utcnow()
        response = await call_next(request)
        end_time = datetime.utcnow()
        
        # Calculate duration
        duration = (end_time - start_time).total_seconds()
        response.headers["X-Process-Time"] = str(duration)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in middleware: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

app.include_router(players.router, prefix="/players", tags=["players"])
app.include_router(prospects.router, prefix="/prospects", tags=["prospects"])
app.include_router(trades.router, prefix="/trades", tags=["trades"])
app.include_router(projections.router, prefix="/projections", tags=["projections"])


@app.get("/")
async def root():
    return {
        "message": "LongBall API",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now().isoformat()
    }

# Enhanced error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {exc}", exc_info=True)
    return {
        "detail": "Internal server error",
        "path": request.url.path,
        "method": request.method,
        "timestamp": datetime.now().isoformat()
    }

# Enhanced health check
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "cors_origins": origins,
    }
    
    try:
        # Test database connection with text() wrapper
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            health_status["database"] = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["database"] = "disconnected"
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
    
    return health_status


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server in {ENVIRONMENT} mode")
    logger.info(f"CORS origins: {origins}")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=PORT,
        workers=WORKERS,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_level="info"
    )