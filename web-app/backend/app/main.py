from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from .routes import players, trades, projections, prospects
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LongBall API",
    description="API for baseball projections and trade analysis",
    version="1.0.0"
)

# Update CORS for Railway deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local development
        "https://*.railway.app",   # Railway domains
        "https://longball-production.up.railway.app"  # Your Railway frontend app
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression
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

app.include_router(
    players.router,
    prefix="/api",
    tags=["players"]
)

app.include_router(
    trades.router,
    prefix="/api",
    tags=["trades"]
)

app.include_router(
    projections.router,
    prefix="/api",
    tags=["projections"]
)

app.include_router(
    prospects.router,
    prefix="/api",
    tags=["prospects"]
)

@app.get("/")
async def root():
    return {
        "message": "MLB Player Evaluation API",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": "production" if "RAILWAY_STATIC_URL" in os.environ else "development"
    }

# Update main entry point for Railway
if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        workers=4,     # Number of worker processes
        log_level="info"
    )