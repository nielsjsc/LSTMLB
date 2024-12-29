from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import players, trades
# Initialize FastAPI app
app = FastAPI(
    title="MLB Player Evaluation API",
    description="API for MLB player valuations and trade analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(players.router, prefix="/api")
app.include_router(trades.router, prefix="/api/trades")
# Base routes
@app.get("/")
async def root():
    return {
        "message": "MLB Player Evaluation API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)