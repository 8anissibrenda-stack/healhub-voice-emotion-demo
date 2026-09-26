"""
FastAPI application entry point for HealHub Voice Emotion Demo.
"""

import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.services.analyzer import analyze_voice_bytes

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm models on server startup
    print("Pre-loading AI models on startup...")
    from backend.services.analyzer import preload_models
    preload_models()
    print("AI models loaded and ready for fast inference!")
    yield

app = FastAPI(
    title="HealHub Voice Emotion & Speech Recognition API",
    description="SIH Prototype backend for voice emotion detection and transcription",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Mount static frontend directories if they exist
if (FRONTEND_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")

if (FRONTEND_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")

if (FRONTEND_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the primary responsive Web UI."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h2>HealHub API Server Running. Frontend index.html not found.</h2>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "HealHub Voice Emotion Backend",
        "emotion_model": settings.EMOTION_MODEL,
        "asr_model": settings.ASR_MODEL,
        "use_hf_api": settings.USE_HF_INFERENCE_API
    }


@app.post("/api/analyze")
def analyze_audio(file: UploadFile = File(...)):
    """
    API Endpoint: Receives voice note audio file, returns transcription & emotion analysis.
    Runs synchronously in FastAPI worker thread pool to prevent blocking event loop.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")
    
    try:
        contents = file.file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        result = analyze_voice_bytes(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False
    )
