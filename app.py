"""
HealHub - Voice Emotion Detection & Transcription Demo (FastAPI + Modern Web UI)
---------------------------------------------------------------------------------
Main application entry point.

To run:
    pip install -r requirements.txt
    python app.py
"""

import sys
import uvicorn
from backend.config import settings

if __name__ == "__main__":
    print(f"Starting HealHub Voice Emotion App on http://{settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False
    )