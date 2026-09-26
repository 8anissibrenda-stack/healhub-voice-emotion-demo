"""
Configuration file for HealHub Voice Emotion Demo backend.
Loads settings from environment variables with sensible defaults.
"""

import os

class Settings:
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    ASR_MODEL: str = os.getenv(
        "ASR_MODEL",
        "distil-whisper/distil-large-v3"
    )
    EMOTION_MODEL: str = os.getenv(
        "EMOTION_MODEL",
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )
    TEXT_EMOTION_MODEL: str = os.getenv(
        "TEXT_EMOTION_MODEL",
        "j-hartmann/emotion-english-distilroberta-base"
    )

    HF_TOKEN: str | None = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    USE_HF_INFERENCE_API: bool = os.getenv("USE_HF_INFERENCE_API", "").lower() in ("true", "1", "yes")

settings = Settings()
