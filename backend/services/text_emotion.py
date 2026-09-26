from __future__ import annotations

from typing import Any

from transformers import pipeline

from backend.config import settings

_text_pipeline = None

TEXT_EMOTION_MAP = {
    "anger": "angry",
    "joy": "happy",
    "sadness": "sad",
    "fear": "fear",
    "neutral": "neutral",
    "surprise": "surprise",
    "disgust": "disgust",
}


def get_text_emotion_pipeline():
    global _text_pipeline
    if _text_pipeline is None:
        print("Loading text emotion model...")
        _text_pipeline = pipeline(
            "text-classification",
            model=settings.TEXT_EMOTION_MODEL,
            top_k=None,
        )
    return _text_pipeline


def analyze_text_emotion(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Text input is empty.")

    pipe = get_text_emotion_pipeline()
    result = pipe(cleaned, truncation=True)

    scores = result[0] if isinstance(result, list) and result and isinstance(result[0], list) else result
    if not isinstance(scores, list):
        raise ValueError("Unexpected text-emotion model output format.")

    best = max(scores, key=lambda item: float(item.get("score", 0.0)))
    label = str(best.get("label", "neutral")).lower()
    normalized = TEXT_EMOTION_MAP.get(label, "neutral")

    return {
        "label": normalized,
        "score": round(float(best.get("score", 0.0)), 3),
        "all_scores": [
            {"label": TEXT_EMOTION_MAP.get(str(item.get("label", "neutral")).lower(), "neutral"), "score": round(float(item.get("score", 0.0)), 3)}
            for item in scores
        ],
    }
