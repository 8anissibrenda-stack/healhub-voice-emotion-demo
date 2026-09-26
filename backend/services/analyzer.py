"""
Voice Emotion & Transcription Analysis Service for HealHub.
Handles robust transcription and multimodal emotion fusion.
"""

import re
import time

import numpy as np
from transformers import pipeline

from backend.config import settings
from backend.services.audio_processor import TARGET_SR, prepare_audio_array
from backend.services.emotion_pipeline import (
    build_multimodal_result,
    compute_acoustic_features,
    normalize_emotion_label,
)
from backend.services.text_emotion import analyze_text_emotion

FRIENDLY_MESSAGE = {
    "happy": "Detected emotional state: positive and upbeat 😊",
    "sad": "Detected emotional state: low energy or distressed 😔",
    "angry": "Detected emotional state: elevated tension / anger 😠",
    "fear": "Detected emotional state: anxious or fearful 😟",
    "disgust": "Detected emotional state: discomfort / unease 😖",
    "surprise": "Detected emotional state: surprised or startled 😮",
    "neutral": "Detected emotional state: calm and neutral 🙂",
}

CONCERN_LEVEL = {
    "angry": "high",
    "fear": "high",
    "sad": "moderate",
    "disgust": "moderate",
    "surprise": "low",
    "happy": "low",
    "neutral": "low",
}

SUPPORT_MESSAGE = {
    "high": "AI-assisted screening suggests significant emotional intensity. Consider supportive conversation or trusted help if needed.",
    "moderate": "The signal is mixed but suggests some emotional strain; a simple check-in may help.",
    "low": "The signal is mild and may reflect ordinary day-to-day variation.",
}

_emotion_pipeline = None
_asr_pipeline = None


def get_emotion_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        print("Loading voice emotion model...")
        _emotion_pipeline = pipeline("audio-classification", model=settings.EMOTION_MODEL, top_k=None)
    return _emotion_pipeline


def get_asr_pipeline():
    global _asr_pipeline
    if _asr_pipeline is None:
        print("Loading ASR model...")
        _asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=settings.ASR_MODEL,
            chunk_length_s=30,
            torch_dtype=None,
        )
    return _asr_pipeline


def preload_models():
    """Warm up models on server startup for faster analysis."""
    if not (settings.USE_HF_INFERENCE_API and settings.HF_TOKEN):
        get_asr_pipeline()
        get_emotion_pipeline()


def _normalize_transcript(raw_text: str) -> str:
    cleaned = (raw_text or "").strip()
    if not cleaned:
        return "No speech detected."
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" .")
    if cleaned.lower().startswith("you seem"):
        return cleaned
    return cleaned if cleaned else "No speech detected."


def _classify_voice_model(processed_audio: np.ndarray) -> dict:
    results = get_emotion_pipeline()(processed_audio, sampling_rate=TARGET_SR, top_k=5)
    aggregated = {}
    for item in results or []:
        label = normalize_emotion_label(str(item.get("label", "neutral")))
        score = float(item.get("score", 0.0))
        aggregated[label] = aggregated.get(label, 0.0) + score
    if not aggregated:
        return {"label": "neutral", "score": 0.25, "all_scores": []}
    best_label, best_score = max(aggregated.items(), key=lambda pair: pair[1])
    return {
        "label": best_label,
        "score": round(float(best_score), 3),
        "all_scores": [
            {"label": k, "score": round(float(v), 3)} for k, v in sorted(aggregated.items(), key=lambda pair: pair[1], reverse=True)
        ],
    }


def analyze_voice_bytes(audio_bytes: bytes) -> dict:
    """Return transcript, voice result, text result and fused final analysis."""
    start_time = time.time()

    if not audio_bytes:
        return {
            "success": False,
            "error": "No audio received. Please record or upload a voice note.",
            "transcript": "",
            "emotion": "",
            "emotion_label": "",
            "voice_label": "",
            "text_label": "",
            "confidence": 0.0,
            "voice_confidence": 0.0,
            "text_confidence": 0.0,
            "concern_level": "low",
            "support_message": "",
            "execution_time_seconds": 0,
        }

    try:
        processed_audio = prepare_audio_array(audio_bytes)
    except Exception as exc:
        return {
            "success": False,
            "error": f"Could not process audio: {str(exc)}",
            "transcript": "",
            "emotion": "",
            "emotion_label": "",
            "voice_label": "",
            "text_label": "",
            "confidence": 0.0,
            "voice_confidence": 0.0,
            "text_confidence": 0.0,
            "concern_level": "low",
            "support_message": "",
            "execution_time_seconds": round(time.time() - start_time, 3),
        }

    max_amplitude = float(np.max(np.abs(processed_audio))) if len(processed_audio) > 0 else 0.0
    is_silent = max_amplitude < 0.005

    if is_silent:
        transcript = "No speech detected (silent audio)."
        voice_result = {"label": "neutral", "score": 0.2}
        text_result = {"label": "neutral", "score": 0.1}
        acoustic_features = compute_acoustic_features(processed_audio)
    else:
        try:
            asr_pipe = get_asr_pipeline()
            asr_result = asr_pipe(
                {"array": processed_audio, "sampling_rate": TARGET_SR},
                generate_kwargs={"language": "en", "task": "transcribe"},
            )
            transcript = _normalize_transcript(asr_result.get("text", ""))
        except Exception as exc:
            transcript = f"Could not transcribe audio reliably: {str(exc)}"

        try:
            voice_result = _classify_voice_model(processed_audio)
        except Exception:
            voice_result = {"label": "neutral", "score": 0.25}

        try:
            text_result = analyze_text_emotion(transcript) if transcript and "No speech detected" not in transcript else {"label": "neutral", "score": 0.15}
        except Exception:
            text_result = {"label": "neutral", "score": 0.15}

        acoustic_features = compute_acoustic_features(processed_audio)

    fused = build_multimodal_result(voice_result, text_result, acoustic_features)
    final_label = fused["final_label"]
    confidence = fused["confidence"]
    confidence_level = "high" if confidence >= 0.7 else "medium" if confidence >= 0.45 else "low"
    message = FRIENDLY_MESSAGE.get(final_label, f"Detected emotional state: {final_label}")

    support_line = SUPPORT_MESSAGE.get(CONCERN_LEVEL.get(final_label, "low"), SUPPORT_MESSAGE["low"])
    evidence = "; ".join(fused.get("evidence", []))
    if evidence:
        message = f"{message}. Why: {evidence}."

    elapsed_time = round(time.time() - start_time, 3)

    return {
        "success": True,
        "error": None,
        "transcript": transcript,
        "emotion": message,
        "emotion_label": final_label,
        "voice_label": voice_result.get("label", "neutral"),
        "text_label": text_result.get("label", "neutral"),
        "confidence": round(float(confidence), 3),
        "voice_confidence": round(float(voice_result.get("score", 0.25)), 3),
        "text_confidence": round(float(text_result.get("score", 0.15)), 3),
        "confidence_level": confidence_level,
        "concern_level": CONCERN_LEVEL.get(final_label, "low"),
        "support_message": support_line,
        "explanation": fused.get("evidence", []),
        "execution_time_seconds": elapsed_time,
    }
