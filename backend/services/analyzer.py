"""
Voice Emotion & Transcription Analysis Service for HealHub.
Handles Hugging Face model inference (local pipeline or HF Inference API).
"""

import time
import requests
import numpy as np
from transformers import pipeline
from backend.config import settings
from backend.services.audio_processor import prepare_audio_array, TARGET_SR

# Layman-friendly messages and risk levels for SIH prototype
FRIENDLY_MESSAGE = {
    "happy": "You seem to be in a good, positive mood 😊",
    "sad": "You seem to be feeling low or down right now 😔",
    "angry": "You seem to be feeling frustrated or angry right now 😠",
    "fear": "You seem to be feeling anxious or scared right now 😟",
    "disgust": "You seem to be feeling uneasy or uncomfortable right now 😖",
    "surprise": "You seem to be feeling surprised or caught off guard 😮",
    "neutral": "You seem calm and neutral right now 🙂",
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
    "high": "If these feelings persist, please consider reaching out to someone you trust or a support helpline.",
    "moderate": "It might help to talk to someone about how you're feeling.",
    "low": "",
}

# Singleton instances for local pipeline lazy loading
_emotion_pipeline = None
_asr_pipeline = None


def get_emotion_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        print("Loading emotion classification model...")
        _emotion_pipeline = pipeline("audio-classification", model=settings.EMOTION_MODEL)
    return _emotion_pipeline


def get_asr_pipeline():
    global _asr_pipeline
    if _asr_pipeline is None:
        print("Loading speech recognition ASR model...")
        _asr_pipeline = pipeline("automatic-speech-recognition", model=settings.ASR_MODEL)
    return _asr_pipeline


def preload_models():
    """Pre-load models on server startup for instant request response."""
    if not (settings.USE_HF_INFERENCE_API and settings.HF_TOKEN):
        get_asr_pipeline()
        get_emotion_pipeline()


def analyze_voice_bytes(audio_bytes: bytes) -> dict:
    """
    Analyzes raw voice audio bytes.
    Returns structured result with transcript, emotion analysis, risk assessment, and metadata.
    
    Time Complexity: O(N) where N is audio samples for preprocessing + transformer inference overhead.
    Space Complexity: O(N) array memory + O(1) per request model inference state.
    """
    start_time = time.time()
    
    if not audio_bytes:
        return {
            "success": False,
            "error": "No audio received. Please record or upload a voice note.",
            "transcript": "",
            "emotion": "",
            "emotion_label": "",
            "concern_level": "low",
            "support_message": "",
            "execution_time_seconds": 0
        }

    # Prepare audio array
    try:
        processed_audio = prepare_audio_array(audio_bytes)
    except Exception as e:
        return {
            "success": False,
            "error": f"Could not process audio: {str(e)}",
            "transcript": "",
            "emotion": "",
            "emotion_label": "",
            "concern_level": "low",
            "support_message": "",
            "execution_time_seconds": round(time.time() - start_time, 3)
        }

    # Check if using HF Inference API
    if settings.USE_HF_INFERENCE_API and settings.HF_TOKEN:
        transcript, emotion_text, top_label, concern, support_line = _analyze_via_hf_api(audio_bytes)
    else:
        transcript, emotion_text, top_label, concern, support_line = _analyze_via_local_pipeline(processed_audio)

    elapsed_time = round(time.time() - start_time, 3)

    return {
        "success": True,
        "error": None,
        "transcript": transcript,
        "emotion": emotion_text,
        "emotion_label": top_label,
        "concern_level": concern,
        "support_message": support_line,
        "execution_time_seconds": elapsed_time
    }


def _analyze_via_local_pipeline(processed_audio: np.ndarray):
    """Local inference using Hugging Face Transformers pipeline."""
    # Check for silent or empty audio input
    max_amplitude = float(np.max(np.abs(processed_audio))) if len(processed_audio) > 0 else 0
    is_silent = max_amplitude < 0.005

    # 1. Transcription
    if is_silent:
        transcript = "No speech detected (silent audio)."
    else:
        try:
            asr_pipe = get_asr_pipeline()
            asr_result = asr_pipe(
                {"array": processed_audio, "sampling_rate": TARGET_SR},
                generate_kwargs={
                    "language": "english",
                    "task": "transcribe"
                }
            )
            raw_text = asr_result.get("text", "").strip()
            # Clean trailing dot hallucination on silence if any
            if raw_text.replace(".", "").strip() == "":
                transcript = "No speech detected."
            else:
                transcript = raw_text
        except Exception as e:
            transcript = f"Could not transcribe audio: {str(e)}"

    # 2. Emotion Detection
    try:
        emo_pipe = get_emotion_pipeline()
        results = emo_pipe(processed_audio, sampling_rate=TARGET_SR, top_k=5)
        top_label = results[0]["label"].lower()
        main_sentence = FRIENDLY_MESSAGE.get(
            top_label, f"Detected emotional tone: {top_label}"
        )
        concern = CONCERN_LEVEL.get(top_label, "low")
        support_line = SUPPORT_MESSAGE.get(concern, "")
        
        emotion_text = main_sentence
        if support_line:
            emotion_text += f"\n\n{support_line}"
    except Exception as e:
        top_label = "unknown"
        concern = "low"
        support_line = ""
        emotion_text = f"Could not analyze emotion: {str(e)}"

    return transcript, emotion_text, top_label, concern, support_line


def _analyze_via_hf_api(audio_bytes: bytes):
    """Remote inference using Hugging Face Inference API."""
    headers = {"Authorization": f"Bearer {settings.HF_TOKEN}"}
    
    # Transcription API call
    try:
        asr_url = f"https://api-inference.huggingface.co/models/{settings.ASR_MODEL}"
        res_asr = requests.post(asr_url, headers=headers, data=audio_bytes, timeout=15)
        if res_asr.status_code == 200:
            transcript = res_asr.json().get("text", "").strip() or "Could not transcribe audio."
        else:
            transcript = "Could not transcribe audio via HF API."
    except Exception:
        transcript = "Could not transcribe audio."

    # Emotion Detection API call
    try:
        emo_url = f"https://api-inference.huggingface.co/models/{settings.EMOTION_MODEL}"
        res_emo = requests.post(emo_url, headers=headers, data=audio_bytes, timeout=15)
        if res_emo.status_code == 200:
            emo_results = res_emo.json()
            if isinstance(emo_results, list) and len(emo_results) > 0:
                top_label = emo_results[0].get("label", "").lower()
            else:
                top_label = "neutral"
            
            main_sentence = FRIENDLY_MESSAGE.get(
                top_label, f"Detected emotional tone: {top_label}"
            )
            concern = CONCERN_LEVEL.get(top_label, "low")
            support_line = SUPPORT_MESSAGE.get(concern, "")
            emotion_text = main_sentence
            if support_line:
                emotion_text += f"\n\n{support_line}"
        else:
            top_label = "unknown"
            concern = "low"
            support_line = ""
            emotion_text = "Could not analyze emotion via HF API."
    except Exception as e:
        top_label = "unknown"
        concern = "low"
        support_line = ""
        emotion_text = f"Could not analyze emotion: {str(e)}"

    return transcript, emotion_text, top_label, concern, support_line
