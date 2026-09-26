import math
import re
from typing import Any

import numpy as np

LABEL_ALIASES = {
    "angry": "angry",
    "mad": "angry",
    "rage": "angry",
    "frustrated": "angry",
    "irritated": "angry",
    "agitated": "angry",
    "happy": "happy",
    "joy": "happy",
    "joyful": "happy",
    "excited": "happy",
    "enthusiastic": "happy",
    "glad": "happy",
    "sad": "sad",
    "sadness": "sad",
    "upset": "sad",
    "melancholy": "sad",
    "depressed": "sad",
    "depressive": "sad",
    "low": "sad",
    "fear": "fear",
    "fearful": "fear",
    "afraid": "fear",
    "anxious": "fear",
    "scared": "fear",
    "nervous": "fear",
    "neutral": "neutral",
    "calm": "neutral",
    "balanced": "neutral",
    "bored": "neutral",
    "surprise": "surprise",
    "surprised": "surprise",
    "disgust": "disgust",
    "disgusted": "disgust",
    "distressed": "sad",
    "stressed": "sad",
    "tired": "sad",
}

SUPPORTED_LABELS = ["angry", "happy", "sad", "fear", "neutral", "surprise", "disgust"]


def normalize_emotion_label(label: str | None) -> str:
    if not label:
        return "neutral"
    cleaned = re.sub(r"[^a-z]+", " ", label.lower()).strip()
    if cleaned in LABEL_ALIASES:
        return LABEL_ALIASES[cleaned]
    for alias, mapped in LABEL_ALIASES.items():
        if cleaned.startswith(alias) or alias in cleaned.split():
            return mapped
    return "neutral"


def compute_acoustic_features(audio: Any) -> dict[str, float]:
    signal = np.asarray(audio, dtype=np.float32).ravel()
    if signal.size == 0:
        return {
            "intensity": 0.0,
            "pitch_variation": 0.0,
            "speech_rate": 0.0,
            "pause_ratio": 0.0,
            "voiced_ratio": 0.0,
            "energy": 0.0,
        }

    rms = float(np.sqrt(np.mean(np.square(signal))))
    energy = float(np.mean(np.square(signal)))
    voiced = np.abs(signal) > max(0.02, 0.05 * np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else 0.02)
    voiced_ratio = float(np.mean(voiced)) if signal.size > 0 else 0.0
    zero_crossing = np.mean(np.abs(np.diff(np.signbit(signal)))) if signal.size > 1 else 0.0
    pitch_variation = float(np.std(np.abs(np.diff(signal)))) if signal.size > 1 else 0.0
    pause_ratio = float(np.mean(np.abs(signal) < 0.01)) if signal.size > 0 else 0.0
    speech_rate = float(np.clip((voiced_ratio * 1.8) + (zero_crossing * 0.8), 0.0, 1.0))
    intensity = float(np.clip((rms * 3.5), 0.0, 1.0))

    return {
        "intensity": intensity,
        "pitch_variation": float(np.clip(pitch_variation * 2.0, 0.0, 1.0)),
        "speech_rate": speech_rate,
        "pause_ratio": float(np.clip(pause_ratio, 0.0, 1.0)),
        "voiced_ratio": float(np.clip(voiced_ratio, 0.0, 1.0)),
        "energy": float(np.clip(energy * 2.0, 0.0, 1.0)),
    }


def infer_acoustic_bias(features: dict[str, float]) -> dict[str, float]:
    bias = {label: 0.0 for label in SUPPORTED_LABELS}
    intensity = features.get("intensity", 0.0)
    pitch_variation = features.get("pitch_variation", 0.0)
    speech_rate = features.get("speech_rate", 0.0)
    pause_ratio = features.get("pause_ratio", 0.0)
    voiced_ratio = features.get("voiced_ratio", 0.0)

    if intensity > 0.45 and pitch_variation > 0.18 and speech_rate > 0.5:
        bias["angry"] += 0.22
    elif intensity > 0.30 and pitch_variation > 0.22:
        bias["happy"] += 0.12
    elif intensity < 0.18 and pitch_variation < 0.12 and pause_ratio > 0.42:
        bias["sad"] += 0.20
    elif intensity < 0.15 and speech_rate < 0.28 and voiced_ratio < 0.35:
        bias["neutral"] += 0.10
    elif intensity > 0.28 and speech_rate > 0.58:
        bias["surprise"] += 0.08

    if intensity < 0.12 and pause_ratio > 0.55:
        bias["sad"] += 0.10

    return bias


def build_multimodal_result(voice_result: dict[str, Any] | None, text_result: dict[str, Any] | None = None, acoustic_features: dict[str, float] | None = None) -> dict[str, Any]:
    voice_label = "neutral"
    voice_score = 0.25
    if voice_result:
        voice_label = normalize_emotion_label(str(voice_result.get("label", "neutral")))
        voice_score = float(voice_result.get("score", 0.25))
        if voice_score <= 0:
            voice_score = 0.25

    text_label = "neutral"
    text_score = 0.0
    if text_result:
        text_label = normalize_emotion_label(str(text_result.get("label", "neutral")))
        text_score = float(text_result.get("score", 0.0))

    combined = {label: 0.0 for label in SUPPORTED_LABELS}
    combined[voice_label] += voice_score * 0.7
    if text_score > 0:
        combined[text_label] += text_score * 0.3

    if acoustic_features:
        acoustic_bias = infer_acoustic_bias(acoustic_features)
        for label in SUPPORTED_LABELS:
            combined[label] += acoustic_bias.get(label, 0.0) * 0.35

    if voice_label == "neutral" and text_label != "neutral":
        combined[text_label] += 0.1

    if max(combined.values()) <= 0:
        combined["neutral"] = 0.5

    final_label = max(combined, key=combined.get)
    confidence = min(0.98, max(0.38, combined[final_label]))

    evidence = []
    if acoustic_features:
        if acoustic_features.get("intensity", 0.0) > 0.35:
            evidence.append("Elevated vocal intensity")
        if acoustic_features.get("pitch_variation", 0.0) > 0.18:
            evidence.append("High pitch variation")
        if acoustic_features.get("speech_rate", 0.0) > 0.45:
            evidence.append("Faster speech pacing")
        if acoustic_features.get("pause_ratio", 0.0) > 0.35:
            evidence.append("Noticeable pauses / low-energy phrasing")

    if text_result and text_label == final_label:
        evidence.append("Transcript content aligns with the detected vocal tone")
    elif text_result:
        evidence.append("Transcript content contributes a secondary signal")

    if not evidence:
        evidence.append("Model confidence remains moderate and may be ambiguous")

    return {
        "final_label": final_label,
        "confidence": round(float(confidence), 3),
        "voice_label": voice_label,
        "voice_confidence": round(float(voice_score), 3),
        "text_label": text_label,
        "text_confidence": round(float(text_score), 3),
        "evidence": evidence[:5],
    }
