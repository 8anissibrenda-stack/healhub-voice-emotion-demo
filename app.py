"""
HealHub - Voice Emotion Detection Demo
----------------------------------------
A minimal Hugging Face + Gradio demo that takes a voice note (recorded live
or uploaded, e.g. exported from WhatsApp) and predicts the speaker's
emotional state (happy, sad, angry, fearful, neutral, etc.).

This is a STANDALONE PROOF-OF-CONCEPT for the SIH26094 "HealHub" idea.
It demonstrates the "voice -> distress signal" piece of the pipeline.

Run:
    pip install -r requirements.txt
    python app.py

Then open the local URL Gradio prints (usually http://127.0.0.1:7860)
"""

import numpy as np
import librosa
import gradio as gr
from transformers import pipeline

# ---------------------------------------------------------------------------
# 1. Load the pretrained Speech Emotion Recognition (SER) model from
#    Hugging Face. This downloads automatically the first time you run it.
# ---------------------------------------------------------------------------
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

print("Loading model... (first run may take a minute to download)")
classifier = pipeline(
    "audio-classification",
    model=MODEL_NAME,
)
print("Model loaded.")

TARGET_SR = 16000  # this model expects 16kHz mono audio

# A simple mapping so we can show a friendlier "distress-oriented" label
# alongside the raw model output. Tune this for your own tiering logic.
DISTRESS_WEIGHT = {
    "angry": "High concern",
    "fear": "High concern",
    "sad": "Moderate concern",
    "disgust": "Moderate concern",
    "surprise": "Low concern",
    "happy": "Low concern",
    "neutral": "Baseline / Low concern",
}


def predict_emotion(audio):
    """
    audio: tuple (sample_rate, numpy_array) as provided by Gradio's
    microphone/upload audio component.
    """
    if audio is None:
        return "No audio received. Please record or upload a voice note."

    sr, data = audio

    # Gradio can give int16 PCM data; convert to float32 in [-1, 1]
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 1.0:
            data = data / 32768.0

    # Convert to mono if stereo
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample to 16kHz if needed (required by the model)
    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)

    # Run inference
    results = classifier(data, sampling_rate=TARGET_SR, top_k=5)

    # Format results nicely
    output_lines = ["Predicted emotions (ranked):\n"]
    for r in results:
        label = r["label"].lower()
        score = r["score"] * 100
        concern = DISTRESS_WEIGHT.get(label, "Unclassified")
        output_lines.append(f"- {label.capitalize():<10} {score:5.1f}%   -> {concern}")

    top_label = results[0]["label"].lower()
    top_concern = DISTRESS_WEIGHT.get(top_label, "Unclassified")
    output_lines.append(f"\nOverall signal: {top_label.upper()} ({top_concern})")

    return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# 2. Build a simple Gradio UI: record or upload -> get emotion prediction
# ---------------------------------------------------------------------------
demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record or upload a voice note"),
    outputs=gr.Textbox(label="Emotion Analysis Result", lines=10),
    title="HealHub - Voice Emotion Detection (Demo)",
    description=(
        "Prototype for SIH26094: record your voice or upload a voice note "
        "(e.g. exported from WhatsApp) to see the predicted emotional tone. "
        "This demonstrates the acoustic-distress-signal component of the "
        "HealHub pipeline (voice -> emotion -> distress tier)."
    ),
)

if __name__ == "__main__":
    demo.launch()
