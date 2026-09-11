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

import spaces
import numpy as np
import librosa
import gradio as gr
from transformers import pipeline

# ---------------------------------------------------------------------------
# 1. Load the pretrained Speech Emotion Recognition (SER) model from
#    Hugging Face. This downloads automatically the first time you run it.
# ---------------------------------------------------------------------------
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
WHISPER_MODEL = "openai/whisper-base"

print("Loading emotion model... (first run may take a minute to download)")
classifier = pipeline(
    "audio-classification",
    model=MODEL_NAME,
)
print("Emotion model loaded.")

print("Loading Whisper transcription model...")
transcriber = pipeline(
    "automatic-speech-recognition",
    model=WHISPER_MODEL,
)
print("Whisper model loaded.")

TARGET_SR = 16000  # this model expects 16kHz mono audio

# Human-readable descriptions mapped to each emotion label.
EMOTION_DESCRIPTION = {
    "happy":   "You seem to be in a good, positive mood 😊",
    "sad":     "You seem to be feeling low or down right now 😔",
    "angry":   "You seem to be feeling frustrated or angry right now 😠",
    "fear":    "You seem to be feeling anxious or scared right now 😟",
    "disgust": "You seem to be feeling uneasy or uncomfortable right now 😖",
    "surprise":"You seem to be feeling surprised or caught off guard 😮",
    "neutral": "You seem calm and neutral right now 🙂",
}

# Optional supportive follow-up line keyed by concern level.
SUPPORT_MESSAGE = {
    "high":     "If these feelings persist, please consider reaching out to someone you trust or a support helpline.",
    "moderate": "It might help to talk to someone about how you're feeling.",
    "low":      "",
}

# Maps each emotion to its concern level for the supportive message.
CONCERN_LEVEL = {
    "angry":   "high",
    "fear":    "high",
    "sad":     "moderate",
    "disgust": "moderate",
    "surprise":"low",
    "happy":   "low",
    "neutral": "low",
}


@spaces.GPU
def predict_emotion(audio):
    """
    audio: tuple (sample_rate, numpy_array) as provided by Gradio's
    microphone/upload audio component.
    """
    if audio is None:
        return (
            "No audio received. Please record or upload a voice note.",
            "No audio received. Please record or upload a voice note.",
        )

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

    # --- Transcription (Whisper) ---
    try:
        asr_result = transcriber({"array": data, "sampling_rate": TARGET_SR})
        transcript = asr_result.get("text", "").strip()
        if not transcript:
            transcript = "Could not transcribe audio."
    except Exception:
        transcript = "Could not transcribe audio."

    # --- Emotion classification ---
    results = classifier(data, sampling_rate=TARGET_SR, top_k=1)

    # Extract the top emotion label
    top_label = results[0]["label"].lower()

    # Build the friendly main sentence
    main_sentence = EMOTION_DESCRIPTION.get(
        top_label,
        "We couldn't quite read the emotional tone. Please try again. 🎙️"
    )

    # Build the optional supportive follow-up line
    concern = CONCERN_LEVEL.get(top_label, "low")
    support_line = SUPPORT_MESSAGE.get(concern, "")

    emotion_result = f"{main_sentence}\n\n{support_line}" if support_line else main_sentence

    return transcript, emotion_result


# ---------------------------------------------------------------------------
# 2. Build a simple Gradio UI: record or upload -> transcription + emotion
# ---------------------------------------------------------------------------
transcript_box = gr.Textbox(
    label="Transcript",
    lines=4,
    placeholder="Your spoken words will appear here...",
)
emotion_box = gr.Textbox(
    label="Emotion Analysis",
    lines=4,
    placeholder="Emotion result will appear here...",
)

demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record or upload a voice note"),
    outputs=[transcript_box, emotion_box],
    title="HealHub - Voice Emotion & Transcription Demo",
    description=(
        "Prototype for SIH26094: record your voice or upload a voice note "
        "(e.g. exported from WhatsApp) to see both the transcribed text and "
        "the predicted emotional tone. Powered by OpenAI Whisper (transcription) "
        "and wav2vec2 (emotion detection)."
    ),
)

if __name__ == "__main__":
    demo.launch()
