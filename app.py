"""
HealHub - Voice Emotion Detection + Transcription Demo
--------------------------------------------------------
A Hugging Face + Gradio demo that takes a voice note (recorded live or
uploaded, e.g. exported from WhatsApp) and shows:
  1. A transcript of the speech (via Whisper)
  2. A simple, layman-friendly emotional-state reading (via a
     speech-emotion-recognition model)

This is a STANDALONE PROOF-OF-CONCEPT for the SIH26094 "HealHub" idea.
Built with gr.Blocks for a cleaner, more reliable layout than the
default gr.Interface.

Run:
    pip install -r requirements.txt
    python app.py
"""

try:
    import spaces  # only available on HuggingFace ZeroGPU Spaces
    HF_SPACES = True
except ImportError:
    HF_SPACES = False  # running locally — no GPU decorator needed

import numpy as np
import librosa
import gradio as gr
from transformers import pipeline

# ---------------------------------------------------------------------------
# 1. Load models once at startup
# ---------------------------------------------------------------------------
EMOTION_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
ASR_MODEL = "openai/whisper-base"

print("Loading emotion model...")
emotion_classifier = pipeline("audio-classification", model=EMOTION_MODEL)
print("Loading transcription model...")
transcriber = pipeline("automatic-speech-recognition", model=ASR_MODEL)
print("Models loaded.")

TARGET_SR = 16000

# Friendly, layman explanations for each raw emotion label
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


def _prepare_audio(audio):
    """Convert a Gradio (sample_rate, numpy_array) tuple into mono float32
    audio resampled to 16kHz, as required by both models."""
    sr, data = audio

    if data.dtype != np.float32:
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 1.0:
            data = data / 32768.0

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)

    return data


def analyze_voice(audio):
    """
    Takes Gradio audio input, returns (transcript_text, emotion_text).
    Wrapped defensively so a failure in one part doesn't take down the whole
    response.
    """
    if audio is None:
        return "No audio received. Please record or upload a voice note.", ""

    try:
        processed = _prepare_audio(audio)
    except Exception as e:
        return f"Could not process audio: {e}", ""

    # --- Transcription ---
    try:
        asr_result = transcriber({"array": processed, "sampling_rate": TARGET_SR})
        transcript = asr_result.get("text", "").strip()
        if not transcript:
            transcript = "Could not transcribe audio."
    except Exception:
        transcript = "Could not transcribe audio."

    # --- Emotion detection ---
    try:
        results = emotion_classifier(processed, sampling_rate=TARGET_SR, top_k=5)
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
        emotion_text = f"Could not analyze emotion: {e}"

    return transcript, emotion_text


# ---------------------------------------------------------------------------
# 2. Build the UI with gr.Blocks for a cleaner, more controllable layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="HealHub - Voice Emotion Detection") as demo:
    gr.Markdown(
        """
        # HealHub — Voice Emotion Detection (Demo)
        Prototype for **SIH26094**. Record your voice or upload a voice note
        (e.g. exported from WhatsApp) to see a transcript and a simple
        reading of the emotional tone. This demonstrates the
        **voice → emotion → distress signal** piece of the HealHub pipeline.
        """
    )

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="numpy",
            label="Record or upload a voice note",
        )

    submit_btn = gr.Button("Analyze", variant="primary")

    with gr.Row():
        transcript_output = gr.Textbox(label="Transcript", lines=4)
        emotion_output = gr.Textbox(label="Emotion Analysis", lines=4)

    # Fire on button click OR as soon as audio is ready (upload/record done)
    # — the .change() handler fixes the "No audio received" issue on mobile
    # where the button click can fire before Gradio registers the audio value.
    for event in [submit_btn.click, audio_input.change]:
        event(
            fn=analyze_voice,
            inputs=audio_input,
            outputs=[transcript_output, emotion_output],
            show_progress="minimal",
        )

    gr.Markdown(
        """
        ---
        *Note: this is a proof-of-concept using general-purpose pretrained
        models. For production use, models would be fine-tuned on
        consented, regional-language distress speech data.*
        """
    )

if __name__ == "__main__":
    demo.launch()
