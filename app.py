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


@(spaces.GPU if HF_SPACES else lambda f: f)
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
# 2. Responsive Custom CSS
# ---------------------------------------------------------------------------
custom_css = """
/* Reset & Base bounds */
body, .gradio-container {
    background-color: #f8fafc !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 auto !important;
}

/* Centered Main Application Container */
#main-container {
    max-width: 1100px !important;
    width: 92% !important;
    margin: 2rem auto !important;
    padding: 2.5rem 2rem !important;
    background: #ffffff !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05), 0 1px 4px rgba(0, 0, 0, 0.03) !important;
    border: 1px solid #e2e8f0 !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 1.75rem !important;
}

/* Header & Logo styling */
.header-box {
    text-align: center !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    margin-bottom: 0.5rem !important;
}

.logo-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 1rem;
}

.logo-svg {
    width: 120px !important;
    max-width: 100% !important;
    height: auto !important;
    display: block !important;
    margin: 0 auto !important;
    filter: drop-shadow(0 4px 12px rgba(251, 133, 0, 0.25));
}

.header-title {
    color: #0f172a !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    text-align: center !important;
}

.header-subtitle {
    color: #475569 !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    max-width: 760px !important;
    margin: 0 auto !important;
    text-align: center !important;
}

/* Audio Recording Input Component Fixes */
.audio-card {
    width: 100% !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    box-sizing: border-box !important;
}

div[data-testid="audio"] {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important;
    background: transparent !important;
    border: none !important;
}

div[data-testid="audio"] svg {
    max-width: 100% !important;
}

/* Analyze Button Styling */
.analyze-btn {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    color: #ffffff !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 0.85rem 2.5rem !important;
    border-radius: 10px !important;
    border: none !important;
    cursor: pointer !important;
    box-shadow: 0 4px 14px rgba(249, 115, 22, 0.3) !important;
    transition: all 0.2s ease-in-out !important;
    margin: 0.5rem auto !important;
    display: inline-block !important;
}

.analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(249, 115, 22, 0.4) !important;
}

/* Results Row & Output Textboxes */
.results-row {
    display: flex !important;
    gap: 1.5rem !important;
    width: 100% !important;
}

.output-box textarea {
    font-size: 0.975rem !important;
    line-height: 1.5 !important;
    color: #1e293b !important;
    background-color: #f8fafc !important;
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
}

.output-box label span {
    font-weight: 600 !important;
    color: #334155 !important;
    font-size: 0.95rem !important;
}

/* Footer Section */
.footer-text {
    text-align: center !important;
    color: #64748b !important;
    font-size: 0.875rem !important;
    border-top: 1px solid #f1f5f9 !important;
    padding-top: 1.25rem !important;
    margin-top: 0.5rem !important;
}

/* Responsive Breakpoints */
@media (max-width: 1023px) {
    #main-container {
        width: 95% !important;
        margin: 1.5rem auto !important;
        padding: 1.75rem 1.25rem !important;
    }
    .logo-svg {
        width: 100px !important;
    }
    .header-title {
        font-size: 1.6rem !important;
    }
}

@media (max-width: 767px) {
    #main-container {
        width: 100% !important;
        margin: 0 auto !important;
        padding: 1.25rem 0.85rem !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        border: none !important;
    }
    .logo-svg {
        width: 85px !important;
    }
    .header-title {
        font-size: 1.4rem !important;
    }
    .results-row {
        flex-direction: column !important;
        gap: 1rem !important;
    }
    .analyze-btn {
        width: 100% !important;
    }
}
"""

# ---------------------------------------------------------------------------
# 3. Build UI with gr.Blocks + responsive layout wrapper
# ---------------------------------------------------------------------------
with gr.Blocks(title="HealHub - Voice Emotion Detection", css=custom_css) as demo:
    with gr.Column(elem_id="main-container"):
        # Header with Logo & Description
        with gr.Column(elem_classes=["header-box"]):
            gr.HTML("""
                <div class="logo-wrapper">
                    <svg class="logo-svg" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M50 10 L85 45 L50 80 L15 45 Z" stroke="url(#orange-grad)" stroke-width="8" fill="none" stroke-linejoin="round" />
                        <path d="M50 25 L75 50 L50 75 L25 50 Z" stroke="url(#orange-grad)" stroke-width="8" fill="none" stroke-linejoin="round" />
                        <defs>
                            <linearGradient id="orange-grad" x1="0" y1="0" x2="100" y2="100">
                                <stop offset="0%" stop-color="#FFB703" />
                                <stop offset="100%" stop-color="#FB8500" />
                            </linearGradient>
                        </defs>
                    </svg>
                </div>
            """)
            gr.Markdown(
                """
                <div class="header-title">HealHub — Voice Emotion Detection</div>
                <div class="header-subtitle">
                    Prototype for <strong>SIH26094</strong>. Record your voice or upload a voice note (e.g. exported from WhatsApp) to see a transcript and an emotional tone reading.
                </div>
                """
            )

        # Audio Input Section
        with gr.Row(elem_classes=["audio-card"]):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Record or upload a voice note",
            )

        # Action Button
        with gr.Row():
            submit_btn = gr.Button("Analyze Voice Note", variant="primary", elem_classes=["analyze-btn"])

        # Results Side-by-Side (Desktop/Tablet) or Column (Mobile)
        with gr.Row(elem_classes=["results-row"]):
            transcript_output = gr.Textbox(
                label="Speech Transcript",
                lines=4,
                placeholder="Spoken words will appear here...",
                interactive=False,
                elem_classes=["output-box"]
            )
            emotion_output = gr.Textbox(
                label="Emotional State Analysis",
                lines=4,
                placeholder="Emotion reading and guidance will appear here...",
                interactive=False,
                elem_classes=["output-box"]
            )

        # Event Binding
        for event in [submit_btn.click, audio_input.change]:
            event(
                fn=analyze_voice,
                inputs=audio_input,
                outputs=[transcript_output, emotion_output],
                show_progress="minimal",
            )

        # Footer / Disclaimer
        with gr.Column(elem_classes=["footer-text"]):
            gr.Markdown(
                """
                *Note: This is a proof-of-concept using pretrained speech models (`openai/whisper-base` and `wav2vec2-lg-xlsr-en`). For production use, models are fine-tuned on consented regional speech datasets.*
                """
            )

if __name__ == "__main__":
    demo.launch()
