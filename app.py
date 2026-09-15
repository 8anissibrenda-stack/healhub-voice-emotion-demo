"""
HealHub - Voice Emotion Detection + Transcription Demo (Streamlit version)
---------------------------------------------------------------------------
A Hugging Face + Streamlit demo that takes a voice note (recorded live or
uploaded, e.g. exported from WhatsApp) and shows:
  1. A transcript of the speech (via Whisper)
  2. A simple, layman-friendly emotional-state reading (via a
     speech-emotion-recognition model)

This is a STANDALONE PROOF-OF-CONCEPT for the SIH26094 "HealHub" idea.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io

import librosa
import numpy as np
import streamlit as st
from transformers import pipeline

# ---------------------------------------------------------------------------
# 1. Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HealHub - Voice Emotion Detection",
    page_icon="🎙️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# 2. Load models once, cached across reruns/sessions
# ---------------------------------------------------------------------------
EMOTION_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
ASR_MODEL = "openai/whisper-base"
TARGET_SR = 16000


@st.cache_resource(show_spinner="Loading models... this may take a minute on first run")
def load_models():
    emotion_classifier = pipeline("audio-classification", model=EMOTION_MODEL)
    transcriber = pipeline("automatic-speech-recognition", model=ASR_MODEL)
    return emotion_classifier, transcriber


emotion_classifier, transcriber = load_models()

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


# ---------------------------------------------------------------------------
# 3. Audio processing + inference (same logic as the Gradio version)
# ---------------------------------------------------------------------------
def _prepare_audio(audio_bytes: bytes) -> np.ndarray:
    """Decode raw audio bytes (wav/mp3/m4a/ogg) into mono float32 audio
    resampled to 16kHz, as required by both models."""
    data, _ = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR, mono=True)
    return data


def analyze_voice(audio_bytes):
    """
    Takes raw audio bytes, returns (transcript_text, emotion_text).
    Wrapped defensively so a failure in one part doesn't take down the whole
    response.
    """
    if audio_bytes is None:
        return "No audio received. Please record or upload a voice note.", ""

    try:
        processed = _prepare_audio(audio_bytes)
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
# 4. Custom CSS (same look as the Gradio version, adapted for Streamlit)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container {
        max-width: 900px !important;
        padding-top: 2.5rem !important;
    }
    .header-title {
        color: #0f172a;
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .header-subtitle {
        color: #475569;
        font-size: 1rem;
        line-height: 1.6;
        max-width: 760px;
        margin: 0 auto 1.5rem auto;
        text-align: center;
    }
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.875rem;
        border-top: 1px solid #f1f5f9;
        padding-top: 1.25rem;
        margin-top: 1.5rem;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.7rem 1.5rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 14px rgba(249, 115, 22, 0.3);
        width: 100%;
    }
    div.stButton > button:hover {
        box-shadow: 0 6px 20px rgba(249, 115, 22, 0.4);
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 5. Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="logo-wrapper" style="display:flex; justify-content:center; margin-bottom:0.5rem;">
        <svg width="100" height="100" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
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
    <div class="header-title">HealHub — Voice Emotion Detection</div>
    <div class="header-subtitle">
        Prototype for <strong>SIH26094</strong>. Record your voice or upload a voice note
        (e.g. exported from WhatsApp) to see a transcript and an emotional tone reading.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 6. Audio input — record or upload
# ---------------------------------------------------------------------------
audio_bytes = None

tab_record, tab_upload = st.tabs(["🎙️ Record", "📁 Upload"])

with tab_record:
    recorded = st.audio_input("Record your voice note")
    if recorded is not None:
        audio_bytes = recorded.read()

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a voice note (wav, mp3, m4a, ogg)",
        type=["wav", "mp3", "m4a", "ogg"],
    )
    if uploaded is not None:
        audio_bytes = uploaded.read()

if audio_bytes is not None:
    st.audio(audio_bytes)

# ---------------------------------------------------------------------------
# 7. Analyze button + results
# ---------------------------------------------------------------------------
if st.button("Analyze Voice Note", type="primary"):
    if audio_bytes is None:
        st.warning("Please record or upload a voice note first.")
    else:
        with st.spinner("Analyzing your voice note..."):
            transcript, emotion_text = analyze_voice(audio_bytes)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Speech Transcript")
            st.text_area(
                "Transcript", transcript, height=160,
                disabled=True, label_visibility="collapsed",
            )
        with col2:
            st.subheader("Emotional State Analysis")
            st.text_area(
                "Emotion", emotion_text, height=160,
                disabled=True, label_visibility="collapsed",
            )

# ---------------------------------------------------------------------------
# 8. Footer / disclaimer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-text">
        Note: This is a proof-of-concept using pretrained speech models
        (<code>openai/whisper-base</code> and <code>wav2vec2-lg-xlsr-en</code>).
        For production use, models are fine-tuned on consented regional speech datasets.
    </div>
    """,
    unsafe_allow_html=True,
)