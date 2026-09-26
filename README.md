# HealHub — Voice Emotion Detection & Speech Transcription Demo (SIH26094)

A lightweight, high-performance, responsive AI application prototype for **SIH26094 (HealHub)**.

The application allows users to record live audio or upload voice notes (WAV, MP3, M4A, OGG, WEBM) and receive:
1. **Speech Transcription** via OpenAI's Whisper model (`openai/whisper-base`).
2. **Emotional State Analysis** via Speech Emotion Recognition (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`).
3. **Layman-Friendly Interpretations & Support Guidance** mapped to detected emotions.

---

## 🏗 Architecture Overview

The codebase follows a clean, platform-independent **Responsive Web UI + FastAPI REST Server + Hugging Face AI Pipeline** architecture:

```text
User Device (Desktop / Laptop / Tablet / Mobile)
       │
       ▼  [HTTP REST / Web Audio API]
Responsive Web Frontend (HTML5, Vanilla CSS3, Modern ES6 JS)
       │
       ▼  [POST /api/analyze]
FastAPI Application Backend (Python Uvicorn ASGI Server)
       │
       ├── Audio Processing (SoundFile + SciPy 16kHz resampling)
       └── AI Inference (Transformers Local Pipelines or HF Inference API)
```

---

## 🚀 Local Quickstart

### 1. Prerequisites
- Python 3.10 or higher
- `pip`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
python app.py
```
Or directly using Uvicorn:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open your browser and navigate to: `http://localhost:8000`

---

## ⚙️ Environment Variables (Optional)

| Variable | Description | Default |
|---|---|---|
| `HOST` | Server bind host address | `0.0.0.0` |
| `PORT` | Server bind port | `8000` |
| `HF_TOKEN` | Optional Hugging Face API token for remote inference | `None` |
| `USE_HF_INFERENCE_API` | Set to `true` to use remote HF API instead of local model weights | `false` |
| `EMOTION_MODEL` | Hugging Face model identifier for emotion detection | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` |
| `ASR_MODEL` | Hugging Face model identifier for speech transcription | `openai/whisper-base` |

---

## ⚡ API Endpoints

- **`GET /`**: Serves the responsive single-page web UI.
- **`GET /api/health`**: Health check and current model configuration.
- **`POST /api/analyze`**: Accepts multipart audio file upload (`file`), returns transcription and emotion analysis JSON.

### Sample API Response (`POST /api/analyze`)
```json
{
  "success": true,
  "error": null,
  "transcript": "Hello, I am feeling a bit anxious about the upcoming interview.",
  "emotion": "You seem to be feeling anxious or scared right now 😟\n\nIf these feelings persist, please consider reaching out to someone you trust or a support helpline.",
  "emotion_label": "fear",
  "concern_level": "high",
  "support_message": "If these feelings persist, please consider reaching out to someone you trust or a support helpline.",
  "execution_time_seconds": 1.24
}
```
