
# 🇮🇳 Indian Text-to-Speech (TTS) API

**FastAPI-based multilingual TTS service for Indian languages** powered by [Sarvam AI](https://sarvam.ai/) and backed up with Google Gemini for translation fallback.

---

## 🔊 Overview

This API provides a **Text-to-Speech (TTS)** solution for Indian languages, allowing users to:
- Translate text across 20+ Indian languages.
- Generate realistic speech with customizable pitch, pace, and loudness.
- Automatically handle long texts with chunking.
- Failover translation using Google Gemini if Sarvam fails.

---

## 🚀 Features

- ✅ Supports **20+ Indian languages**
- ✅ Choose from **7 high-quality speakers**
- ✅ Smart chunking for long text synthesis
- ✅ Gemini fallback for translation
- ✅ Built-in `/health`, `/languages`, and `/speakers` endpoints
- ✅ FastAPI-powered with clean design

---

## 📦 Requirements

- Python 3.9+
- Sarvam AI API Key
- Gemini API Key (for fallback)
- FFmpeg installed (for audio processing)

---

## 🔧 Installation

```bash
git clone https://github.com/yourusername/indian-tts-api.git
cd indian-tts-api
pip install -r requirements.txt
```

> ✅ Make sure to set up your `.env` file:

```env
SARVAM_API_KEY=your_sarvam_api_key
GEMINI_API_KEY=your_gemini_api_key
```

---

## ▶️ Run the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 📂 Endpoints

### 🔹 `GET /`

Health info and welcome message.

---

### 🔹 `POST /tts`

Generate speech from text.

**Request Body:**

```json
{
  "input_text": "Hello, how are you?",
  "source_lang": "auto",
  "target_lang": "hi-IN",
  "speaker": "abhilash",
  "pitch": 0.0,
  "pace": 1.0,
  "loudness": 1.0,
  "speech_sample_rate": 24000,
  "enable_preprocessing": false
}
```

**Returns:** WAV audio stream.

---

### 🔹 `GET /speakers`

Returns available speaker names categorized by gender.

```json
{
  "female": ["anushka", "manisha", "vidya", "arya"],
  "male": ["abhilash", "karun", "hitesh"]
}
```

---

### 🔹 `GET /languages`

Returns supported language codes (e.g., `hi-IN`, `ta-IN`, `bn-IN`).

---

### 🔹 `GET /health`

Health check and config validation for API keys.

---

## 💡 Example Use (Python)

```python
import requests

url = "http://localhost:8000/tts"
payload = {
    "input_text": "Good morning!",
    "source_lang": "auto",
    "target_lang": "hi-IN",
    "speaker": "abhilash"
}

response = requests.post(url, json=payload)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

---

## 🧠 Tech Stack

- **FastAPI** – Lightning-fast web framework
- **Sarvam AI Bulbul v2** – TTS Engine
- **Google Gemini 2.0 Flash** – Fallback translator
- **Pydub** – Audio manipulation
- **Pydantic** – Request validation
- **Uvicorn** – ASGI server

---



## ❤️ Acknowledgments

- [Sarvam AI](https://sarvam.ai)
- [Google Gemini](https://ai.google.dev)
- [FastAPI](https://fastapi.tiangolo.com)
