
# 🇮🇳 Indian Text-to-Speech (TTS) API

**FastAPI-based multilingual TTS service for Indian and International languages**, powered by [Sarvam AI](https://sarvam.ai/) for Indian languages, AWS Polly for international languages, and backed up with Google Gemini for translation fallback. It also features in-memory caching for translations and audio.

Deployed at(Polly integrated): https://tts-api-x738.onrender.com

---

## 🔊 Overview

This API provides a comprehensive **Text-to-Speech (TTS)** solution, allowing users to:
- Translate text across 20+ Indian languages and numerous international languages.
- Generate realistic speech using Sarvam AI for Indian languages and AWS Polly for international languages.
- Customize speech with pitch, pace, and loudness (Sarvam).
- Select different voices and engines (Polly).
- Automatically handle long texts with smart chunking.
- Utilize failover translation using Google Gemini if Sarvam's translation fails.
- Benefit from in-memory caching for frequently requested translations and audio.

---

## 🚀 Features

- ✅ Supports **20+ Indian languages** (via Sarvam AI)
- ✅ Supports a wide range of **international languages** (via AWS Polly)
- ✅ Choose from **7 high-quality Sarvam speakers** and numerous **AWS Polly voices**
- ✅ Smart chunking for long text synthesis
- ✅ Gemini fallback for translation
- ✅ In-memory caching for translations and audio
- ✅ Built-in `/health`, `/languages`, `/speakers`, `/cache` endpoints
- ✅ FastAPI-powered with clean design

---

## 📦 Requirements

- Python 3.9+
- Sarvam AI API Key (for Indian languages)
- Gemini API Key (for translation fallback)
- AWS Access Key ID & Secret Access Key (for international languages via Polly)
- FFmpeg installed (for audio processing by Pydub)
- `boto3` library (for AWS Polly)

---

## 🔧 Installation

```bash
git clone https://github.com/hyperion912/Sarvam-TTS.git
cd Sarvam-TTS
pip install -r requirements.txt
```

> ✅ Make sure to set up your `.env` file:

```env
SARVAM_API_KEY=your_sarvam_api_key
GEMINI_API_KEY=your_gemini_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=your_aws_region # e.g., us-west-2
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
  "source_lang": "auto", // "auto" or specific language code e.g., "en-IN", "en-US"
  "target_lang": "hi-IN", // Target language for TTS, e.g., "hi-IN", "en-US", "es-ES"
  "speaker": "abhilash", // For Sarvam TTS. For Polly, this can be used if `voice` is not provided.
  "voice": null, // Optional: Specific voice for AWS Polly (e.g., "Joanna", "Matthew"). Overrides `speaker` for Polly.
  "pitch": 0.0, // For Sarvam TTS
  "pace": 1.0, // For Sarvam TTS
  "loudness": 1.0, // For Sarvam TTS
  "speech_sample_rate": 24000, // For Sarvam TTS
  "enable_preprocessing": false, // For Sarvam TTS
  // Polly-specific parameters (used if target_lang is not an Indian language)
  "engine": "standard", // Polly engine: "standard", "neural", "long-form", "generative"
  "output_format": "mp3" // Polly output: "mp3", "ogg_vorbis", "pcm"
}
```

**Returns:**
- WAV audio stream for Sarvam AI (Indian languages).
- MP3 (default), OGG Vorbis, or PCM audio stream for AWS Polly (international languages), based on `output_format`.

---

### 🔹 `GET /speakers`

Returns available speaker names for Sarvam and voice details for Polly.

```json
{
  "sarvam": {
    "female": ["anushka", "manisha", "vidya", "arya"],
    "male": ["abhilash", "karun", "hitesh"]
  },
  "polly": {
    "en-US": {"voices": ["Joanna", "Matthew", ...]},
    "en-GB": {"voices": ["Amy", "Emma", ...]},
    // ... other Polly supported languages and voices
  }
}
```

---

### 🔹 `GET /languages`

Returns supported language codes for Indian (Sarvam) and international (Polly) languages.

```json
{
  "indian": ["bn-IN", "en-IN", "gu-IN", ...],
  "international": ["en-US", "en-GB", "es-ES", ...]
}
```

---

### 🔹 `GET /health`

Health check, config validation for API keys, and cache statistics.

```json
{
  "status": "healthy",
  "sarvam_configured": true,
  "gemini_configured": true,
  "aws_polly_configured": true, // or false if AWS keys are not set
  "cache_stats": {
    "translation_cache_size": 0,
    "audio_cache_size": 0
  }
}
```

---

### 🔹 `DELETE /cache`

Clears the in-memory translation and audio caches.

**Returns:**

```json
{
  "message": "Cache cleared successfully"
}
```

---

### 🔹 `GET /cache/stats`

Returns statistics about the current cache usage.

**Returns:**

```json
{
  "translation_cache": {
    "size": 0,
    "keys": [] // Shows up to 10 keys
  },
  "audio_cache": {
    "size": 0,
    "keys": [] // Shows up to 10 keys
  }
}
```

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
- **Sarvam AI Bulbul v2** – TTS Engine (Indian Languages)
- **AWS Polly** – TTS Engine (International Languages)
- **Google Gemini 2.0 Flash** – Fallback translator
- **Pydub** – Audio manipulation
- **Pydantic** – Request validation
- **Uvicorn** – ASGI server
- **Boto3** – AWS SDK for Python

---
