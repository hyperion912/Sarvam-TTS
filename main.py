from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import os
import base64
import io
import re
from typing import List
from dotenv import load_dotenv

load_dotenv()

from sarvamai import SarvamAI
from pydub import AudioSegment
import google.generativeai as genai

app = FastAPI(title="TTS API")

# Load API keys from environment
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Safe limits for payload sizes
TTS_MAX_CHARS = 250 
"""I used 250 chars as a safe limit for TTS requests as after testing it works well for most sentences, even though Sarvam's Bulbul V2 supports 500 chars, but it can lead to issues with longer sentences."""
TRANSLATE_MAX_CHARS = 1000

class TTSRequest(BaseModel):
    input_text: str
    source_lang: str = "auto"
    target_lang: str = "hi-IN"
    speaker: str = "abhilash"
    pitch: float = 0.0
    pace: float = 1.0
    loudness: float = 1.0
    speech_sample_rate: int = 24000
    enable_preprocessing: bool = False

# Utility to break long text at sentence boundaries
def chunk_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    
    sentences = re.split(r'(?<=[.!?।])\s+|\n\s*\n', text.strip())
    chunks, current_chunk = [], ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk + " " + sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Sentence too long — break at word level
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk + " " + word) <= max_chars:
                        temp_chunk += " " + word if temp_chunk else word
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                if temp_chunk:
                    current_chunk = temp_chunk
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Final safety split
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(chunk), max_chars):
                final_chunks.append(chunk[i:i + max_chars])
    return final_chunks

# Handles translation, tries Sarvam first, falls back to Gemini if needed
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    target_code = target_lang
    chunks = chunk_text(text, TRANSLATE_MAX_CHARS)
    translated_chunks = []

    for chunk in chunks:
        try:
            response = client.text.translate(
                input=chunk,
                source_language_code=source_lang,
                target_language_code=target_code,
                model="mayura:v1",
                mode="modern-colloquial",
                enable_preprocessing=True
            )
            translated_chunks.append(response.translated_text)
        except Exception as e:
            print(f"Sarvam failed. Trying Gemini: {e}")
            try:
                if GEMINI_API_KEY is None:
                    raise Exception("Missing Gemini key")
                model = genai.GenerativeModel('gemini-2.0-flash-exp')

                lang_names = {
                "bn-IN": "Bengali",
                "en-IN": "English",
                "gu-IN": "Gujarati",
                "hi-IN": "Hindi",
                "kn-IN": "Kannada",
                "ml-IN": "Malayalam",
                "mr-IN": "Marathi",
                "od-IN": "Odia",
                "pa-IN": "Punjabi",
                "ta-IN": "Tamil",
                "te-IN": "Telugu",
                "as-IN": "Assamese",
                "brx-IN": "Bodo",
                "doi-IN": "Dogri",
                "kok-IN": "Konkani",
                "ks-IN": "Kashmiri",
                "mai-IN": "Maithili",
                "mni-IN": "Manipuri (Meiteilon)",
                "ne-IN": "Nepali",
                "sa-IN": "Sanskrit",
                "sat-IN": "Santali",
                "sd-IN": "Sindhi",
                "ur-IN": "Urdu"
                }

                source_name = lang_names.get(source_lang, source_lang)
                target_name = lang_names.get(target_code, target_code)

                prompt = f"""Translate this text from {source_name} to {target_name}. Just the translated output:
{chunk}"""

                response = model.generate_content(prompt)
                translated_chunks.append(response.text.strip())

            except Exception as fallback_err:
                print(f"Gemini also failed: {fallback_err}")
                translated_chunks.append(chunk)

    return " ".join(translated_chunks)

# Synthesizes speech from text, returns WAV byte stream
def synthesize_speech(text: str, speaker: str, pitch: float, target_language_code: str, 
                      pace: float, loudness: float, sample_rate: int, enable_preprocessing: bool) -> bytes:

    chunks = chunk_text(text, TTS_MAX_CHARS)
    print(f"Total chunks: {len(chunks)}")

    if len(chunks) == 1:
        audio = client.text_to_speech.convert(
            text=chunks[0],
            model="bulbul:v2",
            speaker=speaker.lower(),
            pitch=pitch,
            target_language_code=target_language_code,
            pace=pace,
            loudness=loudness,
            speech_sample_rate=sample_rate,
            enable_preprocessing=enable_preprocessing
        )
        return base64.b64decode("".join(audio.audios))
    
    # Handle multiple chunks with merging
    audio_segments = []

    for i, chunk in enumerate(chunks):
        try:
            audio = client.text_to_speech.convert(
                text=chunk,
                model="bulbul:v2",
                speaker=speaker.lower(),
                pitch=pitch,
                target_language_code=target_language_code,
                pace=pace,
                loudness=loudness,
                speech_sample_rate=sample_rate,
                enable_preprocessing=enable_preprocessing
            )

            audio_data = base64.b64decode("".join(audio.audios))

            try:
                segment = AudioSegment.from_wav(io.BytesIO(audio_data))
            except Exception:
                try:
                    segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
                except Exception:
                    segment = AudioSegment.from_raw(
                        io.BytesIO(audio_data),
                        sample_width=2,
                        frame_rate=sample_rate,
                        channels=1
                    )

            audio_segments.append(segment)
            print(f"Chunk {i+1} processed — {len(segment)} ms")

        except Exception as e:
            print(f"Chunk {i+1} failed: {e}")
            audio_segments.append(AudioSegment.silent(duration=1000))

    if not audio_segments:
        raise Exception("Failed to generate audio")

    final_audio = audio_segments[0]
    for seg in audio_segments[1:]:
        final_audio += AudioSegment.silent(duration=200)
        final_audio += seg

    out_buffer = io.BytesIO()
    final_audio.export(out_buffer, format="wav")
    return out_buffer.getvalue()

@app.get("/")
def root():
    return {"message": "Indian TTS API", "status": "running"}

@app.post("/tts")
def text_to_speech(request: TTSRequest):
    try:
        # Auto-translate if source and target languages differ
        text_to_speak = request.input_text
        if request.source_lang == "auto" or request.source_lang != request.target_lang.split('-')[0]:
            text_to_speak = translate_text(
                request.input_text, 
                request.source_lang, 
                request.target_lang
            )

        audio_data = synthesize_speech(
            text=text_to_speak,
            speaker=request.speaker,
            target_language_code=request.target_lang,
            pitch=request.pitch,
            pace=request.pace,
            loudness=request.loudness,
            sample_rate=request.speech_sample_rate,
            enable_preprocessing=request.enable_preprocessing
        )

        return Response(
            content=audio_data,
            media_type="audio/wav"
        )


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/speakers")
def get_speakers():
    return {
        "female": ["anushka", "manisha", "vidya", "arya"],
        "male": ["abhilash", "karun", "hitesh"]
    }

@app.get("/languages")
def get_languages():
    return [
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN",
    "pa-IN", "ta-IN", "te-IN", "as-IN", "brx-IN", "doi-IN", "kok-IN", "ks-IN",
    "mai-IN", "mni-IN", "ne-IN", "sa-IN", "sat-IN", "sd-IN", "ur-IN"
]

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "sarvam_configured": SARVAM_API_KEY is not None,
        "gemini_configured": GEMINI_API_KEY is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
