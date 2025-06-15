from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import os
import base64
import io
import re
import hashlib
from typing import List, Dict, Optional
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

load_dotenv()

from sarvamai import SarvamAI
from pydub import AudioSegment
import google.generativeai as genai

app = FastAPI(title="TTS API")

# Load API keys from environment
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

# Initialize clients
client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Initialize AWS Polly client
try:
    aws_session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    polly_client = aws_session.client('polly', region_name=AWS_REGION)
except Exception as e:
    print(f"Warning: AWS Polly not configured properly: {e}")
    polly_client = None

# Safe limits for payload sizes
TTS_MAX_CHARS = 250
"""I used 250 chars as a safe limit for TTS requests as after testing it works well for most sentences, even though Sarvam's Bulbul V2 supports 500 chars, but it can lead to issues with longer sentences."""
POLLY_MAX_CHARS = 3000
TRANSLATE_MAX_CHARS = 1000

# In-memory cache for translations and audio
translation_cache: Dict[str, str] = {}
audio_cache: Dict[str, bytes] = {}

# Indian language codes supported by Sarvam
INDIAN_LANGUAGES = {
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN",
    "pa-IN", "ta-IN", "te-IN", "as-IN", "brx-IN", "doi-IN", "kok-IN", "ks-IN",
    "mai-IN", "mni-IN", "ne-IN", "sa-IN", "sat-IN", "sd-IN", "ur-IN"
}

# AWS Polly supported languages and voices
POLLY_LANGUAGES = {
    "en-US": {"voices": ["Joanna", "Matthew", "Ivy", "Justin", "Kendra", "Kimberly", "Salli", "Joey", "Ruth", "Stephen"]},
    "en-GB": {"voices": ["Amy", "Emma", "Brian", "Arthur"]},
    "en-AU": {"voices": ["Nicole", "Russell", "Olivia"]},
    "es-ES": {"voices": ["Conchita", "Lucia", "Enrique"]},
    "es-MX": {"voices": ["Mia"]},
    "es-US": {"voices": ["Penelope", "Miguel", "Lupe"]},
    "fr-FR": {"voices": ["Celine", "Lea", "Mathieu"]},
    "fr-CA": {"voices": ["Chantal", "Gabrielle"]},
    "de-DE": {"voices": ["Marlene", "Vicki", "Hans", "Daniel"]},
    "it-IT": {"voices": ["Carla", "Bianca", "Giorgio"]},
    "pt-BR": {"voices": ["Vitoria", "Camila", "Ricardo"]},
    "pt-PT": {"voices": ["Ines", "Cristiano"]},
    "ja-JP": {"voices": ["Mizuki", "Takumi"]},
    "ko-KR": {"voices": ["Seoyeon"]},
    "zh-CN": {"voices": ["Zhiyu"]},
    "ar-AE": {"voices": ["Zeina"]},
    "hi-IN": {"voices": ["Aditi", "Raveena"]},
    "tr-TR": {"voices": ["Filiz"]},
    "ru-RU": {"voices": ["Tatyana", "Maxim"]},
    "nl-NL": {"voices": ["Lotte", "Ruben"]},
    "sv-SE": {"voices": ["Astrid"]},
    "da-DK": {"voices": ["Naja", "Mads"]},
    "no-NO": {"voices": ["Liv"]},
    "pl-PL": {"voices": ["Ewa", "Maja", "Jacek", "Jan"]},
    "ro-RO": {"voices": ["Carmen"]},
    "ca-ES": {"voices": ["Arlet"]},
    "is-IS": {"voices": ["Dora", "Karl"]},
    "cy-GB": {"voices": ["Gwyneth"]},
    "cmn-CN": {"voices": ["Zhiyu"]},
    "yue-CN": {"voices": ["Hiujin"]},
    "nb-NO": {"voices": ["Liv"]},
}

class TTSRequest(BaseModel):
    input_text: str
    source_lang: str = "auto"
    target_lang: str = "hi-IN"
    speaker: str = "abhilash"  # For Sarvam, voice for Polly
    voice: Optional[str] = None  # Alternative field for Polly voice
    pitch: float = 0.0
    pace: float = 1.0
    loudness: float = 1.0
    speech_sample_rate: int = 24000
    enable_preprocessing: bool = False
    # Polly-specific parameters
    engine: str = "standard"  # standard, neural, long-form, generative
    output_format: str = "mp3"  # mp3, ogg_vorbis, pcm

def generate_cache_key(text: str, **kwargs) -> str:
    """Generate a hash key for caching"""
    key_data = f"{text}_{str(sorted(kwargs.items()))}"
    return hashlib.md5(key_data.encode()).hexdigest()

def is_indian_language(lang_code: str) -> bool:
    """Check if language code is for Indian languages"""
    return lang_code in INDIAN_LANGUAGES

def get_polly_voice(lang_code: str, requested_voice: Optional[str] = None) -> str:
    """Get appropriate Polly voice for language"""
    if lang_code not in POLLY_LANGUAGES:
        # Default to English US if language not supported
        lang_code = "en-US"
    
    available_voices = POLLY_LANGUAGES[lang_code]["voices"]
    
    if requested_voice and requested_voice in available_voices:
        return requested_voice
    
    # Return first available voice as default
    return available_voices[0]

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

# Handles translation with caching
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    # Check cache first
    cache_key = generate_cache_key(text, source=source_lang, target=target_lang)
    if cache_key in translation_cache:
        print("Using cached translation")
        return translation_cache[cache_key]
    
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
                "bn-IN": "Bengali", "en-IN": "English", "gu-IN": "Gujarati", "hi-IN": "Hindi",
                "kn-IN": "Kannada", "ml-IN": "Malayalam", "mr-IN": "Marathi", "od-IN": "Odia",
                "pa-IN": "Punjabi", "ta-IN": "Tamil", "te-IN": "Telugu", "as-IN": "Assamese",
                "brx-IN": "Bodo", "doi-IN": "Dogri", "kok-IN": "Konkani", "ks-IN": "Kashmiri",
                "mai-IN": "Maithili", "mni-IN": "Manipuri (Meiteilon)", "ne-IN": "Nepali",
                "sa-IN": "Sanskrit", "sat-IN": "Santali", "sd-IN": "Sindhi", "ur-IN": "Urdu",
                # International languages
                "en-US": "English", "es-ES": "Spanish", "fr-FR": "French", "de-DE": "German",
                "it-IT": "Italian", "pt-BR": "Portuguese", "ja-JP": "Japanese", "ko-KR": "Korean",
                "zh-CN": "Chinese", "ar-AE": "Arabic", "tr-TR": "Turkish", "ru-RU": "Russian"
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

    result = " ".join(translated_chunks)
    
    # Cache the result
    translation_cache[cache_key] = result
    return result

# Synthesizes speech using Sarvam API
def synthesize_speech_sarvam(text: str, speaker: str, pitch: float, target_language_code: str, 
                            pace: float, loudness: float, sample_rate: int, enable_preprocessing: bool) -> bytes:

    chunks = chunk_text(text, TTS_MAX_CHARS)
    print(f"Total chunks for Sarvam: {len(chunks)}")

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
            print(f"Sarvam chunk {i+1} processed — {len(segment)} ms")

        except Exception as e:
            print(f"Sarvam chunk {i+1} failed: {e}")
            audio_segments.append(AudioSegment.silent(duration=1000))

    if not audio_segments:
        raise Exception("Failed to generate audio with Sarvam")

    final_audio = audio_segments[0]
    for seg in audio_segments[1:]:
        final_audio += AudioSegment.silent(duration=200)
        final_audio += seg

    out_buffer = io.BytesIO()
    final_audio.export(out_buffer, format="wav")
    return out_buffer.getvalue()

# Synthesizes speech using AWS Polly
def synthesize_speech_polly(text: str, voice_id: str, language_code: str, 
                           engine: str = "standard", output_format: str = "mp3") -> bytes:
    
    if not polly_client:
        raise Exception("AWS Polly client not configured")
    
    chunks = chunk_text(text, POLLY_MAX_CHARS)
    print(f"Total chunks for Polly: {len(chunks)}")
    
    if len(chunks) == 1:
        try:
            response = polly_client.synthesize_speech(
                Engine=engine,
                OutputFormat=output_format,
                Text=chunks[0],
                VoiceId=voice_id,
                LanguageCode=language_code
            )
            
            audio_stream = response['AudioStream']
            return audio_stream.read()
            
        except ClientError as e:
            raise Exception(f"Polly synthesis failed: {e}")
    
    # Handle multiple chunks
    audio_segments = []
    
    for i, chunk in enumerate(chunks):
        try:
            response = polly_client.synthesize_speech(
                Engine=engine,
                OutputFormat=output_format,
                Text=chunk,
                VoiceId=voice_id,
                LanguageCode=language_code
            )
            
            audio_data = response['AudioStream'].read()
            
            try:
                if output_format == "mp3":
                    segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
                else:
                    segment = AudioSegment.from_wav(io.BytesIO(audio_data))
            except Exception:
                segment = AudioSegment.from_raw(
                    io.BytesIO(audio_data),
                    sample_width=2,
                    frame_rate=22050,
                    channels=1
                )
            
            audio_segments.append(segment)
            print(f"Polly chunk {i+1} processed — {len(segment)} ms")
            
        except Exception as e:
            print(f"Polly chunk {i+1} failed: {e}")
            audio_segments.append(AudioSegment.silent(duration=1000))
    
    if not audio_segments:
        raise Exception("Failed to generate audio with Polly")
    
    final_audio = audio_segments[0]
    for seg in audio_segments[1:]:
        final_audio += AudioSegment.silent(duration=200)
        final_audio += seg
    
    out_buffer = io.BytesIO()
    if output_format == "mp3":
        final_audio.export(out_buffer, format="mp3")
    else:
        final_audio.export(out_buffer, format="wav")
    
    return out_buffer.getvalue()

@app.get("/")
def root():
    return {"message": "Enhanced Indian TTS API with International Language Support", "status": "running"}

@app.post("/tts")
def text_to_speech(request: TTSRequest):
    try:
        # Check cache first
        cache_key = generate_cache_key(
            request.input_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            speaker=request.speaker,
            voice=request.voice,
            pitch=request.pitch,
            pace=request.pace,
            loudness=request.loudness,
            engine=request.engine,
            output_format=request.output_format
        )
        
        if cache_key in audio_cache:
            print("Using cached audio")
            media_type = "audio/mp3" if request.output_format == "mp3" else "audio/wav"
            return Response(content=audio_cache[cache_key], media_type=media_type)
        
        # Auto-translate if source and target languages differ
        text_to_speak = request.input_text
        if request.source_lang == "auto" or request.source_lang != request.target_lang.split('-')[0]:
            text_to_speak = translate_text(
                request.input_text, 
                request.source_lang, 
                request.target_lang
            )

        # Determine which TTS service to use
        if is_indian_language(request.target_lang):
            print(f"Using Sarvam API for Indian language: {request.target_lang}")
            audio_data = synthesize_speech_sarvam(
                text=text_to_speak,
                speaker=request.speaker,
                target_language_code=request.target_lang,
                pitch=request.pitch,
                pace=request.pace,
                loudness=request.loudness,
                sample_rate=request.speech_sample_rate,
                enable_preprocessing=request.enable_preprocessing
            )
            media_type = "audio/wav"
        else:
            print(f"Using AWS Polly for international language: {request.target_lang}")
            voice_id = get_polly_voice(request.target_lang, request.voice or request.speaker)
            audio_data = synthesize_speech_polly(
                text=text_to_speak,
                voice_id=voice_id,
                language_code=request.target_lang,
                engine=request.engine,
                output_format=request.output_format
            )
            media_type = "audio/mp3" if request.output_format == "mp3" else "audio/wav"

        # Cache the result
        audio_cache[cache_key] = audio_data

        return Response(
            content=audio_data,
            media_type=media_type
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/speakers")
def get_speakers():
    return {
        "sarvam": {
            "female": ["anushka", "manisha", "vidya", "arya"],
            "male": ["abhilash", "karun", "hitesh"]
        },
        "polly": POLLY_LANGUAGES
    }

@app.get("/languages")
def get_languages():
    return {
        "indian": list(INDIAN_LANGUAGES),
        "international": list(POLLY_LANGUAGES.keys())
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "sarvam_configured": SARVAM_API_KEY is not None,
        "gemini_configured": GEMINI_API_KEY is not None,
        "aws_polly_configured": polly_client is not None,
        "cache_stats": {
            "translation_cache_size": len(translation_cache),
            "audio_cache_size": len(audio_cache)
        }
    }

@app.delete("/cache")
def clear_cache():
    """Clear translation and audio caches"""
    translation_cache.clear()
    audio_cache.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/cache/stats")
def cache_stats():
    """Get cache statistics"""
    return {
        "translation_cache": {
            "size": len(translation_cache),
            "keys": list(translation_cache.keys())[:10] 
        },
        "audio_cache": {
            "size": len(audio_cache),
            "keys": list(audio_cache.keys())[:10]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)