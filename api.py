import os
import json
import tempfile
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends, Request, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import soundfile as sf
import uvicorn
from kokoro_onnx import Kokoro
from misaki import espeak
from misaki.espeak import EspeakG2P

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="TTS API", description="Text-to-Speech API using Kokoro")

# Settings for temp files and model paths
class Settings:
    temp_file_dir = os.path.join(tempfile.gettempdir(), "tts_api")
    kokoro_model_path = "kokoro-v1.0.onnx"
    voices_bin_path = "voices-v1.0.bin"

settings = Settings()

# Create temp directory if it doesn't exist
os.makedirs(settings.temp_file_dir, exist_ok=True)

# Models for request validation
class CaptionedSpeechRequest(BaseModel):
    input: str
    voice: str
    response_format: str = "wav"
    speed: float = 1.0
    lang_code: Optional[str] = None

# Helper class for temporary file storage
class TempFileWriter:
    def __init__(self, ext):
        self.ext = ext
        self.temp_dir = settings.temp_file_dir
        self.file = None
        self.path = None
        self.download_path = None

    async def __aenter__(self):
        os.makedirs(self.temp_dir, exist_ok=True)
        fd, self.path = tempfile.mkstemp(suffix=f".{self.ext}", dir=self.temp_dir)
        self.file = os.fdopen(fd, "wb")
        self.download_path = self.path
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    async def write(self, data):
        if self.file:
            self.file.write(data)

    async def finalize(self):
        if self.file:
            self.file.flush()

# Helper class for audio normalization
class AudioNormalizer:
    async def normalize(self, audio):
        # Simple passthrough implementation, can be extended with actual normalization
        return audio

# Helper class for streaming audio writing
class StreamingAudioWriter:
    def __init__(self, format="wav", sample_rate=24000, channels=1):
        self.format = format
        self.sample_rate = sample_rate
        self.channels = channels

    def write_chunk(self, audio_chunk=None, finalize=False):
        # For simplicity, we'll just convert the entire audio to bytes
        # In a real implementation, this would handle streaming properly
        if audio_chunk is not None:
            with tempfile.NamedTemporaryFile(suffix=f".{self.format}") as tmp:
                sf.write(tmp.name, audio_chunk, self.sample_rate)
                tmp.flush()
                with open(tmp.name, "rb") as f:
                    return f.read()
        return b""

# Helper function to find files in directories
async def _find_file(filename: str, search_paths: List[str]):
    for path in search_paths:
        file_path = os.path.join(path, filename)
        if os.path.exists(file_path):
            return file_path
    raise FileNotFoundError(f"File {filename} not found in any of the search paths")

# TTS Service for audio generation
class TTSService:
    def __init__(self):
        try:
            self.kokoro = Kokoro(settings.kokoro_model_path, settings.voices_bin_path)
            # Use espeak for both English and Spanish to avoid spaCy download issues
            self.g2p_es = EspeakG2P(language='es')
            self.g2p_en = EspeakG2P(language='en-us')
            logger.info("TTS Service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TTS Service: {e}")
            raise RuntimeError(f"Failed to initialize TTS Service: {e}")

    async def _get_voice_path(self, voice_name):
        # In this implementation, voice_name is directly used with Kokoro
        # This would need to be extended for a more complex voice system
        return voice_name, voice_name

    async def generate_speech(self, text, voice, speed=1.0, lang_code=None):
        # Determine language from voice or lang_code
        language = lang_code or voice[0].lower()
        
        logger.info(f"Generating speech with language code: {language}, voice: {voice}")
        
        # Select appropriate g2p based on language
        if language == 'e':  # English
            phonemes, _ = self.g2p_en(text)
        elif language == 's':  # Spanish
            phonemes, _ = self.g2p_es(text)
        else:
            # Default to English if language not recognized
            logger.warning(f"Unknown language code: {language}, defaulting to English")
            phonemes, _ = self.g2p_en(text)
        
        # Generate audio using Kokoro
        samples, sample_rate = self.kokoro.create(
            phonemes, voice=voice, speed=speed, is_phonemes=True
        )
        
        return samples, sample_rate

# Helper for dependency injection
def get_tts_service():
    return TTSService()

# Simplified smart text splitter
async def smart_split(text):
    # For simplicity, we'll just yield the entire text as one chunk
    # A real implementation would split text intelligently
    yield text, []

# API Endpoints
@app.get("/dev/timestamps/{filename}")
async def get_timestamps(filename: str):
    """Download timestamps from temp storage"""
    try:
        # Search for file in temp directory
        file_path = await _find_file(
            filename=filename, search_paths=[settings.temp_file_dir]
        )

        return FileResponse(
            file_path,
            media_type="application/json",
            filename=filename,
            headers={
                "Cache-Control": "no-cache",
                "Content-Disposition": f"attachment; filename={filename}",
            },
        )

    except Exception as e:
        logger.error(f"Error serving timestamps file {filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to serve timestamps file",
                "type": "server_error",
            },
        )

@app.post("/dev/captioned_speech")
async def create_captioned_speech(
    request: CaptionedSpeechRequest,
    client_request: Request,
    x_raw_response: str = Header(None, alias="x-raw-response"),
    tts_service: TTSService = Depends(get_tts_service),
):
    """Generate audio with word-level timestamps"""
    try:
        # Set content type based on format
        content_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }.get(request.response_format, f"audio/{request.response_format}")

        # Create streaming audio writer and normalizer
        writer = StreamingAudioWriter(
            format=request.response_format, sample_rate=24000, channels=1
        )
        normalizer = AudioNormalizer()

        # Get voice path
        voice_name, voice_path = await tts_service._get_voice_path(request.voice)

        # Use provided lang_code or determine from voice name
        pipeline_lang_code = request.lang_code if request.lang_code else request.voice[0].lower()
        logger.info(
            f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in text processing"
        )

        # Create temp file writer for timestamps
        temp_writer = TempFileWriter("json")
        await temp_writer.__aenter__()  # Initialize temp file
        # Get just the filename without the path
        timestamps_filename = Path(temp_writer.download_path).name

        # Initialize variables for timestamps
        word_timestamps = []
        current_offset = 0.0

        # For our simplified implementation, we'll generate timestamps based on word lengths
        # This is a very basic approximation
        text = request.input
        words = text.split()
        
        # Generate audio
        samples, sample_rate = await tts_service.generate_speech(
            text, request.voice, request.speed, request.lang_code
        )
        
        # Normalize audio
        normalized_audio = await normalizer.normalize(samples)
        
        # Generate simple word timestamps (approximation)
        total_duration = len(samples) / sample_rate
        avg_word_duration = total_duration / len(words) if words else 0
        
        for i, word in enumerate(words):
            start_time = i * avg_word_duration
            end_time = (i + 1) * avg_word_duration
            
            word_timestamps.append({
                "word": word,
                "start_time": start_time,
                "end_time": end_time,
            })
        
        # Write timestamps to temp file
        timestamps_json = json.dumps(word_timestamps)
        await temp_writer.write(timestamps_json.encode())
        await temp_writer.finalize()
        
        # Return audio file with timestamp reference
        def generate_audio_stream():
            with tempfile.NamedTemporaryFile(suffix=f".{request.response_format}") as tmp:
                sf.write(tmp.name, normalized_audio, sample_rate)
                tmp.flush()
                with open(tmp.name, "rb") as f:
                    yield from f
                    
        return StreamingResponse(
            generate_audio_stream(),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "X-Timestamps-Path": timestamps_filename,
            },
        )

    except ValueError as e:
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error",
            },
        )
    except RuntimeError as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )
    except Exception as e:
        logger.error(f"Unexpected error in speech generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )

# Run the API
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8880, reload=True)