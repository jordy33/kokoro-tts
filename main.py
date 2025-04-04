# --- START OF main.py ---

# ... (Keep all existing imports and setup code) ...
import os
import json
import tempfile
from pathlib import Path
from typing import List, Optional
import io
import asyncio # Ensure asyncio is imported

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Header, Depends, Request, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Gradio imports
import gradio as gr

# Audio and TTS imports
import soundfile as sf
from kokoro_onnx import Kokoro
from misaki import espeak
from misaki.espeak import EspeakG2P
import numpy as np # Needed for audio manipulation if you extend later

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Settings ---
# ... (Keep existing Settings class) ...
class Settings:
    temp_file_dir = os.path.join(tempfile.gettempdir(), "tts_api")
    kokoro_model_path = "kokoro-v1.0.onnx"
    voices_bin_path = "voices-v1.0.bin"

settings = Settings()
os.makedirs(settings.temp_file_dir, exist_ok=True)


# --- Helper Classes (from api.py) ---
# ... (Keep existing TempFileWriter and AudioNormalizer classes) ...
class TempFileWriter:
    def __init__(self, ext):
        self.ext = ext
        self.temp_dir = settings.temp_file_dir
        self.file = None
        self.path = None
        self.download_path = None # This will be the full path

    async def __aenter__(self):
        os.makedirs(self.temp_dir, exist_ok=True)
        fd, self.path = tempfile.mkstemp(suffix=f".{self.ext}", dir=self.temp_dir)
        self.file = os.fdopen(fd, "wb")
        self.download_path = self.path
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        # Optional: Add cleanup logic here if needed, but temp files are often handled by OS

    async def write(self, data):
        if self.file and not self.file.closed:
            self.file.write(data)

    async def finalize(self):
        if self.file and not self.file.closed:
            self.file.flush()


class AudioNormalizer:
    async def normalize(self, audio):
        # Simple passthrough implementation
        return audio


# --- TTS Service Initialization ---
# ... (Keep existing TTSService class and instantiation) ...
class TTSService:
    def __init__(self):
        try:
            # Ensure model files exist
            if not os.path.exists(settings.kokoro_model_path):
                 raise FileNotFoundError(f"Kokoro model not found at {settings.kokoro_model_path}")
            if not os.path.exists(settings.voices_bin_path):
                 raise FileNotFoundError(f"Voices file not found at {settings.voices_bin_path}")

            self.kokoro = Kokoro(settings.kokoro_model_path, settings.voices_bin_path)
            # Use espeak for both English and Spanish
            self.g2p_es = EspeakG2P(language='es')
            self.g2p_en = EspeakG2P(language='en-us')
            logger.info("TTS Service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TTS Service: {e}")
            # Make the error more visible during startup
            print(f"FATAL ERROR: Could not initialize TTS Service: {e}")
            raise RuntimeError(f"Failed to initialize TTS Service: {e}")

    async def _get_voice_path(self, voice_name):
        # In this implementation, voice_name is directly used
        return voice_name, voice_name # Placeholder for potential future expansion

    async def generate_speech(self, text, voice, speed=1.0, lang_code=None):
        language = lang_code or voice[0].lower() # Determine language from voice if not provided
        logger.info(f"Generating speech with language code: '{language}', voice: '{voice}', speed: {speed}") # Log speed

        if language == 'e':
            phonemes, _ = self.g2p_en(text)
        elif language == 's':
            phonemes, _ = self.g2p_es(text)
        else:
            logger.warning(f"Unknown language code: {language}, defaulting to English")
            phonemes, _ = self.g2p_en(text)

        logger.debug(f"Generated phonemes: {phonemes}")

        samples, sample_rate = self.kokoro.create(
            phonemes, voice=voice, speed=speed, is_phonemes=True # Pass speed here
        )

        logger.info(f"Generated {len(samples)} samples at {sample_rate} Hz")
        return samples, sample_rate

try:
    tts_service_instance = TTSService()
except RuntimeError as e:
    logger.error(f"Could not start application due to TTS service initialization failure: {e}")
    import sys
    sys.exit(1)

# --- FastAPI Dependency Injection ---
def get_tts_service():
    return tts_service_instance

# --- FastAPI App Initialization ---
app = FastAPI(title="TTS API", description="Text-to-Speech API using Kokoro")
# --- Mount Static Files ---
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add a specific route for favicon.ico
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

# --- FastAPI Request Models ---
class CaptionedSpeechRequest(BaseModel):
    input: str
    voice: str
    response_format: str = "wav"
    speed: float = 1.0
    lang_code: Optional[str] = None

# --- FastAPI Helper Functions ---
# ... (Keep existing _find_file function) ...
async def _find_file(filename: str, search_paths: List[str]):
    for path in search_paths:
        file_path = os.path.join(path, filename)
        if os.path.exists(file_path):
            return file_path
    logger.error(f"File {filename} not found in search paths: {search_paths}")
    raise FileNotFoundError(f"File {filename} not found in any of the search paths")


# --- FastAPI Endpoints ---
# ... (Keep existing /dev/timestamps/{filename} endpoint) ...
@app.get("/dev/timestamps/{filename}")
async def get_timestamps(filename: str):
    """Download timestamps from temp storage"""
    try:
        logger.info(f"Request received for timestamp file: {filename}")
        # Basic sanitization
        if ".." in filename or "/" in filename or "\\" in filename:
             logger.warning(f"Attempt to access invalid path: {filename}")
             raise HTTPException(status_code=400, detail="Invalid filename")

        file_path = await _find_file(
            filename=filename, search_paths=[settings.temp_file_dir]
        )
        logger.info(f"Found timestamp file at: {file_path}")

        return FileResponse(
            file_path,
            media_type="application/json",
            filename=filename, # Keep original filename for download
            headers={
                "Cache-Control": "no-cache",
                "Content-Disposition": f"attachment; filename={filename}",
            },
        )
    except FileNotFoundError:
         logger.error(f"Timestamp file not found: {filename}")
         raise HTTPException(status_code=404, detail="Timestamps file not found")
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


# ... (Keep existing /dev/captioned_speech endpoint, ensure it uses request.speed) ...
@app.post("/dev/captioned_speech")
async def create_captioned_speech(
    request: CaptionedSpeechRequest,
    tts_service: TTSService = Depends(get_tts_service),
):
    """Generate audio with word-level timestamps (API endpoint)"""
    try:
        logger.info(f"Received API request: Voice={request.voice}, Format={request.response_format}, Lang={request.lang_code}, Speed={request.speed}")

        content_type_map = {
            "mp3": "audio/mpeg", "opus": "audio/opus", "aac": "audio/aac",
            "flac": "audio/flac", "wav": "audio/wav", "pcm": "audio/pcm",
        }
        content_type = content_type_map.get(request.response_format, f"audio/{request.response_format}")
        if request.response_format not in content_type_map:
             logger.warning(f"Unsupported audio format requested: {request.response_format}, defaulting to audio/{request.response_format}")

        normalizer = AudioNormalizer()

        # Determine language code (prefer explicit, fallback to voice prefix)
        pipeline_lang_code = request.lang_code if request.lang_code else request.voice[0].lower()
        logger.info(
            f"Using lang_code '{pipeline_lang_code}' for voice '{request.voice}' in text processing"
        )

        # --- Generate Audio (PASSING SPEED from request) ---
        samples, sample_rate = await tts_service.generate_speech(
            request.input, request.voice, request.speed, pipeline_lang_code # Use request.speed
        )
        if samples is None or len(samples) == 0:
             raise RuntimeError("TTS generation produced no audio samples.")

        # --- Normalize Audio ---
        normalized_audio = await normalizer.normalize(samples)

        # --- Generate Approximate Timestamps ---
        word_timestamps = []
        text = request.input
        words = text.split()
        total_duration = len(normalized_audio) / sample_rate if sample_rate > 0 else 0
        avg_word_duration = total_duration / len(words) if words and total_duration > 0 else 0.1 # Avoid division by zero, provide small default

        current_time = 0.0
        for word in words:
            start_time = current_time
            # Estimate duration based on average (simple approach)
            word_duration = avg_word_duration
            end_time = start_time + word_duration
            word_timestamps.append({
                "word": word,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
            })
            current_time = end_time
         # Ensure last word end_time doesn't exceed total duration
        if word_timestamps:
            word_timestamps[-1]["end_time"] = min(word_timestamps[-1]["end_time"], round(total_duration, 3))

        # --- Save Timestamps ---
        timestamps_filename = ""
        temp_writer_path = None # Store path for logging
        try:
            async with TempFileWriter("json") as temp_writer:
                temp_writer_path = temp_writer.path # Get path for logging before potential errors
                timestamps_json = json.dumps(word_timestamps, indent=2)
                await temp_writer.write(timestamps_json.encode('utf-8'))
                await temp_writer.finalize() # Ensure data is written
                timestamps_filename = Path(temp_writer.download_path).name # Get only the filename part
                logger.info(f"Timestamps saved to temporary file: {timestamps_filename} (full path: {temp_writer.download_path})")
        except Exception as ts_err:
            logger.error(f"Failed to write timestamps file (path: {temp_writer_path}): {ts_err}")
            # Continue without timestamps if saving failed
            timestamps_filename = ""


        # --- Prepare Audio Response ---
        audio_bytes_io = io.BytesIO()
        try:
            sf.write(audio_bytes_io, normalized_audio, sample_rate, format=request.response_format.upper())
            audio_bytes_io.seek(0) # Rewind the buffer to the beginning
            logger.info(f"Audio data prepared in {request.response_format} format.")
        except Exception as audio_err:
             logger.error(f"Failed to format audio to {request.response_format}: {audio_err}")
             raise HTTPException(status_code=500, detail=f"Failed to create audio in format {request.response_format}")

        # --- Return Streaming Response ---
        headers = {
            "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
            "X-Accel-Buffering": "no", # Useful for nginx proxying
            "Cache-Control": "no-cache",
        }
        if timestamps_filename:
            headers["X-Timestamps-Path"] = f"/dev/timestamps/{timestamps_filename}"
            logger.info(f"Included timestamp path in header: {headers['X-Timestamps-Path']}")
        else:
             logger.warning("No timestamp file generated or saved, header omitted.")


        return StreamingResponse(
            audio_bytes_io,
            media_type=content_type,
            headers=headers,
        )

    # Keep existing exception handling
    except ValueError as e:
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={"error": "validation_error", "message": str(e), "type": "invalid_request_error"},
        )
    except FileNotFoundError as e:
        logger.error(f"File not found error during processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "server_error", "message": f"Missing required file: {e}", "type": "server_error"},
        )
    except RuntimeError as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "processing_error", "message": str(e), "type": "server_error"},
        )
    except Exception as e:
        logger.exception(f"Unexpected error in speech generation: {str(e)}") # Log full traceback for unexpected errors
        raise HTTPException(
            status_code=500,
            detail={"error": "unexpected_error", "message": "An unexpected error occurred", "type": "server_error"},
        )


# --- Gradio Interface Definition ---

# ... (Keep VOICES and FORMATS constants) ...
VOICES = {
    "English": ["af_heart", "af_sarah", "af_nova", "af_bella", "am_eric"],
    "Spanish": ["ef_dora", "im_nicola", "if_sara", "bf_emma", "bm_daniel"],
}
FORMATS = ["wav", "mp3", "opus", "flac"]

# ... (Keep update_voices function) ...
def update_voices(language):
    """Update the list of available voices based on the selected language"""
    choices = VOICES.get(language, [])
    value = choices[0] if choices else None
    return gr.Dropdown(choices=choices, value=value)


# MODIFIED Gradio function to accept and use speed
def gradio_text_to_speech(text, voice, language, audio_format, speed): # <-- Added speed parameter
    """Convert text to speech using the internal TTS service for Gradio"""
    # Log the received speed value
    logger.info(f"Gradio request: Voice='{voice}', Lang='{language}', Format='{audio_format}', Speed={speed}, Text='{text[:30]}...'")
    if not text:
        return None, "Error: Text input cannot be empty."
    if not voice:
        return None, "Error: Please select a voice."

    try:
        # Determine language code (use selected language name)
        lang_code = language[0].lower() if language else None

        # --- Call Core TTS Generation Logic (using the global instance) ---
        # Run the async generate_speech in a sync context for Gradio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Pass the speed parameter from Gradio to the service
        samples, sample_rate = loop.run_until_complete(
            tts_service_instance.generate_speech(text, voice, speed, lang_code) # <-- Use speed here
        )

        if samples is None or len(samples) == 0:
             return None, "Error: TTS generation failed (no audio)."

        # --- Normalize Audio ---
        # Assuming passthrough normalizer
        normalized_audio = samples

        # --- Generate Approximate Timestamps ---
        word_timestamps = []
        words = text.split()
        total_duration = len(normalized_audio) / sample_rate if sample_rate > 0 else 0
        avg_word_duration = total_duration / len(words) if words and total_duration > 0 else 0.1

        current_time = 0.0
        for word in words:
            start_time = current_time
            word_duration = avg_word_duration
            end_time = start_time + word_duration
            word_timestamps.append({
                "word": word,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
            })
            current_time = end_time
        if word_timestamps:
            word_timestamps[-1]["end_time"] = min(word_timestamps[-1]["end_time"], round(total_duration, 3))

        timestamps_text = json.dumps(word_timestamps, indent=2)

        # --- Save Audio to a Temporary File for Gradio ---
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}")
        try:
            sf.write(temp_audio_file.name, normalized_audio, sample_rate, format=audio_format.upper())
            logger.info(f"Gradio audio saved temporarily to: {temp_audio_file.name}")
            # Return file path for Gradio Audio component
            return temp_audio_file.name, timestamps_text
        except Exception as write_err:
            logger.error(f"Error writing audio file for Gradio: {write_err}")
            return None, f"Error saving audio: {write_err}"
        finally:
            temp_audio_file.close() # Close the file handle

    except Exception as e:
        logger.exception(f"Error during Gradio TTS processing: {e}")
        return None, f"An unexpected error occurred: {str(e)}"


# --- Define Gradio Blocks (Adding Speed Slider) ---
with gr.Blocks(title="Text-to-Speech System") as gradio_interface:
    gr.Markdown("# Text-to-Speech System")
    gr.Markdown("Enter text, select language/voice/speed, and generate speech.")

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Text to convert",
                placeholder="Enter your text here...",
                lines=5
            )

        with gr.Column(scale=1):
            language_dropdown = gr.Dropdown(
                choices=list(VOICES.keys()),
                label="Language",
                value="English"
            )

            initial_voices = VOICES.get("English", [])
            voice_dropdown = gr.Dropdown(
                choices=initial_voices,
                label="Voice",
                value=initial_voices[0] if initial_voices else None
            )

            format_dropdown = gr.Dropdown(
                choices=FORMATS,
                label="Audio Format",
                value="wav"
            )

            # --- ADD SPEED SLIDER ---
            speed_slider = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                step=0.1,
                value=1.0,
                label="Speech Speed"
            )
            # --- END SPEED SLIDER ---


    with gr.Row():
        convert_btn = gr.Button("Convert to Speech", variant="primary")
        upload_btn = gr.UploadButton("Upload Text File (.txt)", file_types=[".txt"])
        clear_btn = gr.Button("Clear")


    with gr.Row():
        audio_output = gr.Audio(label="Generated Speech", type="filepath", interactive=False)
        timestamps_output = gr.Textbox(label="Word Timestamps (Approximate)", lines=10, interactive=False)


    # --- Gradio Event Handlers ---
    language_dropdown.change(
        fn=update_voices,
        inputs=[language_dropdown],
        outputs=[voice_dropdown]
    )

    # MODIFIED: Add speed_slider to inputs
    convert_btn.click(
        fn=gradio_text_to_speech,
        inputs=[
            text_input,
            voice_dropdown,
            language_dropdown,
            format_dropdown,
            speed_slider # <-- Pass slider value
        ],
        outputs=[audio_output, timestamps_output]
    )

    # ... (Keep clear_outputs and clear_btn.click) ...
    def clear_outputs():
        return "", None, "" # Clears text, audio, and timestamps

    clear_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[text_input, audio_output, timestamps_output]
    )

    # ... (Keep process_uploaded_file and upload_btn.upload) ...
    def process_uploaded_file(file):
        if file is None:
            return ""
        try:
            with open(file.name, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"Loaded text from uploaded file: {file.name}")
                return content
        except Exception as e:
            logger.error(f"Error reading uploaded file {file.name}: {e}")
            gr.Warning(f"Error reading file: {str(e)}") # Show warning in UI
            return ""

    upload_btn.upload(
        fn=process_uploaded_file,
        inputs=[upload_btn],
        outputs=[text_input]
    )


# --- Mount Gradio App onto FastAPI ---
# ... (Keep existing mount call) ...
app = gr.mount_gradio_app(
    app,
    gradio_interface,
    path="/web"
)

# --- Run the Combined Application ---
# ... (Keep existing run call) ...
if __name__ == "__main__":
    print("Starting TTS API and Gradio Web Interface...")
    print(f"API available at: http://localhost:8880")
    print(f"Web UI available at: http://localhost:8880/web")
    uvicorn.run("main:app", host="0.0.0.0", port=8880, reload=True)

# --- END OF main.py ---