# --- START OF main.py ---

import os
import json
import tempfile
from pathlib import Path
from typing import List, Optional
import io

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

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Settings ---
class Settings:
    temp_file_dir = os.path.join(tempfile.gettempdir(), "tts_api")
    kokoro_model_path = "kokoro-v1.0.onnx"  # Ensure these files are in the same directory or provide full path
    voices_bin_path = "voices-v1.0.bin"

settings = Settings()

# Create temp directory if it doesn't exist
os.makedirs(settings.temp_file_dir, exist_ok=True)

# --- Helper Classes (from api.py) ---

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
        # Ensure file exists before trying to remove - might be closed already
        # Optional: Add cleanup logic here if needed, but temp files are often handled by OS

    async def write(self, data):
        if self.file and not self.file.closed:
            self.file.write(data)

    async def finalize(self):
        if self.file and not self.file.closed:
            self.file.flush()
            # Don't close here, __aexit__ handles it


class AudioNormalizer:
    async def normalize(self, audio):
        # Simple passthrough implementation
        return audio

# --- TTS Service Initialization ---
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
        logger.info(f"Generating speech with language code: '{language}', voice: '{voice}'")

        if language == 'e':
            phonemes, _ = self.g2p_en(text)
        elif language == 's':
            phonemes, _ = self.g2p_es(text)
        else:
            logger.warning(f"Unknown language code: {language}, defaulting to English")
            phonemes, _ = self.g2p_en(text)

        logger.debug(f"Generated phonemes: {phonemes}")

        samples, sample_rate = self.kokoro.create(
            phonemes, voice=voice, speed=speed, is_phonemes=True
        )

        logger.info(f"Generated {len(samples)} samples at {sample_rate} Hz")
        return samples, sample_rate

# --- Instantiate TTS Service ONCE ---
# Make it globally accessible for both FastAPI and Gradio logic
try:
    tts_service_instance = TTSService()
except RuntimeError as e:
    logger.error(f"Could not start application due to TTS service initialization failure: {e}")
    # Exit if TTS cannot be initialized, as the app is unusable
    import sys
    sys.exit(1)


# --- FastAPI Dependency Injection ---
def get_tts_service():
    # Return the globally instantiated service
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
async def _find_file(filename: str, search_paths: List[str]):
    for path in search_paths:
        file_path = os.path.join(path, filename)
        if os.path.exists(file_path):
            return file_path
    logger.error(f"File {filename} not found in search paths: {search_paths}")
    raise FileNotFoundError(f"File {filename} not found in any of the search paths")

# --- FastAPI Endpoints ---

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


@app.post("/dev/captioned_speech")
async def create_captioned_speech(
    request: CaptionedSpeechRequest,
    # client_request: Request, # Removed as it wasn't used
    # x_raw_response: str = Header(None, alias="x-raw-response"), # Removed as it wasn't used
    tts_service: TTSService = Depends(get_tts_service),
):
    """Generate audio with word-level timestamps (API endpoint)"""
    try:
        logger.info(f"Received API request for captioned speech: Voice={request.voice}, Format={request.response_format}, Lang={request.lang_code}")

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

        # --- Generate Audio ---
        samples, sample_rate = await tts_service.generate_speech(
            request.input, request.voice, request.speed, pipeline_lang_code
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
            # Refinement: could adjust based on word length, but keep simple for now
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
        try:
            async with TempFileWriter("json") as temp_writer:
                timestamps_json = json.dumps(word_timestamps, indent=2)
                await temp_writer.write(timestamps_json.encode('utf-8'))
                await temp_writer.finalize() # Ensure data is written
                timestamps_filename = Path(temp_writer.download_path).name # Get only the filename part
                logger.info(f"Timestamps saved to temporary file: {timestamps_filename} (full path: {temp_writer.download_path})")
        except Exception as ts_err:
            logger.error(f"Failed to write timestamps file: {ts_err}")
            # Continue without timestamps if saving failed
            timestamps_filename = ""


        # --- Prepare Audio Response ---
        # Use BytesIO for in-memory conversion, avoid hitting disk twice if possible
        audio_bytes_io = io.BytesIO()
        try:
            # Use soundfile to write to the BytesIO object
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
            # IMPORTANT: Provide the path RELATIVE to the API root for the client
            headers["X-Timestamps-Path"] = f"/dev/timestamps/{timestamps_filename}"
            logger.info(f"Included timestamp path in header: {headers['X-Timestamps-Path']}")
        else:
             logger.warning("No timestamp file generated or saved, header omitted.")


        return StreamingResponse(
            audio_bytes_io,
            media_type=content_type,
            headers=headers,
        )

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

# Gradio specific constants (from gradio_interface.py)
VOICES = {
    "English": ["af_heart", "af_sarah", "af_nova", "af_bella", "am_eric"], # Added more from README
    "Spanish": ["ef_dora", "im_nicola", "if_sara", "bf_emma", "bm_daniel"], # Added more from README
    # Update these lists based on the actual voices in your voices-v1.0.bin if different
}
FORMATS = ["wav", "mp3", "opus", "flac"] # Expanded formats based on API

# Gradio helper function to update voice choices
def update_voices(language):
    """Update the list of available voices based on the selected language"""
    return gr.Dropdown(choices=VOICES.get(language, []), value=VOICES.get(language, [""])[0])

# Gradio function to call TTS logic (MODIFIED to call internal functions)
# NOTE: This runs synchronously within Gradio's processing thread.
# For long tasks, consider making the internal logic async and using asyncio.run
# or running Gradio with more workers.
def gradio_text_to_speech(text, voice, language, audio_format):
    """Convert text to speech using the internal TTS service for Gradio"""
    logger.info(f"Gradio request: Text='{text[:30]}...', Voice='{voice}', Lang='{language}', Format='{audio_format}'")
    if not text:
        return None, "Error: Text input cannot be empty."
    if not voice:
        return None, "Error: Please select a voice."

    try:
        # Determine language code (use selected language name)
        lang_code = language[0].lower() if language else None

        # --- Call Core TTS Generation Logic (using the global instance) ---
        # This needs to run synchronously or be handled carefully in Gradio's event loop
        # Using asyncio.run here might be complex within Gradio's sync handler.
        # Keep it simple for now, assuming generate_speech is reasonably fast.
        # If generate_speech was async:
        # import asyncio
        # samples, sample_rate = asyncio.run(tts_service_instance.generate_speech(text, voice, 1.0, lang_code))
        # But since TTSService.generate_speech is not async currently, call directly:

        # We need to run the async generate_speech in a sync context for Gradio
        import asyncio
        try:
            # Get or create an event loop for the current thread if needed
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        samples, sample_rate = loop.run_until_complete(
            tts_service_instance.generate_speech(text, voice, 1.0, lang_code)
        )

        if samples is None or len(samples) == 0:
             return None, "Error: TTS generation failed (no audio)."

        # --- Normalize Audio ---
        # normalizer = AudioNormalizer() # Instantiated locally for sync context
        # normalized_audio = loop.run_until_complete(normalizer.normalize(samples)) # If normalize was async
        # Since normalize is currently sync pass-through:
        normalized_audio = samples # Assuming passthrough normalizer

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
        # Gradio's Audio component needs a file path
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}")
        try:
            sf.write(temp_audio_file.name, normalized_audio, sample_rate, format=audio_format.upper())
            logger.info(f"Gradio audio saved temporarily to: {temp_audio_file.name}")
            return temp_audio_file.name, timestamps_text
        except Exception as write_err:
            logger.error(f"Error writing audio file for Gradio: {write_err}")
            return None, f"Error saving audio: {write_err}"
        finally:
            # Temp file should be cleaned up by Gradio or OS eventually,
            # but closing the handle is good practice if sf.write didn't.
            # sf.write often handles closing, but check soundfile docs if unsure.
            temp_audio_file.close() # Close the file handle

    except Exception as e:
        logger.exception(f"Error during Gradio TTS processing: {e}")
        return None, f"An unexpected error occurred: {str(e)}"


# --- Define Gradio Blocks ---
with gr.Blocks(title="Text-to-Speech System") as gradio_interface:
    gr.Markdown("# Text-to-Speech System")
    gr.Markdown("Enter text, select language/voice, and generate speech.")

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
                value="English"  # Default language
            )

            # Initial voices based on default language
            initial_voices = VOICES.get("English", [])
            voice_dropdown = gr.Dropdown(
                choices=initial_voices,
                label="Voice",
                value=initial_voices[0] if initial_voices else None
            )

            format_dropdown = gr.Dropdown(
                choices=FORMATS,
                label="Audio Format",
                value="wav" # Default format
            )

    with gr.Row():
        convert_btn = gr.Button("Convert to Speech", variant="primary")
        # Keep upload and clear buttons as defined before
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

    convert_btn.click(
        fn=gradio_text_to_speech,
        inputs=[text_input, voice_dropdown, language_dropdown, format_dropdown],
        outputs=[audio_output, timestamps_output]
    )

    def clear_outputs():
        return "", None, "" # Clears text, audio, and timestamps

    clear_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[text_input, audio_output, timestamps_output]
    )

    # Function to handle file upload and update text input
    def process_uploaded_file(file):
        if file is None:
            return ""
        try:
            # Gradio UploadButton provides a temp file object
            with open(file.name, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"Loaded text from uploaded file: {file.name}")
                return content
        except Exception as e:
            logger.error(f"Error reading uploaded file {file.name}: {e}")
            gr.Warning(f"Error reading file: {str(e)}") # Show warning in UI
            return "" # Return empty string on error

    upload_btn.upload(
        fn=process_uploaded_file,
        inputs=[upload_btn],
        outputs=[text_input]
    )


# --- Mount Gradio App onto FastAPI ---
# The key step for integration!
app = gr.mount_gradio_app(
    app,                # The existing FastAPI app
    gradio_interface,   # The Gradio Blocks interface
    path="/web"         # The path where Gradio UI will be served
)

# --- Run the Combined Application ---
if __name__ == "__main__":
    print("Starting TTS API and Gradio Web Interface...")
    print(f"API available at: http://localhost:8880")
    print(f"Web UI available at: http://localhost:8880/web")
    # Use reload=True only for development, disable in production
    uvicorn.run("main:app", host="0.0.0.0", port=8880, reload=True)

# --- END OF main.py ---