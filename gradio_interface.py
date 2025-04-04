import gradio as gr
import requests
import json
import os
import tempfile
from pathlib import Path

# API Configuration
API_URL = "http://localhost:8880"

# Available voices and languages - Updated to use actual available voices
VOICES = {
    "English": ["af_heart", "af_sarah", "af_nova", "af_bella", "am_eric"],
    "Spanish": ["ef_dora", "im_nicola", "if_sara", "bf_emma", "bm_daniel"],
}

# Available audio formats
FORMATS = ["wav", "mp3"]

def text_to_speech(text, voice, language, audio_format):
    """Convert text to speech using the TTS API"""
    try:
        # Determine language code from language selection
        lang_code = language[0].lower()  # 'E' for English, 'S' for Spanish
        
        # Prepare the API request
        url = f"{API_URL}/dev/captioned_speech"
        data = {
            "input": text,
            "voice": voice,
            "response_format": audio_format,
            "speed": 1.0,
            "lang_code": lang_code
        }
        
        # Call the API
        response = requests.post(url, json=data)
        
        if response.status_code != 200:
            return None, f"Error: {response.status_code} - {response.text}"
        
        # Get the timestamps filename from the header
        timestamps_path = response.headers.get("X-Timestamps-Path")
        
        # Save the audio to a temporary file
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}")
        temp_audio.write(response.content)
        temp_audio.close()
        
        # If we have timestamps, fetch them
        timestamps_text = "No timestamps available"
        if timestamps_path:
            try:
                timestamps_response = requests.get(f"{API_URL}/dev/timestamps/{timestamps_path}")
                if timestamps_response.status_code == 200:
                    timestamps = json.loads(timestamps_response.text)
                    timestamps_text = json.dumps(timestamps, indent=2)
            except Exception as e:
                timestamps_text = f"Error fetching timestamps: {str(e)}"
        
        return temp_audio.name, timestamps_text
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def update_voices(language):
    """Update the list of available voices based on the selected language"""
    return gr.Dropdown(choices=VOICES.get(language, []))

def clear_text():
    """Clear the text input and results"""
    return "", "", ""

# Create Gradio interface
with gr.Blocks(title="Text-to-Speech System") as demo:
    gr.Markdown("# Text-to-Speech System")
    gr.Markdown("Enter text to convert to speech")
    
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
            
            voice_dropdown = gr.Dropdown(
                choices=VOICES["English"],
                label="Voice",
                value=VOICES["English"][0]
            )
            
            format_dropdown = gr.Dropdown(
                choices=FORMATS,
                label="Audio Format",
                value="wav"
            )
    
    with gr.Row():
        convert_btn = gr.Button("Convert to Speech", variant="primary")
        upload_btn = gr.Button("Upload Text")
        clear_btn = gr.Button("Clear")
    
    with gr.Row():
        audio_output = gr.Audio(label="Generated Speech", interactive=False, type="filepath")
        timestamps_output = gr.Textbox(label="Word Timestamps", lines=10)
    
    # Set up event handlers
    language_dropdown.change(
        update_voices, 
        inputs=[language_dropdown], 
        outputs=[voice_dropdown]
    )
    
    convert_btn.click(
        text_to_speech,
        inputs=[text_input, voice_dropdown, language_dropdown, format_dropdown],
        outputs=[audio_output, timestamps_output]
    )
    
    clear_btn.click(
        clear_text,
        inputs=[],
        outputs=[text_input, audio_output, timestamps_output]
    )
    
    # File upload component and handler
    file_input = gr.File(label="Upload text file", file_types=[".txt"], visible=False)
    
    def process_file(file):
        if file is None:
            return ""
        try:
            with open(file.name, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    upload_btn.click(
        lambda: gr.update(visible=True),
        inputs=[],
        outputs=[file_input]
    )
    
    file_input.change(
        process_file,
        inputs=[file_input],
        outputs=[text_input]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8881)