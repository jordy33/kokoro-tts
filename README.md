# Text-to-Speech API with Gradio Interface

A Text-to-Speech API and web interface using Kokoro ONNX for high-quality speech synthesis.

## Features

- FastAPI backend with two endpoints:
  - `/dev/timestamps/{filename}`: Download word-level timestamps
  - `/dev/captioned_speech`: Generate speech with timestamps
- Gradio web interface for easy interaction
- Support for multiple languages (English, Spanish)
- Various voice options 
- Multiple output formats (WAV, MP3)

## Requirements

- Python 3.12 or higher
- The required model files:
  - `kokoro-v1.0.onnx`
  - `voices-v1.0.bin`

## Installation

1. Install Python dependencies:

```bash
uv venv --seed -p 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

2. Download the required model files:

```bash
# If you don't already have them
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

## Running the System

1. Start the API server:

```bash
python api.py
```

This will start the API on http://localhost:8880.

2. Start the Gradio interface:

```bash
python gradio_interface.py
```

This will start a web interface on http://localhost:8881.

## API Usage

### Generate Captioned Speech

```python
import requests
import json

response = requests.post(
    "http://localhost:8880/dev/captioned_speech",
    json={
        "input": "Hello, this is a test.",
        "voice": "af_heart",
        "response_format": "wav",
        "speed": 1.0,
        "lang_code": "e"
    }
)

# Save the audio
with open("output.wav", "wb") as f:
    f.write(response.content)

# Get the timestamps file
timestamps_path = response.headers.get("X-Timestamps-Path")
if timestamps_path:
    timestamps_response = requests.get(f"http://localhost:8880/dev/timestamps/{timestamps_path}")
    timestamps = json.loads(timestamps_response.text)
    print(json.dumps(timestamps, indent=2))
```

## Available Voices

### English Voices
- af_heart
- af_sarah
- em_grace
- em_jenny
- em_ryan

### Spanish Voices
- sm_miguel
- sm_pedro
- sf_luna
- sf_dora
- sf_clara

## License

See the LICENSE file for details.