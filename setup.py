from setuptools import setup

setup(
    name="tts",
    version="0.1.0",
    description="Text-to-Speech API using Kokoro and Gradio interface",
    py_modules=["api", "gradio_interface"],
    python_requires=">=3.12",
    install_requires=[
        "kokoro-onnx>=0.4.5",
        "misaki>=0.9.3",
        "num2words>=0.5.14",
        "soundfile>=0.13.1",
        "spacy>=3.8.5",
        "fastapi>=0.104.0",
        "uvicorn>=0.23.2",
        "gradio>=4.0.0",
        "requests>=2.31.0",
        "python-multipart>=0.0.6",
    ],
)