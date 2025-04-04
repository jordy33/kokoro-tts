"""Test script for sf_dora voice"""

import soundfile as sf
from kokoro_onnx import Kokoro
from misaki.espeak import EspeakG2P

# Initialize components
g2p = EspeakG2P(language='es')
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# Test access to voices
print("Attempting to access voices method:")
try:
    voices = kokoro.get_voices()
    print(f"Available voices: {voices}")
except Exception as e:
    print(f"Error getting voices: {e}")

# Actual voice test
text = "Prueba de voz en español."
print(f"Converting text: '{text}'")

# Phonemize
print("Converting to phonemes...")
phonemes, _ = g2p(text)
print(f"Phonemes: {phonemes}")

# Create speech with sf_dora
print("Generating speech with sf_dora...")
try:
    samples, sample_rate = kokoro.create(phonemes, "sf_dora", is_phonemes=True)
    print(f"Successfully generated audio with sf_dora. Sample rate: {sample_rate}, Samples shape: {samples.shape}")
    
    # Save
    sf.write("sf_dora_test.wav", samples, sample_rate)
    print("Created sf_dora_test.wav")
except Exception as e:
    print(f"Error generating audio with sf_dora: {e}")

# Try with ef_dora (the voice used in main.py)
print("\nGenerating speech with ef_dora...")
try:
    samples, sample_rate = kokoro.create(phonemes, "ef_dora", is_phonemes=True)
    print(f"Successfully generated audio with ef_dora. Sample rate: {sample_rate}, Samples shape: {samples.shape}")
    
    # Save
    sf.write("ef_dora_test.wav", samples, sample_rate)
    print("Created ef_dora_test.wav")
except Exception as e:
    print(f"Error generating audio with ef_dora: {e}")