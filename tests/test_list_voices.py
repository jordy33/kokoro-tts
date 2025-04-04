"""Script to list all available voices in Kokoro model"""

from kokoro_onnx import Kokoro

# Initialize Kokoro model
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# Get available voices
print("Available voices in Kokoro model:")
if hasattr(kokoro, 'available_voices'):
    voices = kokoro.available_voices
    for voice in voices:
        print(f"- {voice}")
else:
    print("Kokoro model doesn't expose available_voices directly")
    # Try to list voice files from the voices binary
    print("Attempting to access voices in other ways...")
    
    # Get voice object attributes that might indicate loaded voices
    voice_attributes = [attr for attr in dir(kokoro) if 'voice' in attr.lower()]
    print(f"Voice-related attributes: {voice_attributes}")
    
    # Try to check if we can use voices directly
    test_voices = ["af_heart", "af_sarah", "em_grace", "em_jenny", "sf_dora", "sm_miguel"]
    
    print("\nTesting specific voices:")
    for voice in test_voices:
        try:
            # Try to access the voice (without generating audio)
            # Just check if using the voice name causes an error
            print(f"Checking: {voice}", end=" - ")
            kokoro._get_voice_embedding(voice)
            print("Available")
        except Exception as e:
            print(f"Error: {str(e)}")