import requests
import json
import os

# Configuration
API_URL = "http://localhost:8880"  # Your API base URL
SPEECH_ENDPOINT = f"{API_URL}/dev/captioned_speech"
TIMESTAMP_ENDPOINT_BASE = f"{API_URL}/dev/timestamps" # Base URL for timestamps

# Request Payload
payload = {
    "input": "Hello from the Python test script!",
    "voice": "af_sarah",        # Try a different voice
    "response_format": "mp3",   # Try a different format
    "speed": 1.0,
    "lang_code": "e"          # Explicitly English
}

# --- Step 1: Generate Speech ---
print(f"Sending request to: {SPEECH_ENDPOINT}")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    response = requests.post(SPEECH_ENDPOINT, json=payload)
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

    print(f"\nSpeech generation successful (Status Code: {response.status_code})")

    # Save the audio content
    audio_filename = f"output_python.{payload['response_format']}"
    with open(audio_filename, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {audio_filename}")

    # Extract timestamp path from headers
    timestamp_relative_path = response.headers.get("X-Timestamps-Path")
    print(f"Timestamp path from header: {timestamp_relative_path}")

    if not timestamp_relative_path:
        print("Warning: X-Timestamps-Path header not found in the response.")
        timestamp_filename = None
    else:
        # Extract just the filename from the path (e.g., /dev/timestamps/tmpXYZ.json -> tmpXYZ.json)
        timestamp_filename = os.path.basename(timestamp_relative_path)
        print(f"Extracted timestamp filename: {timestamp_filename}")


    # --- Step 2: Get Timestamps (if path exists) ---
    if timestamp_filename:
        timestamp_url = f"{TIMESTAMP_ENDPOINT_BASE}/{timestamp_filename}"
        print(f"\nFetching timestamps from: {timestamp_url}")

        try:
            ts_response = requests.get(timestamp_url)
            ts_response.raise_for_status() # Check for errors fetching timestamps

            print(f"Timestamp fetch successful (Status Code: {ts_response.status_code})")

            # Parse and print the timestamps JSON
            try:
                timestamps = ts_response.json()
                print("\n--- Timestamps ---")
                print(json.dumps(timestamps, indent=2))
                print("------------------")

                # Save timestamps to a file
                ts_filename = f"timestamps_{os.path.splitext(timestamp_filename)[0]}.json"
                with open(ts_filename, "w", encoding="utf-8") as f:
                    json.dump(timestamps, f, indent=2)
                print(f"Timestamps saved to: {ts_filename}")

            except json.JSONDecodeError:
                print("Error: Failed to decode JSON from timestamp response.")
                print("Raw content:", ts_response.text)
            except Exception as e:
                 print(f"Error processing timestamp data: {e}")


        except requests.exceptions.RequestException as e:
            print(f"Error fetching timestamps: {e}")
            if hasattr(e, 'response') and e.response is not None:
                 print(f"Timestamp server response: {e.response.status_code} - {e.response.text}")

    else:
        print("\nSkipping timestamp fetch because no path was provided.")

except requests.exceptions.RequestException as e:
    print(f"\nError during API request: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Server response: {e.response.status_code}")
        try:
            # Try to print JSON error detail if available
            print(json.dumps(e.response.json(), indent=2))
        except json.JSONDecodeError:
            # Otherwise print raw text
            print(e.response.text)
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

print("\nScript finished.")