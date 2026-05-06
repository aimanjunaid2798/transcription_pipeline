import os
from faster_whisper import WhisperModel

def accept_audio(file_path):
    valid_ext = ('.wav', '.mp3', '.m4a')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")
    if not file_path.lower().endswith(valid_ext):
        raise ValueError("Unsupported file format. Please use .wav, .mp3, or .m4a")
    return file_path

def transcribe_audio(file_path):
    try:
        valid_path = accept_audio(file_path)

        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(valid_path, beam_size=5)
        full_text = " ".join([segment.text.strip() for segment in segments])
        return {"status": "success", "data": full_text}

    except FileNotFoundError as e:
        return {"status": "error", "message": str(e)}
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}

def transcribe_with_timestamps(file_path):
    try:
        valid_path = accept_audio(file_path)

        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(valid_path)

        result = []
        for segment in segments:
            result.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })
        return {"status": "success", "data": result}

    except FileNotFoundError as e:
        return {"status": "error", "message": str(e)}
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Transcription failed: {str(e)}"}

if __name__ == "__main__":
    test_file = "/content/harvard.wav" 

    print("--- Audio Transcription CLI ---")
    print("1. Full Transcription (Text Only)")
    print("2. Detailed Transcription (With Timestamps)")
    
    choice = input("Select an option (1 or 2): ").strip()

    print("\nThe Transcription Process has been Started!")

    if choice == "1":
        response = transcribe_audio(test_file)
        if response["status"] == "success":
            print("\n--- Full Audio Transcription ---")
            print(response["data"])
        else:
            print(f"Error: {response['message']}")

    elif choice == "2":
        response = transcribe_with_timestamps(test_file)
        if response["status"] == "success":
            print("\n--- Transcription with Timestamps ---")
            for entry in response["data"]:
                print(f"[{entry['start']}s - {entry['end']}s]: {entry['text']}")
        else:
            print(f"Error: {response['message']}")

    else:
        print("Invalid selection. Please run the script again and choose 1 or 2.")
