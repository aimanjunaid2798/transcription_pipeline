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
        return {"status": "success", "text": full_text}
        
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
    # Add File
    test_file = "/content/harvard.wav" 
    print("The Transcription Process has been Started!")
    
    response = transcribe_with_timestamps(test_file)
    
    if response["status"] == "success":
        print("Transcription Complete:")
        for entry in response["data"]:
            print(f"[{entry['start']}s - {entry['end']}s]: {entry['text']}")
    else:
        print(f"Failed to process: {response['message']}")
