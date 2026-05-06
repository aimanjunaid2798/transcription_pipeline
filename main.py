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
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(file_path, beam_size=5)
    full_text = " ".join([segment.text.strip() for segment in segments])
    return full_text

def transcribe_with_timestamps(file_path):
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(file_path)
    result = []
    for segment in segments:
        result.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })
    return result

if __name__ == "__main__":
    # Add the audio file
    test_file = "audio.mp3" 
    if os.path.exists(test_file):
        print("The transcription process has been started")
        print(transcribe_with_timestamps(test_file))
    else:
        print("Please place a 'audio.mp3' in the folder to test.")
