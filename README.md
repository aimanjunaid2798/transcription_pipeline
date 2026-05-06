# Transcription Pipeline

A clean, efficient Python implementation for transcribing audio files with precise timestamps using the Faster-Whisper engine.

---

## Design Decisions

### 1. STT Engine: Faster-Whisper
I chose **Faster-Whisper** over the standard OpenAI implementation for its production-ready speed and CTranslate2 backend.
- **Optimization:** It utilizes CTranslate2, which allows for `int8` quantization. This ensures high-speed transcription even on CPU-bound environments.
- **Memory Efficiency:** It maintains a low memory footprint while providing the same accuracy as the original Whisper models.

### 2. Precise Segment-Level Timestamps
The pipeline is designed to return a structured list of dictionaries rather than just raw text.
- **Granularity:** By iterating through the `segments` generator, the system captures `start` and `end` times for every sentence.
- **Utility:** This makes the output ready for downstream features like video subtitling or "click-to-play" audio players.

### 3. Data Validation
Before processing, the script performs a series of checks:
- **Existence Check:** Verifies the file exists to prevent path errors.
- **Format Filtering:** Explicitly allows only `.mp3`, `.wav`, and `.m4a` to ensure the model doesn't attempt to process incompatible binary data.

### 4. Text Post-Processing
The script implements automatic whitespace management and string stripping:
- **Clean Joins:** Segments are joined with single spaces to prevent "word-clumping."
- **Formatting:** Leading and trailing whitespace is removed from each segment to ensure the final output is ready for display.

---

## How to Run

### 1. Installation
```bash
pip install faster-whisper
