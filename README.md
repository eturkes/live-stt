# live-stt

Real-time Japanese speech-to-text transcription with English translation, powered by the Gemini API.

Captures microphone audio, detects speech using RMS-based voice activity detection, and sends audio chunks to Gemini for transcription and translation. Displays Japanese transcriptions alongside English translations as you speak.

## Requirements

- Python >= 3.10 (developed on 3.14)
- A working microphone
- System audio libraries for `sounddevice` (PortAudio)
  - Debian/Ubuntu: `sudo apt install libportaudio2`
  - Fedora/openSUSE: `sudo dnf install portaudio` / `sudo zypper install portaudio`
  - macOS: `brew install portaudio`
- A [Gemini API key](https://aistudio.google.com/apikey)

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for environment management.

```sh
# Install dependencies (creates .venv automatically)
uv sync

# Set your API key
export GEMINI_API_KEY="your-key-here"
```

## Usage

```sh
# Run with defaults (gemini-3.1-flash-live-preview, auto-calibrated threshold)
live-stt

# Or run directly
python live_stt.py
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model MODEL` | `gemini-3.1-flash-live-preview` | Gemini model to use for transcription |
| `--threshold FLOAT` | auto-calibrate | RMS silence threshold; if omitted, 1 second of ambient noise is recorded to set it automatically |
| `--workers INT` | `5` | Number of concurrent API worker threads |
| `--max-chunk FLOAT` | `5.0` | Maximum seconds of audio before force-sending to the API |
| `--no-translate` | off | Transcribe only (no English translation) |
| `-o`, `--output FILE` | none | Append transcriptions to a text file |

### Examples

```sh
# Use a specific model
live-stt --model gemini-2.0-flash

# Manual silence threshold (skip calibration)
live-stt --threshold 0.01

# Fewer workers, longer chunks
live-stt --workers 2 --max-chunk 10

# Transcribe only (no English translation)
live-stt --no-translate

# Save transcriptions to a file
live-stt --no-translate -o transcript.txt
```

## How It Works

### Audio Pipeline

1. **Capture** - `sounddevice` records from the default input device at its native sample rate.
2. **VAD** - Each 100ms block is measured by RMS amplitude against a silence threshold. Speech onset starts buffering; silence lasting >= 0.8s triggers a send.
3. **Resample** - Buffered audio is downsampled to 16 kHz via linear interpolation before sending.
4. **Force-cut** - Continuous speech exceeding `--max-chunk` seconds is force-sent with a 1-second overlap to preserve context across boundaries.
5. **Transcribe** - WAV-encoded audio is sent to the Gemini API with a prompt requesting `JA:` / `EN:` formatted output.

### Concurrency

A pool of `--workers` background threads consumes from a bounded work queue. Each worker builds a prompt that includes up to 3 previous Japanese transcription lines as rolling context for continuity.

### Display

A live audio level meter is rendered in the terminal:

```
  [#########|                               ] 0.0082 * REC q=1
```

- `#` bars show current RMS level
- `|` marks the silence threshold
- `* REC` indicates speech is being recorded
- `q=N` shows pending items in the transcription queue

Completed transcriptions print above the meter:

```
  JA: こんにちは
  EN: Hello
------------------------------------------------------------
```

## Project Structure

```
live-stt/
├── live_stt.py          # Main application
├── list_live_models.py  # Utility: list Gemini models supporting the Live API
├── pyproject.toml       # Project metadata and dependencies
├── uv.lock              # Locked dependency versions
└── .venv/               # Virtual environment (not committed)
```

## Key Constants

Defined at the top of `live_stt.py` and tunable for different environments:

| Constant | Value | Purpose |
|----------|-------|---------|
| `SEND_RATE` | 16000 | Target sample rate sent to the API |
| `BLOCK_DURATION` | 0.1s | Size of each audio capture block |
| `SILENCE_DURATION` | 0.8s | Silence needed to trigger a send |
| `MIN_SPEECH_DURATION` | 0.3s | Minimum speech length to bother transcribing |
| `MAX_CHUNK_DURATION` | 5.0s | Force-send ceiling for continuous speech |
| `OVERLAP_DURATION` | 1.0s | Overlap retained between force-cut chunks |
| `NUM_WORKERS` | 5 | Default concurrent API threads |
| `CONTEXT_SIZE` | 3 | Rolling context window (previous transcriptions) |

## Utilities

### list_live_models.py

Lists Gemini models that support the bidirectional streaming API (`bidiGenerateContent`):

```sh
GEMINI_API_KEY="your-key" python list_live_models.py
```

## Development Notes

- The prompts `PROMPT_TRANSLATE` and `PROMPT_TRANSCRIBE` can be edited to support other source languages by changing the transcription instructions.
- The calibration step (`calibrate_threshold`) is defined but the main loop currently uses it only when `--threshold` is not provided and falls back to `0.0`—meaning all audio is treated as speech unless a manual threshold is set. To enable auto-calibration, call `calibrate_threshold(native_rate)` before the main loop (the function exists and works).
- Worker threads are daemonized and shut down automatically on `Ctrl+C`.
- The transcription queue has a bounded size of `workers * 2`. When full, new chunks are silently dropped to avoid unbounded memory growth.
