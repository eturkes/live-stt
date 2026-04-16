# live-stt

Real-time Japanese speech-to-text transcription with English translation, powered by the Gemini Live API.

Streams microphone audio over a persistent bidirectional WebSocket session to Gemini and prints the Japanese transcription alongside the English translation as you speak.

## Requirements

- Python >= 3.11 (developed on 3.14)
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

# Set your API key (loaded automatically from .env)
echo 'GEMINI_API_KEY=your-key-here' > .env
```

## Usage

```sh
# Run with defaults
live-stt

# Or run directly
python live_stt.py
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model MODEL` | `gemini-3.1-flash-live-preview` | Gemini Live model (must support `bidiGenerateContent`) |
| `--no-translate` | off | Transcribe only (no English translation) |
| `-o`, `--output FILE` | none | Append transcriptions to a text file |

### Examples

```sh
# Transcribe + translate with the default Live model
live-stt

# Transcribe only (no English translation)
live-stt --no-translate

# Save transcriptions to a file
live-stt --no-translate -o transcript.txt
```

## How It Works

### Audio Pipeline

1. **Capture** - `sounddevice` records from the default input device at its native sample rate, in 100 ms blocks.
2. **Resample** - Each block is downsampled to 16 kHz via linear interpolation and converted to 16-bit PCM.
3. **Stream** - PCM bytes are pushed onto an asyncio queue and sent to Gemini over a single persistent Live API session via `send_realtime_input`.
4. **Transcribe** - A system instruction configures the model as a live interpreter. Gemini's native voice-activity detection decides turn boundaries; `output_audio_transcription` returns text for each utterance.
5. **Display** - Complete JA / EN blocks are printed on `turn_complete`.

### Session Model

`run_session` runs an outer reconnect loop around `client.aio.live.connect(...)`. Each connection owns a per-session `TaskGroup` with two tasks:

- **sender** – drains the audio queue and forwards PCM chunks to Gemini.
- **receiver** – consumes `LiveServerMessage` events, accumulates output-transcription deltas, emits each turn as a block, and handles `go_away` / session-resumption updates.

One long-lived **meter** task sits outside the TaskGroup and refreshes the terminal level meter every 100 ms across reconnects.

The config passed to every `connect()` enables:

- `SessionResumptionConfig(transparent=True)` — the server issues a resumption handle (`new_handle`) as state accumulates. The client stores it and passes it back on the next connect, so conversation context carries across reconnects. Handles are valid for ~2 hours.
- `ContextWindowCompressionConfig(sliding_window=SlidingWindow())` — lifts the 15-minute audio-only session cap. The server truncates oldest user turns when the context window fills; the system instruction is exempt.

On `go_away` (sent ~60 s before the server will disconnect) or an unexpected close, the receiver flips `state.should_reconnect` and the outer loop reconnects with the stored handle. Audio keeps flowing into the bounded send-queue during the reconnect gap — up to ~10 s of buffered speech survives the swap.

Native-audio Live models return the `AUDIO` modality; we read `output_audio_transcription.text` and discard the audio bytes. The in-session conversation history is maintained server-side; across reconnects, it's preserved by the resumption handle.

### Display

A live audio level meter is rendered in the terminal:

```
  [#########                               ] 0.0082 * LIVE q=1
```

- `#` bars show current RMS level
- `* LIVE` indicates the session is connected; `* RECONNECT` during the reconnect gap
- `rc=N` shows cumulative reconnect count (appears once non-zero)
- `q=N` shows pending audio chunks in the send queue
- `drop=N` appears if chunks were dropped (queue saturation)

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
| `SEND_RATE` | 16000 | Target sample rate streamed to the Live API |
| `BLOCK_DURATION` | 0.1s | Size of each audio capture block |
| `METER_INTERVAL` | 0.1s | Level-meter refresh rate |
| `AUDIO_QUEUE_MAX` | 100 | Max buffered 100 ms blocks before dropping (≈10 s) |
| `RECONNECT_BACKOFF_S` | 1.0s | Delay between reconnect attempts after a session closes |

## Utilities

### list_live_models.py

Lists Gemini models that support the bidirectional streaming API (`bidiGenerateContent`):

```sh
python list_live_models.py
```

## Development Notes

- The system instructions `SYSTEM_INSTRUCTION_TRANSLATE` / `SYSTEM_INSTRUCTION_TRANSCRIBE` can be edited to support other source languages.
- `Ctrl+C` triggers a signal handler that flips `state.stopping`, drains the audio queue, sends `audio_stream_end=True`, and exits the reconnect loop cleanly.
- Audio-only Live sessions cap at 15 minutes of wall-clock per connection, and the underlying WebSocket times out at ~10 minutes. The reconnect loop + `SessionResumptionConfig` + `ContextWindowCompressionConfig` together lift both limits — sessions can run indefinitely while preserving conversation context across reconnects.
- Resumption handles are valid for ~2 hours. After that, a reconnect starts fresh (conversation history lost) but transcription continues.
- Gemini's native-audio Live models only emit the `AUDIO` response modality. Text arrives via `output_audio_transcription`, but audio-output tokens are still billed (~$0.018/min at list price).
