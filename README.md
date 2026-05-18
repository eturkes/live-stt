# live-stt

Real-time Japanese speech-to-text + English translation, powered by the [Gemini Live API](https://ai.google.dev/gemini-api/docs/live).

Streams microphone audio over a persistent bidirectional WebSocket to Gemini and prints JA transcription alongside EN translation as you speak.

## Requirements

- Python ≥ 3.11 (developed on 3.14)
- Working microphone
- PortAudio system library for `sounddevice`:
  - Debian/Ubuntu: `sudo apt install libportaudio2`
  - Fedora/openSUSE: `sudo dnf install portaudio` / `sudo zypper install portaudio`
  - macOS: `brew install portaudio`
- [Gemini API key](https://aistudio.google.com/apikey)

## Setup

```sh
uv sync                                       # install deps (uses uv)
echo 'GEMINI_API_KEY=your-key-here' > .env    # set API key
```

## Usage

```sh
live-stt                          # default: transcribe + translate
python live_stt.py                # equivalent
```

### CLI

| Flag | Default | Description |
|---|---|---|
| `--model MODEL` | `gemini-3.1-flash-live-preview` | Gemini Live model (must support `bidiGenerateContent`) |
| `--no-translate` | off | Transcribe only (skip English translation) |
| `-o`, `--output FILE` | none | Append blocks to a text file (each prefixed with ISO-8601 timestamp) |
| `--device N` | system default | Input device index (see `--list-devices`) |
| `--list-devices` | off | Print audio devices and exit |

### Examples

```sh
live-stt --no-translate                       # transcribe only
live-stt --no-translate -o transcript.txt     # write to file
live-stt --list-devices                       # see devices
live-stt --device 3                           # use device #3
```

## How it works

### Audio pipeline

1. **Capture** — `sounddevice` records from the chosen input at native sample rate; PortAudio picks block size.
2. **Resample** — Each block is downsampled to 16 kHz (linear interp; integer-decim fast path for 48k/32k) and converted to PCM16.
3. **Stream** — PCM bytes enqueued onto an `asyncio.Queue` and sent over a single persistent Live session via `send_realtime_input`.
4. **Transcribe** — System instruction configures the model as a live interpreter. Gemini's native VAD decides turn boundaries; `output_audio_transcription` returns text per utterance.
5. **Display** — Complete JA/EN blocks print on `turn_complete`.

### Session model

`run_session()` runs an outer reconnect loop around `client.aio.live.connect(...)`. Each connection owns an `asyncio.TaskGroup` with:

- **sender** — drains the audio queue, forwards PCM to Gemini (with coalescing when the queue has backed up after reconnect).
- **receiver** — consumes `LiveServerMessage` events, accumulates `output_transcription` deltas, emits each turn as a block, handles `go_away` / session-resumption updates.

A long-lived **meter** task sits outside the TaskGroup so the terminal-level meter survives reconnects.

Live connect config enables:

- `SessionResumptionConfig(handle=handle)` — server issues a resumption handle as state accumulates; client passes it back on the next connect, preserving conversation context across reconnects (~2 h handle TTL).
- `ContextWindowCompressionConfig(sliding_window=SlidingWindow())` — lifts the 15-min audio-only session cap. Oldest user turns are truncated when context fills; system instruction is exempt.

On `go_away` (sent ~60 s pre-disconnect) or an unexpected close, the receiver flips `state.should_reconnect` and the outer loop reconnects with the stored handle. Up to ~10 s of buffered audio survives the swap (bounded by `AUDIO_QUEUE_MAX`).

Native-audio Live models return the `AUDIO` modality; we read `output_audio_transcription.text` and discard the audio bytes. Audio-output tokens are billed regardless (~$0.018/min at list price).

### Display

Live audio level meter:

```
  [#########                               ] 0.0082 * LIVE q=1
```

- `#` bars: current RMS level
- `* LIVE` / `* RECONNECT`: connection status
- `rc=N`: cumulative reconnect count (appears once non-zero)
- `q=N`: pending audio chunks in send queue
- `drop=N`: chunks dropped on queue saturation (appears once non-zero)

Completed blocks print above the meter:

```
  JA: こんにちは
  EN: Hello
------------------------------------------------------------
```

## Project structure

```
live-stt/
├── live_stt.py              # main app
├── list_live_models.py      # list Gemini Live-capable models
├── tests/                   # pytest suite for pure functions
├── .githooks/               # project-local git hooks (pre-commit: pytest)
├── pyproject.toml           # deps, entry points, ruff/pytest config
├── uv.lock                  # locked deps
├── PLAN.md                  # roadmap
├── SPIKE_REPORT.md          # T3.1 (REST → Live) decision record
├── SPIKE_REPORT_BACKENDS.md # alternative backends comparison
├── spike/                   # spike prototypes + research
├── CLAUDE.md                # agent meta-instructions
└── .agent/                  # agent memory/notetaking
```

### Development

```sh
uv run pytest                                 # run pure-function tests
git config --local core.hooksPath .githooks   # one-time: enable pre-commit hook
```

Tests cover pure audio helpers (`resample`, `pcm16_bytes`) and the JA/EN parsing in `emit_block`. No network or mic required.

The pre-commit hook (`.githooks/pre-commit`) runs `uv run pytest -q` and blocks the commit on failure. The `core.hooksPath` setup is per-clone and not auto-applied by `uv sync` — run it once after cloning.

## Key constants

Defined at the top of `live_stt.py`:

| Constant | Value | Purpose |
|---|---|---|
| `SEND_RATE` | 16000 | Target sample rate streamed to Live API |
| `METER_INTERVAL` | 0.1 s | Level-meter refresh rate |
| `AUDIO_QUEUE_MAX` | 100 | Max buffered audio blocks before dropping |
| `RECONNECT_BACKOFF_MIN_S` | 1.0 s | Initial reconnect delay (doubles per failure) |
| `RECONNECT_BACKOFF_MAX_S` | 30.0 s | Cap on reconnect delay |
| `RECONNECT_RESET_AFTER_S` | 10.0 s | Session alive at least this long resets backoff |

## Utilities

```sh
python list_live_models.py    # list models supporting bidiGenerateContent
```

## Notes

- `SYSTEM_INSTRUCTION_TRANSLATE` / `SYSTEM_INSTRUCTION_TRANSCRIBE` are editable. The tool is Japanese-only by design; a `--language` flag was considered and deferred (see `PLAN.md` § Deferred).
- `Ctrl+C` flips `state.stopping`, drains the queue, sends `audio_stream_end=True`, exits the reconnect loop cleanly.
- Audio-only Live sessions cap at 15 min wall-clock per connection; WS times out at ~10 min. Reconnect + `SessionResumptionConfig` + `ContextWindowCompressionConfig` together lift both — sessions run indefinitely with context preserved.
- Resumption handles valid ~2 h. After expiry, reconnect starts fresh (history lost) but transcription continues.
- This project's primary developers are AI agents. See `CLAUDE.md` and `.agent/` for context on how it's maintained.

## License

Apache-2.0 WITH LLVM-exception. See `LICENSE`.
