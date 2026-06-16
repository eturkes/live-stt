# live-stt

Real-time Japanese speech-to-text + English translation. **No API keys.**

- **STT** runs fully local: [silero VAD](https://github.com/snakers4/silero-vad) endpointing + [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) offline decode (reazonspeech-k2-v2 by default), CPU-only.
- **Translation** rides a ChatGPT/Codex subscription via a persistent `codex app-server` subprocess (zero marginal cost, GPT-5.x quality). Without it, the tool falls back to JA-only.

Each utterance prints as a numbered `JA n:` line the moment decoding ends (≤0.1 s after you stop speaking); its `EN n:` line follows when the translation turn completes (~1 s).

## Requirements

- Python ≥ 3.11 (developed on 3.14)
- Working microphone
- PortAudio system library for `sounddevice`:
  - Debian/Ubuntu: `sudo apt install libportaudio2`
  - Fedora/openSUSE: `sudo dnf install portaudio` / `sudo zypper install portaudio`
  - macOS: `brew install portaudio`
- ~800 MB of model weights in `models/` (one-time download, see below)
- *Optional, for translation:* [codex CLI](https://github.com/openai/codex) ≥ 0.137 on `PATH`, authenticated against a ChatGPT plan entitled to `gpt-5.3-codex-spark` (`codex login` / `codex login --device-auth`)

## Setup

```sh
uv sync                  # install Python deps
cat models/README.md     # download model weights (curl commands inside)
codex login              # optional: enable the EN leg
```

## Usage

```sh
live-stt                          # transcribe + translate
python live_stt.py                # equivalent
```

Startup prints the translation status: `Translation: gpt-5.3-codex-spark via codex app-server` (a ~3 s warm-up turn runs first), `unavailable (JA-only, see log)`, or `disabled (--no-translate)`.

### CLI

| Flag | Default | Description |
|---|---|---|
| `--engine {k2v2,parakeet}` | `k2v2` | STT model (see `models/README.md`; selection rationale: `.agent/decisions.md` D-010) |
| `--no-translate` | off | Transcribe only (skip Codex translation) |
| `-o`, `--output FILE` | none | Append lines to a text file (each prefixed with ISO-8601 timestamp) |
| `--device N` | system default | Input device index (see `--list-devices`) |
| `--list-devices` | off | Print audio devices and exit |

## How it works

### Audio → JA pipeline (all local)

1. **Capture.** `sounddevice` records at the device's native rate; each block is resampled to 16 kHz (linear interp; integer-decim fast path for 48k/32k) and enqueued onto an `asyncio.Queue`.
2. **Endpoint.** silero VAD splits speech on ≥0.5 s silences. Every fed sample also lands in a 60 s `RingBuffer` with absolute indexing.
3. **Re-slice.** silero opens segments 0.2-0.7 s late, clipping leading syllables; each segment is re-sliced from the ring with 0.4 s pre-pad (`VAD_PRE_PAD_S`).
4. **Decode.** sherpa-onnx `OfflineRecognizer` runs in a thread-pool executor (decode RTF ≈ 0.05 on 8 cores, so it never falls behind the mic).
5. **Emit.** `JA n:` prints immediately; the text is queued for translation.

### JA → EN leg (Codex subscription)

`CodexTranslator` spawns `codex app-server` and speaks newline-delimited JSON-RPC over stdio: one thread per session (`ephemeral`, read-only sandbox, approvals denied, tool features off), one `turn/start` per utterance, sequential so EN lines keep JA order. Disabling the tool features is the latency lever (see D-011). The translator role is pinned via `developerInstructions`, which outranks imperatives inside the speech being translated (injection-resistant: "delete all files" gets translated, not obeyed).

Degradation, in order:

- codex CLI missing / init fails → session runs JA-only from the start.
- A turn exceeds 15 s → it's aborted and skipped; 3 consecutive failures → JA-only for the rest of the session.
- Backlog over 50 utterances → oldest dropped.
- The thread is rotated every 100 turns to keep the cached prompt prefix small.

### Diagnostics

Runtime warnings/errors go to stderr via Python `logging`. On a terminal each message clears the level-meter line in place; with stderr redirected (`live-stt 2> errors.log`) the log gets clean `[timestamp] LEVEL message` lines and no ANSI escapes.

### Display

```
JA 1: こんにちは、今日はいい天気ですね。
EN 1: Hello, the weather is nice today.
  [#########                               ] 0.0082 q=1
```

- `#` bars: current RMS level
- `q=N`: pending audio blocks (appears when non-zero)
- `drop=N`: blocks dropped on queue saturation (appears once non-zero)

Numbered lines tie JA/EN pairs together even when the next utterance's JA prints before the previous EN arrives.

## Project structure

```
live-stt/
├── live_stt.py              # main app (single file)
├── replay.py                # deterministic WAV replay through the live pipeline (dev/regression)
├── models/                  # STT weights (gitignored; README.md has download cmds)
├── tests/                   # pytest suite (pure functions + replay regression)
├── .githooks/               # project-local git hooks (pre-commit: pytest)
├── pyproject.toml           # deps, entry point, ruff/pytest config
├── PLAN.md                  # roadmap
├── SPIKE_REPORT.md          # historical: REST → Gemini Live decision (superseded)
├── SPIKE_REPORT_BACKENDS.md # historical: streaming-STT backend comparison
├── spike/                   # historical research notes + gitignored bench WAV corpus
├── CLAUDE.md                # agent meta-instructions
└── .agent/                  # agent memory/notetaking
```

### Development

```sh
uv run pytest                                 # run pure-function tests
git config --local core.hooksPath .githooks   # one-time: enable pre-commit hook
```

Tests cover `resample`, `RingBuffer`, and `emit_line`. No network, mic, or model weights required.

The pre-commit hook (`.githooks/pre-commit`) runs `uv run pytest -q` and blocks the commit on failure. The `core.hooksPath` setup is per-clone and not auto-applied by `uv sync`. Run it once after cloning.

#### Regression testing (WAV replay)

`replay.py` feeds a WAV through the **exact** live STT pipeline (VAD + `RingBuffer` + sherpa decode, no mic or translation) and reports per-segment segmentation, decode latency + RTF, and transcript:

```sh
uv run python replay.py path/to.wav --engine k2v2   # human-readable report
uv run python replay.py path/to.wav --json          # machine-readable
```

`tests/test_replay.py` replays the cached corpus (synthetic bench + real Common Voice clips) and asserts segment count + per-segment transcript + boundary against `tests/replay_goldens.json` (a characterization snapshot of the real pipeline). Decode latency is reported but never asserted, since it is CPU-variable. The golden test skips cleanly when model weights or the gitignored clips are absent. After an intentional pipeline change (VAD tuning, engine swap), regenerate the snapshot and review the JSON diff: `uv run python tests/gen_replay_goldens.py`. The corpus mixes synthetic bench clips with real Common Voice clips (CC0); the latter are (re)fetched via `uv run --with soundfile python tests/fetch_real_clips.py`.

## Key constants

Defined at the top of `live_stt.py` (the config surface, no config files by design):

| Constant | Value | Purpose |
|---|---|---|
| `SAMPLE_RATE` | 16000 | VAD + recognizer rate; mic native rate resampled to this |
| `AUDIO_QUEUE_MAX` | 100 | Max buffered audio blocks before dropping |
| `NUM_THREADS` | 4 | onnxruntime intra-op threads |
| `VAD_MIN_SILENCE_S` | 0.5 s | Silence that closes an utterance |
| `VAD_MIN_SPEECH_S` | 0.25 s | Shorter blips discarded |
| `VAD_PRE_PAD_S` | 0.4 s | Lead-in re-sliced from the ring (silero onset clipping fix) |
| `RING_SECONDS` | 60 | Ring buffer capacity |
| `TRANSLATE_MODEL` / `_EFFORT` | `gpt-5.3-codex-spark` / `low` | Codex model+effort (fallback: `gpt-5.4-mini` / `none`) |
| `TRANSLATE_TIMEOUT_S` | 15 s | Per-turn cap before abort |
| `TRANSLATE_MAX_FAILURES` | 3 | Consecutive failures → JA-only |
| `TRANSLATE_ROTATE_TURNS` | 100 | Fresh thread cadence |
| `TRANSLATE_QUEUE_MAX` | 50 | Translation backlog cap (drop-oldest) |

## Notes

- Japanese-only by design; a `--language` flag was considered and deferred (see `PLAN.md` § Deferred).
- `Ctrl+C` stops the stream, flushes any in-flight VAD segment, waits for pending translations, and shuts the app-server down cleanly.
- Translation uses your Codex subscription quota: ~180 uncached input + ~7-60 output tokens per utterance (prompt prefix cached). A long session barely moves the 5 h window.
- This project's primary developers are AI agents. See `CLAUDE.md` and `.agent/` for context on how it's maintained.

## License

Apache-2.0 WITH LLVM-exception. See `LICENSE`.
