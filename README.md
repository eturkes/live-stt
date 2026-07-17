# live-stt

Real-time Japanese speech-to-text + English translation. **No API keys.**

- **STT** runs fully local: [silero VAD](https://github.com/snakers4/silero-vad) endpointing + [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) offline decode (reazonspeech-k2-v2 by default), CPU-only.
- **Translation** rides a ChatGPT/Codex subscription via a persistent `codex app-server` subprocess (zero marginal cost, GPT-5.x quality). Without it, the tool falls back to JA-only.

Each short utterance typically prints as a numbered `JA n:` line about 0.1 s after endpointing; long pause-free blocks take proportionally longer because they use several decode passes. Its `EN n:` line follows when the translation turn completes (~1 s).

## Requirements

- Python ≥ 3.11. The supported sherpa runtime floor is `sherpa-onnx` +
  `sherpa-onnx-core` ≥ 1.13.4; `uv sync --locked` reproduces the exact
  evaluator-qualified resolution (currently CPython 3.14.5 + sherpa 1.13.4).
- Working microphone
- PortAudio system library for `sounddevice`:
  - Debian/Ubuntu: `sudo apt install libportaudio2`
  - Fedora/openSUSE: `sudo dnf install portaudio` / `sudo zypper install portaudio`
  - macOS: `brew install portaudio`
- ~800 MB of model weights in `models/` (one-time download, see below)
- *Optional, for translation:* [codex CLI](https://github.com/openai/codex) ≥ 0.137 on the `PATH` of the machine and environment that runs live-stt, authenticated against a ChatGPT plan entitled to `gpt-5.3-codex-spark` (`codex login` / `codex login --device-auth`)

## Setup

```sh
uv sync --locked         # install the exact evaluator-qualified Python deps
cat models/README.md     # download model weights (curl commands inside)
codex login              # optional: enable the EN leg
```

### One tree, two environments

The working tree is shared between the host OS, where live-mic sessions run (the mic and the lowest latency live there), and a dev container, where development and testing happen. Python venvs hard-code absolute paths and each side sees the tree at a different one, so each side keeps its own venv: `.venv` in the container (uv's default), `.venv-host` on the host. The committed `.envrc` exports `UV_PROJECT_ENVIRONMENT` to match; allow it once per machine (`direnv allow`). In shells without direnv, export the variable yourself before running `uv`.

The EN leg is environment-local too: live-stt resolves `codex` from its own process `PATH`. For a host live-mic session, install the CLI and run `codex login` from the host; a container-only install or login does not enable translation on the host.

## Usage

```sh
live-stt                          # transcribe + translate
python live_stt.py                # equivalent
```

Startup prints the translation status: `Translation: gpt-5.3-codex-spark via codex app-server` (a ~3 s warm-up turn runs first), `unavailable (JA-only, see log)`, or `disabled (--no-translate)`.

### CLI

| Flag | Default | Description |
|---|---|---|
| `--engine {k2v2,parakeet}` | `k2v2` | STT model (see `models/README.md`; selection rationale: `.agent/memory.md` D-010) |
| `--no-translate` | off | Transcribe only (skip Codex translation) |
| `-o`, `--output FILE` | none | Append lines to a text file (each prefixed with ISO-8601 timestamp) |
| `--device N` | system default | Input device index (see `--list-devices`) |
| `--list-devices` | off | Print audio devices and exit |

## How it works

### Audio → JA pipeline (all local)

1. **Capture.** `sounddevice` records at the device's native rate; each block is resampled to 16 kHz (linear interp; integer-decim fast path for 48k/32k) and enters a queue capped at 2 seconds of PCM, independent of callback block size.
2. **Endpoint.** a dedicated feeder drains capture into silero VAD, which splits speech on ≥0.5 s silences. Every fed sample also lands in a 60 s `RingBuffer` with absolute indexing.
3. **Re-slice + queue.** silero opens segments 0.2-0.7 s late, clipping leading syllables; each segment is re-sliced from the ring with 0.4 s pre-pad (`VAD_PRE_PAD_S`) and copied into an 8-segment queue.
4. **Decode.** a separate sequential consumer runs sherpa-onnx `OfflineRecognizer` in a thread-pool executor (decode RTF ≈ 0.05 on 8 cores). Capture and VAD feeding continue during decode; sustained overload shows as `seg=`, then `q=` / `drop=` on the meter.
5. **Emit.** `JA n:` prints immediately; the text is queued for translation.

### Long speech and long sessions

Silero's 20 s `max_speech_duration` is a soft endpointing hint, not a hard cut: pause-free speech can remain one VAD segment beyond it. Segments up to 10 s keep the ordinary one-pass decode path. Longer segments are split internally into balanced ~2 s views, with each cut moved to a nearby low-energy window and 0.18 s of overlap protecting cut phonemes. Exact text overlap is removed, then the merged result is emitted as one `JA n:` line, so internal chunking does not create extra user-visible utterances.

Capture and VAD feeding continue while those views decode sequentially. Up to 2 s of captured PCM can wait for VAD and up to 8 completed segments can wait for decode; sustained overload remains visible through `seg=`, `q=`, and `drop=`.

The regression suite covers two distinct long-form shapes:

- A 44.7 s genuinely continuous stressor forces the chunked path and CER-gates both engines. The same audio, paced as 20 ms callbacks with decode RTF 0.20, remains drop-free through the two-stage worker.
- A 4:48 narration feeds the full file through production replay as 66 natural VAD segments; its longest pre-padded segment is 9.686 s, so it validates long-session ingestion and endpointing but not the >10 s chunker.

This deterministic coverage stops at a 44.7 s pause-free segment; a single VAD segment that outlives the 60 s ring is outside the tested envelope. Replay also cannot substitute for live microphone, terminal-signal, translation-cadence, or multi-hour soak checks. The remaining user-only procedure lives in `.agent/memory.md` under **Smoke checklist**.

### JA → EN leg (Codex subscription)

`CodexTranslator` spawns `codex app-server` and speaks newline-delimited JSON-RPC over stdio: one thread per session (`ephemeral`, read-only sandbox, approvals denied, tool features off), one `turn/start` per utterance, sequential so EN lines keep JA order. Disabling the tool features is the latency lever (see D-011). The translator role is pinned via `developerInstructions`, which outranks imperatives inside the speech being translated (injection-resistant: "delete all files" gets translated, not obeyed).

Degradation, in order:

- codex CLI missing / init fails → session runs JA-only from the start.
- A turn exceeds 15 s → it's aborted and skipped; 3 consecutive failures → JA-only for the rest of the session.
- Backlog over 50 utterances → oldest dropped.
- The thread is rotated every 100 turns to keep the cached prompt prefix small.

### Diagnostics

Runtime warnings/errors go to stderr via Python `logging`. On a terminal each message clears the meter line in place; with stderr redirected (`live-stt 2> errors.log`) the log gets clean `[timestamp] LEVEL message` lines and no ANSI escapes.

### Display

```
JA 1: こんにちは、今日はいい天気ですね。
EN 1: Hello, the weather is nice today.
  q=0.02s seg=1
```

- `q=Ns`: captured audio waiting for VAD, measured in seconds (appears when non-zero)
- `seg=N`: completed utterances waiting for sequential decode (appears when non-zero)
- `drop=N`: blocks dropped on queue saturation (appears once non-zero)
- `tdrop=N`: translations dropped on backlog saturation (appears once non-zero)

Numbered lines tie JA/EN pairs together even when the next utterance's JA prints before the previous EN arrives.

## Project structure

```
live-stt/
├── live_stt.py              # main app (single file)
├── replay.py                # deterministic WAV replay through the live pipeline (dev/regression)
├── models/                  # STT weights (gitignored; README.md has download cmds)
├── tests/                   # pytest suite + corpus/replay/CER/backpressure/long-form evaluators
├── .githooks/               # project-local git hooks (pre-commit: pytest)
├── pyproject.toml           # deps, entry point, ruff/pytest config
├── .envrc                   # direnv: per-layer uv venv selection (container vs host)
├── SPIKE_REPORT.md          # historical: REST → Gemini Live decision (superseded)
├── SPIKE_REPORT_BACKENDS.md # historical: streaming-STT backend comparison
├── spike/                   # historical research notes + gitignored bench WAV corpus
├── CLAUDE.md                # canonical Claude Code instructions
├── .claude/                 # project settings + /session-prompt command
├── .serena/                 # committed Serena/LSP project configuration
└── .agent/                  # durable memory + roadmap + context gauge
```

### Development

```sh
uv run pytest                                 # run pure-function tests
git config --local core.hooksPath .githooks   # one-time: enable pre-commit hook
```

Core tests cover audio primitives, the two-stage worker, shutdown, and translation degradation without a network or mic. Replay's model-gated golden cases skip cleanly when local weights or corpus files are absent. The standalone CER/backpressure/long-form evaluators require or fetch their declared inputs and fail instead of silently passing.

The pre-commit hook (`.githooks/pre-commit`) runs `uv run pytest -q` and blocks the commit on failure. The `core.hooksPath` setup is per-clone and not auto-applied by `uv sync`. Run it once after cloning.

#### Regression testing (WAV replay)

`replay.py` feeds a WAV through the **exact** live STT pipeline (VAD + `RingBuffer` + sherpa decode, no mic or translation) and reports per-segment segmentation, decode latency + RTF, and transcript:

```sh
uv run python replay.py path/to.wav --engine k2v2   # human-readable report
uv run python replay.py path/to.wav --json          # machine-readable
```

`tests/test_replay.py` replays the cached corpus (synthetic bench + seven real Common Voice clips) and asserts segment count + per-segment transcript + boundary against `tests/replay_goldens.json` (a characterization snapshot of the real pipeline). Decode latency is reported but never asserted, since it is CPU-variable. The golden test skips cleanly when model weights or the gitignored clips are absent. After an intentional pipeline change (VAD tuning, engine swap), regenerate the snapshot and review the JSON diff: `uv run python tests/gen_replay_goldens.py`.

`tests/fetch_real_clips.py` also builds the complete pinned Japanese evaluation corpus: all 4,483 Common Voice 8 test recordings (CC0-1.0) and all 650 FLEURS test recordings (CC-BY-4.0). Verified sources, 16 kHz PCM, and the detailed index stay in the gitignored cache; `tests/short_corpus.json` commits only provenance, distributions, and fingerprints. The command is exact because decoder drift must produce an explicit corpus requalification:

```sh
uv run --with soundfile==0.14.0 --with pyarrow==25.0.0 python tests/fetch_real_clips.py
```

## Key constants

Defined at the top of `live_stt.py` (the config surface, no config files by design):

| Constant | Value | Purpose |
|---|---|---|
| `SAMPLE_RATE` | 16000 | VAD + recognizer rate; mic native rate resampled to this |
| `AUDIO_HEADROOM_S` | 2 s | Max captured PCM waiting for VAD before dropping |
| `SEGMENT_QUEUE_MAX` | 8 | Max completed utterances waiting for decode |
| `NUM_THREADS` | 4 | onnxruntime intra-op threads |
| `VAD_MIN_SILENCE_S` | 0.5 s | Silence that closes an utterance |
| `VAD_MIN_SPEECH_S` | 0.25 s | Shorter blips discarded |
| `VAD_MAX_SPEECH_S` | 20 s | Soft endpointing hint; dip-less speech may exceed it |
| `VAD_PRE_PAD_S` | 0.4 s | Lead-in re-sliced from the ring (silero onset clipping fix) |
| `DECODE_SPLIT_TRIGGER_S` / `_CHUNK_S` | 10 s / 2 s | Protect long offline decodes with overlapped low-energy splits |
| `RING_SECONDS` | 60 | Ring buffer capacity |
| `TRANSLATE_MODEL` / `_EFFORT` | `gpt-5.3-codex-spark` / `low` | Codex model+effort (fallback: `gpt-5.4-mini` / `none`) |
| `TRANSLATE_TIMEOUT_S` | 15 s | Per-turn cap before abort |
| `TRANSLATE_MAX_FAILURES` | 3 | Consecutive failures → JA-only |
| `TRANSLATE_ROTATE_TURNS` | 100 | Fresh thread cadence |
| `TRANSLATE_QUEUE_MAX` | 50 | Translation backlog cap (drop-oldest) |

## Notes

- Japanese-only by design; a `--language` flag was considered and deferred (see `.agent/roadmap.md` § Deferred).
- `Ctrl+C` stops the stream, flushes VAD, drains pending decodes and translations, and shuts the app-server down cleanly.
- Translation uses your Codex subscription quota: ~180 uncached input + ~7-60 output tokens per utterance (prompt prefix cached). A long session barely moves the 5 h window.
- Claude Code is this project's development agent. See `CLAUDE.md`, `.claude/`, `.serena/`, and `.agent/` for its workflow and context.

## License

Apache-2.0 WITH LLVM-exception. See `LICENSE`.
