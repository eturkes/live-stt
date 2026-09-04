# live-stt

Real-time Japanese speech-to-text + English translation. **No API keys.**

- **STT** runs fully local: [silero VAD](https://github.com/snakers4/silero-vad) controls a streaming decode of [Whisper](https://github.com/openai/whisper) large-v3-turbo (INT8) on the Intel NPU through OpenVINO. Japanese appears while you are still speaking, about 2.5 s behind the voice. The [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) CPU engines stay available with `--engine`.
- **Translation** rides a ChatGPT/Codex subscription via a persistent `codex app-server` subprocess (zero marginal cost, GPT-5.x quality). Without it, the tool falls back to JA-only.

On the default engine, Japanese builds on the status line while you speak. The numbered `JA n:` line lands when you stop, and its `EN n:` line follows about 1 s later. The sherpa engines show no partial text: they wait for the pause, then print `JA n:` about 0.6 s after you stop.

## Requirements

- Python ≥ 3.11. `uv sync --locked` reproduces the exact qualified resolution
  (currently CPython 3.14.5).
- Working microphone
- An OpenVINO device for the default Whisper engine. `uv sync` installs
  OpenVINO as a hard dependency. The default device is the Intel NPU; pass
  `--asr-device GPU` or `--asr-device CPU` to use another one. OpenVINO ships
  wheels only for CPython 3.11 to 3.14 on macOS arm64, Linux x86_64 and
  aarch64, and Windows amd64. If your shell sets `PYTHONPATH` to a different
  OpenVINO build, clear it first. That entry hides the installed package. The
  OpenVINO import then fails with a message that does not name `PYTHONPATH`.
  The committed `.envrc` clears `PYTHONPATH` when direnv is active.
- PortAudio system library for `sounddevice`:
  - Debian/Ubuntu: `sudo apt install libportaudio2`
  - Fedora/openSUSE: `sudo dnf install portaudio` / `sudo zypper install portaudio`
  - macOS: `brew install portaudio`
- ~790 MB of model weights in `models/` for the default engine (one-time
  download, see below). Each sherpa fallback engine adds its own weights, and
  needs `sherpa-onnx` + `sherpa-onnx-core` ≥ 1.13.4.
- *Optional, for translation:* [codex CLI](https://github.com/openai/codex) ≥ 0.137 on the `PATH` of the machine and environment that runs live-stt, authenticated against a ChatGPT plan entitled to `gpt-5.6-luna` (`codex login` / `codex login --device-auth`)

## Setup

```sh
uv sync --locked         # install the exact qualified Python deps, OpenVINO included
                         # then download model weights: see models/README.md
codex login              # optional: enable the EN leg
```

### One tree, two environments

The working tree is shared between the host OS, where live-mic sessions run (the mic and the lowest latency live there), and a dev container, where development and testing happen. Python venvs hard-code absolute paths and each side sees the tree at a different one, so each side keeps its own venv: `.venv` in the container (uv's default), `.venv-host` on the host. The committed `.envrc` exports `UV_PROJECT_ENVIRONMENT` to match. Run `direnv allow` on each machine, and again after the file changes. In shells without direnv, export the variable yourself before you run `uv`, and clear `PYTHONPATH` as above.

The EN leg is environment-local too: live-stt resolves `codex` from its own process `PATH`. For a host live-mic session, install the CLI and run `codex login` from the host; a container-only install or login does not enable translation on the host.

## Usage

```sh
live-stt                          # transcribe + translate
python live_stt.py                # equivalent
```

Startup prints the translation status: `Translation: gpt-5.6-luna via codex app-server` (a ~3 s warm-up turn runs first), `unavailable (JA-only, see log)`, or `disabled (--no-translate)`.

### CLI

| Flag | Default | Description |
|---|---|---|
| `--engine {k2v2,parakeet,whisper}` | `whisper` | STT model (see `models/README.md`). `whisper` streams partial text; the sherpa engines decode each closed utterance. Rationale: `.agent/memory.md` D-016 (default), D-010 (sherpa pair) |
| `--asr-device DEV` | `NPU` | OpenVINO device for `--engine whisper`. `GPU` or `CPU` also enable session term biasing, which the NPU rejects. |
| `--context TEXT` | empty | Japanese topic line for this session. Name anything that must be spelled correctly. The tool trusts these terms at once and keeps them for the whole run. It also learns recurring terms from its own captions, and gives both to the recognizer and to the translator. Everything is forgotten when the session ends. |
| `--no-translate` | off | Transcribe only (skip Codex translation) |
| `-o`, `--output FILE` | new file in `transcripts/` | Append lines to this file instead of the session file |
| `--no-save` | off | Do not save the transcript to disk (conflicts with `-o`) |
| `--device N` | system default | Input device index (see `--list-devices`) |
| `--list-devices` | off | Print audio devices and exit |

### Saved transcripts

live-stt saves every session by default. It writes one file per run to `transcripts/`, named by the start time, and prints that path at startup:

```
Transcript: /home/you/Projects/live-stt/transcripts/2026-08-31T13-40-55.txt

[2026-08-31T13:40:58+09:00] JA 1: 今日はいい天気ですね
[2026-08-31T13:41:00+09:00] EN 1: The weather is nice today.
```

Each line holds an ISO-8601 timestamp and the same `n` as the terminal line, so JA and EN pairs stay matched. One file per run keeps that numbering unambiguous. Every line is flushed as it lands, so a killed session keeps what it already transcribed. The file is created with the first transcribed line, so a session that decodes nothing leaves no file behind.

`transcripts/` is gitignored. To write somewhere else, use `-o FILE`. To keep a session off disk, use `--no-save`.

## How it works

### Audio → JA pipeline (all local)

1. **Capture.** `sounddevice` records at the device's native rate; each block is resampled to 16 kHz (linear interp; integer-decim fast path for 48k/32k) and enters a queue capped at 2 seconds of PCM, independent of callback block size.
2. **Endpoint.** Capture drains into silero VAD, which splits speech on ≥0.5 s silences. Every fed sample also lands in a 60 s `RingBuffer` with absolute indexing.
3. **Re-slice.** silero opens segments 0.2-0.7 s late, clipping leading syllables. Both engine paths re-slice from the ring with 0.4 s pre-pad (`VAD_PRE_PAD_S`) to recover the lead-in. The sherpa path then copies each closed segment into an 8-segment queue; the whisper path has no such queue.
4. **Decode.** On `--engine whisper`, silero controls one growing buffer instead of closing segments. Speech-start opens the buffer, each further second of audio re-decodes the whole of it, and a character is committed once two consecutive decodes agree on it (LocalAgreement-2, `VAC_CHUNK_S`). Committed text appears on the status line as it lands and is never rewritten; the numbered `JA n:` line follows at the end of the utterance. The buffer is trimmed against fully-decoded spans (`VAC_TRIM_S`), which capped it at 11.2 s on the measured clips. Decode RTF is 0.48-0.61 on the NPU. The sherpa engines instead decode each closed VAD segment in one pass (RTF ≈ 0.05 on 8 cores) and emit no partial text.

   The two paths hold the real-time line differently. On the sherpa engines a separate feeder keeps capture and VAD running through each decode. The whisper path has no such feeder: it waits for every decode, and capture buffers into the 2 s queue meanwhile. Measured on the NPU, the longest single wait was 1.006 s over 182 s of pause-free speech, and nothing was dropped. Sustained overload shows as `seg=` (sherpa only), then `q=` and `drop=` on the meter.
5. **Screen.** A caption the recognizer invented is dropped whole, before anything else sees it. See the next section.
6. **Emit.** `JA n:` prints immediately; the text is queued for translation.

### Captions the recognizer invented

The recognizer is pinned to Japanese. Audio that it cannot account for therefore still comes back as Japanese text. It arrives in two shapes.

The first shape is a loop: one short unit, repeated until the model reaches its own length limit. Long silence and spoken English both cause it. One live caption reached 714 characters against a caption median of 19. Across four live sessions the loops were 15-31 % of every Japanese character printed, which scrolls the conversation out of the terminal. The second shape is spoken English, transcribed as English.

Two defences apply, in order:

- **Decode.** Every decode carries `ASR_REPETITION_PENALTY`. This is the only repetition control that the NPU honors. It accepts `no_repeat_ngram_size` and then ignores it silently. On an English clip the penalty cuts a 528-character loop to zero, and it costs 3 substitutions in 1166 characters of the retention corpus.
- **Publication.** A caption that still shows a defect is dropped whole. The tool does not print it, save it, number it, or translate it. It keeps its line numbers dense, so `JA 7` is always the seventh caption you spoke. One warning names the reason, and the meter counts the caption as `skip=`.

A caption is dropped when either rule matches:

- 40 or more of its characters are one unit of at most 8 characters, repeated back to back.
- Its Latin letters outnumber its Japanese characters by more than `CAPTION_LATIN_RATIO` to 1.

Both bounds sit in an empty gap in 1073 live captions, which the rules drop 4.0 % of. The smallest looped caption carries 252 repeated characters and the longest survivor carries 32. A Latin letter is one phoneme where a Japanese character is a whole syllable, so one loanword outnumbers the kana around it: at 1 to 1 the second rule reads `Discordで送ります。` as English. The 18 spoken-English captions stay below 0.15 Japanese characters per character, and the 6 Japanese captions that carry loanwords stay above 0.27.

### Long speech and long sessions

Silero's 20 s `max_speech_duration` is a soft endpointing hint, not a hard cut: pause-free speech can remain one VAD segment beyond it. Offline decoders drop content wholesale once a single segment passes roughly 15-20 s, so each engine path bounds what it hands the model.

The sherpa engines bound it after the fact. Segments up to 10 s keep the ordinary one-pass decode path. Longer segments are split internally into balanced ~2 s views, with each cut moved to a nearby low-energy window and 0.18 s of overlap protecting cut phonemes. Exact text overlap is removed, then the merged result is emitted as one `JA n:` line, so internal chunking does not create extra user-visible utterances. Capture and VAD feeding continue while those views decode sequentially. Up to 2 s of captured PCM can wait for VAD and up to 8 completed segments can wait for decode; sustained overload remains visible through `seg=`, `q=`, and `drop=`.

The whisper path bounds it by construction and never uses that splitter: the streaming buffer is trimmed against fully-decoded spans as it goes, so the model always sees a bounded buffer rather than one long closed segment.

The regression suite covers three distinct long-form shapes:

- A 44.7 s genuinely continuous stressor forces the chunked path and CER-gates both engines. The same audio, paced as 20 ms callbacks with decode RTF 0.20, remains drop-free through the two-stage worker.
- The shipped whisper path is paced on real NPU decode costs, one per streaming update, recorded from both pause-free clips. Both replays drop nothing: the queue peaks at 0.760 s and 1.060 s of the 2 s headroom, and no trim discards un-emitted audio. The same replays drop once every decode is slowed by 1.5x, which is the margin the measurement leaves.
- A 14:08 narration in six pinned sections feeds each full file through production replay. The six sections give 213 natural VAD segments, and the longest pre-padded segment of any of them is 9.686 s. The corpus therefore validates long-session ingestion and endpointing, but not the >10 s chunker.

Deterministic coverage now reaches 182 s of pause-free audio on the shipped path and 44.7 s on the sherpa path. A VAD segment that outlives the 60 s ring stays outside the tested envelope. Replay also cannot substitute for live microphone, terminal-signal, translation-cadence, or multi-hour soak checks. The remaining user-only procedure lives in `.agent/memory.md` under **Smoke checklist**.

### JA → EN leg (Codex subscription)

`CodexTranslator` spawns `codex app-server` and speaks newline-delimited JSON-RPC over stdio: one thread per session (`ephemeral`, read-only sandbox, approvals denied, tool features off), one `turn/start` per utterance, sequential so EN lines keep JA order. Disabling the tool features is the latency lever (see D-011). Each thread also asks for Codex's "Fast" service tier (`serviceTier: "priority"`, 1.5x speed at higher quota burn), so live-stt gets it without changing your global `~/.codex/config.toml`; the server echoes the tier it applied, and a tier it does not recognize is dropped silently, so a mismatch logs one warning and translation continues at the account default. The translator role is pinned via `developerInstructions`, which outranks imperatives inside the speech being translated (injection-resistant: "delete all files" gets translated, not obeyed). Two of those instructions target defects measured on clinical Japanese: Japanese brand-name drugs come back as the international generic (プレドニン as prednisolone, not the different molecule "prednisone") with the dose and schedule untouched, and the English never invents a patient's sex the Japanese did not state.

The translator declines a repeated caption independently, as a backstop to the publication screen above. Such a caption makes the model generate without ever stopping: a run of one character never finished a turn at 120 characters, while 480 characters of real speech took 7 s. The rule is repetition, not length, and it is the same rule and the same threshold. One warning names the reason, and the meter counts the caption as `tskip=`. On the shipped path the screen drops those captions first, so `tskip=` stays at 0 unless a caption reaches the queue by another route.

Degradation, in order:

- codex CLI missing / init fails → session runs JA-only from the start.
- A caption that repeats one short unit for 40 characters or more → not translated, and no failure is counted.
- A turn exceeds 15 s → it's aborted and skipped; 3 consecutive failures → JA-only for the rest of the session.
- Backlog over 50 utterances → oldest dropped.
- The thread is rotated every 100 turns to keep the cached prompt prefix small.

### Diagnostics

Runtime warnings/errors go to stderr via Python `logging`. On a terminal each message clears the meter line in place; with stderr redirected (`live-stt 2> errors.log`) the log gets clean `[timestamp] LEVEL message` lines and no ANSI escapes.

### Display

```
JA 1: こんにちは、今日はいい天気ですね。
EN 1: Hello, the weather is nice today.
  q=0.02s   では次の議題に
```

The last line is the status line. It rewrites itself in place and holds two things: the backlog counters, then the partial caption of the utterance you are still speaking. Only the default engine produces that caption. Committed characters never change once they appear, so the caption only grows until the utterance ends and becomes the next `JA n:` line. A long caption is truncated from the left to fit the terminal. The whole status line is written only to a terminal, so a redirected stdout stays clean.

- `q=Ns`: captured audio waiting for VAD, measured in seconds (appears when non-zero)
- `seg=N`: completed utterances waiting for sequential decode (sherpa engines only; appears when non-zero)
- `skip=N`: captions dropped as loops or as spoken English (appears once non-zero)
- `drop=N`: blocks dropped on queue saturation (appears once non-zero)
- `tdrop=N`: translations dropped on backlog saturation (appears once non-zero)
- `tskip=N`: captions declined as repetition loops and not translated (appears once non-zero)

Numbered lines tie JA/EN pairs together even when the next utterance's JA prints before the previous EN arrives.

## Project structure

```
live-stt/
├── live_stt.py              # main app (single file)
├── streaming.py             # streaming hypothesis buffer for the default engine
├── replay.py                # deterministic WAV replay through the live pipeline (dev/regression)
├── cer.py                   # character-error-rate scorer shared by the evaluators
├── gate.py                  # the quality gate (dev tool, not shipped in the wheel)
├── models/                  # STT weights (gitignored; README.md has download cmds)
├── transcripts/             # saved sessions, one file per run (gitignored, created on first line)
├── tests/                   # pytest suite + corpus/replay/CER/backpressure/long-form/retention evaluators
├── .githooks/               # project-local git hooks (pre-commit: pytest)
├── pyproject.toml           # deps, entry point, ruff/pytest config
├── .envrc                   # direnv: per-layer uv venv selection (container vs host)
├── spike/                   # gitignored bench WAV corpus (D-014 replay/test); superseded spike docs pruned
├── CLAUDE.md                # canonical Claude Code instructions
├── .claude/                 # project settings + session slash commands
├── .serena/                 # committed Serena/LSP project configuration
└── .agent/                  # durable memory + roadmap + polish register + closed-milestone archive
```

### Development

```sh
uv run python gate.py                         # the quality gate (tests, lint, format, typecheck, import)
uv run pytest                                 # just the test suite
git config --local core.hooksPath .githooks   # one-time: enable pre-commit hook
```

`gate.py` owns the exact step set, so the gate cannot be shortened by describing it differently. Every step is fast and hermetic: it needs no weights, no accelerator, and no network. Run one step with `--only NAME`.

Core tests cover audio primitives, the streaming buffer, the two-stage worker, shutdown, and translation degradation without a network or mic. Replay's model-gated golden cases skip cleanly when local weights, the accelerator, or corpus files are absent.

The evaluators under `tests/` are separate and deliberately outside the gate. Run one when a decode or streaming change puts its number in question. The first four need the gitignored weights and minutes of compute. The last two replay a committed trace instead, so they need no weights and no accelerator and finish in a second.

```sh
uv run python tests/eval_cer.py               # 2-engine CER + RTF over the short corpus and stressors
uv run --with soundfile python tests/eval_long_form.py  # build the pinned 14:08 narration (--score adds CER)
uv run python tests/eval_backpressure.py      # paced replay: bounded queues, drop-free
uv run python tests/eval_retention.py         # shipped path over 182 s of pause-free speech
uv run python tests/eval_vac_lag.py           # per-character caption lag of the streaming path
uv run python tests/eval_term_census.py       # what the recognizer gives session context as a key
```

Each one requires or fetches its declared inputs and fails instead of silently passing. `eval_retention.py` additionally checks the probe WAV against its committed hash first, so its number always belongs to the pinned input.

The pre-commit hook (`.githooks/pre-commit`) runs `uv run pytest -q` and blocks the commit on failure. The `core.hooksPath` setup is per-clone and not auto-applied by `uv sync`. Run it once after cloning.

#### Regression testing (WAV replay)

`replay.py` feeds a WAV through the **exact** live STT pipeline (VAD + `RingBuffer` + decode, no mic or translation) and reports per-segment segmentation, decode latency + RTF, and transcript. The engine you pass picks the path, exactly as it does live: `--engine whisper` drives the shipped streaming path, the sherpa engines drive the closed-segment path.

```sh
uv run python replay.py path/to.wav                 # human-readable report
uv run python replay.py path/to.wav --engine whisper  # the shipped streaming path
uv run python replay.py path/to.wav --json          # machine-readable
```

Note that `replay.py` defaults to `--engine k2v2`, which is **not** live-stt's default. The goldens key on the deterministic CPU engines, so that is the default the tool inherits.

`tests/test_replay.py` replays the cached corpus and asserts segment count + per-segment transcript + boundary against `tests/replay_goldens.json` (a characterization snapshot of the real pipeline), currently 25 rows. Most rows cover the sherpa engines, which reproduce on any CPU. One row covers the whisper path and records the device it was produced on (`NPU`), because that path reproduces per device rather than everywhere. Decode latency is reported but never asserted, since it is CPU-variable. The golden test skips cleanly when model weights, the accelerator, or the gitignored clips are absent. After an intentional pipeline change (VAD tuning, engine swap), regenerate the snapshot and review the JSON diff: `uv run python tests/gen_replay_goldens.py`.

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
| `ASR_DEVICE` | `NPU` | OpenVINO device for the default engine (`--asr-device` overrides) |
| `ASR_REPETITION_PENALTY` | 1.2 | Decode-side loop brake; the only repetition knob the NPU honors |
| `VAC_CHUNK_S` | 1 s | New audio between streaming re-decodes (default engine) |
| `VAC_TRIM_S` | 8 s | Past this, the streaming buffer commits finished spans and trims them away |
| `DECODE_SPLIT_TRIGGER_S` / `_CHUNK_S` | 10 s / 2 s | Protect long offline decodes with overlapped low-energy splits (sherpa engines) |
| `RING_SECONDS` | 60 | Ring buffer capacity |
| `TRANSLATE_MODEL` / `_EFFORT` | `gpt-5.6-luna` / `low` | Codex model+effort (runner-up: `gpt-5.6-terra` / `medium`) |
| `TRANSLATE_SERVICE_TIER` | `priority` | Codex "Fast" tier, requested per thread (`"default"` for the standard tier) |
| `TRANSLATE_TIMEOUT_S` | 15 s | Per-turn cap before abort |
| `TRANSLATE_MAX_FAILURES` | 3 | Consecutive failures → JA-only |
| `TRANSLATE_ROTATE_TURNS` | 100 | Fresh thread cadence |
| `TRANSLATE_QUEUE_MAX` | 50 | Translation backlog cap (drop-oldest) |
| `CAPTION_REPEAT_MAX_CHARS` | 40 | Repetition span that drops a caption before it is published |
| `CAPTION_REPEAT_UNIT_CHARS` | 8 | Longest repeated unit that screen counts as a decode loop |
| `CAPTION_LATIN_RATIO` | 4 | Latin letters per Japanese character above which a caption is English |

## Notes

- Japanese-only by design; a `--language` flag was considered and deferred (see `.agent/roadmap.md` § Deferred).
- `Ctrl+C` stops the stream, flushes VAD, drains pending decodes and translations, and shuts the app-server down cleanly.
- Translation uses your Codex subscription quota: ~180 uncached input + ~7-60 output tokens per utterance (prompt prefix cached). A long session barely moves the 5 h window. The "Fast" service tier trades quota for speed ("1.5x speed, increased usage"), so it burns that window faster than the per-turn token counts alone suggest; set `TRANSLATE_SERVICE_TIER = "default"` to drop back to the standard tier.
- Claude Code is this project's development agent. See `CLAUDE.md`, `.claude/`, `.serena/`, and `.agent/` for its workflow and context.

## License

Apache-2.0 WITH LLVM-exception. See `LICENSE`.
