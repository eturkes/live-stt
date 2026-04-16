# T3.1 Spike Report — Gemini Live API

**Scope:** PLAN.md T3.1. Does a persistent bidirectional `client.aio.live.connect` session beat the existing 5-worker REST-with-chunking pipeline on latency and code size enough to justify the cost?

**Outcome:** Rewrite landed in `live_stt.py`. The Live path is materially better on perceived latency and code size. It is **~10× more expensive per minute** at list price, driven by audio-output tokens we cannot opt out of on native-audio Live models. Recommendation below.

## What changed

| | Before | After |
|---|---|---|
| Lines in `live_stt.py` | 270 | 240 |
| Threads | 1 capture + N workers | main async loop only |
| Queues | unbounded `audio_queue` + bounded `transcribe_queue` | single bounded `asyncio.Queue(maxsize=100)` |
| Chunking | 5 s chunks with 1 s overlap | continuous 100 ms PCM16 frames |
| Context | manual rolling window (`CONTEXT_SIZE=3`) prepended to every prompt | session-native (server-side history) |
| WAV framing | per-chunk WAV wrap | raw `audio/pcm;rate=16000` bytes |
| CLI flags | `--model --workers --max-chunk --no-translate -o` | `--model --no-translate -o` |
| Python floor | 3.10 | 3.11 (`asyncio.TaskGroup`) |
| SDK entry | `client.models.generate_content_stream` | `client.aio.live.connect` |

Deleted: the worker pool, chunk buffer/overlap logic, rolling context dict, per-chunk WAV encoding, `queue.Full` silent-drop, the two prior prompt templates. Kept: `resample()`, RMS level meter (now labeled `LIVE` instead of `REC`), output file format (`JA:`/`EN:` blocks separated by a blank line), graceful `Ctrl+C`.

## Benchmark

One canned Japanese utterance synthesized by `gemini-2.5-flash-preview-tts`:
> こんにちは、今日はライブAPIのテストをしています。よろしくお願いします。

Clip length: **5.25 s** at 16 kHz.

| Metric | REST (`gemini-2.5-flash`, single chunk upload) | LIVE (`gemini-3.1-flash-live-preview`, 100 ms stream) |
|---|---|---|
| Connect | — | **0.72 s** |
| Time-to-first-text (from upload/stream start) | 5.95 s | 6.27 s |
| **Time-to-first-text, adjusted for real-time stream** | **5.95 s** (blocking upload) | **1.02 s** (6.27 − 5.25 s of real-time audio) |
| Total wall-clock to final text | 5.98 s | 8.13 s |
| **Total post-audio latency** | **~0.7 s** | **~2.9 s** (8.13 − 5.25) |
| Transcript quality | correct, spacing artifacts in katakana | correct, cleaner katakana |

Raw transcripts:

```
REST : JA: こんにちは。今日はライブ API の テスト を し て い ます。よろしくお願いします。
       EN: Hello. Today, I'm testing the live API. Thank you in advance.

LIVE : JA: こんにちは。今日はLive APIのテストをしています。よろしくお願いします。
       EN: Hello. Today I'm testing the Live API. Nice to meet you.
```

**Interpretation.** The REST-path TTFT above (5.95 s) is on a single-chunk upload — i.e. the *best case* for REST since the whole clip is sent at once. In the production pipeline, the old `live_stt.py` buffers audio for `--max-chunk` seconds (default 5) *before* sending, so the user-perceived latency to first text is **max-chunk + network + inference** — ~6–10 s in practice. The LIVE path starts receiving transcript ~1 s after speech ends, regardless of total utterance length. For live captioning this is the whole point.

## Cost (list price, April 2026)

| | `gemini-2.5-flash` audio generate_content | `gemini-3.1-flash-live-preview` |
|---|---|---|
| Audio input | $1.00 / 1M tok ≈ $0.0018/min | $3.00 / 1M tok ≈ $0.005/min |
| Audio output | n/a (text out) | $12.00 / 1M tok ≈ $0.018/min (billed whether discarded or not) |
| Estimated total for 1 hr continuous speech | **~$0.15–0.25** | **~$1.40** |

LIVE is ~6–10× more expensive per hour. The audio-output billing is the dominant term and is unavoidable today — native-audio Live models don't support a text-only response modality. If Google ships a text-only Live SKU (half-cascade lineage), the cost gap collapses to ~3×.

## Gotchas encountered / deferred

- **`TaskGroup` requires Python 3.11.** Bumped `requires-python` and `ruff.target-version` accordingly. README updated.
- **Session duration cap: 15 min audio-only.** MVP disconnects on expiry (loses history). `SessionResumptionConfig` handle-based resumption + client-side ring buffer for the reconnect gap is a follow-up.
- **Format drift risk.** System prompt asks the model to *speak* `JA: ... / EN: ...`, which then gets captured via `output_audio_transcription`. Forum reports note prompt adherence is non-strict on 3.1 Live. Parser in `emit_block()` tolerates a missing `JA:` prefix by wrapping the whole text as the JA line; missing `EN:` is silently dropped.
- **Audio-output tokens billed even when discarded.** No knob to disable. Documented in README.
- **Known open SDK bugs watched but not hit in spike:** `python-genai#1859` (scrambled transcripts on >20 s continuous speech), `#1224` (receive silent after first `turn_complete`). Neither reproduced in benchmark or connect test — needs real-mic soak.

## Did not verify (user smoke-test required)

- Real microphone capture path. Spike tested the Gemini side only: `sd.InputStream.start()` + `audio_callback` boundary crossing, threadsafe `loop.call_soon_threadsafe` into `asyncio.Queue`, `Ctrl+C` signal handler on a real terminal.
- Sustained >2 min live sessions (scramble bug, VAD turn-taking under mixed silence/speech).
- Behavior when the mic has background noise that triggers spurious VAD turns.

## Recommendation

**Adopt LIVE.** The latency delta is the feature. The cost delta is bearable for a personal tool — $1.40/hr is $0.023/min, rounding to cents per session. The code is shorter and the concurrency model is simpler (no thread pool, no manual context, no chunking).

If this tool ever sees high-volume usage, revisit by either:
1. Reintroducing the REST path behind a `--rest` flag for low-cost / high-latency mode.
2. Migrating to a half-cascade Live model with text-only output modality if/when Google ships one.

## Follow-ups worth one PR each

- **Session reconnect with resumption handle** (removes 15 min cliff).
- **Audio ring buffer across reconnect** (so no words are lost in the swap).
- **`--language` flag** — T2.2 in PLAN.md, trivial to add now that the system instruction is parameterized.
- **Remove or fix the `list_live_models.py` line-length lint warning** (pre-existing; line 19).
