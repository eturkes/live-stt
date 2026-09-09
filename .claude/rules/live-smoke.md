---
paths:
  - "live_stt.py"
  - "streaming.py"
---

# Live smoke — agent-covered vs user-only (L-004)

This is the fixed procedure a "**Did not verify (L-004)**" disclaimer points at. Each item names the
`live_stt.py` observable that decides pass/fail.

**Agent-covered, so do not send these to the user.** The two-engine replay/CER gates own short-transcript
stability, continuous >10 s decode behaviour and the full-file narration path. Paced production replay
owns the 44.722 s / 20 ms-callback / decode-RTF 0.20 backpressure case and all 7 short clips, and it
extends to the shipped VAC branch on real NPU per-update cost for both pause-free clips: `drop=0`,
`forced_trims=0`, segment queue 0 (VAC owns none), audio queue peak 0.760 s / 1.060 s of the 2.000 s
headroom, longest contiguous decode 0.764 s / 1.006 s — and contiguous == max single decode, so one
update fires per drain and updates never bunch. The long-form carry arm confirms the reserve with no
corpus at all. `eval_retention.py` is the shipped path's accuracy gate. Pure tests own drain-on-shutdown,
sentinel landing, stage-failure cancellation and translator degradation. **Terminal close is
agent-covered**: `test_shipped_path.py` forks a real pty, closes the master, and asserts the child exits
0 with both lines in its transcript.

**User-only boundary:** mic capture, device select, latency feel, real Ctrl+C flush, `-o` persistence,
Codex cadence, multi-hour soak.

**Setup:** models downloaded; `codex` installed on the PATH of the machine running live-stt; `codex
login` there once for the EN leg (without it: `Translation: unavailable — JA-only`, EN checks N/A). One
utterance = speech + a ≥0.5 s pause (`VAD_MIN_SILENCE_S`).

## Live-mic pass (~5 min)

1. **Devices** — `uv run live-stt --list-devices` prints the `sd.query_devices()` table and exits. Pass:
   your mic shows with input channels.
2. **Capture + backlog** — `uv run live-stt`, speak. Pass: `JA n:` lines print; `q=` / `seg=` are absent
   or brief and clear after each utterance; any `drop=N` fails.
3. **Device select** — `uv run live-stt --device N`. Pass: prints `Mic: #N <name> @ <rate> Hz`; capture
   works as in (2).
4. **Latency + endpointing (VAC cadence, NOT the old 0.6 s VAD-segment rule)** — one sentence, then stop.
   Pass: partial text grows on the meter status line *while you are still speaking* (first characters
   ~1.0-1.4 s in), the trailing DIM run being the unconfirmed tail, which may be rewritten between
   ticks while the normal-intensity run never is; then the numbered `JA n:` line lands after you stop; a brief mid-sentence pause does
   not split it; committed characters are never rewritten or duplicated once shown. On
   `--engine k2v2|parakeet` the old rule still applies: no partials, `JA n:` ~0.6 s after you stop.
5. **Translation cadence** — Codex up. Pass: each `EN n:` trails its `JA n:` by ~1 s, shared `n` keeps
   pairs matched; `--no-translate` suppresses every EN line.
6. **Ctrl+C mid-utterance** — start speaking, Ctrl+C while still talking. Pass: the in-progress `JA n:`
   still prints (the worker flushes the VAD in `finally`), its `EN n:` still lands if Codex is up (the
   translator drains last), then `Stopped.` with no hang.
7. **Ctrl+C mid-decode** — `uv run live-stt --engine parakeet`, speak continuously for >20 s, pause, then
   Ctrl+C while the slower long-block decode runs. Pass: that block's `JA n:` still lands, its `EN n:`
   follows if Codex is up, then `Stopped.` with no hang — VAD feeder and sequential decoder both drain
   before translation shutdown.
8. **Transcript persistence** — `uv run live-stt` (no flags), speak, Ctrl+C. Pass: startup prints
   `Transcript: <repo>/transcripts/<start-time>.txt`; that file holds `[<ISO-8601>] JA n: …` / `EN n: …`
   lines, one per event, flushed immediately, closed in `finally`. Then `-o /tmp/stt.txt` → same lines
   appended there and nothing new in `transcripts/`; `--no-save` → prints
   `Transcript: not saved (--no-save)` and writes no file; start-then-immediate-Ctrl+C with no speech
   leaves no file (lazy creation).

## Soak (1-3 h) — watch at start and end

- **Backlog / drops** — `q=` / `seg=` / `drop=` / `tdrop=` stay absent (`q=` / `seg=` may blip and
  clear). A standing `q=Ns` means PCM awaits VAD within `AUDIO_HEADROOM_S`=2 s; `seg=N` means completed
  utterances await decode within `SEGMENT_QUEUE_MAX`=8; any `drop=N` means ingestion fell behind;
  `tdrop=N` means translation fell behind. **`tskip=N` is a CONTENT decision, never backpressure** —
  reading it as backlog corrupts the soak result. Redirected stdout carries no status line, so the same
  counters ride stderr as `backlog peak:` HIGH-WATER marks, logged only when a peak moves (a clean run
  logs nothing), which is what makes `> stt.log` a usable soak record. A peak never clears ⇒ read the
  LAST such line as the session's worst backlog and its timestamp as when that worst arrived.
- **Thread rotation** — about every 100 EN turns one `EN` lands a few seconds slower, then cadence
  resumes with no error. EN must keep flowing across the bump.
- **Quota** — out of band via `account/rateLimits/read`; expect ≈0 % primary-window movement.
- **Memory** (external `ps`/`top`): RSS flat — ring, audio headroom and segment queue are bounded,
  `_RESAMPLE_CACHE` ≤8, codex `_notes`/`_pending` drain per turn. Steady growth signals a leak.

**Run `uv run python session_report.py --log stt.log` after any soak** — it answers every question in
this section mechanically off the saved transcript and the redirected stderr, so a session stays
diagnosable once the scrollback is gone. Redirect stderr (`> stt.log 2>&1`) to give it the counters.

EN stopping while JA continues is the sanctioned JA-only degrade (D-009); the transcript marker names
which trigger fired, and a `-- translation restored: codex app-server probed|respawned` marker below
it means the leg came back and names the arm (`translation-leg.md`). One caption's EN arriving ~5-6 s
late right after a disable marker is that recovery working, not a stall.
