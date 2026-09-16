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
login` there once for the EN leg (without it: `Translation: unavailable — source-only`, TGT checks N/A). One
utterance = speech + a ≥0.5 s pause (`VAD_MIN_SILENCE_S`).

## Live-mic pass (~5 min)

1. **Devices** — `uv run live-stt --list-devices` prints the `sd.query_devices()` table and exits. Pass:
   your mic shows with input channels.
2. **Capture + backlog** — `uv run live-stt 2> stt.log`, speak, and KEEP the log. Pass: `SRC n:` lines
   print; `q=` / `seg=` are absent or brief and clear after each utterance; any `drop=N` fails. A live
   session HAS produced `drop=` with no established cause (`.agent/deferred.md` → *Explain the live
   audio drops*), and the surviving high-water digest of that session cannot attribute a drop to a
   caption, which is why the row is still open. **One redirect now captures the whole timeline
   without hiding the status line** — the peak log gates on the stream PAIR (L-006,
   `asr-pipeline.md`), so the status line stays on screen while `backlog peak:` and
   `caption dropped (…)` lines land in the file. What it is NOT is free: the peak re-logs whenever
   the rendered string moves, so a moving peak can format+write+flush up to 1/`METER_INTERVAL` = 10
   lines/s synchronously in the meter thread, unmeasured under a real drop storm. Then
   `uv run python session_report.py --log stt.log` attributes every drop increase to the captions
   bracketing it and to any screened caption inside that gap. Capture from the start: a drop that
   reproduces only once is lost if the first run kept no log. Several runs can share one log file.
   Each run stamps `session: <transcript path>` when it starts, so the report separates them and a
   startup drop stays with the run that made it, `-o PATH` included.
3. **Device select** — `uv run live-stt --device N`. Pass: prints `Mic: #N <name> @ <rate> Hz`; capture
   works as in (2).
4. **Latency + endpointing (VAC cadence, NOT the old 0.6 s VAD-segment rule)** — one sentence, then stop.
   Pass: partial text grows on the meter status line *while you are still speaking* (first characters
   ~1.0-1.4 s in), the trailing DIM run being the unconfirmed tail, which may be rewritten between
   ticks while the normal-intensity run never is; then the numbered `SRC n:` line lands after you stop; a brief mid-sentence pause does
   not split it; committed characters are never rewritten or duplicated once shown. On
   `--engine k2v2|parakeet` the old rule still applies: no partials, `SRC n:` ~0.6 s after you stop.
5. **Translation cadence** — Codex up. Pass: each `TGT n:` trails its `SRC n:` by ~1 s, shared `n` keeps
   pairs matched; `--no-translate` suppresses every TGT line.
6. **Ctrl+C mid-utterance** — start speaking, Ctrl+C while still talking. Pass: the in-progress `SRC n:`
   still prints (the worker flushes the VAD in `finally`), its `TGT n:` still lands if Codex is up (the
   translator drains last), then `Stopped.` with no hang. **The last `TGT` is the load-bearing half** —
   it was lost in 4 of 7 saved sessions, because Ctrl+C killed the app-server along with the terminal's
   process group before the drain could use it (`translation-leg.md`). A missing final `TGT`, or a
   `-- translation disabled: codex app-server exited` marker at the foot of the transcript, is that
   regression. **M14's `start_new_session=True` fix is VALIDATED on a real mic** over a 26-minute
   143-caption session: `SRC 143` then `TGT 143` six seconds later, drain order SRC 142 → SRC 143 →
   TGT 142 → TGT 143, zero `-- translation` markers, `Stopped.`, no hang, process fully exited. That
   retires this item's L-004 debt; items 1, 3, 4, 5, 7 and 8 stay outstanding.
7. **Ctrl+C mid-decode** — `uv run live-stt --engine parakeet`, speak continuously for >20 s, pause, then
   Ctrl+C while the slower long-block decode runs. Pass: that block's `SRC n:` still lands, its `TGT n:`
   follows if Codex is up, then `Stopped.` with no hang — VAD feeder and sequential decoder both drain
   before translation shutdown.
8. **Transcript persistence** — `uv run live-stt` (no flags), speak, Ctrl+C. Pass: startup prints
   `Transcript: <repo>/transcripts/<start-time>.txt`; that file holds `[<ISO-8601>] SRC n: …` / `TGT n: …`
   lines, one per event, flushed immediately, closed in `finally`. Then `-o /tmp/stt.txt` → same lines
   appended there and nothing new in `transcripts/`; `--no-save` → prints
   `Transcript: not saved (--no-save)` and writes no file; start-then-immediate-Ctrl+C with no speech
   leaves no file (lazy creation).

**Every `SRC n:` criterion above is the ONE-WAY grammar**, and `--two-way` is default OFF ⇒ no marker
can appear without the flag. Under it, an utterance whose language was HELD rather than detected
publishes `SRC n <!>: text` — space, `<!>`, colon, source line only, screen and transcript alike; its
`TGT n:` stays unmarked and no caption is ever withheld. Read that mark as the held-label signal it is,
never as a defect: `session_report.py` counts those utterances as `held`, and the short utterances that
carry it are the ones the detector had under 2 s of audio to judge.

## Soak (1-3 h) — watch at start and end

- **Backlog / drops** — `q=` / `seg=` / `drop=` / `tdrop=` stay absent (`q=` / `seg=` may blip and
  clear). A standing `q=Ns` means PCM awaits VAD within `AUDIO_HEADROOM_S`=2 s; `seg=N` means completed
  utterances await decode within `SEGMENT_QUEUE_MAX`=8; any `drop=N` means ingestion fell behind, and
  its cause is currently UNKNOWN — a 26-minute session reached
  `backlog peak: q=2.00s drop=9033 skip=20` with every burst inside a ≥17 s publication gap, yet 16 of
  the 20 such gaps dropped nothing, so keep `stt.log` rather than diagnosing from the meter;
  `tdrop=N` means translation fell behind. **`tskip=N` is a CONTENT decision, never backpressure** —
  reading it as backlog corrupts the soak result. Wherever the status line is not the reader's, the same
  counters ride stderr as `backlog peak:` HIGH-WATER marks, logged only when a peak moves (a clean run
  logs nothing), which is what makes `2> stt.log` a usable soak record while the status line keeps
  drawing. A peak never clears ⇒ read the LAST such line as the session's worst backlog and its
  timestamp as when that worst arrived.
- **Thread rotation** — about every 100 translation turns one `TGT` lands a few seconds slower, then cadence
  resumes with no error. TGT must keep flowing across the bump.
- **Quota** — out of band via `account/rateLimits/read`; expect ≈0 % primary-window movement.
- **Memory** (external `ps`/`top`): RSS flat — ring, audio headroom and segment queue are bounded,
  `_RESAMPLE_CACHE` ≤8, codex `_notes`/`_pending` drain per turn. Steady growth signals a leak.

**Run `uv run python session_report.py --log stt.log` after any soak** — it answers every question in
this section mechanically off the saved transcript and the redirected stderr, so a session stays
diagnosable once the scrollback is gone. Redirect stderr (`2> stt.log`) to give it the counters; the
status line keeps drawing, so the record costs the soak no screen output — not zero work, which no
live run has measured (item 2).

TGT stopping while SRC continues is the sanctioned source-only degrade (D-009); the transcript marker names
which trigger fired, and a `-- translation restored: codex app-server probed|respawned` marker below
it means the leg came back and names the arm (`translation-leg.md`). One caption's TGT arriving ~5-6 s
late right after a disable marker is that recovery working, not a stall.
