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
`forced_trims=0`, segment queue 0 (VAC owns none), audio queue peak 1.000 s / 1.240 s of the 8.000 s
headroom, longest contiguous decode 1.003 s / 1.107 s (whisper-ja-760M) — and contiguous == max single decode, so one
update fires per drain and updates never bunch. The `live` arm replays the machine state that dropped
audio live (×1.75 decode cost plus one 3.53 s blocking span) drop-free under the catch-up rule, and
the long-form carry arm confirms the reserve with no corpus at all. `eval_retention.py` is the shipped path's accuracy gate. Pure tests own drain-on-shutdown,
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
2. **Capture + backlog** — `uv run live-stt --save-log`, speak, and KEEP the log
   (`transcripts/<start>.log`). Pass: `SRC n:` lines print; `q=` / `seg=` are absent or brief and
   clear after each utterance; any `drop=N` fails. The
   drops the 09-24 session logged have a measured mechanism and a fix (catch-up rule + 8 s headroom,
   `asr-pipeline.md` `## Real-time cost`): on a slow machine state (battery,
   `low-power`) `q=` may climb past 2 s during a long utterance or a runaway caption and must
   clear once it ends, partial captions updating less often meanwhile. **The log captures the whole
   timeline
   without hiding the status line** — the peak log gates on the stream PAIR (L-006,
   `asr-pipeline.md`), so the status line stays on screen while `backlog peak:` and
   `caption dropped (…)` lines land in the file. What it is NOT is free: the peak re-logs whenever
   the rendered string moves, so a moving peak can format+write+flush up to 1/`METER_INTERVAL` = 10
   lines/s synchronously in the meter thread, unmeasured under a real drop storm. Then a bare
   `uv run python session_report.py` (it reads `transcripts/*.log` + `stt.log`) attributes every drop
   increase to the captions bracketing it and to any screened caption inside that gap. Capture from
   the start: a drop that reproduces only once is lost if the first run kept no log. `2> stt.log`
   records the same lines (`--save-log` = that redirect from the session stamp on, named like the
   transcript), and several runs can share that one file. Each run stamps
   `session: <transcript path>` when it starts, so the report separates them and a startup drop
   stays with the run that made it, `-o PATH` included.
   **VALIDATED on a real mic in the slow machine state** (60 min, battery + `low-power`, 534
   captions): no `drop=`, `q=` peak 1.18 s, cleared after each utterance; the `backlog peak:` lines
   landed in the `--save-log` file while the status line kept drawing (`script -T` typescript).
3. **Device select** — `uv run live-stt --device N`. Pass: prints `Mic: #N <name> @ <rate> Hz`; capture
   works as in (2).
4. **Latency + endpointing (VAC cadence, NOT the old 0.6 s VAD-segment rule)** — one sentence, then stop.
   Pass: partial text grows on the meter status line *while you are still speaking* (first characters
   ~1.0-1.4 s in), the trailing DIM run being the unconfirmed tail, which may be rewritten between
   ticks while the normal-intensity run never is; then the numbered `SRC n:` line lands after you
   stop — speech past ~8 s also publishes settled pieces as their own lines while you talk, each with
   its own `TGT`; a brief mid-sentence pause does not split it; committed characters are never
   rewritten or duplicated once shown, across piece lines included. On
   `--engine k2v2|parakeet` the old rule still applies: no partials, `SRC n:` ~0.6 s after you stop.
   Cadence measured on the item-2 session's typescript (1,541 distinct partial renders): partial text
   changes every 1.01 s p50 / 1.83 s p90 / 2.94 s max — the 1 s cadence stretched by catch-up on
   `low-power` decodes — and the first partial leads its `SRC` line by 4.46 s p50. Latency feel and
   the never-rewritten criterion stay the user's.
5. **Translation cadence** — Codex up. Pass: each `TGT n:` trails its `SRC n:` by ~2-3 s, shared `n` keeps
   pairs matched; `--no-translate` suppresses every TGT line. **Cadence VALIDATED** on the item-2
   session (codex 0.163.0): 533 of 534 translated, p50 2 s / p90 3 s / max 8 s; the one miss was a
   DNS lookup failure in codex's websocket connect at a thread rotation (host
   `~/.codex/logs_2.sqlite`), the leg translating the next caption. `--no-translate` unrun.
6. **Ctrl+C mid-utterance** — start speaking, Ctrl+C while still talking. Pass: the in-progress `SRC n:`
   still prints (the worker flushes the VAD in `finally`), its `TGT n:` still lands if Codex is up (the
   translator drains last), then `Stopped.` with no hang. **The last `TGT` is the load-bearing half** —
   it was lost in 4 of 7 saved sessions, because Ctrl+C killed the app-server along with the terminal's
   process group before the drain could use it (`translation-leg.md`). A missing final `TGT`, or a
   `-- translation disabled: codex app-server exited` marker at the foot of the transcript, is that
   regression. **M14's `start_new_session=True` fix is VALIDATED on a real mic** over a 26-minute
   143-caption session: `SRC 143` then `TGT 143` six seconds later, drain order SRC 142 → SRC 143 →
   TGT 142 → TGT 143, zero `-- translation` markers, `Stopped.`, no hang, process fully exited. That
   retires this item's L-004 debt; items 1, 3, 4, 7, 8, 9 and 10 stay outstanding (2 + 5 VALIDATED).
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
9. **Quiet speech + an empty room (`LeveledVad`)** — `uv run live-stt --save-audio
   --save-log` through a meeting whose speakers sit far from the mic, then ≥5 min of the empty room.
   Pass: the far speech publishes `SRC` lines from the first seconds on; the empty room publishes
   nothing — no `ご視聴ありがとうございました`-family caption, no `caption dropped` burst. The leveled
   copy boosts room tone up to 16× once speech stops, while the two saved sessions hold only 48 s and
   25 s of whisper-silent gaps; keep the WAV, `replay.py` re-derives either half.
   The item-2 meeting covers the far-speech half at the 10-08 level alone — p90 −34.7 dBFS against
   10-09's −43.8 (512-sample window RMS over windows above −70 dBFS, which reproduces both recorded
   p90s): `SRC 1` 9 s after start. The quiet −44 dBFS shape and the empty room stay unrun.
10. **GPU mode (`--asr-device GPU`, never met a mic or the host)** — on the host, with no
   `OCL_ICD_VENDORS` exported, `uv run live-stt --asr-device GPU --save-log`, speak Japanese, then
   English under `--two-way`. Pass: startup prints `Engine: whisper (local OpenVINO GPU; …)` and no
   process dies at the model load (the host OpenCL loader crashed GPU starts before `check_device`
   named its vendor dir, `openvino-accel.md`); partial text appears sooner than in NPU mode (agent
   trace: first dim character ~0.8 s vs ~1.2 s); English captions under `--two-way` publish as
   they do in NPU mode. A first GPU run compiles in seconds, not minutes. Run it with no other GPU
   workload on the box: a concurrent GPU benchmark stretches forced decodes 2-7×, turbo's runaway
   span past the 8 s capture headroom (`asr-pipeline.md` `## GPU mode`), so a slow or dropping GPU
   session is first a contention question.

**Every `SRC n:` criterion above is the ONE-WAY grammar**, and `--two-way` is default OFF ⇒ no marker
can appear without the flag. Under it, an utterance whose language was HELD rather than detected
publishes `SRC n <!>: text` — space, `<!>`, colon, source line only, screen and transcript alike; its
`TGT n:` stays unmarked and no caption is ever withheld. Read that mark as the held-label signal it is,
never as a defect: `session_report.py` counts those utterances as `held`, and the short utterances that
carry it are the ones the detector had under 2 s of audio to judge.

## Soak (1-3 h) — watch at start and end

- **Backlog / drops** — `q=` / `seg=` / `drop=` / `tdrop=` / `tstale=` stay absent (`q=` / `seg=`
  may blip and clear). A standing `q=Ns` means PCM awaits VAD within `AUDIO_HEADROOM_S`=8 s; `seg=N` means completed
  utterances await decode within `SEGMENT_QUEUE_MAX`=8; any `drop=N` means ingestion fell behind — on the
  default engine one blocking span exhausted what the queue had left (≤ `VAC_BACKLOG_S` of backlog
  plus the span reaching 8 s) or the catch-up rule stopped holding updates back
  (`asr-pipeline.md` `## Real-time cost`), on sherpa the decode stage backing up past `seg=` — so keep
  the session log and the `script -T` typescript whose `q=` sawtooth times each span;
  `tdrop=N` means translation fell behind. **`tskip=N` is a CONTENT decision, never backpressure** —
  reading it as backlog corrupts the soak result. Wherever the status line is not the reader's, the same
  counters ride stderr as `backlog peak:` HIGH-WATER marks, logged only when a peak moves (a clean run
  logs nothing), which is what makes `--save-log` a usable soak record while the status line keeps
  drawing. A peak never clears ⇒ read the LAST such line as the session's worst backlog and its
  timestamp as when that worst arrived.
- **Stale skips (never met a mic, L-004)** — after any degrade, read the transcript for
  `-- translation skipped (stale): <n>` lines and the session log for their
  `caption <n> not translated: queued <N> s, over TRANSLATE_MAX_STALENESS_S` twins. They are the
  bound working, not a fault: the leg came back, and captions that waited over
  `TRANSLATE_MAX_STALENESS_S`=15 s went source-only rather than publishing a translation beside
  captions it has nothing to do with. The meter counts them as `tstale=N`, its own field and never
  `tdrop=`/`tskip=` (`translation-leg.md`). Expected shape, from the one live cascade on record: a
  short run of consecutive numbers right after a restore marker, then normal pairs. A stale skip
  OUTSIDE a degrade window is the defect — it would mean ordinary turns are exceeding the bound.
- **Thread rotation** — about every 100 translation turns one `TGT` lands a few seconds slower, then cadence
  resumes with no error. TGT must keep flowing across the bump.
- **Quota** — out of band via `account/rateLimits/read`; expect ≈0 % primary-window movement.
- **Memory** (external `ps`/`top`): RSS flat — ring, audio headroom and segment queue are bounded,
  one soxr stream per session, codex `_notes`/`_pending` drain per turn. Steady growth signals a leak.
  Held live over the item-2 session: session child 1,951 → 1,957 MB across 60 min, where an
  undrained VAD segment queue grows ~3.8 MB per speech-minute (`asr-pipeline.md`).

**Run `uv run python session_report.py` after any soak** — it answers every question in this
section mechanically off the saved transcript and the saved log, so a session stays diagnosable once
the scrollback is gone; pass `--two-way` for a `--two-way` session, or its English lines read as
`declined` captions. Run the soak with `--save-log` to give it the counters; the
status line keeps drawing, so the record costs the soak no screen output — not zero work, which no
live run has measured (item 2).

TGT stopping while SRC continues is the sanctioned source-only degrade (D-009); the transcript marker names
which trigger fired, and a `-- translation restored: codex app-server probed|respawned` marker below
it means the leg came back and names the arm (`translation-leg.md`). One caption's TGT arriving ~5-6 s
late right after a disable marker is that recovery working, not a stall. **The probe arm is now
live-validated** — it fired on 2026-09-18 and restored the leg on attempt 1, 6 s from disable to
restore — so the shape to expect below a restore marker is a few `-- translation skipped (stale): <n>`
lines for the captions that waited out the outage, then paired SRC/TGT again.
