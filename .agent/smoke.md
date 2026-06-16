# Live-path smoke + soak checklist

The L-004 paths the agent cannot run: mic capture, device selection, latency feel,
Ctrl+C flush, `-o` persistence, multi-hour soak. This is the fixed procedure a
"Did not verify (L-004)" disclaimer points at. Every item names the `live_stt.py`
observable that decides pass/fail.

**Setup:** models downloaded (`models/README.md`); `codex login` once for the EN leg
(D-011) — without it the tool prints `Translation: unavailable — JA-only` and the EN
checks are N/A. One utterance = speech followed by a ≥0.5 s pause (`VAD_MIN_SILENCE_S`).

## Live-mic pass (~5 min, one run)

1. **Devices** — `uv run live-stt --list-devices` prints the `sd.query_devices()`
   table and exits (no models needed). *Pass:* your mic shows with input channels.
2. **Capture + meter** — `uv run live-stt`, then speak. *Pass:* the bar
   `[#### … ] 0.0xxx` rises with your voice and falls in silence (it renders
   `sqrt(state.latest_ms)` set in `audio_callback`); no `drop=` suffix appears and
   `q=` stays absent (a brief `q=1`–`2` during a decode is fine; a standing `q=N`
   or any `drop=N` is a fail — audio_q drained, `state.dropped` 0).
3. **Device select** — `uv run live-stt --device N`. *Pass:* startup prints
   `Mic: #N <name> @ <rate> Hz`; meter reacts as in (2).
4. **Latency + endpointing** — say one sentence, then stop. *Pass:* `JA n: …`
   prints ~0.6 s after you stop — `VAD_MIN_SILENCE_S` (0.5 s) to close the segment
   plus ~0.1 s decode (D-010); a brief (<0.5 s) mid-sentence pause does not split it.
5. **Translation cadence** — Codex up. *Pass:* each `EN n:` trails its `JA n:` by
   ~1 s; the shared `n` keeps a pair matched even when the next `JA` prints first;
   `--no-translate` suppresses every EN line.
6. **Ctrl+C mid-utterance** — start a sentence, press Ctrl+C while still speaking.
   *Pass:* the in-progress utterance still prints `JA n:` (worker flushes the VAD in
   its `finally`), its `EN n:` still lands if Codex is up (translator drains last),
   then `Stopped.` — no hang (shutdown sentinel is non-blocking, T8.1).
7. **`-o` persistence** — `uv run live-stt -o /tmp/stt.txt`, speak, Ctrl+C. *Pass:*
   the file holds `[<ISO-8601>] JA n: …` / `EN n: …` lines, one per event, written
   immediately (`emit_line` flushes each line; file closed in `finally`).

## Soak (1–3 h, intermittent or continuous speech)

Watch the three in-code observables; note them at start and end.

- **Backlog / drops** — meter `q=` stays absent (or blips small and clears) and
  `drop=` never appears. A standing `q=N` or any `drop=N` means decode fell behind
  the mic (audio_q hit `AUDIO_QUEUE_MAX`=100 → `state.dropped`).
- **Thread rotation** — roughly every 100 EN turns one `EN` lands a few seconds
  slower, then cadence resumes with no error (`_translate` rotates the codex thread
  at `_turns % TRANSLATE_ROTATE_TURNS`; the fresh thread eats one uncached turn,
  D-011). EN must keep flowing across the bump.
- **Quota** — out-of-band via the `account/rateLimits/read` app-server RPC (D-011;
  live_stt.py does not surface it). Expect ≈0 % primary-window movement (D-011
  measured ~0 %); a climbing window means revisit D-011.

Supplementary (external, no in-code metric): RSS via `ps`/`top` should stay flat —
the ring is fixed-capacity (`RING_SECONDS`), `_RESAMPLE_CACHE` caps at 8 entries, and
the codex `_notes`/`_pending` drain per turn, so steady growth signals a leak.

If EN stops while JA continues, that is the D-009 JA-only degrade: the
3-consecutive-failure path logs `translation disabled after N…`; a codex EOF in an
idle gap currently degrades silently (T8.5 will log it).
