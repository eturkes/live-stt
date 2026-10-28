# live-stt — deferral queue

The funding menu, read on demand. Unattached ⇒ `.agent/spec.md` stays the sole attached state, and
its `Tasks` = one open `- [ ]` row per row below, same rank and title, then the pointer here. Rank =
funding order. Acceptance is written at deferral time while the evidence is fresh, and the funded row
is that unit's whole contract (`assurance-posture.md`). A `steered/maintain.md` Queue body names
the rows it funds, and the `auto/maintain.md` Queue body funds every row in rank order; closing a
row deletes it from this file and from `.agent/spec.md`'s `Tasks` in one
commit. `tests/test_law_consistency.py` locks that pairing and rejects naming a row by `rank N`
anywhere else, since a rank retargets onto a different unit the moment an earlier row dies.

1. **Live-mic validation pass** — user-only (L-004): M14's `_respawn` arm, the degrade marker and
   the SIGHUP shutdown have never met a mic (`_probe` has); standing debt = `live-smoke.md` items 1,
   3, 4 (latency feel), 7 (Ctrl+C mid-decode), 8 (`-o`), 9 (quiet far speech + empty room), 10 (GPU
   mode), 11 (`--playback` in a real call) and a multi-hour soak.
   **Accept:** the user runs `live-smoke.md` and reports; each item lands verified or defective.

2. **Untranslated sessions read as transient failures** — `session_report.py` attributes every
   caption of a session that never translated (`--no-translate`, or `Translation: unavailable` at
   startup, which writes no transcript marker by design) to `failed` "transient failure, leg
   recovered" — 22 of the 23 captions of a `--no-translate --playback` run, the last `shutdown`.
   The transcript carries no TGT and no `--` translation marker, so "leg recovered" claims what no
   record shows. Tier `data` (the report is re-read by the user).
   **Accept:** a session whose transcript holds no `TGT` line and no translation marker reports its
   captions under their own reason (not `failed`, `shutdown` or `disabled`), text and `--json`
   alike; a session with any TGT keeps today's attribution byte-identical
   (`tests/test_session_report.py` unchanged and green); the new lock seen red first.

3. **Playback leveling across digital silence** — `--playback` hears EXACT zeros whenever nothing
   plays, and `LeveledVad` admits no window ≤ `VAD_LEVEL_FLOOR_DBFS`, so its 10 s history never
   ages and a quiet talker after a loud one is heard at the loud one's gain 1.0: live, 5 quiet CV
   clips (speech p90 −37..−43 dBFS) between loud ones with 10 s zero gaps, 3 plays each — cv489 lost
   whole, cv894 lost `先月`, cv904 gained `はい。`; loud clips kept every caption, none landing past a clip's 4 s tail. Every arm tried
   trades quiet against loud (`asr-pipeline.md`); `LeveledVad` is a user ruling ⇒ the fix
   needs the user's ruling first. Tier `kernel` (decode-quality mechanism, no downstream re-check).
   **Accept:** a playback-mode-only change (mic path byte-identical by construction: the replay
   goldens and retention CER 0.0532 unmoved) recovers cv489, cv894 and cv904 in
   `tests/eval_playback_silence.py`'s shipped arm with no loud clip losing a character against
   today's shipped arm, confirmed by one `tests/eval_playback_routing.py --e2e probe` and one
   `--e2e pinned` run; the user rules the shape before it ships.

4. **Python 3.15 interpreter** — `.python-version` pins `3.14` while CPython 3.15.1 is the latest
   release: the wheel-only hard dependencies openvino, openvino-genai, onnxruntime and sherpa-onnx
   publish no cp315 wheel at their latest releases.
   **Accept:** every package in `uv.lock` resolves a cp315 wheel → `.python-version` 3.15, `.venv`
   re-synced, `tests/fetch_fleurs_parallel.py` rerun so its Python stamp re-qualifies on a
   byte-identical rebuild, the full gate green under the accelerator prelude with zero skips,
   retention CER 0.0532 re-derived on the NPU.

5. **Export recipe at the latest transformers and Python** — `models/README.md`'s whisper-ja-760M
   export pins `transformers==4.57.6` in a Python 3.13 venv. optimum-intel 2.2.0 caps
   `transformers<5.6`, and the newest it admits, 5.5.4, exports a model whose retention hypothesis
   differs from the shipped weights' by one inserted `。` (CER 0.0532 either way). Under Python 3.14
   optimum 2.3.0's exporter raises `TypeError: NormalizedConfig.__init__() got multiple values for
   argument 'allow_new'` (a `functools.partial` class attribute binds as a method from 3.14 on).
   The pinned recipe decodes byte-identically to the shipped weights on 150 + 150 FLEURS clips and
   on retention.
   **Accept:** each unblock moves its own pin — an optimum-intel release admitting the latest
   transformers moves `transformers==`, an optimum stack running under the latest Python moves
   `--python` — once an export under the moved pin into scratch, run through
   `tests/eval_asr_screen.py --arms fleurs,retention --device NPU` (private `OPENVINO_CACHE_DIR`),
   decodes every hypothesis byte-identically to the shipped weights; any differing hypothesis → a
   re-export ruling for the user with the D-016 JA legs re-measured.
