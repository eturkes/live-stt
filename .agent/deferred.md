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
   mode) and a multi-hour soak.
   **Accept:** the user runs `live-smoke.md` and reports; each item lands verified or defective.

2. **Python 3.15 interpreter** — `.python-version` pins `3.14` while CPython 3.15.1 is the latest
   release: the wheel-only hard dependencies openvino, openvino-genai, onnxruntime and sherpa-onnx
   publish no cp315 wheel at their latest releases.
   **Accept:** every package in `uv.lock` resolves a cp315 wheel → `.python-version` 3.15, `.venv`
   re-synced, `tests/fetch_fleurs_parallel.py` rerun so its Python stamp re-qualifies on a
   byte-identical rebuild, the full gate green under the accelerator prelude with zero skips,
   retention CER 0.0532 re-derived on the NPU.

3. **Export recipe at the latest transformers and Python** — `models/README.md`'s whisper-ja-760M
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

4. **Screen compile-cache flag** — the export recipe re-qualifies through
   `tests/eval_asr_screen.py` on a private `OPENVINO_CACHE_DIR`, a module constant no committed
   entry point moves: the index-isolation qualification reassigned it from a gitignored `runpy`
   wrapper, which `packaging-deps.md` describes — a scratch-local validator. Data tier.
   **Accept:** `eval_asr_screen.py --cache-dir DIR` compiles into DIR — an NPU run on an empty DIR
   leaves its blobs there while `models/openvino/cache` keeps its listing and mtime — and a
   hardware-free lock on the flag's wiring, seen red on the prior revision; `packaging-deps.md`
   names the flag in place of the wrapper; the gate stays green.
