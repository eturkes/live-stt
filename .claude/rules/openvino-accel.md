---
paths:
  - "live_stt.py"
  - "replay.py"
  - "tests/eval_*.py"
  - "tests/build_*.py"
  - "tests/gen_replay_goldens.py"
  - "models/README.md"
---

# OpenVINO accelerator path

Device-selection law (exact names, static shapes, `EXECUTION_DEVICES` readback) is machine-scoped in
`CLAUDE.local.md`. This file carries what is specific to this repo and this container.

## The whisper prelude — both halves, every time

```sh
source /var/home/eturkes/.local/app/intel-accel/env.sh   # NPU compiler loader
unset PYTHONPATH                                          # or: env -u PYTHONPATH …
```

- **Without `env.sh` the NPU aborts**: `Cannot load library ".../libopenvino_intel_npu_compiler_loader.so"`.
  The PyPI wheel does not carry the NPU compiler loader; the farm does.
- **The `PYTHONPATH` half has two sources, either alone fatal**: the host `~/.profile` sources the
  GenAI tarball's `setupvars.sh`, and `env.sh` itself prepends `openvino_genai_container/python`. Both
  trees ship `py_openvino_genai` for CPython 3.10-3.13 and none for this project's 3.14 ⇒
  `ModuleNotFoundError: No module named 'openvino_genai.py_openvino_genai'`.
- **`import openvino` still succeeds and is the trap.** OpenVINO extends `__path__`, so its `__init__`
  comes from the shadowing tree while `_pyopenvino` resolves to the venv's cp314 build;
  `openvino_genai` extends nothing and dies on its relative import. A stray entry can also surface as
  a confusing `AxisSet` ImportError rather than a missing module.
- `.envrc` unsets `PYTHONPATH` for direnv-hooked interactive shells ALONE — agent Bash, scripts and
  git hooks still need `env -u PYTHONPATH`. `gate.py` clears it itself and runs no whisper step, so
  the gate needs no farm.

## Trees, layers, hardware

- Core Ultra 7 268V: Arc 140V iGPU (`8086:64a0`, `xe`, `/dev/dri/renderD128`) + AI Boost NPU 4
  (`8086:643e`, `intel_vpu`, `/dev/accel/accel0`).
- Two GenAI trees exist and only one loads in the container: `~/.local/app/openvino_genai` is the HOST
  build and needs `GLIBC_2.43` against the container's 2.41; `~/.local/app/openvino_genai_container` is
  what `env.sh` selects. Never confuse the unusable tarball with the deployment path.
- **The deployment path is the PyPI wheels.** `openvino` / `openvino-genai` publish cp314
  manylinux_2_28 wheels, so the project's own py3.14 `.venv` enumerates `['CPU','GPU','NPU']` and
  compiles + infers on each. Only `LD_LIBRARY_PATH` / `OCL_ICD_VENDORS` / `ZE_ENABLE_ALT_DRIVERS`
  matter, and all three are container-only shims: the host carries `libze_loader.so.1`,
  `libze_intel_npu.so.1`, `libze_intel_gpu.so.1` and `libOpenCL.so.1` system-wide with glibc 2.43, so
  **on the host OpenVINO is an ordinary `pyproject.toml` dependency installed by plain `uv sync`** —
  no farm, no `env.sh`, no second environment.
- `intel-accel/` is a reversible container-only symlink farm over the host Intel drivers, with a pinned
  Ubuntu IGC that avoids the host IGC's newer-glibc requirement. Its `/run/host/...` targets dangle
  harmlessly on the host and stay outside project repos. After a host Intel-driver update, rebuild with
  `python3 /var/home/eturkes/.local/app/intel-accel/make_farm.py`, then rerun the self-test.

## Placement + benchmarking

- `openvino_genai.WhisperPipeline` wraps its compiled model without exposing it, so there is no
  `EXECUTION_DEVICES` to query: placement is an **exact-target inference** — `requested_device="NPU"`
  plus a successful decode — and `check_device` admits exact names alone, since an `AUTO:` spelling
  would relocate silently. `ASR_DEVICE` is therefore a bare name.
- **Never infer acceleration from a requested provider.** sherpa-onnx 1.13.2 and 1.13.4 do not expose
  Intel/OpenVINO at all, and `provider="openvino"` logs unsupported then silently falls back to CPU.
  Record the ACTUAL execution device in every accelerator benchmark.
- Decode cost varies ~20 % run to run on identical inputs; treat any single trace as one sample and
  carry a scale ladder as the margin around it.
- Energy is unmeasurable here: RAPL `energy_uj` is mode 400 `nobody:nogroup` in this container and
  unreadable even under sudo ⇒ NPU-vs-GPU perf/W is a host measurement.

## Model cache

`models/openvino/cache` is `OPENVINO_CACHE_DIR` and fully regenerable — `live_stt.py` mkdirs it before
every compile, so deleting it whole is safe. **Measured cold-compile cost of an empty cache = 105.7 s**
for a whisper NPU replay against 12.2 s warm ⇒ clear it only when reclaiming the disk is worth ~93 s on
the next run.
