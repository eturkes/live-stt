# Post-mortem: live-stt stuck in kernel audio calls

**Status: unresolved; terminals recovered, audio-driver recovery unverified.**
Incident: 2026-09-11, host time **JST (UTC+09:00)**. Scope of this change = evidence + investigation
handoff, not an implementation fix. Observations below came from the affected host, `/proc`, `ps`,
installed package metadata, and source inspection; hypotheses are labeled separately.

## Summary + impact

Two `uv run live-stt` instances became unresponsive. Both main processes were in Linux `D` state
inside audio-related kernel calls. Targeted `SIGTERM`, then `SIGKILL`, failed to remove either main
process. Stopping each `uv` launcher and its multiprocessing helpers returned that terminal to its
shell, but left the main process blocked with `SIGKILL` pending. A second launch was also blocked;
relaunching was not a demonstrated recovery.

Actual transcription activity, the last printed startup line, transcript loss, and impact on other
audio applications were not established. No audio-service restart, driver reset, reboot, dependency
change, or fresh reproduction was performed. A reboot was suggested as a possible recovery, not
tested or proven necessary.

## Environment snapshot

Snapshot collected at **13:11:13 JST**; repository HEAD at inspection:
`0674ca5ef7e804a11e483f0edddbad9b33e329cc`. Tracked files were clean before this documentation change;
the revision actually loaded by the already-running processes was not independently established.

| Component | Observed value |
| --- | --- |
| Host | Aeon/openSUSE, `VERSION_ID=20260908`, x86_64 |
| Kernel | `7.2.3-1-default`; RPM `kernel-default-7.2.3-1.1.x86_64` |
| Launch | `uv run live-stt`, no CLI flags, host `.venv-host` |
| Python | Executable resolves to uv-managed CPython `3.14.5` |
| Python audio | `sounddevice 0.5.5`, `cffi 2.0.0` |
| Loaded PortAudio | `/usr/lib64/libportaudio.so.2.0.0`; RPM `libportaudio2-190700_20210406-1.17.x86_64` |
| Audio card | Card 0 = `sofsoundwire`; platform driver `sof_sdw`, module `snd_soc_sof_sdw` |
| Installed audio stack | `pipewire-1.6.8-2.1`, `wireplumber-0.5.17-1.1`, `sof-firmware-2025.12.2-1.2` |
| Other Python dependencies | `numpy 2.4.6`, `openvino 2026.3.1`, `openvino-genai 2026.3.1.0`, `sherpa-onnx`/`sherpa-onnx-core 1.13.4` |

Installed PipeWire/WirePlumber versions do not establish that either session reached those servers.
The physical microphone, selected PortAudio host API, and triggering device operation remain unknown.

## Timeline + process evidence

PIDs identify this incident only; rediscover and verify process identity before any future signaling.

| Event | Older instance | Newer instance |
| --- | --- | --- |
| Process start, from `ps` | 13:02:57 JST | 13:05:49 JST |
| `uv` launcher | PID `599249` | PID `629657` |
| Main Python/live-stt | PID `599252`, terminal `pts/11` | PID `629660`, terminal `pts/16` |
| Multiprocessing resource tracker | PID `599437` | PID `629691` |
| Multiprocessing forkserver | PID `599438` | PID `629692` |
| Initial observation, 13:06:50 JST | `Dl+`, 9 threads | `Dl+`, 9 threads |
| Wait channel, subsequent `/proc` inspection | `dpcm_fe_dai_open` | `snd_pcm_control_ioctl` |
| Open audio FD, before second containment | FD 7 → `/dev/snd/controlC0` | FD 18 → `/dev/snd/controlC0` |
| Final sample, 13:11:13 JST | `D`, 1 thread, same wait channel | `D`, 1 thread, same wait channel |

Before intervention, the older process had `ShdPnd: 0000000000000002` (pending `SIGINT`);
the newer had no pending shared signal. Both had `SigCgt: 0000000100000002`, without the
`SIGTERM`/`SIGHUP` bits that the application's installed handlers would catch.

Containment, first for the older instance and then for the newer at the user's request:

1. Verified the main executable, launcher arguments, parent relationship, and process start identity;
   enumerated only that instance's descendants.
2. Sent `SIGTERM` to the main process; waited approximately 3 seconds; sent `SIGKILL` when it remained.
3. Sent `SIGTERM` to its helpers and launcher; escalated remaining runnable targets to `SIGKILL`.
4. Verified both shells returned to the foreground (`Ss+`). Both launchers disappeared.

Final retained process state:

```text
PID     PPID  TTY     STAT  WCHAN                  COMMAND
599252  3029  pts/11  D     dpcm_fe_dai_open       live-stt
599437  599252 pts/11 Z     -                      python3 <defunct>
599438  599252 pts/11 Z     -                      python3 <defunct>
629660  3029  pts/16  D     snd_pcm_control_ioctl  live-stt
629691  629660 pts/16 Z     -                      python3 <defunct>
629692  629660 pts/16 Z     -                      python3 <defunct>
```

Both main processes: `SigPnd: 0000000000000100` (`SIGKILL`),
`ShdPnd: 0000000000004002` (`SIGTERM` + `SIGINT`), 1 remaining thread. Parent 3029 was the user's
systemd manager. Helpers were stopped but remained **zombies**, awaiting reaping; they were not
still-running workers. “Terminal free” therefore did **not** mean “instance fully exited.”

`D` is labeled “disk sleep” by `/proc/status`; the observed wait symbols and device descriptors point
to audio, not evidence of a storage problem. Pending `SIGKILL` plus persistent `D` state is the
important distinction from an ordinary Python cancellation/shutdown hang.

## Source findings; exact blocking phase still unknown

Line numbers refer to the inspected revision of [live_stt.py](../../live_stt.py).

| Entry point | Relevant ordering / gap |
| --- | --- |
| `run_session`, lines 1848–1857 | Synchronous `import sounddevice` → recognizer/VAD initialization → `sd.query_devices(...)`, before application signal handlers. |
| Signal setup, around line 1902 | `_install_signal_handlers(state)` follows device query and translator startup. Its asyncio callbacks require the event loop to run. |
| Stream lifecycle, lines 1907–1934 | Synchronous `sd.InputStream(...)` construction is before the cleanup `try`; `start()`, `stop()`, and `close()` also execute in the event-loop thread. |
| `main --list-devices`, around line 2038 | Imports `sounddevice` and queries devices directly; this path is not a hardware-free diagnostic. |

Installed `sounddevice.py` confirms import-time `_initialize()` at line 2972, calling
`_lib.Pa_Initialize()` at line 2936. That initializer temporarily redirects stderr to `/dev/null`.
Missing PortAudio initialization diagnostics would therefore not prove that initialization succeeded.
The source was read as text; importing it again was deliberately avoided on the affected host.

Existing [L-010](../../.claude/rules/asr-pipeline.md) already records audio-control probes during
`sounddevice` import hanging an offline evaluator. Lazy import protects offline consumers, but the
live/device entry points still perform that initialization in the main process.

**Working hypotheses, not a root-cause verdict:**

- Audio initialization/device access is blocked below Python, in the ALSA/SOF path. Import-time
  PortAudio probing is a strong lead; device query/open remain candidates. The uncaught
  `SIGTERM`/`SIGHUP` masks are consistent with not reaching application handler setup, but are not a
  Python stack trace or proof of the exact phase.
- The newer instance may be waiting behind a card/control lock held by the older operation. Shared
  card 0 and different wait channels support investigating contention; they do not identify a lock
  owner or establish whether the older process caused or encountered the driver fault.
- Neither multiprocessing helpers, translation, nor OpenVINO are established causes. Their presence
  or absence alone is insufficient attribution.

## Evidence gaps + next investigation

1. **Capture the blocked stack before recovery.** Obtain per-thread kernel stacks, syscall state,
   and kernel ALSA/SOF/SoundWire messages with the necessary host access. The attempted unprivileged
   `journalctl -k --since '15 minutes ago' --no-pager -n 250` reported restricted journal visibility
   and yielded no usable kernel evidence. This is missing evidence, not a clean kernel log. No
   Python/native backtrace or syscall trace was captured.
2. **Locate the boundary.** Add flushed before/after markers around import/initialization, query,
   stream construction/start/stop/close, and audio-library teardown in a controlled diagnostic run.
   Record the last visible marker, exact input/host API, active audio clients, and recent
   suspend/resume or device changes. These conditions were not captured for this incident.
3. **Research from the captured stack.** Compare the actual kernel/firmware/PortAudio versions with
   upstream reports for those symbols and this card. A dependency upgrade alone is not evidence of
   a fix; the existing maintenance queue's `sounddevice` update has not been linked to this fault.
4. **Reproduce after host recovery, with the user.** Cover a cold default launch, `--list-devices`,
   the selected-input path, interruption during startup, and close/relaunch. Follow the existing
   [live-smoke boundary](../../.claude/rules/live-smoke.md); record actual results. Avoid accumulating
   additional blocked processes while the original driver state persists.

Useful read-only capture after selecting a freshly verified PID:

```sh
live_stt_pid=12345  # replace with the verified main process PID
ps -p "$live_stt_pid" -o pid,ppid,lstart,tty,stat,wchan:32,comm
rg '^(State|PPid|Threads|SigPnd|ShdPnd|SigCgt):' "/proc/$live_stt_pid/status"
cat "/proc/$live_stt_pid/wchan"
```

## Candidate mitigation + acceptance for the fixing agent

Choose the smallest mitigation justified by the captured boundary; separate application containment
from host-driver repair.

- Evaluate audio-backend process isolation with bounded startup/shutdown and explicit failure
  reporting. An asyncio timeout around a synchronous call, or moving signal-handler installation
  earlier, does not by itself make a kernel-blocked call cancellable. A supervisor must also bound
  its wait after `SIGKILL`: this incident proves a child may remain unreapable while blocked.
- Evaluate avoiding broad ALSA probing through a supported host-audio route, if tracing confirms
  probing as the trigger. Preserve working device selection, capture format, latency, and local STT.
- Evaluate duplicate-start protection before backend initialization. Bound retries and report any
  residual blocked PID; a restart loop would multiply this incident rather than recover from it.
- Keep driver resets/reboots as explicit host-recovery operations coordinated with the user.
  Preserve available evidence first; document the observed result rather than promising recovery.

**Acceptance:**

1. State the evidenced blocking layer, remaining unknowns, and whether the result is containment or
   a demonstrated trigger fix.
2. For application containment, a deterministic blocked-backend regression exercises the chosen
   boundary without real hardware. The CLI returns within a documented bound, identifies the failed
   phase, bounds cleanup/retries, and reports residual processes accurately. Exercise direct launch
   and `uv run`. For a host-only fix, record the changed component and before/after reproduction
   evidence instead of adding an application supervisor without demonstrated need.
3. Preserve hardware-free module import and existing shutdown behavior: queued audio → final JA →
   final EN → translator close, including the real-pty terminal-close regression. Existing tests in
   [test_audio.py](../../tests/test_audio.py) and
   [test_shipped_path.py](../../tests/test_shipped_path.py) cover these pieces, not this driver fault.
4. Complete the applicable gate and user-run host checks. Healthy startup, device listing,
   interruption, and relaunch must leave no unexpected live/zombie descendants or `D`-state process.
   If real-driver verification is unavailable, label the result unverified at that boundary rather
   than declaring this incident fixed.

This report adds no runtime fix. Transcripts, models, dependencies, and unrelated workspace files
were left unchanged.
