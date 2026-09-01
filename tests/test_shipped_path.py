"""Coverage for the shipped default path: whisper + OpenVINO device + VAC (D-016).

`test_streaming.py` owns the LocalAgreement policy and the VAC loop over fully
stubbed parts. This file owns the path a default `live-stt` run actually takes:
CLI defaults -> `check_models` -> `load_recognizer` -> a REAL `WhisperEngine`
over a fake OpenVINO pipeline -> `worker`'s duck dispatch -> the VAC loop's
publication and status-line contracts.

Running the real engine class against a fake pipeline is the point. The device
rules that D-016 shipped -- NPU by default, hotwords dropped there rather than
sent -- are then decided by the production constants at every call, so widening
`ASR_HOTWORDS_DEVICES` or changing `ASR_DEVICE` breaks a test instead of quietly
sending a parameter the NPU raises on.

No test here needs a microphone, a downloaded model, or a usable OpenVINO
install: `openvino_genai` is replaced in `sys.modules` before the engine's
function-local import runs.
"""

import asyncio
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import live_stt  # noqa: E402
from streaming import SAMPLE_RATE  # noqa: E402

TRANSCRIPT = "あいうえおかきくけこさしすせそたちつてと"


# --- fake OpenVINO pipeline ---------------------------------------------------


class _Chunk:
    def __init__(self, start_ts, end_ts, text):
        self.start_ts, self.end_ts, self.text = start_ts, end_ts, text


class _Result:
    def __init__(self, texts, chunks=None):
        self.texts = texts
        if chunks is not None:
            self.chunks = chunks


class _FakePipeline:
    """Stands in for `openvino_genai.WhisperPipeline`.

    Mirrors the one production-relevant refusal of the real binding: the NPU's
    StaticWhisperPipeline raises on `hotwords`, so passing it there must fail
    here too rather than pass silently (D-016 (b)).
    """

    instances: list["_FakePipeline"] = []

    def __init__(self, model_dir, device, **kwargs):
        self.model_dir, self.device, self.kwargs = model_dir, device, kwargs
        self.calls: list[dict] = []
        self.result = _Result(["ダミー"], [_Chunk(0.0, 1.0, "ダミー")])
        _FakePipeline.instances.append(self)

    def generate(self, samples, **kwargs):
        if "hotwords" in kwargs and self.device == "NPU":
            raise RuntimeError("'hotwords' parameter is not supported on NPU device")
        self.calls.append({"samples": np.asarray(samples), **kwargs})
        return self.result(samples) if callable(self.result) else self.result


@pytest.fixture
def openvino(monkeypatch, tmp_path):
    """Install the fake binding + a private cache dir; yields the pipeline registry."""
    _FakePipeline.instances = []
    monkeypatch.setitem(
        sys.modules, "openvino_genai", types.SimpleNamespace(WhisperPipeline=_FakePipeline)
    )
    monkeypatch.setattr(live_stt, "OPENVINO_CACHE_DIR", tmp_path / "ov-cache")
    return _FakePipeline.instances


def transcribing_pipeline(pipeline, segment_chars=4):
    """Make `pipeline` transcribe one TRANSCRIPT character per second of audio."""

    def result(samples):
        n = max(1, int(round(len(np.asarray(samples)) / SAMPLE_RATE)))
        text = TRANSCRIPT[:n]
        chunks, at = [], 0.0
        for i in range(0, len(text), segment_chars):
            piece = text[i : i + segment_chars]
            chunks.append(_Chunk(at, at + len(piece), piece))
            at += len(piece)
        return _Result([text], chunks)

    pipeline.result = result


# --- engine + device routing (P1) ---------------------------------------------


@pytest.mark.parametrize("engine_name", sorted(live_stt.WHISPER_ENGINES))
def test_whisper_engine_routes_to_openvino_with_the_requested_device(openvino, engine_name):
    engine = live_stt.load_recognizer(engine_name, "GPU")

    assert isinstance(engine, live_stt.WhisperEngine)
    (pipeline,) = openvino
    assert pipeline.model_dir == str(live_stt.ENGINE_DIRS[engine_name])
    assert pipeline.device == "GPU"


def test_whisper_engine_defaults_to_the_shipped_accelerator(openvino):
    """A default run must place on the NPU, which is what D-016 measured."""
    engine = live_stt.load_recognizer("whisper")

    assert isinstance(engine, live_stt.WhisperEngine)
    assert engine.device == "NPU"
    assert openvino[0].device == "NPU"


def test_sherpa_engines_route_to_their_own_model_directory(monkeypatch):
    seen = {}

    def from_transducer(**kwargs):
        seen["transducer"] = kwargs
        return "k2v2-recognizer"

    def from_nemo_ctc(**kwargs):
        seen["nemo"] = kwargs
        return "parakeet-recognizer"

    monkeypatch.setattr(
        live_stt,
        "sherpa_onnx",
        types.SimpleNamespace(
            OfflineRecognizer=types.SimpleNamespace(
                from_transducer=from_transducer, from_nemo_ctc=from_nemo_ctc
            )
        ),
    )

    assert live_stt.load_recognizer("k2v2") == "k2v2-recognizer"
    assert live_stt.load_recognizer("parakeet") == "parakeet-recognizer"
    assert seen["transducer"]["encoder"].startswith(str(live_stt.ENGINE_DIRS["k2v2"]))
    assert seen["nemo"]["model"].startswith(str(live_stt.ENGINE_DIRS["parakeet"]))
    # The sherpa branch must never reach OpenVINO, whatever device is asked for.
    assert live_stt.load_recognizer("k2v2", "NPU") == "k2v2-recognizer"


# --- WhisperEngine (P2) -------------------------------------------------------


def test_engine_creates_its_compilation_cache_and_passes_it_to_the_pipeline(openvino, tmp_path):
    cache = tmp_path / "ov-cache"
    assert not cache.exists()

    live_stt.WhisperEngine(tmp_path / "model", "CPU")

    assert cache.is_dir()
    assert openvino[0].kwargs == {"CACHE_DIR": str(cache)}


@pytest.mark.parametrize("device", sorted(live_stt.ASR_HOTWORDS_DEVICES))
def test_biasing_capable_devices_keep_the_term_list(openvino, tmp_path, device):
    engine = live_stt.WhisperEngine(tmp_path / "model", device)
    engine.decode(np.zeros(SAMPLE_RATE, dtype=np.float32))
    # An unset list is omitted, never sent as "": the keyword itself is the switch.
    assert "hotwords" not in openvino[0].calls[0]

    engine.set_hotwords("東京、タワー")
    engine.decode(np.zeros(SAMPLE_RATE, dtype=np.float32))

    assert engine.supports_hotwords
    assert engine.hotwords == "東京、タワー"
    assert openvino[0].calls[1]["hotwords"] == "東京、タワー"


def test_the_default_device_drops_the_term_list_instead_of_sending_it(openvino, tmp_path):
    """The NPU raises on `hotwords`; the setter is what keeps the call site safe."""
    engine = live_stt.WhisperEngine(tmp_path / "model", live_stt.ASR_DEVICE)
    engine.set_hotwords("東京、タワー")

    assert not engine.supports_hotwords
    assert engine.hotwords == ""
    engine.decode(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert "hotwords" not in openvino[0].calls[0]


def test_decode_asks_for_japanese_transcription_without_timestamps(openvino, tmp_path):
    engine = live_stt.WhisperEngine(tmp_path / "model", "CPU")
    openvino[0].result = _Result([" こんにちは ", "世界 "])

    assert engine.decode(np.zeros(SAMPLE_RATE, dtype=np.float32)) == "こんにちは 世界"
    call = openvino[0].calls[0]
    assert call["language"] == "<|ja|>"
    assert call["task"] == "transcribe"
    assert call["return_timestamps"] is False


def test_the_audio_reaches_the_pipeline_unaltered(openvino, tmp_path):
    """Every other test feeds zeros, so a decode that substituted its input would pass."""
    engine = live_stt.WhisperEngine(tmp_path / "model", "CPU")
    samples = np.linspace(-0.75, 0.75, SAMPLE_RATE, dtype=np.float32)

    engine.decode(samples)

    forwarded = openvino[0].calls[0]["samples"]
    assert forwarded.dtype == np.float32
    assert forwarded.shape == samples.shape
    assert np.array_equal(forwarded, samples)


def test_decode_segments_returns_spans_the_trim_rule_can_use(openvino, tmp_path):
    engine = live_stt.WhisperEngine(tmp_path / "model", "CPU")
    openvino[0].result = _Result(["あいうえ"], [_Chunk(0.0, 2.0, "あい"), _Chunk(2.0, 4.0, "うえ")])

    text, segments = engine.decode_segments(np.zeros(SAMPLE_RATE, dtype=np.float32))

    assert text == "あいうえ"
    assert [(s.start_s, s.end_s, s.text) for s in segments] == [
        (0.0, 2.0, "あい"),
        (2.0, 4.0, "うえ"),
    ]
    # The Japanese pin rides every call, not just the untimestamped one VAC skips.
    call = openvino[0].calls[0]
    assert call["return_timestamps"] is True
    assert call["language"] == "<|ja|>"
    assert call["task"] == "transcribe"


def test_spans_are_dropped_when_they_do_not_reconstruct_the_transcript(openvino, tmp_path):
    """A cut point that is not in the transcript would lose or repeat text."""
    engine = live_stt.WhisperEngine(tmp_path / "model", "CPU")
    openvino[0].result = _Result(["あいうえ"], [_Chunk(0.0, 2.0, "あい"), _Chunk(2.0, 4.0, "XX")])

    text, segments = engine.decode_segments(np.zeros(SAMPLE_RATE, dtype=np.float32))

    assert text == "あいうえ"
    assert segments == []


def test_spans_are_empty_when_the_pipeline_reports_none(openvino, tmp_path):
    engine = live_stt.WhisperEngine(tmp_path / "model", "CPU")
    openvino[0].result = _Result(["あいうえ"])

    assert engine.decode_segments(np.zeros(SAMPLE_RATE, dtype=np.float32)) == ("あいうえ", [])


# --- check_models marker routing (P4) -----------------------------------------


@pytest.fixture
def model_tree(monkeypatch, tmp_path):
    """A models/ tree with nothing in it; tests create only what they mean to."""
    dirs = {engine: tmp_path / engine for engine in live_stt.ENGINE_DIRS}
    for d in dirs.values():
        d.mkdir()
    monkeypatch.setattr(live_stt, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(live_stt, "VAD_MODEL", tmp_path / "silero_vad.onnx")
    monkeypatch.setattr(live_stt, "ENGINE_DIRS", dirs)
    (tmp_path / "silero_vad.onnx").touch()
    return dirs


WHISPER_MARKER = "openvino_encoder_model.xml"
SHERPA_MARKER = "tokens.txt"


def test_whisper_is_ready_on_its_own_marker(model_tree):
    (model_tree["whisper"] / WHISPER_MARKER).touch()
    assert live_stt.check_models("whisper") is None


@pytest.mark.parametrize(
    "engine_name", sorted(set(live_stt.ENGINE_DIRS) - live_stt.WHISPER_ENGINES)
)
def test_sherpa_is_ready_on_its_own_marker(model_tree, engine_name):
    (model_tree[engine_name] / SHERPA_MARKER).touch()
    assert live_stt.check_models(engine_name) is None


def test_a_sherpa_marker_does_not_satisfy_whisper(model_tree):
    """The two engines ship different files; the wrong one must still report missing."""
    (model_tree["whisper"] / SHERPA_MARKER).touch()
    err = live_stt.check_models("whisper")
    assert err is not None
    assert "whisper/" in err


def test_a_whisper_marker_does_not_satisfy_sherpa(model_tree):
    (model_tree["parakeet"] / WHISPER_MARKER).touch()
    err = live_stt.check_models("parakeet")
    assert err is not None
    assert "parakeet/" in err


def test_a_missing_vad_is_reported_for_every_engine(model_tree):
    live_stt.VAD_MODEL.unlink()
    (model_tree["whisper"] / WHISPER_MARKER).touch()
    err = live_stt.check_models("whisper")
    assert err is not None
    assert "silero_vad.onnx" in err
    assert "models/README.md" in err


def test_both_missing_assets_are_named_in_one_message(model_tree):
    """Reporting one at a time would cost a second failed run to discover the other."""
    live_stt.VAD_MODEL.unlink()

    err = live_stt.check_models("whisper")

    assert err is not None
    assert "silero_vad.onnx" in err
    assert "whisper/" in err


# --- worker duck dispatch (P6) ------------------------------------------------


class _SpanEngine:
    def decode(self, samples):
        return ""

    def decode_segments(self, samples):
        return "", []


class _TextOnlyEngine:
    def decode(self, samples):
        return ""


class _SherpaShapedEngine:
    def create_stream(self):
        raise AssertionError("not reached")

    def decode_stream(self, stream):
        raise AssertionError("not reached")


async def _route(monkeypatch, rec):
    took = []

    async def vac(*args, **kwargs):
        took.append("vac")

    async def feed(*args, **kwargs):
        took.append("feed")

    async def decode(*args, **kwargs):
        took.append("decode")

    monkeypatch.setattr(live_stt, "_vac_segments", vac)
    monkeypatch.setattr(live_stt, "_feed_segments", feed)
    monkeypatch.setattr(live_stt, "_decode_segments", decode)
    state = live_stt.State()
    await live_stt.worker(rec, None, 1600, asyncio.Queue(), state, None)
    assert not state.stopping  # a routing failure would surface as a dead worker
    return sorted(took)


def test_an_engine_with_spans_runs_the_streaming_policy(monkeypatch):
    assert asyncio.run(_route(monkeypatch, _SpanEngine())) == ["vac"]


def test_an_engine_without_spans_runs_the_vad_segment_path(monkeypatch):
    """VAC trims against segment spans, so a text-only engine cannot use it."""
    assert asyncio.run(_route(monkeypatch, _TextOnlyEngine())) == ["decode", "feed"]


def test_a_sherpa_recognizer_runs_the_vad_segment_path(monkeypatch):
    assert asyncio.run(_route(monkeypatch, _SherpaShapedEngine())) == ["decode", "feed"]


# --- VAC over the real engine: publication + biasing (P3, P7) -----------------


class _StubVad:
    """Speech is on while `script` says so, one entry per accepted window."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def accept_waveform(self, _block):
        self.calls += 1

    def is_speech_detected(self):
        i = min(self.calls, len(self.script)) - 1
        return self.script[i] if i >= 0 else False


class _RecordingContext:
    """Records what the recognizer was offered and what came back as evidence."""

    def __init__(self, terms=""):
        self._offer = (terms, frozenset(terms.split("、")) if terms else frozenset())
        self.observed: list[tuple[str, frozenset[str]]] = []

    def asr_hotwords(self):
        return self._offer

    def observe_ja(self, text, prompted=frozenset()):
        self.observed.append((text, prompted))


class _RecordingTranslator:
    def __init__(self):
        self.submitted: list[tuple[int, str]] = []

    def submit(self, seq, text):
        self.submitted.append((seq, text))


def run_vac(monkeypatch, rec, script, window=1600, translator=None, context=None):
    """Drive `_vac_segments` over `script`, one VAD window per entry."""
    vad = _StubVad(script)
    state = live_stt.State()
    lines: list[tuple[str, int, str]] = []
    segments: list[tuple] = []
    monkeypatch.setattr(
        live_stt, "emit_line", lambda tag, seq, text, f: lines.append((tag, seq, text))
    )
    q: asyncio.Queue = asyncio.Queue()
    for _ in script:
        q.put_nowait(np.zeros(window, dtype=np.float32))
    q.put_nowait(None)

    asyncio.run(
        live_stt._vac_segments(
            rec,
            vad,
            window,
            q,
            state,
            None,
            translator=translator,
            on_segment=lambda *a: segments.append(a),
            context=context,
        )
    )
    return types.SimpleNamespace(lines=lines, state=state, segments=segments)


def whisper_engine(openvino, tmp_path, device=None):
    engine = live_stt.WhisperEngine(tmp_path / "model", device or live_stt.ASR_DEVICE)
    transcribing_pipeline(openvino[-1])
    return engine


ONE_UTTERANCE = [True] * 40 + [False]


def test_one_utterance_publishes_exactly_once_through_every_consumer(
    monkeypatch, openvino, tmp_path
):
    engine = whisper_engine(openvino, tmp_path)
    translator, context = _RecordingTranslator(), _RecordingContext()

    run = run_vac(monkeypatch, engine, ONE_UTTERANCE, translator=translator, context=context)

    assert len(openvino[0].calls) > 2, "the utterance must be re-decoded while it is open"
    (line,) = run.lines
    assert line[0] == "JA"
    assert line[1] == 1
    assert line[2]
    assert translator.submitted == [(1, line[2])]
    assert [text for text, _ in context.observed] == [line[2]]
    assert len(run.segments) == 1


def test_a_second_utterance_takes_the_next_number_on_every_consumer(
    monkeypatch, openvino, tmp_path
):
    engine = whisper_engine(openvino, tmp_path)
    translator, context = _RecordingTranslator(), _RecordingContext()

    run = run_vac(
        monkeypatch,
        engine,
        [True] * 30 + [False] * 5 + [True] * 30 + [False],
        translator=translator,
        context=context,
    )

    assert [seq for _, seq, _ in run.lines] == [1, 2]
    assert [seq for seq, _ in translator.submitted] == [1, 2]
    assert [text for _, text in translator.submitted] == [text for _, _, text in run.lines]
    assert len(context.observed) == 2


def test_an_utterance_that_decodes_to_nothing_publishes_nothing_and_burns_no_number(
    monkeypatch, openvino, tmp_path
):
    engine = whisper_engine(openvino, tmp_path)
    openvino[0].result = _Result([""], [])
    translator, context = _RecordingTranslator(), _RecordingContext()

    run = run_vac(
        monkeypatch,
        engine,
        [True] * 20 + [False] * 5 + [True] * 20 + [False],
        translator=translator,
        context=context,
    )

    assert run.lines == []
    assert translator.submitted == []
    assert context.observed == []
    # Silence is still a completed segment, which is what replay's contract counts.
    assert len(run.segments) == 2


def test_a_silent_utterance_does_not_consume_a_line_number(monkeypatch, openvino, tmp_path):
    """Numbering advances on publication, so a caption's `n` counts captions."""
    engine = whisper_engine(openvino, tmp_path)
    speaking = openvino[0].result
    decodes: list[int] = []

    def result(samples):
        decodes.append(1)
        return speaking(samples) if len(decodes) > 3 else _Result([""], [])

    openvino[0].result = result
    translator, context = _RecordingTranslator(), _RecordingContext()

    run = run_vac(
        monkeypatch,
        engine,
        [True] * 20 + [False] * 5 + [True] * 20 + [False],
        translator=translator,
        context=context,
    )

    assert len(run.segments) == 2  # both utterances closed
    assert [seq for _, seq, _ in run.lines] == [1]  # the silent one burned no number
    assert [seq for seq, _ in translator.submitted] == [1]


def test_the_default_device_reports_no_biasing_so_captions_stay_evidence(
    monkeypatch, openvino, tmp_path
):
    """Nothing reached the model, so the caption is un-prompted support (D-015)."""
    engine = whisper_engine(openvino, tmp_path)
    context = _RecordingContext("東京、タワー")

    run = run_vac(monkeypatch, engine, ONE_UTTERANCE, context=context)

    assert engine.hotwords == ""
    assert context.observed == [(run.lines[0][2], frozenset())]


def test_a_biasing_capable_device_discounts_the_terms_it_supplied(monkeypatch, openvino, tmp_path):
    engine = whisper_engine(openvino, tmp_path, device="GPU")
    context = _RecordingContext("東京、タワー")

    run = run_vac(monkeypatch, engine, ONE_UTTERANCE, context=context)

    assert engine.hotwords == "東京、タワー"
    assert context.observed == [(run.lines[0][2], frozenset({"東京", "タワー"}))]


def test_a_session_term_survives_to_the_translator_but_never_to_the_npu(
    monkeypatch, openvino, tmp_path
):
    """The shipped default learns terms and spends them on the translator alone."""
    engine = whisper_engine(openvino, tmp_path)
    context = live_stt.SessionContext("東京タワー")

    run_vac(monkeypatch, engine, ONE_UTTERANCE, context=context)

    assert "東京タワー" in context.translator_brief()
    assert engine.hotwords == ""
    assert all("hotwords" not in call for call in openvino[0].calls)


# --- status line (P5) ---------------------------------------------------------


def test_the_settling_caption_only_ever_grows_while_an_utterance_is_open(
    monkeypatch, openvino, tmp_path
):
    """Committed characters are append-only, so the status line never rewrites."""
    engine = whisper_engine(openvino, tmp_path)
    state_ref: list = []
    seen: list[str] = []
    original = live_stt.WhisperEngine.decode_segments

    def watching(self, samples):
        seen.append(state_ref[0].partial)
        return original(self, samples)

    monkeypatch.setattr(live_stt.WhisperEngine, "decode_segments", watching)

    vad = _StubVad(ONE_UTTERANCE)
    state = live_stt.State()
    state_ref.append(state)
    monkeypatch.setattr(live_stt, "emit_line", lambda *a: None)
    q: asyncio.Queue = asyncio.Queue()
    for _ in ONE_UTTERANCE:
        q.put_nowait(np.zeros(1600, dtype=np.float32))
    q.put_nowait(None)
    asyncio.run(live_stt._vac_segments(engine, vad, 1600, q, state, None))

    assert seen[0] == ""
    assert any(seen), "partial text must reach the meter before the line is published"
    # Growth, not just non-shrinkage: a partial frozen at the first commit is
    # append-only too, and would leave the caption stalled for the whole utterance.
    assert len(set(seen)) >= 3
    for earlier, later in zip(seen, seen[1:], strict=False):
        assert later.startswith(earlier)
    assert state.partial == ""  # cleared once the numbered line is published


class _Screen:
    """stdout double that stops the meter after `ticks` writes."""

    def __init__(self, state, ticks=1):
        self.state, self.ticks, self.writes = state, ticks, []

    def write(self, text):
        self.writes.append(text)
        if len(self.writes) >= self.ticks:
            self.state.stopping = True

    def flush(self):
        pass


def run_meter(monkeypatch, state, audio_q=None, translator=None, columns=80, tty=True):
    screen = _Screen(state)
    monkeypatch.setattr(live_stt, "_STDOUT_TTY", tty)
    monkeypatch.setattr(live_stt, "METER_INTERVAL", 0)
    monkeypatch.setattr(
        live_stt.shutil, "get_terminal_size", lambda *a: types.SimpleNamespace(columns=columns)
    )
    monkeypatch.setattr(sys, "stdout", screen)
    queue = audio_q if audio_q is not None else asyncio.Queue()
    asyncio.run(live_stt.meter(state, queue, translator))
    return screen.writes


def test_the_meter_is_silent_off_a_terminal(monkeypatch):
    """Carriage returns would corrupt a redirected transcript (L-006)."""
    state = live_stt.State()
    state.partial = "あいうえお"
    state.dropped = 7

    assert run_meter(monkeypatch, state, tty=False) == []


def test_an_idle_meter_writes_only_the_line_clear(monkeypatch):
    state = live_stt.State()

    assert run_meter(monkeypatch, state) == [live_stt._LINE_CLEAR]


def test_the_meter_shows_each_backlog_counter_only_when_it_is_nonzero(monkeypatch):
    state = live_stt.State()
    audio_q = live_stt.AudioQueue()
    audio_q.put_nowait(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
    state.segment_queue_depth = 3
    state.dropped = 2
    translator = types.SimpleNamespace(dropped_translations=5)

    written = run_meter(monkeypatch, state, audio_q, translator)[0]

    assert "q=0.50s" in written
    assert "seg=3" in written
    assert "drop=2" in written
    assert "tdrop=5" in written


def test_the_meter_hides_a_translator_with_no_dropped_turns(monkeypatch):
    state = live_stt.State()
    translator = types.SimpleNamespace(dropped_translations=0)

    assert "tdrop" not in run_meter(monkeypatch, state, translator=translator)[0]


def test_a_long_caption_is_tail_truncated_to_the_terminal_width(monkeypatch):
    """A wrapped line survives the next line-clear as residue, so it must fit."""
    state = live_stt.State()
    state.partial = "".join(str(i % 10) for i in range(200))
    state.dropped = 4

    written = run_meter(monkeypatch, state, columns=40)[0]

    body = written[len(live_stt._LINE_CLEAR) :]
    assert len(body) <= 39
    assert state.partial.endswith(body[-10:])  # the newest characters, not the oldest


# --- CLI flags (P1 defaults, mutual exclusion) --------------------------------


def run_cli(monkeypatch, *argv):
    """Run `main()` with the model preflight and the session stubbed out."""
    seen = {}
    monkeypatch.setattr(sys, "argv", ["live-stt", *argv])

    def check_models(engine):
        seen["preflight"] = engine
        return None

    async def run_session(args):
        seen["args"] = args

    monkeypatch.setattr(live_stt, "check_models", check_models)
    monkeypatch.setattr(live_stt, "run_session", run_session)
    live_stt.main()
    return seen


def test_a_bare_run_announces_whisper_on_the_shipped_accelerator(monkeypatch, capsys):
    seen = run_cli(monkeypatch)

    assert seen["preflight"] == "whisper"
    assert "Engine: whisper (local OpenVINO NPU, no network)" in capsys.readouterr().out


def test_the_asr_device_flag_reaches_the_session_and_the_banner(monkeypatch, capsys):
    seen = run_cli(monkeypatch, "--asr-device", "GPU")

    assert seen["args"].asr_device == "GPU"
    assert "local OpenVINO GPU" in capsys.readouterr().out


def test_a_sherpa_engine_is_announced_as_sherpa_not_openvino(monkeypatch, capsys):
    seen = run_cli(monkeypatch, "--engine", "parakeet")

    assert seen["preflight"] == "parakeet"
    assert "Engine: parakeet (local sherpa-onnx, no network)" in capsys.readouterr().out


def test_every_engine_directory_is_selectable_from_the_command_line(monkeypatch):
    for engine in live_stt.ENGINE_DIRS:
        assert run_cli(monkeypatch, "--engine", engine)["preflight"] == engine


def test_the_engine_choices_are_listed_in_a_stable_order(monkeypatch, capsys):
    """`--help` renders `choices` in iteration order, so an unsorted one drifts."""
    monkeypatch.setattr(sys, "argv", ["live-stt", "--help"])

    with pytest.raises(SystemExit) as exit_info:
        live_stt.main()

    assert exit_info.value.code == 0
    assert "{" + ",".join(sorted(live_stt.ENGINE_DIRS)) + "}" in capsys.readouterr().out


def test_an_unknown_engine_is_refused_before_anything_loads(monkeypatch):
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, "--engine", "gemini")
    assert exit_info.value.code == 2


def test_the_context_seed_defaults_to_empty_and_is_passed_through(monkeypatch):
    assert run_cli(monkeypatch)["args"].context == ""
    assert run_cli(monkeypatch, "--context", "医療の会議")["args"].context == "医療の会議"


def test_the_translation_leg_is_on_by_default_and_switchable_off(monkeypatch):
    """Parse level only; the JA-only degrade itself needs a live session (L-004)."""
    assert run_cli(monkeypatch)["args"].no_translate is False
    assert run_cli(monkeypatch, "--no-translate")["args"].no_translate is True


def test_saving_to_a_path_and_not_saving_at_all_cannot_be_asked_for_together(monkeypatch):
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, "-o", "/tmp/x.txt", "--no-save")
    assert exit_info.value.code == 2


def test_a_missing_model_stops_the_run_before_the_session_starts(monkeypatch, capsys):
    started = []
    monkeypatch.setattr(sys, "argv", ["live-stt"])
    monkeypatch.setattr(live_stt, "check_models", lambda engine: "Missing model files")

    async def run_session(args):
        started.append(args)

    monkeypatch.setattr(live_stt, "run_session", run_session)

    with pytest.raises(SystemExit) as exit_info:
        live_stt.main()

    assert exit_info.value.code == 1
    assert started == []
    assert "Missing model files" in capsys.readouterr().err
