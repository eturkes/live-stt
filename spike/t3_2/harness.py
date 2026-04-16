"""Mock Gemini Live client + session for offline benchmarking the T3.2 prototypes.

The mock is intentionally narrow: it reproduces just enough of
`client.aio.live.connect(model, config).__aenter__() -> AsyncSession` and the
subset of `AsyncSession` the prototypes use, driven by a scripted scenario.

Design

    MockClient(scenario) -> .aio.live.connect(...) -> async-cm -> MockSession
    MockSession.send_realtime_input(audio=Blob) | send_client_content(turns, ...)
              .receive() -> async-gen of LiveServerMessage
    A scenario runs an async coroutine that drives the next session via a
    controller handle — pushing transcription messages, issuing go_away,
    forcing WebSocket close. All sends are recorded on the session for
    assertions.

The harness is time-accelerated: scenarios use `ctrl.advance(seconds)` which
does an `asyncio.sleep` but scales by TIME_SCALE so a 45-minute simulated
session runs in ~30s of wall-clock.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from google.genai import types

TIME_SCALE = 0.01  # 1 simulated second = 10 ms wall-clock; 45 min = 27 s
EPS = 1e-9


@dataclass
class SentAudio:
    t: float
    size: int


@dataclass
class SentClientContent:
    t: float
    turn_count: int
    text_bytes: int
    turn_complete: bool


@dataclass
class SessionRecord:
    """Everything that happened to one MockSession, for later assertions."""

    connect_time: float
    config: Any
    handle_passed: str | None
    audio_sent: list[SentAudio] = field(default_factory=list)
    client_content_sent: list[SentClientContent] = field(default_factory=list)
    audio_stream_end_sent: bool = False
    closed_time: float | None = None
    issued_handle: str | None = None
    messages_emitted: int = 0


class MockSession:
    """Mimics google.genai.live.AsyncSession as used by the prototypes."""

    def __init__(
        self,
        config,
        handle_passed: str | None,
        controller: SessionController,
    ):
        self._inbox: asyncio.Queue[types.LiveServerMessage | None] = asyncio.Queue()
        self._closed = asyncio.Event()
        self._ctrl = controller
        self.record = SessionRecord(
            connect_time=controller.now,
            config=config,
            handle_passed=handle_passed,
        )

    async def send_realtime_input(self, *, audio=None, audio_stream_end=None):
        if self._closed.is_set():
            raise RuntimeError("session is closed")
        if audio is not None:
            size = len(audio.data) if hasattr(audio, "data") else 0
            self.record.audio_sent.append(SentAudio(t=self._ctrl.now, size=size))
        if audio_stream_end:
            self.record.audio_stream_end_sent = True

    async def send_client_content(self, *, turns=None, turn_complete=False):
        if self._closed.is_set():
            raise RuntimeError("session is closed")
        turn_list = list(turns or [])
        nbytes = 0
        for t in turn_list:
            for p in getattr(t, "parts", []) or []:
                if getattr(p, "text", None):
                    nbytes += len(p.text.encode("utf-8"))
        self.record.client_content_sent.append(
            SentClientContent(
                t=self._ctrl.now,
                turn_count=len(turn_list),
                text_bytes=nbytes,
                turn_complete=turn_complete,
            )
        )

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        """Stream server-side messages until closed or turn_complete.

        Mirrors real AsyncSession.receive(): yields until turn_complete, at which
        point the generator exits (#1224). Raises ConnectionResetError if called
        after the session has been closed — this forces the client's outer loop
        to notice a dead connection and reconnect, rather than spin on a
        zero-yield generator.
        """
        if self._closed.is_set() and self._inbox.empty():
            raise ConnectionResetError("mock session closed")
        while True:
            get_task = asyncio.create_task(self._inbox.get())
            close_task = asyncio.create_task(self._closed.wait())
            done, pending = await asyncio.wait(
                {get_task, close_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for p in pending:
                p.cancel()
            if get_task in done:
                msg = get_task.result()
            else:
                # Closed while we were waiting.
                raise ConnectionResetError("mock session closed")
            if msg is None:
                return
            self.record.messages_emitted += 1
            yield msg
            if msg.server_content and msg.server_content.turn_complete:
                return

    def _inject(self, msg: types.LiveServerMessage | None):
        """Scenario drops a message into this session's receive stream."""
        self._inbox.put_nowait(msg)

    def _close(self):
        if not self._closed.is_set():
            self.record.closed_time = self._ctrl.now
            self._closed.set()
            self._inbox.put_nowait(None)  # wake up receive()


class SessionController:
    """Lives for one MockSession. Scenario uses it to drive messages."""

    def __init__(self, scenario_clock: ScenarioClock):
        self._clock = scenario_clock
        self.session: MockSession | None = None
        self.handle_issued: str | None = None
        self.done = asyncio.Event()

    @property
    def now(self) -> float:
        return self._clock.now

    def attach(self, session: MockSession):
        self.session = session

    def emit_block(self, ja: str, en: str | None = None, turn_complete: bool = True):
        """Inject a complete JA/EN transcription block."""
        assert self.session is not None
        parts: list[str] = [f"JA: {ja}"]
        if en:
            parts.append(f"EN: {en}")
        text = "\n".join(parts)
        msg = types.LiveServerMessage(
            server_content=types.LiveServerContent(
                output_transcription=types.Transcription(text=text),
                turn_complete=turn_complete,
            )
        )
        self.session._inject(msg)
        if turn_complete:
            # End this receive() invocation; the prototype's while-loop will call again.
            pass

    def emit_partial(self, text: str):
        msg = types.LiveServerMessage(
            server_content=types.LiveServerContent(
                output_transcription=types.Transcription(text=text),
                turn_complete=False,
            )
        )
        self.session._inject(msg)

    def issue_handle(self, handle: str | None = None, resumable: bool = True):
        if handle is None:
            handle = "h_" + secrets.token_hex(4)
        self.handle_issued = handle
        msg = types.LiveServerMessage(
            session_resumption_update=types.LiveServerSessionResumptionUpdate(
                new_handle=handle,
                resumable=resumable,
            )
        )
        self.session._inject(msg)

    def go_away(self, time_left: str = "60s"):
        msg = types.LiveServerMessage(go_away=types.LiveServerGoAway(time_left=time_left))
        self.session._inject(msg)

    def force_close(self):
        self.session._close()
        self.done.set()


class ScenarioClock:
    def __init__(self):
        self._t0 = time.monotonic()

    @property
    def now(self) -> float:
        return (time.monotonic() - self._t0) / TIME_SCALE

    async def advance(self, seconds: float):
        await asyncio.sleep(seconds * TIME_SCALE)


@dataclass
class HarnessLog:
    sessions: list[SessionRecord] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0


class MockLiveNamespace:
    """Fake `client.aio.live` with `.connect(model=..., config=...)`."""

    def __init__(self, client: MockClient):
        self._client = client

    def connect(self, *, model: str, config):
        return _ConnectCM(self._client, model=model, config=config)


class MockAioNamespace:
    def __init__(self, client: MockClient):
        self.live = MockLiveNamespace(client)
        self.models = _MockModels(client)


class _MockModels:
    """Minimal stand-in for client.aio.models used by prototype C's extractor.

    If the owning MockClient was given an `entity_response_fn`, each call routes
    through it so scenarios can return realistic JSON for the extractor.
    """

    def __init__(self, client: MockClient):
        self._client = client

    async def generate_content(self, *, model: str, contents, config=None):
        fn = self._client._entity_response_fn
        if fn is None:
            return _FakeResp(text="[]")
        prompt = contents if isinstance(contents, str) else str(contents)
        self._client._entity_calls += 1
        try:
            result = fn(prompt, self._client._entity_calls)
            import json as _json
            return _FakeResp(text=_json.dumps(result))
        except Exception:
            return _FakeResp(text="[]")


@dataclass
class _FakeResp:
    text: str


class _ConnectCM:
    """Async context manager returned by mock live.connect."""

    def __init__(self, client: MockClient, *, model: str, config):
        self._client = client
        self._model = model
        self._config = config
        self._session: MockSession | None = None

    async def __aenter__(self) -> MockSession:
        if self._client.all_scripts_done.is_set():
            # Refuse further connects once the scenario has played out. Simulates
            # a shut-down server; lets the test end without empty churn sessions.
            raise ConnectionRefusedError("mock scenario exhausted")
        handle = (
            self._config.session_resumption.handle
            if self._config and self._config.session_resumption
            else None
        )
        ctrl = SessionController(self._client._clock)
        session = MockSession(self._config, handle, ctrl)
        ctrl.attach(session)
        self._session = session
        self._client._log.sessions.append(session.record)
        task = asyncio.create_task(self._client._drive_next_session(ctrl))
        self._driver_task = task
        return session

    async def __aexit__(self, exc_type, exc, tb):
        if self._session is not None:
            self._session._close()
        try:
            await asyncio.wait_for(self._driver_task, timeout=1.0)
        except (TimeoutError, asyncio.TimeoutError):
            self._driver_task.cancel()
        except Exception:
            pass
        return False  # do not suppress


SessionScript = Callable[[SessionController], Awaitable[None]]


class MockClient:
    """Top-level fake for genai.Client(api_key=...)."""

    def __init__(
        self,
        scripts: list[SessionScript],
        entity_response_fn: Callable[[str, int], list[dict]] | None = None,
    ):
        self._entity_response_fn = entity_response_fn
        self._entity_calls = 0
        self.aio = MockAioNamespace(self)
        self._scripts = list(scripts)
        self._total_scripts = len(self._scripts)
        self._scripts_completed = 0
        self.all_scripts_done = asyncio.Event()
        self._clock = ScenarioClock()
        self._log = HarnessLog()
        self._log.start_time = self._clock.now

    async def _drive_next_session(self, ctrl: SessionController):
        if not self._scripts:
            # No more scripts — keep session idle briefly then close.
            await self._clock.advance(1.0)
            ctrl.force_close()
            return
        script = self._scripts.pop(0)
        try:
            await script(ctrl)
        except Exception as e:
            print(f"[harness] script failed: {e!r}")
        finally:
            if not ctrl.done.is_set():
                ctrl.force_close()
            self._scripts_completed += 1
            if self._scripts_completed >= self._total_scripts:
                self.all_scripts_done.set()

    @property
    def log(self) -> HarnessLog:
        return self._log

    def finalize(self):
        self._log.end_time = self._clock.now


def mock_client_factory(
    scripts: list[SessionScript],
    entity_response_fn: Callable[[str, int], list[dict]] | None = None,
):
    """Builder matching prototype run_session(client_factory=...) signature."""

    holder: dict = {}

    def _factory(_api_key):
        client = MockClient(scripts, entity_response_fn=entity_response_fn)
        holder["client"] = client
        return client

    _factory.holder = holder  # caller can read holder['client'] after run
    return _factory


class FakeStream:
    """No-op stand-in for sounddevice.InputStream during benchmarks."""

    def __init__(self):
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def fake_stream_factory(_state, _audio_q, _loop):
    return FakeStream()
