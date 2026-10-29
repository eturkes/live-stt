"""Observe outgoing Codex configuration and RPC policy without starting a server."""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

import live_stt

_TOOL_DISABLES = {
    "web_search": "disabled",
    **dict.fromkeys(
        (
            "features.image_generation",
            "features.browser_use",
            "features.browser_use_external",
            "features.computer_use",
            "features.apps",
            "features.shell_tool",
            "features.view_image",
            "features.goals",
            "features.sleep_tool",
        ),
        False,
    ),
}


def _assert_tool_disables(config):
    for key, value in _TOOL_DISABLES.items():
        assert key in config and config[key] == value, f"tool disable missing or enabled: {key}"


def test_start_passes_every_tool_disable_to_the_app_server(monkeypatch):
    argv = []

    async def unavailable(*args, **_kwargs):
        argv.extend(args)
        raise FileNotFoundError("fake spawn boundary")

    monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", unavailable)
    assert asyncio.run(live_stt.CodexTranslator().start()) is False
    assert argv[:2] == ["codex", "app-server"]
    assert argv[2::2] == ["-c"] * len(live_stt._CODEX_CONFIG)
    config = {}
    for setting in argv[3::2]:
        key, value = setting.split("=", 1)
        assert key not in config
        config[key] = json.loads(value)
    _assert_tool_disables(config)


@pytest.mark.parametrize("key", _TOOL_DISABLES)
@pytest.mark.parametrize("plant", ["missing", "enabled"])
def test_the_config_lock_rejects_each_removed_or_enabled_tool(key, plant):
    config = {key: json.loads(value) for key, value in live_stt._CODEX_CONFIG.items()}
    if plant == "missing":
        config.pop(key)
    else:
        config[key] = "enabled" if key == "web_search" else True
    with pytest.raises(AssertionError, match=key):
        _assert_tool_disables(config)


def _capture_rpc(source):
    async def scenario():
        translator = live_stt.CodexTranslator(sources=("ja", "en"))
        leg = translator._legs[source]
        captured = {}

        async def request(method, params):
            captured[method] = params
            if method == "turn/start":
                return {"turn": {"id": "turn"}}
            return {
                "thread": {"id": "thread-" + source},
                "serviceTier": live_stt.TRANSLATE_SERVICE_TIER,
            }

        translator._request = request  # type: ignore[method-assign]
        leg.thread_id = await translator._new_thread(leg)
        translator._notes.put_nowait(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": leg.thread_id,
                    "turn": {"id": "turn", "status": "completed"},
                },
            }
        )
        assert await translator._turn("spoken text", leg) == ""
        return captured, leg.instructions

    return asyncio.run(scenario())


def _assert_rpc_policy(captured, instructions, source):
    expected = {
        "thread/start": {
            "model": live_stt.TRANSLATE_MODEL,
            "sandbox": "read-only",
            "approvalPolicy": "never",
            "ephemeral": True,
            "personality": "none",
            "developerInstructions": instructions,
            "serviceTier": live_stt.TRANSLATE_SERVICE_TIER,
        },
        "turn/start": {
            "threadId": "thread-" + source,
            "input": [{"type": "text", "text": "spoken text"}],
            "effort": live_stt.TRANSLATE_EFFORT,
            "summary": "none",
        },
    }
    for method, fields in expected.items():
        for key, value in fields.items():
            assert key in captured[method] and captured[method][key] == value, (
                f"{method} policy missing or changed: {key}"
            )


@pytest.mark.parametrize("source", ["ja", "en"])
def test_each_direction_sends_the_configured_model_effort_and_thread_policy(monkeypatch, source):
    # Sentinels prove routing rather than merely matching today's hard-coded defaults.
    monkeypatch.setattr(live_stt, "TRANSLATE_MODEL", "fixture-model")
    monkeypatch.setattr(live_stt, "TRANSLATE_EFFORT", "fixture-effort")
    monkeypatch.setattr(live_stt, "TRANSLATE_SERVICE_TIER", "fixture-tier")
    captured, instructions = _capture_rpc(source)
    _assert_rpc_policy(captured, instructions, source)


@pytest.mark.parametrize(
    ("method", "key"),
    [
        ("thread/start", key)
        for key in (
            "model",
            "sandbox",
            "approvalPolicy",
            "ephemeral",
            "personality",
            "developerInstructions",
            "serviceTier",
        )
    ]
    + [("turn/start", key) for key in ("threadId", "input", "effort", "summary")],
)
@pytest.mark.parametrize("plant", ["missing", "changed"])
def test_the_rpc_policy_lock_rejects_missing_or_changed_fields(method, key, plant):
    captured, instructions = _capture_rpc("ja")
    planted = copy.deepcopy(captured)
    if plant == "missing":
        del planted[method][key]
    else:
        planted[method][key] = "wrong-policy"
    with pytest.raises(AssertionError, match=f"{method} policy missing or changed: {key}"):
        _assert_rpc_policy(planted, instructions, "ja")


def _assert_frames(writes, objects):
    assert len(writes) == len(objects)
    for data, obj in zip(writes, objects, strict=True):
        assert data.endswith(b"\n") and data.count(b"\n") == 1, (
            "RPC record lacks one terminal newline"
        )
        assert json.loads(data) == obj


def test_every_outgoing_rpc_record_has_its_own_terminal_newline():
    writes = []
    translator = live_stt.CodexTranslator()
    translator._proc = SimpleNamespace(stdin=SimpleNamespace(write=writes.append))  # type: ignore[assignment]
    objects = [{"method": "first", "params": {"text": "spoken\ntext"}}, {"id": 1, "result": {}}]
    for obj in objects:
        translator._write(obj)
    _assert_frames(writes, objects)


@pytest.mark.parametrize("plant", ["missing", "extra"])
def test_the_framing_lock_rejects_unterminated_or_split_records(plant):
    obj = {"method": "fixture"}
    data = json.dumps(obj).encode() + (b"" if plant == "missing" else b"\n\n")
    with pytest.raises(AssertionError, match="terminal newline"):
        _assert_frames([data], [obj])
