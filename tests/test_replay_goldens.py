"""Golden regeneration preserves unavailable cells; external boundaries are fake."""

import json

import pytest

from tests import gen_replay_goldens as goldens
from tests.evaluator_fakes import hermetic_evaluator  # noqa: F401

pytestmark = pytest.mark.usefixtures("hermetic_evaluator")


@pytest.mark.parametrize("absent", ["model", "device", "wav"])
def test_unavailable_cells_keep_prior_rows_available_cells_refresh_and_outside_cells_drop(
    monkeypatch, tmp_path, capsys, absent
):
    cache = tmp_path / "cache"
    cache.mkdir()
    target = tmp_path / "goldens.json"
    kept = dict(
        ja_ref="old ref",
        purpose="old purpose",
        device="NPU",
        n_segments=1,
        segments=[dict(start=3, n=4, text="kept")],
    )
    target.write_text(
        json.dumps(
            {
                "whisper": {"blocked": kept},
                "k2v2": {"refresh": {"old": True}, "obsolete": {"old": True}},
                "retired": {"blocked": {"old": True}},
            }
        ),
        encoding="utf-8",
    )
    for cid in ("refresh", "new", *(() if absent == "wav" else ("blocked", "unseen"))):
        (cache / f"{cid}.wav").touch()
    monkeypatch.setattr(goldens, "ROOT", tmp_path)
    monkeypatch.setattr(goldens, "CACHE", cache)
    monkeypatch.setattr(goldens, "GOLDENS", target)
    monkeypatch.setattr(goldens, "ASR_DEVICE", "NPU")
    monkeypatch.setattr(
        goldens,
        "all_clips",
        lambda: [(cid, f"ref {cid}", "purpose") for cid in ("refresh", "new", "blocked", "unseen")],
    )
    monkeypatch.setattr(
        goldens, "matrix", lambda: {"k2v2": ["refresh", "new"], "whisper": ["blocked", "unseen"]}
    )
    monkeypatch.setattr(
        goldens,
        "check_models",
        lambda engine: "model absent" if absent == "model" and engine == "whisper" else None,
    )
    monkeypatch.setattr(
        goldens,
        "check_device",
        lambda engine, _: "device absent" if absent == "device" and engine == "whisper" else None,
    )
    calls = []

    def replay(path, engine, device):
        calls.append((path.stem, engine, device))
        return dict(
            n_segments=1,
            overall_rtf=0.25,
            segments=[dict(start=10, n=20, text="fresh", decode_s=0.25)],
        )

    monkeypatch.setattr(goldens.replay, "replay_wav", replay)
    goldens.main()
    expected = {
        "k2v2": {
            cid: dict(
                ja_ref=f"ref {cid}",
                purpose="purpose",
                n_segments=1,
                segments=[dict(start=10, n=20, text="fresh")],
            )
            for cid in ("refresh", "new")
        },
        "whisper": {"blocked": kept},
    }
    assert json.loads(target.read_text(encoding="utf-8")) == expected
    assert calls == [("refresh", "k2v2", "NPU"), ("new", "k2v2", "NPU")]
    stderr = capsys.readouterr().err
    for line in (
        "keep whisper/blocked:",
        "omit whisper/unseen:",
        "drop k2v2/obsolete: outside the matrix",
        "drop retired/blocked: outside the matrix",
    ):
        assert line in stderr
    first = target.read_bytes()
    goldens.main()
    assert target.read_bytes() == first
