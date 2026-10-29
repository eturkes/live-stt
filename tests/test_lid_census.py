"""Locks for the committed language-identification census and its regenerator."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from decimal import Decimal
from itertools import groupby
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import live_stt
from tests import build_lid_census as builder

CENSUS_PATH = Path(__file__).with_name("lid_census.json")
COMMITTED_BYTES = CENSUS_PATH.read_bytes()
CENSUS = json.loads(COMMITTED_BYTES)
VIEWS = CENSUS["views"]
SYNTHETIC = CENSUS["synthetic"]


def test_fixture_schema_and_census() -> None:
    view_fields = [
        "spoken",
        "utterance",
        "bucket",
        "argmax",
        "argmax_score",
        "ja",
        "en",
    ]
    synthetic_fields = ["id", "seconds", "argmax", "argmax_score", "ja", "en"]

    assert set(CENSUS) == {
        "model",
        "model_sha256",
        "runtime",
        "labels_total",
        "view_fields",
        "views",
        "utterances",
        "synthetic_fields",
        "synthetic",
    }
    assert CENSUS["view_fields"] == builder.VIEW_FIELDS == view_fields
    assert CENSUS["synthetic_fields"] == builder.SYNTHETIC_FIELDS == synthetic_fields
    assert CENSUS["model"] == builder.MODEL_NAME == "voxlingua107.onnx"
    assert CENSUS["runtime"] == builder.RUNTIME == "onnxruntime CPUExecutionProvider"
    assert CENSUS["labels_total"] == 107
    assert len(CENSUS["model_sha256"]) == 64
    assert CENSUS["utterances"] == 1747
    assert len(VIEWS) == 8286
    assert len(SYNTHETIC) == 25
    assert Counter(row[2] for row in VIEWS) == {
        "1s": 1746,
        "2s": 1500,
        "3s": 1445,
        "5s": 1211,
        "8s": 637,
        "VADfin": 1747,
    }
    assert Counter(row[0] for row in VIEWS) == {"ja": 4377, "en": 3909}
    assert builder.SAMPLE_RATE == live_stt.SAMPLE_RATE == 16_000
    assert builder.PREFIX_SECONDS == (1, 2, 3, 5, 8)
    assert builder.LANGUAGES == ("ja", "en")
    assert builder.BOUNDARY_SECONDS == 1
    assert builder.ROUND == 6


def test_input_map_accounts_for_every_census_utterance() -> None:
    inputs = json.loads(builder.INPUTS.read_bytes())
    assert set(inputs) == {"index_sha256", "clips"}
    assert inputs["index_sha256"] == builder.INDEX_SHA256
    assert list(inputs["clips"]) == list(builder.LANGUAGES)
    for language, clips in inputs["clips"].items():
        names = [clip["clip"] for clip in clips]
        assert names and names == sorted(set(names))
        for clip in clips:
            assert set(clip) == {"clip", "wav_sha256", "utterances"}
            assert clip["clip"].endswith(".wav")
            assert len(clip["wav_sha256"]) == 64
            assert set(clip["wav_sha256"]) <= set("0123456789abcdef")
            assert type(clip["utterances"]) is int and clip["utterances"] >= 0
        expected = {row[1] for row in VIEWS if row[0] == language}
        assert sum(clip["utterances"] for clip in clips) == len(expected)


def test_views_follow_canonical_utterance_and_bucket_order() -> None:
    spoken = [row[0] for row in VIEWS]
    utterances = [row[1] for row in VIEWS]
    first_en = spoken.index("en")

    assert utterances == sorted(utterances)
    assert all(value == "ja" for value in spoken[:first_en])
    assert all(value == "en" for value in spoken[first_en:])

    seen = []
    expected_prefixes = [f"{seconds}s" for seconds in builder.PREFIX_SECONDS]
    for utterance, rows_iter in groupby(VIEWS, key=lambda row: row[1]):
        rows = list(rows_iter)
        seen.append(utterance)
        assert len({row[0] for row in rows}) == 1
        buckets = [row[2] for row in rows]
        assert buckets[-1] == "VADfin"
        assert buckets[:-1] == expected_prefixes[: len(buckets) - 1]

    assert seen == list(range(CENSUS["utterances"]))


def test_probabilities_are_raw_and_numbers_are_six_decimal() -> None:
    rows = [*VIEWS, *SYNTHETIC]
    raw_pairs = 0
    for row in rows:
        argmax, score, ja, en = row[-4:]
        raw_pairs += ja + en < 1.0
        assert (score == max(ja, en)) is (argmax in {"ja", "en"})
        if argmax == "ja":
            assert score == ja
        elif argmax == "en":
            assert score == en

    assert raw_pairs > len(rows) * 4 / 5

    decimal_payload = json.loads(COMMITTED_BYTES, parse_float=Decimal)
    quantum = Decimal("0.000001")
    for row in [*decimal_payload["views"], *decimal_payload["synthetic"]]:
        assert all(value == value.quantize(quantum) for value in row if isinstance(value, Decimal))


def test_synthetic_grid_and_rejections() -> None:
    probe_names = ("silence", "noise--60dB", "noise--30dB", "noise--10dB", "hum-30dB")
    expected = [
        (f"{name}-{seconds}s", seconds)
        for seconds in builder.PREFIX_SECONDS
        for name in probe_names
    ]

    assert [(row[0], row[1]) for row in SYNTHETIC] == expected
    for _, _, argmax, score, ja, en in SYNTHETIC:
        assert live_stt.lid_accept(argmax, score, ja, en) is None


def test_render_round_trips_committed_bytes() -> None:
    committed_bytes = COMMITTED_BYTES

    assert builder.render(json.loads(committed_bytes)) == committed_bytes
    assert builder.render({"z": 1, "a": [2.0]}) == b'{"z":1,"a":[2.0]}\n'


class FakeDetector:
    """Deterministic scores keyed on view length, so every row differs from its neighbours."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.calls = 0

    def score(self, pcm) -> tuple[str, float, float, float]:
        self.calls += 1
        ja = round(len(pcm) / 1_000_000, 6)
        en = round(1.0 - ja, 6)
        argmax = "ja" if ja >= en else "en"
        return argmax, max(ja, en), ja, en


@pytest.fixture
def injected_corpus(tmp_path, monkeypatch):
    """`builder` wired to a two-clip-per-language fake corpus and a weights-free detector.

    Full mode is otherwise unreachable without the gitignored corpora and the LID weights,
    which is how every other lock here stayed green with `census` replaced by a raise.
    """
    (tmp_path / builder.MODEL_NAME).write_bytes(b"fake weights")
    labels = json.dumps({"ja": 0, "en": 1, "fr": 2})
    (tmp_path / "lang_map.json").write_text(labels, encoding="utf-8")
    monkeypatch.setattr(live_stt, "LID_MODEL_DIR", tmp_path)
    monkeypatch.setattr(live_stt, "LanguageDetector", FakeDetector)
    monkeypatch.setattr(builder, "missing_resources", lambda **_kwargs: [])
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "OUT", tmp_path / "lid_census.json")
    inputs = tmp_path / "lid_census_inputs.json"
    monkeypatch.setattr(builder, "INPUTS", inputs, raising=False)
    monkeypatch.setattr(
        builder.corpus, "file_sha256", lambda path: hashlib.sha256(path.name.encode()).hexdigest()
    )
    inputs.write_text(
        json.dumps(
            {
                "index_sha256": {
                    "ja": builder.corpus.EXPECTED_INDEX_SHA256,
                    "en": builder.corpus.EN_EXPECTED_INDEX_SHA256,
                },
                "clips": {
                    language: [
                        {
                            "clip": f"{language}-{index}.wav",
                            "wav_sha256": hashlib.sha256(
                                f"{language}-{index}.wav".encode()
                            ).hexdigest(),
                            "utterances": 2,
                        }
                        for index in range(2)
                    ]
                    for language in builder.LANGUAGES
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        builder,
        "corpus_clips",
        lambda language, limit=None: [Path(f"{language}-{index}.wav") for index in range(2)][
            :limit
        ],
    )
    monkeypatch.setattr(
        builder,
        "read_clip",
        lambda path: np.full(8 * builder.SAMPLE_RATE, len(path.name) / 100, dtype=np.float32),
    )
    monkeypatch.setattr(
        builder,
        "vad_buffers",
        lambda pcm: [pcm[: round(2.5 * builder.SAMPLE_RATE)], pcm[: 6 * builder.SAMPLE_RATE]],
    )
    return builder.OUT


def test_full_mode_scores_renders_and_compares_bytes(injected_corpus, capsys) -> None:
    assert builder.main([]) == 1
    assert "not byte-identical" in capsys.readouterr().err

    assert builder.main(["--write"]) == 0
    written = injected_corpus.read_bytes()
    payload = json.loads(written)

    assert payload["model_sha256"] == hashlib.sha256(b"fake weights").hexdigest()
    assert payload["labels_total"] == 3
    assert payload["utterances"] == 8
    assert [row[1] for row in payload["views"]] == sorted(row[1] for row in payload["views"])
    assert Counter(row[2] for row in payload["views"]) == {
        "1s": 8,
        "2s": 8,
        "3s": 4,
        "5s": 4,
        "VADfin": 8,
    }
    assert len(payload["synthetic"]) == 25
    assert builder.render(payload) == written

    assert builder.main([]) == 0
    assert "byte-identical" in capsys.readouterr().out

    injected_corpus.write_bytes(written.replace(b'"labels_total":3', b'"labels_total":4'))
    assert builder.main([]) == 1


def test_bounded_mode_rejects_a_rebuild_that_dropped_trailing_rows(
    injected_corpus, monkeypatch
) -> None:
    """A missing final view must fail against the independently selected clip extent."""
    assert builder.main(["--write"]) == 0
    detector = cast(live_stt.LanguageDetector, FakeDetector())

    assert builder.bounded_check(detector, "ja", 2) == (4, 16)

    full = builder.language_views
    monkeypatch.setattr(builder, "language_views", lambda *a, **k: full(*a, **k)[:-1])
    with pytest.raises(ValueError, match="differ from the fixture"):
        builder.bounded_check(detector, "ja", 2)


def test_bounded_census_rejects_a_whole_missing_last_utterance(
    injected_corpus, monkeypatch
) -> None:
    assert builder.main(["--write"]) == 0
    detector = cast(live_stt.LanguageDetector, FakeDetector())
    assert builder.bounded_check(detector, "ja", 2) == (4, 16)

    full = builder.language_views
    monkeypatch.setattr(
        builder,
        "language_views",
        lambda *args, **kwargs: [row for row in full(*args, **kwargs) if row[1] < 3],
    )
    with pytest.raises(ValueError, match="differ from the fixture"):
        builder.bounded_check(detector, "ja", 2)


@pytest.mark.parametrize("corruption", ["index", "name", "hash", "short"])
def test_bounded_census_refuses_an_input_map_for_other_clips(
    injected_corpus, monkeypatch, corruption
) -> None:
    assert builder.main(["--write"]) == 0
    detector = cast(live_stt.LanguageDetector, FakeDetector())
    assert builder.bounded_check(detector, "ja", 2) == (4, 16)
    inputs = json.loads(builder.INPUTS.read_bytes())
    rows = inputs["clips"]["ja"]
    if corruption == "index":
        inputs["index_sha256"]["ja"] = "f" * 64
    elif corruption == "name":
        rows[0]["clip"] = "other.wav"
    elif corruption == "hash":
        rows[0]["wav_sha256"] = "f" * 64
    else:
        rows.pop()
    builder.INPUTS.write_text(json.dumps(inputs), encoding="utf-8")
    with pytest.raises(ValueError, match="(fingerprints|input map)"):
        builder.bounded_check(detector, "ja", 2)


def test_input_map_mode_reproduces_the_cut_without_lid_scoring(
    injected_corpus, monkeypatch
) -> None:
    assert builder.main(["--write"]) == 0
    census_bytes = injected_corpus.read_bytes()
    expected = json.loads(builder.INPUTS.read_bytes())

    def unexpected_detector():
        pytest.fail("input-map mode constructed the LID")

    monkeypatch.setattr(live_stt, "LanguageDetector", unexpected_detector)
    assert builder.main(["--input-map", "--write"]) == 0
    written = builder.INPUTS.read_bytes()
    assert json.loads(written) == expected
    assert builder.main(["--input-map"]) == 0
    assert builder.main(["--input-map", "--write"]) == 0
    assert builder.INPUTS.read_bytes() == written
    assert injected_corpus.read_bytes() == census_bytes

    full = builder.vad_buffers
    monkeypatch.setattr(builder, "vad_buffers", lambda pcm: full(pcm)[:-1])
    assert builder.main(["--input-map", "--write"]) == 1
    assert builder.INPUTS.read_bytes() == written


def test_corpus_clips_sorts_fleurs_and_refuses_a_short_corpus(tmp_path, monkeypatch) -> None:
    entries = [
        {"wav": "fleurs/b.wav", "source": "fleurs"},
        {"wav": "fleurs/a.wav", "source": "fleurs"},
        {"wav": "cv/x.wav", "source": "common_voice"},
    ]
    monkeypatch.setattr(builder, "corpus_dir", lambda language: tmp_path / "fleurs")
    monkeypatch.setattr(
        builder.corpus, "validate_cached_index", lambda directory, expected: entries
    )

    assert [path.name for path in builder.corpus_clips("ja")] == ["a.wav", "b.wav"]
    assert [path.name for path in builder.corpus_clips("en", 1)] == ["a.wav"]
    with pytest.raises(ValueError, match="asked for 5 clips, corpus holds 2"):
        builder.corpus_clips("ja", 5)


def test_bounded_views_match_regeneration() -> None:
    missing = builder.missing_resources()
    if missing:
        pytest.skip("absent: " + ", ".join(missing))

    detector = live_stt.LanguageDetector(live_stt.LID_MODEL_DIR)
    for language in builder.LANGUAGES:
        assert builder.bounded_check(detector, language, 3)[1] > 0

    assert builder.synthetic_views(detector) == SYNTHETIC
