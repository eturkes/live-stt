"""Locks for the committed language-identification census and its regenerator."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from itertools import groupby
from pathlib import Path

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
    assert CENSUS["utterances"] == 1926
    assert len(VIEWS) == 8628
    assert len(SYNTHETIC) == 25
    assert Counter(row[2] for row in VIEWS) == {
        "1s": 1925,
        "2s": 1516,
        "3s": 1453,
        "5s": 1188,
        "8s": 620,
        "VADfin": 1926,
    }
    assert Counter(row[0] for row in VIEWS) == {"ja": 4485, "en": 4143}
    assert builder.SAMPLE_RATE == live_stt.SAMPLE_RATE == 16_000
    assert builder.PREFIX_SECONDS == (1, 2, 3, 5, 8)
    assert builder.LANGUAGES == ("ja", "en")
    assert builder.BOUNDARY_SECONDS == 1
    assert builder.ROUND == 6


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


def test_bounded_views_match_regeneration() -> None:
    missing = builder.missing_resources()
    if missing:
        pytest.skip("absent: " + ", ".join(missing))

    detector = live_stt.LanguageDetector(live_stt.LID_MODEL_DIR)
    for language in builder.LANGUAGES:
        regenerated = builder.language_views(detector, language, limit=3)
        assert regenerated
        assert regenerated == builder.committed_views(language)[: len(regenerated)]

    assert builder.synthetic_views(detector) == SYNTHETIC
