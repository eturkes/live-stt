"""Locks for the D-016 claim registry and its omission validator (M11.3a).

Three layers, each answering a different failure:

- Registry shape and disposition census: the registry is the specification M11.3d
  implements against, so a silently retyped row or a quietly cleared `absent` is the
  defect these catch.
- Both-ways fixture grading: a validator that never fires is indistinguishable from a
  green one, so the committed fixtures prove it fires and prove it stops firing.
- Legacy pointer re-derivation: the source census recorded one key path that does not
  exist in its artifact (`C29`), which no amount of value-checking would have found.
  Those tests skip when the gitignored `.scratch/` tree is absent.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import validate_d016_claims as V  # noqa: E402

TESTS = Path(__file__).parent
REPO = TESTS.parent
REGISTRY = TESTS / "d016_claims.json"
FIXTURE_COMPLETE = TESTS / "d016_fixture_complete.json"
FIXTURE_UNMEASURED = TESTS / "d016_fixture_unmeasured.json"
FIXTURE_EVIDENCE = TESTS / "d016_fixture_evidence.json"
SCRATCH = REPO / ".scratch"

# The source census skips C34: it was the enumeration prompt's audit sentinel, not a
# claim. Naming the gap here stops a later renumber from silently closing it.
EXPECTED_IDS = [f"C{n}" for n in range(1, 42) if n != 34]
EXPECTED_DISPOSITIONS = {
    "measured": 31,
    "narrowed": 4,
    "legacy-retained": 1,
    "corrected": 1,
    "absent": 3,
}
# No M11 unit measures these six, so each carries a ruled disposition instead.
RULED_UNMEASURED = {
    "C23": "narrowed",
    "C29": "legacy-retained",
    "C37": "narrowed",
    "C38": "narrowed",
    "C39": "narrowed",
    "C41": "corrected",
}
# M11.5b must re-derive or correct these three; nothing else may become `absent`.
EXPECTED_ABSENT = {"C13", "C32", "C33"}


@pytest.fixture(scope="module")
def registry() -> dict:
    return V.load(REGISTRY)


@pytest.fixture(scope="module")
def rows(registry: dict) -> list[dict]:
    return registry["claims"]


def test_registry_is_structurally_clean(registry: dict) -> None:
    assert V.validate_structure(registry) == []


def test_registry_carries_the_forty_census_ids(rows: list[dict]) -> None:
    assert [row["id"] for row in rows] == EXPECTED_IDS
    assert len(rows) == 40


def test_c34_is_absent_because_it_was_never_a_claim(rows: list[dict]) -> None:
    assert "C34" not in {row["id"] for row in rows}


def test_disposition_census(rows: list[dict]) -> None:
    assert Counter(row["disposition"] for row in rows) == EXPECTED_DISPOSITIONS


def test_the_six_unmeasured_rows_carry_their_ruled_disposition(rows: list[dict]) -> None:
    by_id = {row["id"]: row["disposition"] for row in rows}
    assert {rid: by_id[rid] for rid in RULED_UNMEASURED} == RULED_UNMEASURED


def test_absent_rows_are_exactly_the_three_unsourced_numbers(rows: list[dict]) -> None:
    absent = {row["id"] for row in rows if row["disposition"] == "absent"}
    assert absent == EXPECTED_ABSENT


def test_absent_rows_record_the_failed_search(rows: list[dict]) -> None:
    for row in rows:
        if row["disposition"] == "absent":
            assert row["note"] and row["note"].strip(), row["id"]
            assert row["legacy_source"]["status"] == "absent", row["id"]


def test_no_row_carries_a_measured_value(rows: list[dict]) -> None:
    for row in rows:
        assert set(row) == set(V.ROW_KEYS), row["id"]
        assert "measured_value" not in row, row["id"]


def test_measured_rows_are_the_only_ones_with_a_key_path(rows: list[dict]) -> None:
    for row in rows:
        has_path = row["evaluator_key_path"] is not None
        assert has_path == (row["disposition"] == "measured"), row["id"]


def test_replacements_belong_to_corrected_and_narrowed_alone(rows: list[dict]) -> None:
    for row in rows:
        has_replacement = row["replacement"] is not None
        assert has_replacement == (row["disposition"] in ("corrected", "narrowed")), row["id"]


def test_registry_fails_only_on_its_three_known_omissions(registry: dict) -> None:
    """With every measured path resolvable, `absent` is what is left failing."""
    evidence = _evidence_satisfying_every_measured_row(registry)
    errors = V.validate(registry, evidence)
    assert {error.split(":", 1)[0] for error in errors} == EXPECTED_ABSENT
    assert all("unresolved omission" in error for error in errors)


def _evidence_satisfying_every_measured_row(registry: dict) -> dict:
    """Synthesize the evidence M11.5b will produce, so the omission half is isolated."""
    evidence: dict = {}
    for row in registry["claims"]:
        path = row["evaluator_key_path"]
        if path is None:
            continue
        node = evidence
        for token in path[:-1]:
            node = node.setdefault(token, {})
        node[path[-1]] = row["claim_value"]
    return evidence


def test_complete_fixture_grades_zero() -> None:
    errors = V.validate(V.load(FIXTURE_COMPLETE), V.load(FIXTURE_EVIDENCE))
    assert errors == []


def test_unmeasured_fixture_grades_nonzero() -> None:
    errors = V.validate(V.load(FIXTURE_UNMEASURED), V.load(FIXTURE_EVIDENCE))
    assert len(errors) == 6
    assert all("unresolved omission" in error for error in errors)


def test_the_two_fixtures_describe_the_same_claims() -> None:
    """Otherwise the pair grades two different things and proves nothing."""
    complete = [row["id"] for row in V.load(FIXTURE_COMPLETE)["claims"]]
    unmeasured = [row["id"] for row in V.load(FIXTURE_UNMEASURED)["claims"]]
    assert complete == unmeasured


@pytest.mark.parametrize(
    ("inventory", "expected_rc"),
    [(FIXTURE_COMPLETE, 0), (FIXTURE_UNMEASURED, 1)],
)
def test_cli_exit_codes(inventory: Path, expected_rc: int) -> None:
    done = subprocess.run(
        [
            sys.executable,
            str(TESTS / "validate_d016_claims.py"),
            str(inventory),
            str(FIXTURE_EVIDENCE),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == expected_rc, done.stdout + done.stderr


def test_cli_rejects_a_wrong_argument_count() -> None:
    done = subprocess.run(
        [sys.executable, str(TESTS / "validate_d016_claims.py"), str(FIXTURE_COMPLETE)],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2


# --- adversarial validator probes (contract P8) ---------------------------------------
#
# Row indices into the complete fixture: 0 = measured, 3 = narrowed, 4 = legacy-retained.
# Each probe asserts the offending id appears in some error, never an exact message:
# error text is not frozen, the id-prefix contract is.

PROBES = [
    ("duplicate_id", 0, lambda i: i["claims"][1].update(id="C1"), "C1"),
    ("unknown_row_key", 0, lambda i: i["claims"][0].update(surprise=1), "C1"),
    ("measured_value_key", 0, lambda i: i["claims"][0].update(measured_value=0.5), "C1"),
    ("missing_field", 0, lambda i: i["claims"][0].pop("metric"), "C1"),
    ("bad_disposition", 0, lambda i: i["claims"][0].update(disposition="guessed"), "C1"),
    ("null_claim_value", 0, lambda i: i["claims"][0].update(claim_value=None), "C1"),
    ("blank_metric", 0, lambda i: i["claims"][0].update(metric="   "), "C1"),
    ("spaced_metric", 0, lambda i: i["claims"][0].update(metric="ref chars"), "C1"),
    ("measured_null_path", 0, lambda i: i["claims"][0].update(evaluator_key_path=None), "C1"),
    ("measured_empty_path", 0, lambda i: i["claims"][0].update(evaluator_key_path=[]), "C1"),
    (
        "measured_unresolvable",
        0,
        lambda i: i["claims"][0].update(evaluator_key_path=["nope"]),
        "C1",
    ),
    ("non_measured_has_path", 3, lambda i: i["claims"][3].update(evaluator_key_path=["a"]), "C4"),
    (
        "replacement_blank_field",
        3,
        lambda i: i["claims"][3]["replacement"].update(source="  "),
        "C4",
    ),
    ("replacement_unknown_key", 3, lambda i: i["claims"][3]["replacement"].update(x=1), "C4"),
    (
        "replacement_on_measured",
        0,
        lambda i: i["claims"][0].update(replacement={"claim": "a", "source": "b"}),
        "C1",
    ),
    ("legacy_unknown_key", 0, lambda i: i["claims"][0]["legacy_source"].update(x=1), "C1"),
    (
        "legacy_retained_null_value",
        4,
        lambda i: i["claims"][4]["legacy_source"].update(stored_value=None),
        "C5",
    ),
    (
        "present_source_blank_path",
        0,
        lambda i: i["claims"][0]["legacy_source"].update(path="  "),
        "C1",
    ),
    (
        "absent_source_blank_note",
        0,
        lambda i: _make_absent_source(i["claims"][0], note="   "),
        "C1",
    ),
    ("row_not_an_object", 0, lambda i: i["claims"].__setitem__(0, "text"), "claims[0]"),
]


def _make_absent_source(row: dict, note: str) -> None:
    row["legacy_source"] = {"status": "absent", "path": None, "key": None, "stored_value": None}
    row["note"] = note


@pytest.mark.parametrize(("name", "_row", "mutate", "offender"), PROBES, ids=[p[0] for p in PROBES])
def test_validator_rejects(name: str, _row: int, mutate, offender: str) -> None:
    inventory = json.loads(FIXTURE_COMPLETE.read_text(encoding="utf-8"))
    mutate(inventory)
    errors = V.validate(inventory, V.load(FIXTURE_EVIDENCE))
    assert errors, f"{name} produced no error"
    assert any(error.startswith(f"{offender}:") for error in errors), f"{name}: {errors}"


def test_the_probe_fixture_is_green_unmutated() -> None:
    """The positive control: without it every probe above could pass vacuously."""
    assert V.validate(V.load(FIXTURE_COMPLETE), V.load(FIXTURE_EVIDENCE)) == []


@pytest.mark.parametrize("version", [True, 1.0, "1", 2, None])
def test_schema_version_must_be_the_integer_one(version: object) -> None:
    # `bool` is an `int` subclass and `True == 1`, so an untyped equality test passes it.
    inventory = json.loads(FIXTURE_COMPLETE.read_text(encoding="utf-8"))
    inventory["schema_version"] = version
    assert any("schema_version" in error for error in V.validate_structure(inventory))


def test_unknown_top_level_key_is_rejected() -> None:
    inventory = json.loads(FIXTURE_COMPLETE.read_text(encoding="utf-8"))
    inventory["extra"] = 1
    assert any("unknown top-level key" in error for error in V.validate_structure(inventory))


def test_claims_must_be_a_list_rather_than_raising() -> None:
    assert V.validate_structure({"schema_version": 1, "claims": {}}) != []


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_load_rejects_non_finite_literals(tmp_path: Path, literal: str) -> None:
    """Python's json accepts these by default; a non-finite claim compares unequal to itself."""
    path = tmp_path / "bad.json"
    path.write_text(f'{{"schema_version": 1, "claims": [], "x": {literal}}}', encoding="utf-8")
    with pytest.raises(ValueError):
        V.load(path)


def _single_measured_row(key_path: list[str]) -> dict:
    """The complete fixture reduced to one measured row, so evidence stays a stub."""
    inventory = json.loads(FIXTURE_COMPLETE.read_text(encoding="utf-8"))
    inventory["claims"] = [inventory["claims"][0]]
    inventory["claims"][0]["evaluator_key_path"] = key_path
    return inventory


@pytest.mark.parametrize("value", [0, False, "", [], {}])
def test_falsy_resolutions_are_not_omissions(value: object) -> None:
    """Only a null resolution means the metric was never written."""
    assert V.validate(_single_measured_row(["falsy"]), {"falsy": value}) == []


def test_null_resolution_is_an_omission() -> None:
    errors = V.validate(_single_measured_row(["nulled"]), {"nulled": None})
    assert any(error.startswith("C1:") for error in errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [("corpus", "made_up"), ("arm", "made_up"), ("device", "TPU")],
)
def test_enum_fields_are_strict(field: str, value: str) -> None:
    inventory = json.loads(FIXTURE_COMPLETE.read_text(encoding="utf-8"))
    inventory["claims"][0][field] = value
    assert any(error.startswith("C1:") for error in V.validate_structure(inventory))


@pytest.mark.parametrize("path", [["  "], ["ok", ""], [1]])
def test_key_path_tokens_must_be_non_blank_strings(path: list) -> None:
    inventory = _single_measured_row(path)
    assert any(error.startswith("C1:") for error in V.validate_structure(inventory))


def test_non_measured_rows_never_touch_evidence() -> None:
    """A narrowed row must pass against empty evidence; only `measured` resolves."""
    inventory = json.loads(FIXTURE_COMPLETE.read_text(encoding="utf-8"))
    inventory["claims"] = [inventory["claims"][3]]
    assert V.validate(inventory, {}) == []


def test_key_path_walks_dicts_before_list_positions() -> None:
    """An all-digit key in an evidence object must stay reachable."""
    assert V.resolve_key_path({"0": "by-key"}, ["0"]) == "by-key"
    assert V.resolve_key_path(["by-index"], ["0"]) == "by-index"
    with pytest.raises(LookupError):
        V.resolve_key_path(["only"], ["name"])
    with pytest.raises(LookupError):
        V.resolve_key_path(["only"], ["7"])
    with pytest.raises(LookupError):
        V.resolve_key_path(3, ["anything"])


# --- legacy pointer re-derivation (skips without the gitignored .scratch/ tree) -------

_LINE_KEY = re.compile(r"^lines?\s")


def _present_rows() -> list[dict]:
    return [r for r in V.load(REGISTRY)["claims"] if r["legacy_source"]["status"] == "present"]


def _requires_scratch(path: str) -> Path:
    artifact = REPO / path
    if not artifact.exists():
        pytest.skip(f"{path} absent: .scratch/ is gitignored evidence")
    return artifact


@pytest.mark.parametrize("row", _present_rows(), ids=lambda r: r["id"])
def test_legacy_pointer_resolves_in_its_artifact(row: dict) -> None:
    """A stored value that is right behind a key path that is wrong is still a defect."""
    legacy = row["legacy_source"]
    paths = legacy["path"].split(",")
    keys = legacy["key"].split(",")

    if _LINE_KEY.match(legacy["key"]):
        _check_line_pointer(_requires_scratch(paths[0]), legacy)
        return

    stored = legacy["stored_value"]
    expected = stored if len(paths) > 1 else [stored]
    assert len(paths) == len(keys), f"{row['id']}: {len(paths)} paths against {len(keys)} keys"
    for path, key, want in zip(paths, keys, expected, strict=True):
        node = json.loads(_requires_scratch(path).read_text(encoding="utf-8"))
        for token in key.split("."):
            assert isinstance(node, dict) and token in node, f"{row['id']}: no {key} in {path}"
            node = node[token]
        assert node == want, f"{row['id']}: {path}:{key} holds {node!r}, registry says {want!r}"


def _check_line_pointer(artifact: Path, legacy: dict) -> None:
    lines = artifact.read_text(encoding="utf-8").splitlines()
    numbers = [int(n) for n in re.findall(r"\d+", legacy["key"].split(" ", 1)[1].split()[0])]
    assert numbers, f"unparseable line key {legacy['key']!r}"
    for number in numbers:
        assert 1 <= number <= len(lines), f"{artifact.name} has no line {number}"

    stored = legacy["stored_value"]
    if isinstance(stored, str):
        return  # A prose summary is not byte-comparable; the line reference is the claim.
    wanted = stored if isinstance(stored, list) else [stored]
    body = "\n".join(lines[number - 1] for number in numbers)
    for value in wanted:
        assert str(value) in body, f"{artifact.name} lines {numbers} do not carry {value}"
