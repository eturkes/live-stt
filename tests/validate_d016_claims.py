#!/usr/bin/env python3
"""Structural + omission validator for the D-016 claim registry.

Run it as:

    uv run python tests/validate_d016_claims.py tests/d016_claims.json EVIDENCE.json

Exit 0 means every claim row is present and sourced. Exit 1 prints one error per line.

The registry (`tests/d016_claims.json`) is the single canonical claim->source surface:
M11.3d implements the `measured` key paths, M11.5b populates them, M11.6a renders the
human-facing view from it. A second inventory would let two validators drift.

Two rules carry the judgment and are why this file is kernel-tier:

- The validator never compares a resolved value to `claim_value`. It asserts a claim is
  present and sourced, never that it is true. Treating the registry as an oracle would
  let a wrong claim certify itself.
- `disposition == "absent"` is always an error. That is the omission half: a metric that
  no unit measures fails loudly here instead of yielding a byte-stable but incomplete
  baseline. Three rows are deliberately `absent` today (D-016's retention CER 0.0583 and
  the two append-only post-fix numbers, which no `.scratch/` artifact carries), so this
  validator is deliberately NOT a direct `gate.py` step until M11.5b clears them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

DISPOSITIONS = ("measured", "corrected", "narrowed", "legacy-retained", "absent")
CORPORA = ("long_form", "retention")
ARMS = ("vac", "vad", "stream", "vad_prev", "vad_hotwords", "whole_file")
DEVICES = ("NPU", "GPU", "CPU")
LEGACY_STATUSES = ("present", "absent")

# Every row carries every key; a nullable field is spelled `null`, never omitted. Uniform
# rows are what make a dropped field a structural error rather than a silent default.
ROW_KEYS = (
    "id",
    "claim",
    "claim_value",
    "metric",
    "corpus",
    "arm",
    "device",
    "disposition",
    "evaluator_key_path",
    "replacement",
    "legacy_source",
    "note",
)
LEGACY_KEYS = ("status", "path", "key", "stored_value")
REPLACEMENT_KEYS = ("claim", "source")

_FILE = "<inventory>"


def _reject_non_finite(token: str) -> Any:
    raise ValueError(f"non-finite literal {token} is not JSON and cannot carry a claim")


def load(path: str | Path) -> dict[str, Any]:
    """Parse an inventory or evidence file. Raises on unreadable or malformed JSON.

    `parse_constant` is what rejects `NaN`/`Infinity`: Python accepts those literals by
    default, and a non-finite claim value would compare unequal to itself forever.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_non_finite)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be an object, got {type(data).__name__}")
    return data


def _is_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_token(value: Any) -> bool:
    """Non-blank and whitespace-free: a metric names a key in evaluator output."""
    return _is_str(value) and not any(char.isspace() for char in value)


def _is_schema_version(value: Any) -> bool:
    # `bool` is an `int` subclass and `True == 1`, so an untyped equality test would
    # accept `"schema_version": true`.
    return isinstance(value, int) and not isinstance(value, bool) and value == SCHEMA_VERSION


def _check_row(row: Any, index: int, seen: set[str]) -> list[str]:
    if not isinstance(row, dict):
        # Positional prefix rather than `_FILE`: every row error is addressable by the
        # row it came from, whether or not that row got as far as carrying an id.
        return [f"claims[{index}]: must be an object, got {type(row).__name__}"]

    raw_id = row.get("id")
    errors: list[str] = []

    # One isinstance branch rather than `_is_str`, so pyright narrows `rid` to `str`.
    if not isinstance(raw_id, str) or not raw_id.strip():
        rid = f"claims[{index}]"
        errors.append(f"{rid}: `id` must be a non-empty string")
    else:
        rid = raw_id
        if not (rid.startswith("C") and rid[1:].isdigit()):
            errors.append(f"{rid}: `id` must match C<digits>")
        elif rid in seen:
            errors.append(f"{rid}: duplicate id")
        else:
            seen.add(rid)

    unknown = sorted(set(row) - set(ROW_KEYS))
    if unknown:
        # `measured_value` is the specific key this refusal exists for: a measured value
        # inside the registry would make a claim indistinguishable from its evidence.
        errors.append(f"{rid}: unknown key(s) {', '.join(unknown)}")
    missing = [k for k in ROW_KEYS if k not in row]
    if missing:
        errors.append(f"{rid}: missing key(s) {', '.join(missing)}")
    if missing:
        return errors

    if not _is_str(row["claim"]):
        errors.append(f"{rid}: `claim` must be non-empty verbatim claim text")
    if row["claim_value"] is None:
        errors.append(f"{rid}: `claim_value` must not be null")
    if not _is_token(row["metric"]):
        errors.append(f"{rid}: `metric` must be a non-blank, whitespace-free token")
    for field, allowed in (("corpus", CORPORA), ("arm", ARMS), ("device", DEVICES)):
        value = row[field]
        if value is not None and value not in allowed:
            errors.append(f"{rid}: `{field}` must be null or one of {', '.join(allowed)}")

    disposition = row["disposition"]
    if disposition not in DISPOSITIONS:
        errors.append(f"{rid}: `disposition` must be one of {', '.join(DISPOSITIONS)}")
        return errors

    errors += _check_key_path(rid, row["evaluator_key_path"], disposition)
    errors += _check_replacement(rid, row["replacement"], disposition)
    errors += _check_legacy(rid, row["legacy_source"], disposition, row["note"])
    return errors


def _check_key_path(rid: str, path: Any, disposition: str) -> list[str]:
    if disposition == "measured":
        if not isinstance(path, list) or not path:
            return [f"{rid}: `measured` requires a non-empty `evaluator_key_path`"]
        if not all(_is_str(token) for token in path):
            return [f"{rid}: `evaluator_key_path` tokens must be non-empty strings"]
        return []
    if path is not None:
        return [f"{rid}: `{disposition}` must carry a null `evaluator_key_path`"]
    return []


def _check_replacement(rid: str, replacement: Any, disposition: str) -> list[str]:
    if disposition in ("corrected", "narrowed"):
        if not isinstance(replacement, dict):
            return [f"{rid}: `{disposition}` requires a `replacement` object"]
        unknown = sorted(set(replacement) - set(REPLACEMENT_KEYS))
        if unknown:
            return [f"{rid}: `replacement` has unknown key(s) {', '.join(unknown)}"]
        missing = [k for k in REPLACEMENT_KEYS if not _is_str(replacement.get(k))]
        if missing:
            return [f"{rid}: `replacement` needs non-empty {', '.join(missing)}"]
        return []
    if replacement is not None:
        return [f"{rid}: `{disposition}` must carry a null `replacement`"]
    return []


def _check_legacy(rid: str, legacy: Any, disposition: str, note: Any) -> list[str]:
    if not isinstance(legacy, dict):
        return [f"{rid}: `legacy_source` must be an object"]
    unknown = sorted(set(legacy) - set(LEGACY_KEYS))
    missing = [k for k in LEGACY_KEYS if k not in legacy]
    errors: list[str] = []
    if unknown:
        errors.append(f"{rid}: `legacy_source` has unknown key(s) {', '.join(unknown)}")
    if missing:
        errors.append(f"{rid}: `legacy_source` missing key(s) {', '.join(missing)}")
        return errors

    status = legacy["status"]
    if status not in LEGACY_STATUSES:
        errors.append(f"{rid}: `legacy_source.status` must be one of {', '.join(LEGACY_STATUSES)}")
        return errors

    if status == "present":
        if not _is_str(legacy["path"]) or not _is_str(legacy["key"]):
            errors.append(f"{rid}: a present `legacy_source` needs non-empty `path` and `key`")
        if legacy["stored_value"] is None:
            errors.append(f"{rid}: a present `legacy_source` needs a non-null `stored_value`")
    else:
        populated = [k for k in ("path", "key", "stored_value") if legacy[k] is not None]
        if populated:
            errors.append(f"{rid}: an absent `legacy_source` must null {', '.join(populated)}")
        # An absent provenance is recorded, never silently dropped: the note is what
        # carries the search that failed and the closest artifact found instead.
        if not _is_str(note):
            errors.append(f"{rid}: an absent `legacy_source` requires a non-empty `note`")

    if disposition == "legacy-retained" and status != "present":
        errors.append(f"{rid}: `legacy-retained` requires a present `legacy_source`")
    if note is not None and not _is_str(note):
        errors.append(f"{rid}: `note` must be null or a non-empty string")
    return errors


def validate_structure(inventory: Any) -> list[str]:
    """Schema, ids and per-disposition required fields. Needs no evidence."""
    if not isinstance(inventory, dict):
        return [f"{_FILE}: top level must be an object"]
    errors: list[str] = []
    unknown = sorted(set(inventory) - {"schema_version", "claims"})
    if unknown:
        errors.append(f"{_FILE}: unknown top-level key(s) {', '.join(unknown)}")
    if not _is_schema_version(inventory.get("schema_version")):
        errors.append(f"{_FILE}: `schema_version` must be the integer {SCHEMA_VERSION}")
    claims = inventory.get("claims")
    if not isinstance(claims, list):
        errors.append(f"{_FILE}: `claims` must be a list")
        return errors
    seen: set[str] = set()
    for index, row in enumerate(claims):
        errors += _check_row(row, index, seen)
    return errors


def resolve_key_path(evidence: Any, tokens: list[str]) -> Any:
    """Walk `tokens` through `evidence`. Raises LookupError when the path does not exist.

    A token indexes a dict key, or a list position when it is all digits. Dict lookup wins
    on a dict, so an all-digit key in an evidence object stays reachable.
    """
    node = evidence
    for depth, token in enumerate(tokens):
        if isinstance(node, dict):
            if token not in node:
                raise LookupError(f"no key {token!r} at depth {depth}")
            node = node[token]
        elif isinstance(node, list):
            if not token.isdigit():
                raise LookupError(f"list at depth {depth} needs a numeric token, got {token!r}")
            position = int(token)
            if position >= len(node):
                raise LookupError(f"index {position} out of range at depth {depth}")
            node = node[position]
        else:
            raise LookupError(f"{type(node).__name__} at depth {depth} is not indexable")
    return node


def validate(inventory: Any, evidence: Any) -> list[str]:
    """Structure, plus key-path resolution against `evidence`, plus omission."""
    errors = validate_structure(inventory)
    broken = {error.split(":", 1)[0] for error in errors}
    if not isinstance(inventory, dict) or not isinstance(inventory.get("claims"), list):
        return errors

    for row in inventory["claims"]:
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        if not _is_str(rid) or rid in broken:
            continue
        disposition = row.get("disposition")
        if disposition == "absent":
            errors.append(f"{rid}: unresolved omission - no unit measures this claim yet")
            continue
        if disposition != "measured":
            continue
        try:
            value = resolve_key_path(evidence, row["evaluator_key_path"])
        except LookupError as exc:
            path = ".".join(row["evaluator_key_path"])
            errors.append(f"{rid}: `{path}` does not resolve in evidence ({exc})")
            continue
        # Falsy is fine; only a null resolution means the metric was never written.
        if value is None:
            path = ".".join(row["evaluator_key_path"])
            errors.append(f"{rid}: `{path}` resolves to null in evidence")
    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {Path(argv[0]).name} INVENTORY EVIDENCE", file=sys.stderr)
        return 2
    try:
        inventory = load(argv[1])
        evidence = load(argv[2])
    except (OSError, ValueError) as exc:
        print(f"{_FILE}: {exc}", file=sys.stderr)
        return 1
    errors = validate(inventory, evidence)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
