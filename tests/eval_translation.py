#!/usr/bin/env python3
"""Attribute caption error between the two legs: speech->JA (ASR) and JA->EN.

The user-visible failure is one number ("the English is wrong"), but it is
produced by a cascade, so the interesting quantity is where the meaning dies.
This evaluator holds the corpus and the judge fixed and varies only the
Japanese that reaches the translator:

    oracle_ja        the FLEURS reference sentence, verbatim  -> translation ceiling
    oracle_ja_flat   the same sentence with punctuation removed, i.e. the
                     surface shape ASR actually emits          -> punctuation cost
    k2v2_join        the shipped engine's whole-clip hypothesis -> ASR content cost
    k2v2_segments    that hypothesis split the way live VAD splits it, one turn
                     per segment, English concatenated         -> fragmentation cost
    parakeet_join    the A/B engine's whole-clip hypothesis    -> engine choice

Every arm is translated by the real ``CodexTranslator`` (production model,
effort, service tier, thread options and developerInstructions), one dedicated
``codex app-server`` per arm so no arm can read another's thread history.

Grading is blind: for each item a judge model sees the Japanese reference and
all arms' English anonymized behind [Cn] markers in a per-item shuffled order,
plus the human FLEURS English reference as an unlabelled sixth candidate. That
reference is the instrument's calibration - a judge that does not score it near
the top is not measuring adequacy.

ASR hypotheses are NOT decoded here. They are read from the committed, hash-
locked detail JSONL that ``tests/eval_models.py`` produced (`model_baseline.json`
pins their sha256), so this evaluator adds a translation layer to frozen ASR
evidence instead of re-running a multi-hour decode.

LLM output is not byte-reproducible. `--aggregate-only` recomputes every
published number from the cached details exactly; a fresh run will not
reproduce the details themselves, and the manifest says so.

Usage:
    uv run --with sacrebleu python tests/eval_translation.py --items 24
    uv run --with sacrebleu python tests/eval_translation.py            # all items
    uv run --with sacrebleu python tests/eval_translation.py --aggregate-only
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cer import align, normalize  # noqa: E402
from live_stt import (  # noqa: E402
    TRANSLATE_MODEL,
    TRANSLATOR_INSTRUCTIONS,
    CodexTranslator,
)

SCHEMA_VERSION = 1
CACHE = ROOT / "spike" / "backends" / "cache" / "translation_eval-v1"
CORPUS_MANIFEST = ROOT / "tests" / "short_corpus.json"
PARALLEL_MANIFEST = ROOT / "tests" / "fleurs_parallel.json"
MODEL_MANIFEST = ROOT / "tests" / "model_baseline.json"
BASELINE = ROOT / "tests" / "translation_baseline.json"

# Judges are chosen from outside the candidate set: the translator is Luna, so
# grading with Luna would let one model grade its own output. Two independent
# judges at high effort; the human reference candidate calibrates both.
JUDGES = (("gpt-5.6-sol", "high"), ("gpt-5.6-terra", "high"))
JUDGE_TIMEOUT_S = 300.0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260810

ARMS = ("oracle_ja", "oracle_ja_flat", "k2v2_join", "k2v2_segments", "parakeet_join")
HUMAN = "human_reference"
CANDIDATES = (*ARMS, HUMAN)

JUDGE_INSTRUCTIONS = """You grade English translations of Japanese speech, produced inside a
real-time speech-to-text pipeline. You are given the Japanese sentence that was actually spoken
(the ground truth) and several candidate English translations of it.

Some candidates were translated from a perfect transcript; others were translated from automatic
speech recognition output that may contain recognition errors, may lack punctuation, and may have
been cut into fragments. You are not told which is which, and you must not guess or reward a
candidate for looking like one kind or the other. Grade only what the English says against what the
Japanese says.

For every candidate return:
- adequacy 0-5: does the English convey what the Japanese actually says? 5 = every element of the
  meaning is present and correct; 4 = complete but for a minor nuance; 3 = the gist survives while
  one real element is wrong, dropped or softened; 2 = substantially wrong or half missing;
  1 = only a fragment of the meaning survives; 0 = wrong, empty or unrelated. Judge meaning, not
  wording: different phrasing that carries the same content is fully adequate.
- fluency 0-5: is it natural, idiomatic written English (5 = a native speaker would write this,
  0 = unintelligible)?
- confident_error 0 or 1: 1 if the English reads as confident, fluent, plausible prose while
  asserting something the Japanese does not say - a wrong name, number, entity, action or claim
  presented as fact. This is the dangerous failure: a listener who cannot check the Japanese has no
  signal that anything went wrong. Broken or obviously garbled English is NOT a confident error;
  score it 0 here and let fluency carry the penalty. An honest omission is not a confident error.
- why: at most 12 words naming the decisive error, or "ok".

Reply with ONLY a JSON object mapping each candidate label to
{"adequacy": int, "fluency": int, "confident_error": int, "why": "<=12 words"}.
No markdown fence, no prose outside the JSON."""


@dataclass(frozen=True)
class Item:
    """One FLEURS sentence: audio-derived hypotheses plus both references."""

    item_id: str
    sentence_id: int
    case_id: str
    duration_s: float
    gender: str | None
    ja_reference: str
    en_reference: str
    k2v2_hyp: str
    k2v2_segments: tuple[str, ...]
    k2v2_cer: float
    parakeet_hyp: str
    parakeet_cer: float


def _json_bytes(value: object, *, compact: bool = False) -> bytes:
    sep = (",", ":") if compact else (",", ": ")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=sep).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_ja(text: str) -> str:
    """Drop punctuation and separators, keeping every meaning-bearing character.

    ASR emits an unpunctuated stream, so a reference sentence complete with 、。
    and quotation marks is an easier translation input than anything the live
    pipeline can produce. Removing exactly Unicode P* and Z* turns the reference
    into the same surface shape without touching content, which isolates the
    cost of missing punctuation from the cost of misrecognized words.
    """
    return "".join(c for c in text if not unicodedata.category(c).startswith(("P", "Z")))


def _detail_path(manifest: dict, engine: str) -> tuple[Path, str]:
    details = manifest["deterministic"]["controls"][engine]["details"]
    return ROOT / details["path"], details["sha256"]


def _read_details(path: Path, expected_sha256: str) -> dict[str, dict]:
    raw = path.read_bytes()
    actual = _sha256_bytes(raw)
    if actual != expected_sha256:
        raise RuntimeError(f"{path.name}: sha256 {actual} != committed {expected_sha256}")
    rows = {}
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["case_id"]] = row
    return rows


def load_items(limit: int | None, seed: int) -> tuple[list[Item], dict]:
    """One item per unique FLEURS sentence, joined to both engines' hypotheses.

    FLEURS records some sentences two or three times. Keeping every recording
    would put the same sentence in front of one judge and one translator thread
    repeatedly, so the set is collapsed to one recording per sentence (lowest
    case_id), which is also what makes the English reference join one-to-one.
    """
    corpus = _load_json(CORPUS_MANIFEST)
    index = ROOT / corpus["cache"]["directory"] / corpus["cache"]["index"]
    # English references are needed only to judge; the translation arms run
    # without them, so a missing parallel manifest degrades to empty references
    # and run() refuses to judge rather than blocking the expensive half.
    parallel = _load_json(PARALLEL_MANIFEST) if PARALLEL_MANIFEST.exists() else None
    en_by_sentence: dict[int, str] = {}
    if parallel is not None:
        refs_path = ROOT / parallel["cache"]["directory"] / parallel["cache"]["index"]
        for line in refs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                en_by_sentence[row["sentence_id"]] = row["en_reference"]

    model_manifest = _load_json(MODEL_MANIFEST)
    engines = {}
    for engine in ("k2v2", "parakeet"):
        path, sha = _detail_path(model_manifest, engine)
        engines[engine] = _read_details(path, sha)

    by_sentence: dict[int, dict] = {}
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["source"] != "fleurs":
            continue
        prev = by_sentence.get(row["sentence_id"])
        if prev is None or row["corpus_id"] < prev["corpus_id"]:
            by_sentence[row["sentence_id"]] = row

    items: list[Item] = []
    for sentence_id in sorted(by_sentence):
        row = by_sentence[sentence_id]
        case_id = row["corpus_id"]
        k2 = engines["k2v2"][case_id]
        pk = engines["parakeet"][case_id]
        items.append(
            Item(
                item_id=f"fleurs-s{sentence_id:05d}",
                sentence_id=sentence_id,
                case_id=case_id,
                duration_s=row["duration_seconds"],
                gender=row.get("gender"),
                ja_reference=row["reference"],
                en_reference=en_by_sentence.get(sentence_id, ""),
                k2v2_hyp=k2["hyp"],
                k2v2_segments=tuple(s["text"] for s in k2["segments"]),
                k2v2_cer=k2["cer"],
                parakeet_hyp=pk["hyp"],
                parakeet_cer=pk["cer"],
            )
        )

    population = len(items)
    if limit is not None and limit < population:
        rng = random.Random(seed)
        items = sorted(rng.sample(items, limit), key=lambda i: i.sentence_id)
    scope = {
        "population_sentences": population,
        "selected_sentences": len(items),
        "selection": "all" if len(items) == population else f"random(seed={seed})",
        "seed": seed,
        "corpus_index_sha256": corpus["cache"]["index_sha256"],
        "parallel_index_sha256": parallel["cache"]["index_sha256"] if parallel else None,
        "asr_details": {
            engine: _detail_path(model_manifest, engine)[1] for engine in ("k2v2", "parakeet")
        },
    }
    return items, scope


def arm_inputs(item: Item, arm: str) -> tuple[str, ...]:
    """The Japanese turn(s) an arm submits: one string per translation turn."""
    if arm == "oracle_ja":
        return (item.ja_reference,)
    if arm == "oracle_ja_flat":
        return (flatten_ja(item.ja_reference),)
    if arm == "k2v2_join":
        return (item.k2v2_hyp,) if item.k2v2_hyp else ()
    if arm == "k2v2_segments":
        return tuple(t for t in item.k2v2_segments if t)
    if arm == "parakeet_join":
        return (item.parakeet_hyp,) if item.parakeet_hyp else ()
    raise ValueError(f"unknown arm: {arm}")


# --------------------------------------------------------------------------
# Translation


async def _translate_arm(arm: str, items: list[Item], out_path: Path, log) -> None:
    """Run one arm through its own translator process, resuming what exists."""
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["item_id"])
    pending = [i for i in items if i.item_id not in done]
    if not pending:
        log(f"[{arm}] complete ({len(done)} items cached)")
        return

    translator = CodexTranslator()
    if not await translator.start():
        raise RuntimeError(f"{arm}: codex app-server unavailable")
    out = out_path.open("a", encoding="utf-8")
    try:
        for n, item in enumerate(pending, 1):
            turns = arm_inputs(item, arm)
            parts: list[str] = []
            failures: list[str] = []
            t0 = time.perf_counter()
            for ja in turns:
                en = ""
                for attempt in range(3):
                    en = await translator._translate(ja)
                    if en:
                        break
                    failures.append(f"attempt{attempt}:empty")
                    # _translate swallows the cause and self-disables after
                    # three strikes; an evaluator must keep going and record
                    # the failure instead, so re-arm and retry the same turn.
                    translator.enabled = True
                    translator._failures = 0
                    if translator._proc is None or translator._proc.returncode is not None:
                        await translator.close()
                        translator = CodexTranslator()
                        if not await translator.start():
                            raise RuntimeError(f"{arm}: translator restart failed")
                parts.append(en)
            record = {
                "item_id": item.item_id,
                "arm": arm,
                "ja_turns": list(turns),
                "en_turns": parts,
                "en": " ".join(p for p in parts if p).strip(),
                "turns": len(turns),
                "failures": failures,
                "wall_s": round(time.perf_counter() - t0, 3),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if n % 10 == 0 or n == len(pending):
                log(f"[{arm}] {n}/{len(pending)} (+{len(done)} cached)")
    finally:
        out.close()
        await translator.close()


async def run_translations(items: list[Item], arms: tuple[str, ...], log) -> dict[str, Path]:
    """All arms concurrently: separate processes, so no shared thread history.

    Latency is not a measurement here - only content is - so concurrency costs
    nothing that this evaluator reports, and it turns a serial hour into
    minutes. Each arm writes its own resumable JSONL.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = {arm: CACHE / f"translations-{arm}.jsonl" for arm in arms}
    await asyncio.gather(*(_translate_arm(arm, items, paths[arm], log) for arm in arms))
    return paths


# --------------------------------------------------------------------------
# Judging


class JudgeClient:
    """Minimal app-server client for a judge thread (not the production path).

    Deliberately separate from CodexTranslator: the judge runs a different
    model, different instructions and a much longer timeout, and must not
    inherit the translator's degrade-to-silence policy.
    """

    def __init__(self, model: str, effort: str):
        self.model = model
        self.effort = effort
        self.proc: asyncio.subprocess.Process | None = None
        self.reader: asyncio.Task | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._notes: asyncio.Queue[dict] = asyncio.Queue()
        self.thread_id: str | None = None

    def _write(self, obj: dict) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())

    async def _request(self, method: str, params=None):
        self._next_id += 1
        rid = self._next_id
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        return await fut

    async def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            try:
                line = await self.proc.stdout.readline()
            except (ValueError, OSError):
                break
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                fut = self._pending.pop(msg["id"], None)
                if fut and not fut.done():
                    if "error" in msg:
                        fut.set_exception(RuntimeError(json.dumps(msg["error"])[:300]))
                    else:
                        fut.set_result(msg.get("result"))
            elif "method" in msg and "id" in msg:
                self._write({"jsonrpc": "2.0", "id": msg["id"], "result": {"decision": "denied"}})
            else:
                self._notes.put_nowait(msg)
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("app-server exited"))
        self._pending.clear()
        self._notes.put_nowait({"method": "error", "params": {}})

    async def start(self) -> None:
        from live_stt import _CODEX_CONFIG

        argv = ["codex", "app-server"]
        for key, value in _CODEX_CONFIG.items():
            argv += ["-c", f"{key}={value}"]
        self.proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self.reader = asyncio.create_task(self._read_loop())
        await asyncio.wait_for(
            self._request(
                "initialize",
                {"clientInfo": {"name": "judge", "title": "judge", "version": "1.0"}},
            ),
            30,
        )
        self._write({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        resp = await asyncio.wait_for(
            self._request(
                "thread/start",
                {
                    "model": self.model,
                    "cwd": str(ROOT),
                    "sandbox": "read-only",
                    "approvalPolicy": "never",
                    "ephemeral": True,
                    "personality": "none",
                    "developerInstructions": JUDGE_INSTRUCTIONS,
                },
            ),
            60,
        )
        self.thread_id = resp["thread"]["id"] if "thread" in resp else resp["threadId"]

    async def ask(self, text: str) -> str:
        while not self._notes.empty():
            self._notes.get_nowait()
        await self._request(
            "turn/start",
            {
                "threadId": self.thread_id,
                "input": [{"type": "text", "text": text}],
                "effort": self.effort,
                "summary": "none",
            },
        )
        parts: list[str] = []
        final = None
        while True:
            note = await self._notes.get()
            method = note.get("method", "")
            params = note.get("params", {})
            if method == "item/agentMessage/delta":
                parts.append(params.get("delta", ""))
            elif (
                method == "item/completed" and params.get("item", {}).get("type") == "agentMessage"
            ):
                final = params["item"].get("text")
            elif method == "turn/completed":
                return (final or "".join(parts)).strip()
            elif method == "error" and not params.get("willRetry"):
                raise RuntimeError(json.dumps(params.get("error", params))[:300])

    async def close(self) -> None:
        if self.reader:
            self.reader.cancel()
        if self.proc and self.proc.returncode is None:
            try:
                assert self.proc.stdin
                self.proc.stdin.close()
                await asyncio.wait_for(self.proc.wait(), 5)
            except Exception:
                self.proc.kill()


def parse_judge_json(text: str) -> dict:
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in judge reply")
    return json.loads(body[start : end + 1])


def judge_prompt(item: Item, order: list[tuple[str, str]]) -> str:
    blocks = []
    for i, (_, english) in enumerate(order, 1):
        # Marker-delimited, never quoted: quoting a candidate makes every one
        # look like it added quotes (L-026).
        blocks.append(f"[C{i}]\n{english}\n[/C{i}]")
    return (
        f"Japanese sentence actually spoken:\n{item.ja_reference}\n\n"
        "Candidate English translations. The text between each [Cn] and [/Cn] marker is that "
        "candidate's exact output, byte for byte; the markers are not part of it. An empty block "
        "means that candidate produced no output at all.\n"
        + "\n".join(blocks)
        + f"\n\nGrade all {len(order)} candidates."
    )


def candidate_texts(item: Item, translations: dict[str, dict[str, dict]]) -> dict[str, str]:
    texts = {arm: translations[arm].get(item.item_id, {}).get("en", "") for arm in ARMS}
    texts[HUMAN] = item.en_reference
    return texts


async def _judge_model(
    model: str,
    effort: str,
    items: list[Item],
    translations: dict[str, dict[str, dict]],
    out_path: Path,
    log,
) -> None:
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("ok"):
                    done.add(row["item_id"])
    pending = [i for i in items if i.item_id not in done]
    if not pending:
        log(f"[judge {model}] complete ({len(done)} items cached)")
        return

    client = JudgeClient(model, effort)
    await client.start()
    out = out_path.open("a", encoding="utf-8")
    try:
        for n, item in enumerate(pending, 1):
            texts = candidate_texts(item, translations)
            # Per-item deterministic shuffle keyed on the item alone: label
            # position carries no arm identity, and the order is reproducible.
            rng = random.Random(int(hashlib.sha256(item.item_id.encode()).hexdigest()[:8], 16))
            order = [(name, texts[name]) for name in CANDIDATES]
            rng.shuffle(order)
            labels = {f"C{i}": name for i, (name, _) in enumerate(order, 1)}
            try:
                raw = await asyncio.wait_for(client.ask(judge_prompt(item, order)), JUDGE_TIMEOUT_S)
                scores = parse_judge_json(raw)
                record = {
                    "item_id": item.item_id,
                    "judge": model,
                    "judge_effort": effort,
                    "ok": True,
                    "labels": labels,
                    "scores": {labels[k]: v for k, v in scores.items() if k in labels},
                    "raw": raw[:4000],
                }
                missing = set(CANDIDATES) - set(record["scores"])
                if missing:
                    record["ok"] = False
                    record["error"] = f"ungraded candidates: {sorted(missing)}"
            except Exception as exc:
                record = {
                    "item_id": item.item_id,
                    "judge": model,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if n % 10 == 0 or n == len(pending):
                log(f"[judge {model}] {n}/{len(pending)} (+{len(done)} cached)")
    finally:
        out.close()
        await client.close()


async def run_judges(
    items: list[Item], translations: dict[str, dict[str, dict]], log
) -> dict[str, Path]:
    paths = {model: CACHE / f"judge-{model}.jsonl" for model, _ in JUDGES}
    await asyncio.gather(
        *(
            _judge_model(model, effort, items, translations, paths[model], log)
            for model, effort in JUDGES
        )
    )
    return paths


# --------------------------------------------------------------------------
# Scoring


def load_jsonl_by_item(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["item_id"]] = row
    return rows


def _chrf(hypotheses: list[str], references: list[str]) -> float | None:
    """chrF2++ against the human reference: a judge-independent cross-check."""
    try:
        from sacrebleu.metrics import CHRF  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    metric = CHRF(word_order=2)
    return round(metric.corpus_score(hypotheses, [references]).score, 4)


def _bootstrap_paired(deltas: list[float], draws: int, seed: int) -> dict:
    """Percentile CI over items; the item is the resampling unit, so both
    judges' opinions of one sentence move together and are not counted as
    independent evidence."""
    if not deltas:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(draws):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "mean": round(statistics.fmean(deltas), 4),
        "lo": round(means[int(0.025 * draws)], 4),
        "hi": round(means[int(0.975 * draws) - 1], 4),
        "n": n,
    }


def score(items: list[Item], translations: dict[str, dict], judgements: dict[str, dict]) -> dict:
    by_item: dict[str, dict[str, list[dict]]] = {}
    judge_failures = []
    for model, rows in judgements.items():
        for item_id, row in rows.items():
            if not row.get("ok"):
                judge_failures.append(
                    {"item_id": item_id, "judge": model, "error": row.get("error")}
                )
                continue
            for name, sc in row["scores"].items():
                by_item.setdefault(item_id, {}).setdefault(name, []).append({"judge": model, **sc})

    graded = [i for i in items if len(by_item.get(i.item_id, {})) == len(CANDIDATES)]
    per_candidate: dict[str, dict] = {}
    item_means: dict[str, dict[str, float]] = {}
    for name in CANDIDATES:
        adequacy, fluency, confident = [], [], []
        for item in graded:
            scores = by_item[item.item_id][name]
            a = statistics.fmean(float(s["adequacy"]) for s in scores)
            adequacy.append(a)
            fluency.append(statistics.fmean(float(s["fluency"]) for s in scores))
            confident.append(statistics.fmean(float(s.get("confident_error", 0)) for s in scores))
            item_means.setdefault(item.item_id, {})[name] = a
        texts = [
            (
                translations.get(name, {}).get(i.item_id, {}).get("en", "")
                if name in ARMS
                else i.en_reference
            )
            for i in graded
        ]
        per_candidate[name] = {
            "adequacy_mean": round(statistics.fmean(adequacy), 4) if adequacy else None,
            "fluency_mean": round(statistics.fmean(fluency), 4) if fluency else None,
            "confident_error_rate": round(statistics.fmean(confident), 4) if confident else None,
            "adequacy_le2_rate": (
                round(sum(1 for a in adequacy if a <= 2.0) / len(adequacy), 4) if adequacy else None
            ),
            "chrf2pp": _chrf(texts, [i.en_reference for i in graded]),
            "empty_outputs": sum(1 for t in texts if not t.strip()),
        }

    contrasts = {}
    for label, better, worse in (
        ("punctuation_cost", "oracle_ja", "oracle_ja_flat"),
        ("asr_content_cost", "oracle_ja_flat", "k2v2_join"),
        ("fragmentation_cost", "k2v2_join", "k2v2_segments"),
        ("engine_k2v2_to_parakeet", "parakeet_join", "k2v2_join"),
        ("end_to_end_vs_ceiling", "oracle_ja", "k2v2_segments"),
        ("human_headroom", HUMAN, "oracle_ja"),
    ):
        deltas = [item_means[i.item_id][better] - item_means[i.item_id][worse] for i in graded]
        contrasts[label] = {
            "arms": [better, worse],
            **_bootstrap_paired(deltas, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
        }

    # Does ASR error actually predict caption damage? Bucketed by the frozen
    # strict CER of the shipped engine on that exact clip.
    buckets: dict[str, list[float]] = {}
    for item in graded:
        cer = item.k2v2_cer
        key = "0" if cer == 0 else "0-5%" if cer <= 0.05 else "5-15%" if cer <= 0.15 else ">15%"
        buckets.setdefault(key, []).append(
            item_means[item.item_id]["k2v2_segments"] - item_means[item.item_id]["oracle_ja_flat"]
        )
    cer_response = {
        key: {"n": len(v), "mean_adequacy_delta": round(statistics.fmean(v), 4)}
        for key, v in sorted(buckets.items())
    }

    asr = {
        engine: {
            "micro_cer": round(
                sum(
                    align(normalize(i.ja_reference), normalize(getattr(i, f"{engine}_hyp")))[1]
                    for i in graded
                )
                / max(1, sum(len(normalize(i.ja_reference)) for i in graded)),
                6,
            ),
            "clips_with_any_error": sum(1 for i in graded if getattr(i, f"{engine}_cer") > 0),
        }
        for engine in ("k2v2", "parakeet")
    }
    segments = [len([s for s in i.k2v2_segments if s]) for i in graded]
    return {
        "graded_items": len(graded),
        "judge_failures": judge_failures,
        "candidates": per_candidate,
        "contrasts": contrasts,
        "cer_response": cer_response,
        "asr_on_graded_items": asr,
        "segmentation": {
            "mean_segments_per_clip": round(statistics.fmean(segments), 3) if segments else None,
            "multi_segment_rate": (
                round(sum(1 for s in segments if s > 1) / len(segments), 4) if segments else None
            ),
        },
    }


def build_manifest(items: list[Item], scope: dict, results: dict, paths: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "question": ("Does caption error come from speech->JA recognition or JA->EN translation?"),
        "method": {
            "corpus": "FLEURS ja_jp test, one recording per unique sentence, "
            "English reference = the parallel FLEURS en_us sentence",
            "asr": "frozen hypotheses read from tests/model_baseline.json detail JSONL; "
            "no audio is decoded by this evaluator",
            "translator": {
                "model": TRANSLATE_MODEL,
                "path": "live_stt.CodexTranslator (production config/instructions/tier)",
                "instructions_sha256": _sha256_bytes(TRANSLATOR_INSTRUCTIONS.encode("utf-8")),
                "isolation": "one codex app-server per arm; no shared thread history",
            },
            "judges": [{"model": m, "effort": e} for m, e in JUDGES],
            "judge_instructions_sha256": _sha256_bytes(JUDGE_INSTRUCTIONS.encode("utf-8")),
            "blinding": "per-item deterministic shuffle; arms and the human reference are "
            "presented identically behind [Cn] markers",
            "reproducibility": "LLM output is not byte-reproducible: --aggregate-only "
            "recomputes every number below from the cached details exactly, but a fresh "
            "run produces different details",
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "item"},
        },
        "scope": scope,
        "arms": {
            "oracle_ja": "FLEURS reference sentence verbatim",
            "oracle_ja_flat": "reference with Unicode P*/Z* removed (ASR surface shape)",
            "k2v2_join": "shipped engine whole-clip hypothesis, one turn",
            "k2v2_segments": "shipped engine per-VAD-segment hypotheses, one turn each "
            "(this is the live pipeline)",
            "parakeet_join": "A/B engine whole-clip hypothesis, one turn",
            HUMAN: "FLEURS human English reference, graded blind as calibration",
        },
        "results": results,
        "evidence": {
            "details": {
                name: {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": _sha256_bytes(path.read_bytes()),
                    "rows": sum(
                        1 for line in path.read_text(encoding="utf-8").splitlines() if line
                    ),
                }
                for name, path in sorted(paths.items())
                if path.exists()
            }
        },
    }


def render(manifest: dict) -> str:
    results = manifest["results"]
    lines = [
        f"items graded: {results['graded_items']}  "
        f"judges: {', '.join(j['model'] for j in manifest['method']['judges'])}",
        "",
        f"  {'candidate':<16} {'adequacy':>9} {'fluency':>8} {'conf.err':>9} "
        f"{'adeq<=2':>8} {'chrF2++':>8}",
    ]
    for name in CANDIDATES:
        row = results["candidates"][name]
        chrf = row["chrf2pp"]
        lines.append(
            f"  {name:<16} {row['adequacy_mean']:>9.3f} {row['fluency_mean']:>8.3f} "
            f"{row['confident_error_rate']:>9.3f} {row['adequacy_le2_rate']:>8.3f} "
            f"{(f'{chrf:.2f}' if chrf is not None else 'n/a'):>8}"
        )
    lines += ["", "  contrast (adequacy points, paired, 95% CI)"]
    for label, row in results["contrasts"].items():
        lines.append(
            f"  {label:<26} {row['mean']:>+7.3f}  [{row['lo']:>+.3f}, {row['hi']:>+.3f}]  "
            f"n={row['n']}"
        )
    lines += ["", "  end-to-end adequacy drop vs flat oracle, by clip CER"]
    for key, row in results["cer_response"].items():
        lines.append(f"  {key:<26} {row['mean_adequacy_delta']:>+7.3f}  n={row['n']}")
    return "\n".join(lines)


async def run(args) -> int:
    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    items, scope = load_items(args.items, args.seed)
    log(f"items: {len(items)} of {scope['population_sentences']} FLEURS sentences")
    arms = tuple(args.arms.split(",")) if args.arms else ARMS

    paths = {arm: CACHE / f"translations-{arm}.jsonl" for arm in arms}
    if not args.aggregate_only:
        paths = await run_translations(items, arms, log)
    translations = {arm: load_jsonl_by_item(path) for arm, path in paths.items()}

    if not args.translate_only and any(not i.en_reference for i in items):
        raise RuntimeError(
            f"English references missing: run tests/fetch_fleurs_parallel.py "
            f"to build {PARALLEL_MANIFEST.name} before judging"
        )
    judge_paths = {model: CACHE / f"judge-{model}.jsonl" for model, _ in JUDGES}
    if not args.aggregate_only and not args.translate_only:
        judge_paths = await run_judges(items, translations, log)
    judgements = {model: load_jsonl_by_item(path) for model, path in judge_paths.items()}
    if args.translate_only:
        log("translate-only: skipping judging and aggregation")
        return 0

    results = score(items, translations, judgements)
    manifest = build_manifest(items, scope, results, {**paths, **judge_paths})
    BASELINE.write_bytes(_json_bytes(manifest) + b"\n")
    print(render(manifest))
    log(f"wrote {BASELINE.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--items", type=int, default=None, help="Sentence count (default: all).")
    parser.add_argument("--seed", type=int, default=7, help="Selection seed when --items subsets.")
    parser.add_argument("--arms", default=None, help="Comma-separated arm subset.")
    parser.add_argument("--translate-only", action="store_true", help="Skip judging.")
    parser.add_argument(
        "--aggregate-only", action="store_true", help="Rebuild the manifest from cached details."
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
