---
paths:
  - "live_stt.py"
  - "tests/test_translator.py"
  - "tests/test_context.py"
  - "tests/test_en_pairing.py"
  - "tests/test_term_census.py"
  - "tests/eval_en_pairing.py"
  - "tests/eval_term_census.py"
  - "tests/eval_translate_repeat.py"
---

# Translation leg — codex app-server + the session-context learner

## Transport + configuration (D-011)

Persistent `codex app-server` subprocess, newline-delimited JSON-RPC over stdio: `initialize` →
`thread/start` (one thread per session, `ephemeral`, `sandbox:read-only`, `approvalPolicy:never`,
`personality:none`) → `turn/start` per block → `agentMessage` deltas → `turn/completed`; quota via
`account/rateLimits/read`. Sequential turns, so EN lines keep JA order.

- **Model `gpt-5.6-luna` + `effort:low`** (runner-up `gpt-5.6-terra`+`medium`), picked by tournament
  and held by a clinical re-test: median 1.38 s general / 1.71 s clinical, quality 4.83/5, contract
  2.00/2, 0 format violations, 0 failed turns. **Effort is near-inert** — ≤+0.06/5 across low→max for
  +0.6-1.1 s, and higher effort threw 22.59 s and 15.67 s turns past `TRANSLATE_TIMEOUT_S`=15 ⇒ it
  buys abort risk, not accuracy. The superseded Spark default ranked last (paired quality −0.486,
  11 format violations, a permanent mid-run thread stall in 2 of 4 attempts).
- **Tool-feature disables in `_CODEX_CONFIG` are THE latency lever.** Re-enabling
  web_search/image/browser/computer/apps re-injects ~15 K tokens/turn: p50 3.15 s → **0.99 s**. Keep
  them off.
- **`serviceTier:"priority"`** = Codex's "Fast" tier (1.5× speed, increased usage), set PER THREAD in
  `_new_thread` — startup plus every ~100-turn rotation — so global `~/.codex/config.toml` stays at
  `default` per user directive. **An unrecognized tier is dropped silently** (thread/start succeeds and
  echoes `serviceTier:null`), so the code compares the echo and logs one warning rather than assuming
  it landed. Watch the shared bucket: "increased usage" burns quota faster than token counts imply.
- **Role pinned via `developerInstructions` on thread/start** — injection-resistant, imperatives inside
  the speech get translated, not obeyed. AGENTS.md-in-cwd mode was rejected: the model obeyed "delete
  all files". Two shipped instruction lines fix defects measured on clinical Japanese — the
  international generic name for Japanese brand-name drugs with dose/unit/schedule untouched
  (プレドニン → prednisolone, never the different molecule "prednisone"), and never inventing a
  patient's sex the Japanese did not state (12.3 % → **0.9 %** of turns). **The gender line is what
  fixes gender**: the drug-only arm's rise was noise, so shipping the drug line alone would have left
  the defect — attribute a fix to the line that causes it by running one cumulative arm per candidate.
- Marginal cost ≈180 in + 7-60 out tokens/turn. Luna bills the shared `codex` bucket, so translation
  competes with ordinary Codex usage (~1,300 turns ≈ 1 pp of the weekly window). Thread grows
  ~30 tok/turn ⇒ rotate every ~100 turns, costing one ~2.7 s uncached turn.
- Codex auth is external state (`~/.codex/auth.json` — check existence only, never read). `codex` must
  be on the PATH of the machine RUNNING live-stt; installs and logins are per-machine and user-only, and
  a container login does not carry over.
- Revisit the whole choice if Luna latency or entitlement changes, if sustained quota burn on the
  shared bucket contradicts ~0 % window movement, or if a leaner instructions channel is sanctioned.

## Degradation contract (locked by `tests/test_translator.py`)

- A >64 KiB line, a broken transport, or EOF routes into cleanup that enqueues one `{method:error}`
  sentinel onto `_notes`, so a turn parked mid-collect raises at once instead of waiting out
  `TRANSLATE_TIMEOUT_S`. The enabled→disabled flip logs once. `submit` counts backlog evictions into
  `dropped_translations` (meter `tdrop=`; `TRANSLATE_QUEUE_MAX`=50, drop-oldest).
- **L-033 — validate liveness at the COMMIT point, not only at spawn.** `start()` re-checks
  `_reader_task.done()` / `returncode` immediately before `enabled=True`, so a warm-up health check
  that completes just before the child dies cannot strand later turns over a dead reader.
- Both disable paths run through `_disable` ⇒ one stderr line **plus one
  `-- translation disabled: <reason>` marker in the transcript**, so a session stays diagnosable once
  the scrollback is gone. Reasons: `N consecutive failures (<type>)` = the 3-strike path;
  `codex app-server exited` = a codex EOF in an idle gap, which also wakes a mid-flight turn promptly.
  Startup failure deliberately marks nothing — nothing is decoded yet and a write would defeat
  `TranscriptFile`'s lazy creation. A per-block failure stays stderr-only and names its exception
  TYPE, because `TimeoutError` stringifies to `''` and `%s` alone logged `translation failed ()`.
- **Recovery is `_recover()`, driven inline from `run()` on the caption that finds the leg down, and
  `_alive()` picks its arm (M14.2, M14.3).** Both permanent degrades come through one gate, one
  budget, one backoff and one marker pair; what differs is whether a server survived.
  - **EOF ⇒ `_respawn`.** Nothing is left to talk to, so `start()` re-runs whole. **4.8 s** measured
    against a real app-server, next turn at normal cadence.
  - **3 strikes ⇒ `_probe`**: a fresh thread plus one turn on the SURVIVING process. Measured after
    three genuine 15 s stalls: probe **5.5 s**, the turns behind it 1.6-2.3 s. Latency is not the
    argument — respawning a live process is WRONG. `_respawn` drops `_proc` and `start()` overwrites
    `_reader_task`, leaving the survivor a reader no one owns whose eventual EOF calls `_disable` on
    a translator that by then holds a healthy leg, and enqueues a wake sentinel into its notes. Only
    a dead process makes that abandonment safe.
  - **The fresh thread is the probe, not an extra.** A stalled turn poisons its THREAD and
    interrupt-plus-drain does not clear it (L-026), so a turn on the thread that took the three
    strikes measures the wedge and reports a healthy server dead. `_new_thread` → `_instructions` →
    `translator_brief` is also the only channel the CURRENT glossary rides, on either arm.
  - The design fork was priced and inline won: a background recovery task buys idle-gap repair for
    one more task to order against `close()` and the SIGHUP path, while `run()` inherits that
    ordering whole.
  - **Drain `_notes` before the handshake**, on both arms. The EOF cleanup's own wake sentinel is
    still queued, and a turn that STALLED rather than errored keeps emitting after `_abort_turn`
    drained; the probe turn collects that `{method:error}` and raises — measured as
    `init failed (RuntimeError: {})` in 0.41 s, throwing away a server that had in fact started.
  - `submit` queues while the leg is down but still recoverable, because `run()` is parked on that
    queue and nothing else wakes it; a caption arriving inside the backoff is discarded in `run()`,
    which keeps a degrade out of the backlog and out of `tdrop=`.
  - Bounds, shared by both arms so a leg that dies two ways cannot spend twice:
    `TRANSLATE_MAX_RECOVERIES`=5 per session, never refunded, and `TRANSLATE_RECOVERY_WAIT_S`=5
    doubling per failure. `start()` assigns `_proc` only once the exec succeeds ⇒ `_proc is None`
    after a failed respawn means the BINARY is gone (uninstalled mid-session) and recovery ends there
    rather than spending the budget rediscovering it. A wedged-but-alive server is re-probed until
    the budget runs out; if it exits under a probe, the next attempt routes itself to `_respawn`.
  - **Shutdown latches recovery off at both ends**, `submit_sentinel` and `close()`, each with its own
    lock: the drain window between them is the one place `run()` is alive with recovery armed, and a
    respawn there would spend a handshake on a session that is ending. `start()`'s own failure path
    calls `_end_proc()`, never `close()`, or one failed attempt would latch the rest off.
  - **The app-server spawns `start_new_session=True`, OUT of the terminal's process group.** Ctrl+C
    signals the whole foreground group, so a same-group child dies before the drain reaches the last
    caption and that caption's EN is lost — the shape in 4 of 7 saved sessions, the last logging
    `codex app-server exited` 2 s after its final JA while `run()` was still mid-turn. Latching
    recovery off at shutdown is what makes the loss permanent, correctly: the leg is not coming back
    inside a session that is ending, so the child must not die in the first place. Measured: child
    returncode -2 in-group against alive detached, handshake and a real turn unaffected, and
    `_end_proc()` closes stdin so the child still exits 0 ⇒ detaching orphans nothing.
  - `_restore` mirrors `_disable` into both channels ⇒ `-- translation restored: codex app-server
    probed|respawned (attempt N)`, naming the arm. A transcript carrying only the disable marker
    reads as JA-only from that point while EN lines resume below it. A recovered leg resets
    `_failures` (else it is one strike from dying, and the probe arm arrives carrying a FULL count by
    construction) and `_turns` (else the fresh thread rotates early).
- **A repeated short unit makes the translator generate without terminating.** Measured through the
  real app-server with a fresh thread per turn, a 30 s bound and a real-speech canary after every
  degenerate turn: `"あ"+"は"*n` runs 2.9 s at 20 characters and 3.4 s at 60, then **stalls at
  120/240/480**; `中央の`/`クラブの`/`アーメンの` survive 240 (4.6-10.8 s) and **all stall at 480**;
  real speech scales flat to **7.0 s at 480**. So it is repetition, not length, and not only single
  characters. The publication screen in `asr-pipeline.md` is what keeps such a caption out; `submit`'s
  identical screen is the backstop, and it declines before the queue ⇒ `_turn`, `_failures` and
  `observe_en` are untouched by construction.

## Session context learner (D-015)

Learned from the session's own captions, held in memory, discarded at exit. Locked by
`tests/test_context.py`.

- **Candidates are restricted script runs, not n-grams** (`_TERM_RUN`: katakana ≥2, kanji 2-8, latin
  ≥2). Japanese is unsegmented, so the restriction is what makes extraction usable at all — over the
  same 81 caption lines, unrestricted n-grams produced 9,613 candidates against 100 for script runs,
  nearly all of the excess being grammar fragments. Cost 4.46 µs/line ⇒ runs inline, never a
  background pass.
- **Promotion + lease are one safety property with two halves.** A candidate is promoted after
  `CONTEXT_TERM_SUPPORT`=3 sightings in distinct segments **whose recogniser prompt did not already
  contain it**, and trust is a lease (`CONTEXT_TERM_LEASE`=60 segments) that only an un-prompted
  sighting renews. Conditioning on a mis-recognition reproduces it, so without the exclusion an error
  promotes itself on evidence it manufactured; without the lease a term the prompt keeps producing
  renews its own trust forever.
- Bounds: `CONTEXT_MAX_TERMS`=12 · `CONTEXT_TERM_MEMORY`=40 segments of candidate patience ·
  `CONTEXT_PROMPT_MAX_CHARS`=160, sized against Whisper's 223-token prev-text budget, terms ordered
  longest-first because length is the cheap proxy for what a recogniser gets wrong.
- `--context TEXT` is the user's correction channel: NFKC-normalized, trusted at once, never evicted,
  excluded from support accounting, and listed ONCE — repetition is what makes a `<|startofprev|>`
  payload loop.
- Two consumers. `asr_hotwords()` returns the term list **and** the terms it carries, which the caller
  must hand back to `observe_ja` or the guard is defeated (unreachable on the NPU default —
  `asr-pipeline.md`). `translator_brief()` rides the thread's `developerInstructions`, never turn text,
  because turn text is declared translatable input; a changed glossary rotates the thread.
- **`observe_en` pairs each trusted term with its English spelling, and the pairing is what makes the
  glossary help.** An unpaired list names a term without saying how to write it, and every glossary
  change rotates the codex thread whose own history was holding the spelling ⇒ the unpaired list made
  rendering consistency WORSE than no context at all. Evidence rule: exactly one still-unpaired
  trusted term in the JA and exactly one proper noun in the EN, `CONTEXT_EN_SUPPORT`=2 agreeing turns,
  where a proper noun is a capitalized run that is not sentence-initial. Measured over three corpora ×
  3 sessions/arm through the real translator: distinct spellings of one recurring proper noun 1/1/1 in
  **9 of 9** paired sessions against 3/5/2 · 1/1/1 · 4/1/1 unpaired; adequacy held at Δ **−0.018**
  [−0.089, +0.050]. A rendering expires with its term's lease, which is what bounds the dict.
- **`_EN_STOP` (the pronoun "I" + its four contractions) and `_EN_SENTENCE` are the shipped fix for
  `標柱 = I`.** English capitalizes its first-person pronoun, so "I" passed the not-sentence-initial
  rule and was the most common sole proper noun in a real run — 20 of the 63 single-proper-noun turns,
  ahead of `Gon` at 17 — briefing the translator to spell a mis-recognised key as a pronoun.
  `_EN_SENTENCE` splits on `(?<=[.!?…])["”’»]*\s+`: `…` ends 20 sentences in that stream, and the
  closing-quote allowance is what keeps `カスケ = Kasuke` learnable where a lookbehind demanding
  `[.!?]` immediately before the whitespace read two sentences as one and shut the gate. The
  quote-opening rule is guarded to openers (`(?:^|(?<=\s))`), since a straight `"` closes with the
  character it opens with and a name after quoted speech would otherwise be discarded.
- **Rejected, never re-propose: a JA-side plausibility test on the KEY.** It drops the correct
  `神様 = God` and removes `標柱 = I` only because 標柱 is kanji, blocking the whole kanji-name class
  (兵十 / 加助), and it cannot work in principle — the defect's key is itself name-shaped.
- **Rejected: LLM-written running summaries and background extractor turns** — codex must never become
  a dependency of the local leg (D-009).
- `_TERM_RUN`'s katakana floor is **2**, matching the kanji floor. Over the 215-caption story a floor
  of 2 admits 10 forms floor 3 rejects and **exactly one reaches support** (ゴン, 40 captions) while
  the other 9 die below it; ごん goes from visible in 3 of its 50 occurrences to 43, and every
  pre-existing trust episode stays identical. The 2-character katakana slot is nearly empty of real
  vocabulary (1 token in 6,879 characters of in-tree JA reference against 70 of 3+ characters) while
  the RECOGNISER fills it, which is exactly where a short native name lands ⇒
  `CONTEXT_TERM_SUPPORT`, not the floor, is what rejects vocabulary. **Honest limit:** one
  katakana-sparse corpus — in a loanword-dense domain (ドア, バス, ケア, メモ) short vocabulary would
  reach support and the support threshold alone stands against it.
- **Honest limits, all measured.** A persistent ASR error recurring 3× un-prompted IS learned: the
  guard prevents amplification, not initial learning, and `--context` is the correction channel. The
  learner stabilises whatever the recogniser stably produces, which live is often a mis-recognition.
  A hallucinated but genuine proper noun (`Okkawa`, `Anke`) is unreachable by any lexical or
  positional rule; `CONTEXT_EN_SUPPORT` is the only lever there.
- **Simulation is the learner's best case; check a claim against the live arm.** An offline screen
  found 2 dead pairings (鼻腔, イワシ) that do NOT exist against a real translator — a rendering needs
  a proper noun and both keys are ordinary English nouns, while the controls ゴン → `Gon` and 神様 →
  `God` pair in the same run, making it a refutation rather than a null. Two rules the live arm
  exposes: a term the English never pairs never leaves the gate, so it opens on every later sighting
  AND keeps closing its neighbours' openings; and the JA side is arm-independent, since `observe_ja`
  never reads `renderings`. Read a replay's `trusted`, never a raw sighting count — support also needs
  three sightings inside one `CONTEXT_TERM_MEMORY` window, and openings saturate at
  `CONTEXT_EN_SUPPORT` because a paired term leaves the gate.
