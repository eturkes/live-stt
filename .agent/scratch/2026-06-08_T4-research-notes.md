# T4 research notes (2026-06-08, two opus web-research agents; compressed)

## STT leg — verdict: sherpa-onnx + ReazonSpeech k2-v2

No native frame-sync streaming open JA model exists (June 2026): sherpa streaming catalogs = en/zh/ko/bn; Reazon streaming "planned"; Kyutai = en/fr; Qwen3-ASR streaming = vLLM-only (GPU). **All JA paths are chunked-streaming**: silero VAD + endpoint → fast offline decode; partials via rolling re-decode of open utterance buffer (~300 ms cadence).

| Rank | Model | CER (JSUT/CV8/TEDxJP) | RTF (CPU, 2 thr) | Notes |
|---|---|---|---|---|
| 1 | `reazonspeech-k2-v2` zipformer transducer 159M, sherpa asset `...zipformer-ja-reazonspeech-2024-08-01` | 6.45/7.85/9.09 fp32; int8-enc+fp32-dec ≈ fp32 | 0.054 int8 | Apache-2.0; no punctuation; ≤30 s/window (fine w/ VAD); beats whisper-large-v3 (7.18/8.18/9.96) on all 3 |
| 2 | `parakeet-tdt_ctc-0.6b-ja` (NeMo FastConformer, CTC branch), asset `...parakeet-tdt_ctc-0.6b-ja-35000-int8` | 6.4/7.1/9.0 | 0.106 int8 | CC-BY-4.0; punctuated output; A/B candidate |
| 3 | `cohere-transcribe-03-2026` 2B, asset `...cohere-transcribe-14-lang-int8-2026-04-01` | UNVERIFIED (human-eval: 66% win vs whisper-large-v3 JA) | UNVERIFIED on CPU | Pull only if 1–2 disappoint |

Rejected: kotoba-whisper (stale, loses CER, full large-v3 encoder kills CPU speedup), WhisperLiveKit/SimulStreaming (GPU latency claims; whisper JA trails specialists; valid fallback for within-utterance partials), whisper.cpp (same family), SenseVoice/moonshine/OmnilingualASR (no JA numbers). Transducer architecture avoids whisper-type phonetic-substitution hallucination (reasoning, not benchmarked).

Integration: in-process `sherpa_onnx.OfflineRecognizer.from_transducer(...)` + `VoiceActivityDetector(silero_vad.onnx)`; feed float32 mono 16 kHz. Bilingual ja-en sibling exists: `...zipformer-ja-en-reazonspeech-2025-01-17`.

## Translation leg — verdict: persistent `codex app-server` + `gpt-5.3-codex-spark`

- **Surface ranking** (sanctioned+stable+fast): app-server (stdio JSON-RPC 2.0; `initialize` → `thread/start` → `turn/start` with `input:[{type:text,...}]` → stream `item/agentMessage/delta` → `turn/completed` w/ token usage) ≈ official Python SDK `pip install openai-codex` (v0.1.0b3 2026-06-03, beta; wraps app-server; asyncio support UNVERIFIED) > `codex exec`/`exec resume` (O(seconds) spawn per call) > mcp-server > raw POST `chatgpt.com/backend-api/codex/responses` (Cloudflare TLS-fingerprint breakage ≥4× in 2026; server validates official base instructions → 400 on custom; REJECT).
- **Models via ChatGPT auth** (June 2026): gpt-5.5, gpt-5.4, gpt-5.4-mini, gpt-5.3-codex-spark (Pro-only preview; text-only; 128k ctx; sub-100 ms TTFT, >1,000 tok/s; **separate rate pool** ~300–1,500/5 h, doesn't draw standard limits). gpt-5.2*/5.3-codex deprecated for subscription auth since ~2026-06-02. Fast mode = same model 1.5× velocity at 2.5× credit burn — Spark dominates it for short outputs.
- **Latency**: Spark ≈ 0.2–0.5 s end-to-end for ~200-token output on warm thread (persistent WS since Spark launch: −80% roundtrip overhead). Standard models ≈ sub-500 ms TTFT + ~70 tok/s ⇒ 3–4 s per 200 tokens. Keep context tiny: AGENTS.md small, zero MCP servers (33k-token harness → 25.9 s pathology).
- **Instruction control on subscription auth**: `model_instructions_file` REJECTED server-side (400, issue #3202). Recipe: stock base prompt + tiny `AGENTS.md` in empty dedicated cwd ("JA→EN translator; output only the translation; no tools") + per-turn `outputSchema` forcing `{"translation":"..."}`. Knobs: `--sandbox read-only`, `approval never`, `tools.web_search=false`, `features.multi_agent=false`, `effort minimal`, `personality none`. Tools can't be globally disabled on app-server; with no-tool prompts + empty cwd they don't fire.
- **Quota (token-metered since 2026-04-02; 1 turn = 1 "local message"; persistent threads bill per turn, prompt-cache ≈10% input rate)**: Pro-5x 5 h ranges — 5.5: 80–400, 5.4: 100–500, mini: 300–1,750; Pro-20x ≈ 4×. 400 calls/2 h scenario ⇒ ~18 credits on mini / fits Spark's separate pool. Weekly caps unpublished (UNVERIFIED risk for daily heavy use).
- **Install**: static musl binary from github.com/openai/codex/releases (best for container; no deps) or `curl -fsSL https://chatgpt.com/codex/install.sh | sh` or npm (Node ≥22). SDK: `pip install openai-codex`.
- **Headless auth (sanctioned)**: enable "Allow device code login" at chatgpt.com Settings→Security, then `codex login --device-auth` (beta); or copy `~/.codex/auth.json` from another machine (officially documented for "trusted scripts/private CI"; auto-refresh; >~8 days idle → stale; two machines refreshing one bundle race — dedicate a login per machine).
- **ToS**: official surfaces + private automation = sanctioned. Raw endpoint = gray + technically hostile (Anthropic/Google revoked equivalent 3rd-party OAuth Apr 2026 — precedent risk).
- **Verify on first contact**: Spark entitlement on this Pro account (GH #17642 sync bug); SDK asyncio story; actual per-turn overhead with outputSchema.

Full transcripts: agent output files (session-local). Sources cited inline in agent results; key: developers.openai.com/codex/{app-server,sdk,models,pricing.md,auth/ci-cd-auth,config-reference}; hilab.jp blog (k2-v2 CER); k2-fsa.github.io sherpa docs (RTF tables); huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja.
