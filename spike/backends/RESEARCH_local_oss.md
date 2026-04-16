# Local / Open-Source Streaming STT for live-stt — April 2026

Scope: replace the Gemini Live backend (~$1.40/hr) with an all-local JA transcription +
EN translation pipeline on Linux + consumer hardware. Numbers are current as of April 2026.

## STT engines

### 1. faster-whisper (CTranslate2)
- **Repo**: https://github.com/SYSTRAN/faster-whisper
- **What**: Whisper large-v3 / -v3-turbo re-implemented on CTranslate2. Chunked inference
  (30 s windows); you bolt a VAD + sliding-window strategy on top for streaming.
- **Disk / memory**: large-v3 FP16 ~3.1 GB on disk, ~4–6 GB VRAM; INT8 ~1.5 GB on disk,
  ~2 GB VRAM. large-v3-turbo ~6 GB VRAM FP16 / ~3 GB INT8.
- **RTF**: RTX 3060 12 GB runs large-v3 at RTF ~0.15 (30-s chunk in ~8 s); a 4060 is in
  the same ballpark. Turbo on a 4060 comfortably hits >10× real-time.
- **JA accuracy**: In the neosophie 2026 RTX 5090 benchmark, `whisper-large-v3-turbo`
  scored **WER 0.21 / CER 0.18** on JA broadcast audio — second place behind Qwen3-ASR.
- **Streaming**: not native; use `whisper_streaming` / WhisperLiveKit as a wrapper.

### 2. whisper.cpp (ggml)
- **Repo**: https://github.com/ggml-org/whisper.cpp
- **What**: Pure C/C++ port of Whisper, 4-bit/5-bit/8-bit ggml quantizations. Has a
  `stream` example using a rolling window.
- **Disk / memory**: large-v3 Q5_0 ~1.1 GB; runs in ~2–3 GB RAM.
- **RTF**: on a modern 8-core desktop CPU (Zen 4 / 13th-gen Intel), large-v3 is
  marginal — roughly RTF 0.6–1.0 with AVX2. Realistically you need the **medium** or
  **turbo** model for CPU-only real-time JA; Vulkan / ROCm builds help. 4060 via CUDA
  build closes most of the gap to faster-whisper.
- **JA accuracy**: same underlying weights as Whisper large-v3, so ~same ceiling.
- **Streaming**: yes, built-in, but the example is naive chunk-and-concat — worse than
  LocalAgreement for stable output.

### 3. whisper-streaming (Ufal) / SimulStreaming / WhisperLiveKit
- **Repos**:
  - https://github.com/ufal/whisper_streaming (original, now considered legacy)
  - https://github.com/ufal/SimulStreaming (successor, ~5× faster, AlignAtt policy)
  - https://github.com/QuentinFuxa/WhisperLiveKit (maintained integration layer)
- **What**: Wrapper around faster-whisper / MLX-Whisper / Voxtral with **LocalAgreement-n**
  or **AlignAtt** committing policy + **Silero VAD**. Turns Whisper into a true streaming
  system with ~2–4 s latency and stable output.
- **JA**: inherits from underlying Whisper — works fine, same WER as the backend.
- **Why it matters**: this is the path for `live-stt`. Implementing LocalAgreement
  yourself is ~500 lines; using WhisperLiveKit is ~30.

### 4. NVIDIA NeMo — Canary / Parakeet
- **Models**: `nvidia/parakeet-tdt-0.6b-v3`, Canary V2, Nemotron-Speech-Streaming
- **Status April 2026**: Parakeet V3 + Canary V2 (Aug 2025) support **25 European
  languages only — Japanese is NOT included**. NeMo's streaming infra (chunked RNNT,
  Nemotron-Speech-Streaming with 160 ms min latency) is best-in-class for EN, but JA
  support hasn't shipped. **Rule out for this project.**
- **Note**: reazonspeech-nemo-v2 uses the NeMo framework for JA but is a separate model.

### 5. ReazonSpeech v2 (JA-specialized)
- **Repo**: https://github.com/reazon-research/ReazonSpeech
- **Models**: `reazonspeech-espnet-v2` (120M), `reazonspeech-nemo-v2` (619M FastConformer-
  RNNT), `reazonspeech-k2-v2` (159M Next-gen Kaldi).
- **JA accuracy (neosophie 2026)**: `reazonspeech-espnet-v2` hits **WER 0.23 / CER 0.20**
  — best of the JA-specialists on broadcast audio, slightly behind Whisper-turbo.
- **Streaming**: k2-v2 runs CPU-only with VAD chunking; true streaming variant is
  promised but not yet released. FastConformer-RNNT in NeMo is natively streaming.
- **Disk / memory**: tiny (120–620M), fits under 2 GB VRAM comfortably.

### 6. Kotoba-Whisper v2.x (distilled JA Whisper)
- **Repo**: https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0
- **What**: Whisper large-v3 distilled to a 2-layer decoder using ReazonSpeech v2 (7.2M
  clips). **6.3× faster than large-v3**, CER 9.2 on CommonVoice 8 JA (vs large-v3 ~10.5),
  competitive on JSUT basic 5000.
- **Disk / memory**: ~1.6 GB, fits in ~3 GB VRAM FP16.
- **Streaming**: drop-in faster-whisper CT2 build exists (`kotoba-whisper-v2.0-faster`).
- **Verdict**: the sweet spot for JA on a 4060. v2.2 adds diarization + punctuation.
- **2026 note**: neosophie's single benchmark scored kotoba v2.0 surprisingly badly on
  conversational broadcast clips. Probably domain mismatch — their training set is
  also broadcast-heavy, so investigate with your own audio before committing.

### 7. Qwen3-ASR-1.7B / Voxtral-mini-4B (honorable mentions)
- **Repos**: https://github.com/QwenLM/Qwen3-ASR, Mistral Voxtral-Mini
- **Qwen3-ASR**: #1 on the JA benchmark (WER 0.19). vLLM streaming supported. ~4 GB
  VRAM quantized. **The strongest local JA option as of April 2026** if you can tolerate
  the vLLM stack.
- **Voxtral-mini-4b**: 4 B params, ~8 GB VRAM, decent JA (WER 0.24), supported by
  WhisperLiveKit as a backend.

### 8. Moonshine multilingual
- **Repo**: https://github.com/moonshine-ai/moonshine, arXiv:2509.02523
- JA Moonshine (Feb 2026 release) is **tiny** (~30–70 M params) and beats Whisper-tiny
  by 48 %, matches Whisper-medium on its target domains. Low-latency edge model, but
  accuracy ceiling is below large-v3/turbo — better for embedded, overkill-weak here.

## Translation path (local)

- **NLLB-200-distilled-600M**: https://huggingface.co/facebook/nllb-200-distilled-600M —
  ~1.2 GB, fits in 2–3 GB VRAM FP16, CT2 conversion available for 1.5–2× speedup.
  Quality on JA→EN is good for sentence-level; mediocre on colloquial / context-heavy
  speech. BLEU on FLORES JA-EN ≈ 23–25 (distilled). ~80–200 ms per sentence on GPU.
- **Madlad-400 (3B)**: https://huggingface.co/google/madlad400-3b-mt — ~6 GB VRAM,
  slightly better on low-resource langs; roughly comparable BLEU on JA-EN but bigger
  footprint. Only worth it if you're already GPU-loaded.
- **Small LLM (Gemma-2-2B-jpn-it / Qwen2.5-3B-Instruct)**: ~2–6 GB VRAM depending on
  quant. Prompted MT gives more natural EN than NLLB but is 5–10× slower (token-by-
  token) and adds latency you can feel at utterance boundaries. The webbigdata
  `gemma-2-2b-jpn-it-translate` fine-tune is purpose-built and fits in 2 GB at Q4_K_M.
- **Cascade to cheap cloud LLM**: gpt-5-nano is **$0.05 / $0.40 per 1 M tok**;
  gemini-2.5-flash-lite is **$0.10 / $0.40**. A 1-hour JA conversation is ~15 k JA
  tokens in, ~20 k EN tokens out → **~$0.009 in + $0.008 out = $0.017/hr** for
  translation alone via gpt-5-nano. That's a >80× reduction vs Gemini Live's bundled
  $1.40/hr, for arguably better translation than any local <=3B model.

## Practical recommendation

**Stack**: WhisperLiveKit (SimulStreaming + AlignAtt + Silero VAD) on faster-whisper
`kotoba-whisper-v2.0-faster` for STT, cascaded to gpt-5-nano for EN translation.

- Total VRAM: ~3 GB (STT only; MT is remote).
- Minimum hardware: **RTX 3060 12 GB / RTX 4060 8 GB, 16 GB system RAM**. CPU-only is
  possible with whisper.cpp + medium model but latency will suffer (RTF ~0.8 on 8-core).
- Expected end-to-end latency: ~2–3 s (VAD + AlignAtt commit) + ~0.4 s (nano round-trip)
  ≈ **~3 s total** — comparable to the current Gemini Live path.
- Hourly cost: ~$0.02/hr (nano translation) vs $1.40/hr today.

### Sketch (~40 lines)
```python
import asyncio, sounddevice as sd, numpy as np
from whisperlivekit import TranscriptionEngine, AudioProcessor
from openai import AsyncOpenAI

SR, BLOCK = 16000, int(16000 * 0.1)
oai = AsyncOpenAI()
engine = TranscriptionEngine(
    model="kotoba-tech/kotoba-whisper-v2.0-faster",
    backend="faster-whisper",
    lan="ja",
    min_chunk_size=1.0,
    vad=True, vac=True,
    streaming_strategy="simulstreaming",
)

async def translate(ja: str) -> str:
    r = await oai.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "system", "content": "Translate JA to natural EN. Only output the translation."},
                  {"role": "user", "content": ja}])
    return r.choices[0].message.content

async def main():
    proc = AudioProcessor(engine)
    async def on_commit(text: str):
        print(f"JA: {text}")
        print(f"EN: {await translate(text)}")
        print("-" * 60)
    proc.on_transcription = on_commit
    def cb(indata, frames, t, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata
        pcm = (np.clip(mono, -1, 1) * 32767).astype(np.int16).tobytes()
        asyncio.get_event_loop().call_soon_threadsafe(proc.feed, pcm)
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=BLOCK, callback=cb):
        await proc.run()

asyncio.run(main())
```

### Decisive risk
**Latency under interactivity + first-token warmup.** SimulStreaming's AlignAtt commits
*faster* than LocalAgreement but Whisper still needs a ~1 s stable context window before
committing. Expect 2–3 s between utterance end and committed JA line — acceptable for
monologue / media, noticeable in conversation. Second-order risks: cold start (8–15 s
to load large-v3-turbo into VRAM) and JSUT-trained kotoba-v2 underperforming on
conversational / noisy audio.

## Verdict: **2 / 5** for this project

Given the author's stated preference for simple code and the current single-file,
~380-line shape of `live_stt.py`, this is a bad trade. You replace **one cloud API
call** (Gemini Live doing JA ASR + EN MT + turn boundaries in one go) with:
- a VAD,
- a streaming committing policy (LocalAgreement / AlignAtt),
- a CTranslate2 Whisper load + warmup,
- a translation model or a second API,
- plus the GPU ecosystem overhead (torch, ctranslate2, CUDA drivers, Silero).

Even the "recommended" stack is conceptually ~4 moving parts vs today's 1. For personal
use at current cadence, $1.40/hr is ~$30/mo of daily 1-hour sessions — not nothing, but
the switchover cost is a rewrite, and the quality floor (Whisper JA CER ~10; NLLB BLEU
~24) is demonstrably worse than the Gemini Live 3.1 path.

**The one exception**: if privacy, offline use, or unbounded-length sessions without
reconnect drama matters more than dollars, then `live-stt-local` as a second binary
(not a replacement) is justified. A fork, not a rewrite. Or: keep STT local
(WhisperLiveKit + kotoba) and keep translation on a cheap cloud LLM — that actually
gets you to ~$0.02/hr while leaving the author's translation quality intact.

## Sources

- https://github.com/SYSTRAN/faster-whisper
- https://github.com/ggml-org/whisper.cpp
- https://github.com/ufal/whisper_streaming
- https://github.com/ufal/SimulStreaming
- https://github.com/QuentinFuxa/WhisperLiveKit
- https://github.com/NVIDIA-NeMo/NeMo
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://github.com/reazon-research/ReazonSpeech
- https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0
- https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster
- https://github.com/QwenLM/Qwen3-ASR
- https://github.com/moonshine-ai/moonshine
- https://arxiv.org/abs/2509.02523
- https://huggingface.co/facebook/nllb-200-distilled-600M
- https://huggingface.co/google/madlad400-3b-mt
- https://huggingface.co/webbigdata/gemma-2-2b-jpn-it-translate
- https://neosophie.com/en/blog/20260226-japanese-asr-benchmark (JA ASR 2026 benchmark on RTX 5090)
- https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks
- https://arxiv.org/html/2506.17077 (CUNI IWSLT 2025, SimulStreaming + AlignAtt)
- https://arxiv.org/html/2307.14743 (original whisper-streaming paper, LocalAgreement-n)
- https://pricepertoken.com/pricing-page/model/google-gemini-2.5-flash-lite
