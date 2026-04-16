# Deepgram as a Backend for `live-stt`

_Research snapshot: April 2026. JA streaming STT + cascade translation._

## 1. Model lineup

Deepgram's current flagship is **Nova-3** (early 2025), with **Nova-3 Multilingual** rolled out
through 2025-early 2026 to cover 40+ languages. **No Nova-4** has been announced as of April 2026;
the recent cadence is incremental language expansions and keyterm-prompting upgrades, not a new
generation. Tiers, in descending quality: Nova-3 > Nova-2 > Enhanced > Base. Japanese streaming is
supported on Nova-3, Nova-2, Enhanced, and Base; Nova-3 is the only sensible choice for this use
case (industry-leading JA WER, real-time code-switching, ~300 ms streaming latency).

Two relevant APIs:
- **`listen` (speech-to-text)** — the WebSocket streaming endpoint we want. Returns transcripts only.
- **`voice-agent`** — bundled STT + LLM + TTS loop at ~$4.50/hr. Not relevant: we already have a
  translation cascade, we don't want orchestration, and the price is 10x higher.

## 2. JA support

Confirmed **streaming** (not batch-only) on Nova-3 multilingual. Deepgram markets JA handling as
tuned for mixed kana/kanji/loanwords and syllabic rhythm. Nova-3 Multilingual reports a ~21%
relative streaming WER reduction vs Nova-2 averaged over 11 languages including JA. **Published
JA-specific WER numbers are scarce** — no head-to-head against Whisper-v3 or Gemini Live on
CommonVoice-JA or ReazonSpeech that I could find in community benchmarks. Treat the "competitive on
JA" claim as unverified; plan a 30-min A/B against Gemini before committing.

## 3. Pricing (April 2026)

- **Nova-3 Multilingual streaming, PAYG: $0.0092/min = $0.552/hr.**
- Growth plan (annual prepay): $0.0078/min = $0.468/hr.
- Monolingual (EN) Nova-3 is $0.0077/min but doesn't apply — JA forces the multilingual SKU.
- **New accounts get $200 credit, no card, no expiry** — ~360 hours of Nova-3 multilingual
  streaming. More than enough to spike and validate.
- Billed per-second, not rounded up.

Versus the current Gemini Live ~$1.40/hr: Deepgram STT alone is ~2.5x cheaper.

## 4. Translation story (cascade)

**Deepgram does not translate in the streaming path.** The `detect_language` param exists but
"translation" only lives inside the Voice Agent API's LLM leg. For our cascade, pipe each finalized
JA utterance to a cheap LLM over HTTPS.

Cost math at 20 turns/min × 150 JA chars/turn, which is ~150 tokens in + ~200 tokens out
(generous) = 3000 tokens/min in, 4000 out. Per hour: 180k in, 240k out.

| LLM | In $/1M | Out $/1M | Per hour |
|-----|---------|----------|----------|
| gpt-5-nano | $0.05 | $0.40 | $0.009 + $0.096 = **$0.11/hr** |
| Gemini 2.5 Flash-Lite | $0.10 | $0.40 | $0.018 + $0.096 = **$0.11/hr** |

**Total realistic cost: Deepgram STT $0.55/hr + translation $0.11/hr ≈ $0.66/hr PAYG** (≈$0.58/hr
on Growth). That's 2.1x cheaper than the current Gemini Live setup, before counting the $200 free
credit.

## 5. Streaming API shape

- URL: `wss://api.deepgram.com/v1/listen`
- Auth: HTTP header `Authorization: Token <API_KEY>` on the WS upgrade.
- Relevant URL params for us: `model=nova-3`, `language=ja`, `encoding=linear16`,
  `sample_rate=16000`, `channels=1`, `interim_results=true`, `endpointing=300` (ms silence →
  `speech_final:true`), `utterance_end_ms=1000`, `vad_events=true`, `smart_format=true`.
- Message flow (server → client JSON frames):
  - `type:"SpeechStarted"` — VAD onset.
  - `type:"Results"` with `is_final:false` — interim.
  - `type:"Results"` with `is_final:true` — finalized segment.
  - `type:"Results"` with `speech_final:true` — endpoint detected; utterance boundary.
  - `type:"UtteranceEnd"` — silence gap past `utterance_end_ms`.
  - `type:"Metadata"` — request id, duration.
- Binary frames upstream = raw PCM16.
- **KeepAlive**: idle timeout is **10 s** of no data (`NET-0001` close); send
  `{"type":"KeepAlive"}` as a text frame every 3–5 s when mic is silent. Sending KeepAlive alone
  without any audio eventually still fails — gate it behind "we've sent at least one audio chunk".

## 6. Session/duration limits

No documented hard max session duration. Deepgram does auto-close after 10 s idle. **No session
resumption** — STT is stateless, timestamps reset on reconnect. The reconnect story is just: detect
close, open new WS, keep streaming, re-anchor your own wall-clock offset. This matches `live-stt`'s
current reconnect loop; nothing lost since we don't rely on server-side session state.

## 7. Sample Python snippet

```python
# pip install deepgram-sdk>=4 sounddevice
import asyncio, os, sounddevice as sd
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

RATE, BLOCK = 16000, 1600  # 100 ms PCM16

async def main():
    dg = DeepgramClient(os.environ["DEEPGRAM_API_KEY"])
    conn = dg.listen.asyncwebsocket.v("1")

    async def on_msg(_self, result, **_):
        alt = result.channel.alternatives[0]
        if not alt.transcript:
            return
        if result.is_final and result.speech_final:
            print(alt.transcript)
            print("------")

    async def on_utt_end(_self, utt, **_):
        pass  # optional: flush buffered partials

    conn.on(LiveTranscriptionEvents.Transcript, on_msg)
    conn.on(LiveTranscriptionEvents.UtteranceEnd, on_utt_end)

    opts = LiveOptions(
        model="nova-3", language="ja", encoding="linear16",
        sample_rate=RATE, channels=1, interim_results=True,
        endpointing=300, utterance_end_ms=1000,
        vad_events=True, smart_format=True,
    )
    await conn.start(opts)

    loop = asyncio.get_running_loop()
    def cb(indata, *_):
        loop.call_soon_threadsafe(asyncio.create_task, conn.send(bytes(indata)))
    with sd.RawInputStream(samplerate=RATE, blocksize=BLOCK,
                           dtype="int16", channels=1, callback=cb):
        await asyncio.Event().wait()  # run until cancelled

asyncio.run(main())
```

## 8. Known issues

- `deepgram-python-sdk` has ~5 open issues (April 2026); none block streaming or JA.
- Historical JA bug: `nova-2-general` sometimes emitted `勝ちます。---...` (runaway dashes) via
  LiveKit plugin — [issue #496](https://github.com/deepgram/deepgram-python-sdk/issues/496). Not
  confirmed on Nova-3. If we see runaway characters, file and fall back.
- Community threads flag transient WS drops on long sessions (hours); reconnect loop is mandatory.
- SDK has gone through a v3→v4 API reshuffle; pin the version.

## 9. Verdict

**Fit: 4 / 5.** Cheap ($0.66/hr all-in vs $1.40/hr Gemini), proper JA streaming on Nova-3, clean WS
API, $200 free credit erases spike cost. Decisive risks: (a) **no public JA WER benchmark** —
accuracy on real mic input is unverified and the one anecdotal JA bug on Nova-2 is unresolved;
(b) cascade adds a hop of latency (LLM TTFT on each finalized turn) vs Gemini's single-pass
STT+translate, so end-to-end TTFT will likely land 1.3–1.8 s vs current ~1 s. Worth a 1-day spike
gated on an A/B accuracy check against real JA mic audio before any migration.

## Sources

- https://developers.deepgram.com/docs/models-languages-overview
- https://developers.deepgram.com/reference/speech-to-text/listen-streaming
- https://developers.deepgram.com/docs/live-streaming-audio
- https://developers.deepgram.com/docs/endpointing
- https://developers.deepgram.com/docs/interim-results
- https://developers.deepgram.com/docs/utterance-end
- https://developers.deepgram.com/docs/audio-keep-alive
- https://developers.deepgram.com/docs/recovering-from-connection-errors-and-timeouts-when-live-streaming-audio
- https://deepgram.com/pricing
- https://deepgram.com/learn/introducing-nova-3-speech-to-text-api
- https://deepgram.com/learn/deepgram-expands-nova-3-with-11-new-languages-across-europe-and-asia
- https://deepgram.com/learn/nova-3-multilingual-major-wer-improvements-across-languages
- https://deepgram.com/learn/voice-agent-api-generally-available
- https://github.com/deepgram/deepgram-python-sdk
- https://github.com/deepgram/deepgram-python-sdk/issues/496
- https://ai.google.dev/gemini-api/docs/pricing
- https://brasstranscripts.com/blog/deepgram-pricing-per-minute-2025-real-time-vs-batch
