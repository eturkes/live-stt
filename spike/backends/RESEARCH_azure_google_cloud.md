# Backend spike: Azure AI Speech vs. Google Cloud Speech-to-Text v2

Research window: April 2026. Target: `live-stt` (Linux, asyncio, mic -> JA text + EN translation, hours-long sessions). Baseline: Gemini Live API, ~1 s TTFT, ~$1.40/hr.

---

## 1. Microsoft Azure AI Speech (Speech Translation)

**Streaming flavor.** Azure's `TranslationRecognizer` opens a single bidirectional WebSocket doing **STT + translation in one stream**: `recognizing` events for interim JA, `recognized` events for finalized JA with `translations['en']` attached. No cascade. Also supports Live Interpreter and multi-lingual source (unspecified input language).

**JA streaming quality.** `ja-JP` is a GA source language with interim+final streaming. Mature Azure ASR (powers Teams live captions); no widely reported regression in 2026. Community reports place it on par with Google for conversational JA.

**Pricing (April 2026, Standard S0).** Speech Translation list: **$2.50 / audio hour**, bundles source transcription plus up to 2 target languages. Billed per second. F0 free tier: 5 hours/month. **JA -> EN continuous = $2.50/hr** (one target, no extra Translator API). ~1.8x Gemini.

**API shape.** WebSocket under the hood; SDK hides it. Python package `azure-cognitiveservices-speech` (1.46+). Auth via `(subscription_key, region)` or AAD bearer token. Audio via `PushAudioInputStream` fed from our existing 16 kHz PCM16 pipeline. Not async-native -- signal/callback pattern, so we bridge to asyncio via `run_coroutine_threadsafe` or a queue.

**Session/duration limits.** No documented hard cap on continuous translation sessions. Real-time diarization caps at 240 min; plain translation runs until silence or stop. Concurrent limit 100 req/resource (S0, adjustable). Users report intermediate LB resets every 30-60 min -- **reconnect loop still required**, same as Gemini.

**Sample Python (mic -> JA text + EN translation):**
```python
import azure.cognitiveservices.speech as speechsdk
import os, time

cfg = speechsdk.translation.SpeechTranslationConfig(
    subscription=os.environ["AZURE_SPEECH_KEY"],
    region=os.environ["AZURE_SPEECH_REGION"],
    speech_recognition_language="ja-JP")
cfg.add_target_language("en")

push = speechsdk.audio.PushAudioInputStream(
    stream_format=speechsdk.audio.AudioStreamFormat(
        samples_per_second=16000, bits_per_sample=16, channels=1))
reco = speechsdk.translation.TranslationRecognizer(
    translation_config=cfg,
    audio_config=speechsdk.audio.AudioConfig(stream=push))

def on_final(evt):
    print("JA:", evt.result.text)
    print("EN:", evt.result.translations.get("en", ""))

reco.recognized.connect(on_final)
reco.recognizing.connect(lambda e: print(".", end="", flush=True))
reco.canceled.connect(lambda e: print("CANCELED:", e.error_details))
reco.start_continuous_recognition()
# Feed existing 100 ms PCM16 blocks: push.write(block_bytes)
try:
    while True: time.sleep(0.1)
except KeyboardInterrupt:
    reco.stop_continuous_recognition(); push.close()
```

**Known issues.** Issue [#2740](https://github.com/Azure-Samples/cognitive-services-speech-sdk/issues/2740): premature stop with segmentation timeouts set (Python 1.42, open early 2026). Issue [#2760](https://github.com/Azure-Samples/cognitive-services-speech-sdk/issues/2760): continuous recognition self-terminates over long periods -- reconnect on `session_stopped`. Q&A threads note ~30 s idle cutoff if no speech; must feed continuous audio. Threaded SDK is an awkward fit for pure-asyncio `live_stt.py`.

**Verdict: 4/5.** Best-in-class integrated STT+translation stream, one vendor, predictable pricing. Cost penalty is modest. Main risk = threaded SDK wrapper, not accuracy.

---

## 2. Google Cloud Speech-to-Text v2 + Cloud Translation API

**Streaming flavor.** `SpeechClient.streaming_recognize` (gRPC bidi) streams audio -> JA text only. Translation is a **separate call** to `cloud-translate` v3 NMT per finalized utterance. Two paths, two SDKs. Current streaming model: Chirp 3 (`model="chirp_3"`, GA for `ja-JP` in 2026).

**JA streaming quality.** Chirp 3 is strong for JA: Paul Kuo's Feb 2026 benchmark reports 6.4-25.5% CER across Japanese business content, mean 13.5% -- materially better than `latest_long` and competitive with Whisper-large-v3. Streaming interim + final on `ja-JP`.

**Pricing (April 2026).** STT v2 standard (Chirp 3, streaming): **$0.016/min = $0.96/hr**, first 500k min/month, no free tier on v2. Translation v3 NMT: $20/M chars; JA ~200 chars/min -> 12k/hr -> **$0.24/hr** (first 500k chars free). **Total ~$1.20/hr**, slightly under Gemini, half of Azure.

**API shape.** gRPC (HTTP/2). Python: `google-cloud-speech>=2.30` + `google-cloud-translate>=3.20`. Auth via service-account JSON or ADC. gRPC generator pattern: yield `StreamingRecognizeRequest` from a queue. Blocking iterator wraps cleanly with `asyncio.to_thread`; `SpeechAsyncClient` also exists.

**Session/duration limits.** **Hard 5-minute cap per `StreamingRecognize` call** -- unchanged in 2026. Google's "endless streaming" pattern closes/reopens before the cap with a trailing-audio buffer across the seam. Quotas: 300 concurrent streams per 5 min, 3000 req/min per region. Heavier reconnect burden than Azure or Gemini.

**Sample Python (mic -> JA text; translation call omitted):**
```python
import queue, os
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech

PROJECT = os.environ["GCP_PROJECT"]
client = speech_v2.SpeechClient()
recognizer = f"projects/{PROJECT}/locations/global/recognizers/_"
streaming_config = cloud_speech.StreamingRecognitionConfig(
    config=cloud_speech.RecognitionConfig(
        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000, audio_channel_count=1),
        language_codes=["ja-JP"], model="chirp_3",
        features=cloud_speech.RecognitionFeatures(enable_automatic_punctuation=True)),
    streaming_features=cloud_speech.StreamingRecognitionFeatures(interim_results=True))

audio_q: "queue.Queue[bytes]" = queue.Queue()
def requests():
    yield cloud_speech.StreamingRecognizeRequest(
        recognizer=recognizer, streaming_config=streaming_config)
    while (chunk := audio_q.get()) is not None:
        yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

for resp in client.streaming_recognize(requests=requests()):
    for r in resp.results:
        if r.alternatives:
            print("FINAL" if r.is_final else "...", r.alternatives[0].transcript)
# Per final: translate_client.translate_text(contents=[text], target_language_code="en", ...)
```

**Known issues.** Google Developer Forum: `SPEECH_ACTIVITY_END` [not triggered with `ja-JP`](https://discuss.google.dev/t/google-cloud-speech-to-text-v2-android-speech-activity-end-not-triggered-with-ja-jp-but-works/189836) (works for `en-US`) -- breaks semantic endpointing. v2 [CANCELLED errors](https://discuss.google.dev/t/google-cloud-speech-to-text-v2-streamingrecognize-cancelled-when-using-cmn-hans-cn/189835) on some Asian locales. No v2 free tier -- first spike hits the meter. Cascade doubles reconnect surface.

**Verdict: 3/5.** Cheapest of the three, Chirp 3 JA accuracy excellent, but 5-min cap + cascade translation doubles integration work. Reconnect logic tighter than current Gemini loop.

---

## Recommendation
**Azure Speech Translation is the one worth prototyping.** Integrated JA-STT+EN-translation over one WebSocket matches `live-stt`'s existing one-stream model; the 1.8x cost bump buys no 5-minute reconnect cycle and no second SDK. Google v2 + Translate is cheaper but needs seam-spanning reconnect buffering and a second hot-path API -- strictly worse ergonomics than what we already have with Gemini.

## Sources
- [Azure Speech pricing](https://azure.microsoft.com/en-us/pricing/details/speech/)
- [Azure Speech Translation overview](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-translation)
- [Azure Speech quotas and limits (2026-04)](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-services-quotas-and-limits)
- [Azure-Samples translation_sample.py](https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/master/samples/python/console/translation_sample.py)
- [Azure SDK Issue #2740](https://github.com/Azure-Samples/cognitive-services-speech-sdk/issues/2740) / [#2760](https://github.com/Azure-Samples/cognitive-services-speech-sdk/issues/2760)
- [Google Cloud STT pricing](https://cloud.google.com/speech-to-text/pricing) / [STT v2 quotas](https://docs.cloud.google.com/speech-to-text/v2/quotas)
- [Chirp 3 docs](https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3)
- [Chirp 3 JA benchmark (Paul Kuo, Feb 2026)](https://paulkuo.tw/en/articles/google-chirp3-japanese-stt-benchmark/)
- [Cloud Translation v3 pricing](https://cloud.google.com/translate/pricing)
- [ja-JP SPEECH_ACTIVITY_END bug](https://discuss.google.dev/t/google-cloud-speech-to-text-v2-android-speech-activity-end-not-triggered-with-ja-jp-but-works/189836)
