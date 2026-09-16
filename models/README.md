# models/

Local STT model weights. This directory is gitignored, except for this file.

The default engine needs the Whisper model and the silero VAD, which is about
790 MB. Each sherpa fallback engine adds its own directory. All four engines
together are about 2.2 GB.

```
models/
├── silero_vad.onnx                                        # all engines
├── openvino/
│   ├── whisper-large-v3-turbo-int8-ov/                     # --engine whisper (default)
│   │   ├── openvino_encoder_model.xml / .bin
│   │   ├── openvino_decoder_model.xml / .bin
│   │   ├── openvino_tokenizer.xml / .bin
│   │   ├── openvino_detokenizer.xml / .bin
│   │   └── config.json, generation_config.json, tokenizer.json, ...
│   └── cache/                                              # OpenVINO compile cache (generated)
├── lid/
│   └── d2-ecapa/                                           # --two-way language detection
│       ├── voxlingua107.onnx
│       ├── lang_map.json
│       └── manifest.json
├── sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/       # --engine k2v2
│   ├── encoder-epoch-99-avg-1.int8.onnx
│   ├── decoder-epoch-99-avg-1.onnx
│   ├── joiner-epoch-99-avg-1.onnx
│   └── tokens.txt
└── sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/   # --engine parakeet
    ├── model.int8.onnx
    └── tokens.txt
```

## Download

The default engine needs these two commands. Run them from the repository root.

```sh
uvx --from 'huggingface_hub[cli]' hf download \
  OpenVINO/whisper-large-v3-turbo-int8-ov \
  --local-dir models/openvino/whisper-large-v3-turbo-int8-ov
curl -Lo models/silero_vad.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

The sherpa fallback engines are optional. Download one only if you use
`--engine k2v2` or `--engine parakeet`.

```sh
cd models
base=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
curl -L $base/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2 | tar xj
curl -L $base/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8.tar.bz2 | tar xj
```

The language detector is optional. Download it only if you use `--two-way`. It
adds about 87 MB. The revision in the URL is pinned. Always verify the files
after the download.

```sh
mkdir -p models/lid/d2-ecapa
base=https://huggingface.co/crash-sv/scribe-ecapa-voxlingua107/resolve/13a135951a7352386984030d35531563f3473aaa
for f in voxlingua107.onnx lang_map.json manifest.json; do
  curl --fail --location --retry 3 "$base/$f?download=true" -o "models/lid/d2-ecapa/$f"
done
sha256sum --check --strict <<'EOF'
e2c3c3da39b99e3f9196d15fceef6a65f702320038bbc08813a4f21280255ce8  models/lid/d2-ecapa/voxlingua107.onnx
d593825e7e8ce82c8b50c0810a371a3e69cd0293b52751ddf515f6ee2c592182  models/lid/d2-ecapa/lang_map.json
4ff4dff8bb48fe4b68b5f9551c5bb854154e44c5dd836b120a2aa7b2eecca54b  models/lid/d2-ecapa/manifest.json
EOF
```

The downloaded `manifest.json` gives MD5 sums for two of the three files. Use
the SHA-256 sums above instead. They cover all three files.

## Notes

`models/openvino/cache/` holds compiled OpenVINO blobs. live-stt creates it and
fills it on the first run of each engine and device. It is safe to delete. An
empty cache costs about 105 seconds of compile time on the next Whisper run,
against about 12 seconds warm, and it grows to about 2 GB for the default model.

The Whisper model is INT8 weight-compressed to the OpenVINO IR format. It is MIT
licensed, from `openai/whisper-large-v3-turbo`. The sherpa model layouts need
`sherpa-onnx` and `sherpa-onnx-core` 1.13.4 or later. That floor is the
compatibility bound; `uv.lock` pins the exact qualified runtime.

The language detector is an ECAPA VoxLingua107 model. It is Apache-2.0 licensed,
from `crash-sv/scribe-ecapa-voxlingua107`. It reads raw audio, runs on the CPU
with ONNX Runtime, and shares nothing with the Whisper model.

Engine selection rationale is in `.claude/rules/asr-pipeline.md`. D-016 covers
the Whisper default and its device. D-010 covers the choice between the two
sherpa engines.
