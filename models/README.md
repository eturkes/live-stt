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

## Notes

`models/openvino/cache/` holds compiled OpenVINO blobs. live-stt creates it and
fills it on the first run of each engine and device. It is safe to delete. An
empty cache costs about 105 seconds of compile time on the next Whisper run,
against about 12 seconds warm, and it grows to about 2 GB for the default model.

The Whisper model is INT8 weight-compressed to the OpenVINO IR format. It is MIT
licensed, from `openai/whisper-large-v3-turbo`. The sherpa model layouts need
`sherpa-onnx` and `sherpa-onnx-core` 1.13.4 or later. That floor is the
compatibility bound; `uv.lock` pins the exact qualified runtime.

Engine selection rationale is in `.agent/memory.md`. D-016 covers the Whisper
default and its device. D-010 covers the choice between the two sherpa engines.
