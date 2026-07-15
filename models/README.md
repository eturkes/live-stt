# models/

Local STT model weights (gitignored; ~800 MB total). `live_stt.py` expects:

```
models/
├── silero_vad.onnx
├── sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/   # --engine k2v2 (default)
│   ├── encoder-epoch-99-avg-1.int8.onnx
│   ├── decoder-epoch-99-avg-1.onnx
│   ├── joiner-epoch-99-avg-1.onnx
│   └── tokens.txt
└── sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/   # --engine parakeet
    ├── model.int8.onnx
    └── tokens.txt
```

Runtime compatibility: these model layouts are supported with `sherpa-onnx` +
`sherpa-onnx-core` ≥ 1.13.4. That declared floor is the compatibility/security
bound; `uv.lock` is the exact evaluator-qualified runtime and currently resolves
both packages to 1.13.4. Re-run the evaluator gate before qualifying a newer lock.

Download (engine selection rationale: `.agent/memory.md` D-010):

```sh
cd models
base=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
curl -LO $base/silero_vad.onnx
curl -L $base/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2 | tar xj
curl -L $base/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8.tar.bz2 | tar xj  # optional A/B
```
