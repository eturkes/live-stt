# T3.2 Benchmark Results

## Benchmark 1: 45-min normal session

Scenario: 3 sessions, goAway between 1-2, unexpected close between 2-3, resumption handle carried throughout (no cold reconnects).

| proto | blocks_ok | reconnects | sessions | seed_events | seed_bytes | client_content | extractor_runs | entities | wall_s | err |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 29/29 | 2 | 3 | 0 | 0 | 0 | 0 | 0 | 17.69 | - |
| B | 29/29 | 2 | 3 | 0 | 0 | 0 | 0 | 0 | 17.66 | - |
| C | 29/29 | 2 | 3 | 0 | 0 | 0 | 58 | 8 | 17.66 | - |

## Entity extraction (C only)
- TP=8, FP=0, FN=0; precision=1.00, recall=1.00, F1=1.00
- Captured: ['ソニー', '京都', '山田さん', '新宿駅', '東京大学', '渋谷', '田中先生', '鈴木部長']
- Missed:   []
- Spurious: []

## Per-session mock log
### A
- session 0: handle_in=None, msgs_emitted=11, audio_chunks_sent=0, no seed
- session 1: handle_in='h_7a3deacb', msgs_emitted=9, audio_chunks_sent=0, no seed
- session 2: handle_in='h_e6c3d98e', msgs_emitted=14, audio_chunks_sent=0, no seed
### B
- session 0: handle_in=None, msgs_emitted=11, audio_chunks_sent=0, no seed
- session 1: handle_in='h_21e5d024', msgs_emitted=9, audio_chunks_sent=0, no seed
- session 2: handle_in='h_ff84ffa4', msgs_emitted=14, audio_chunks_sent=0, no seed
### C
- session 0: handle_in=None, msgs_emitted=11, audio_chunks_sent=0, no seed
- session 1: handle_in='h_b8f7f660', msgs_emitted=9, audio_chunks_sent=0, no seed
- session 2: handle_in='h_941793e2', msgs_emitted=14, audio_chunks_sent=0, no seed

## Benchmark 2: cold start with prior history

Scenario: fresh State (handle=None), but recent_blocks pre-populated with 4 prior JA/EN pairs (simulating restore-from-disk). One scripted session of 3 blocks. This is the only case where prototype B's cold-seed path fires.

| proto | blocks_ok | reconnects | sessions | seed_events | seed_bytes | client_content | extractor_runs | entities | wall_s | err |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 3/3 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0.97 | - |
| B | 3/3 | 0 | 1 | 1 | 373 | 1 | 0 | 0 | 0.97 | - |
| C | 3/3 | 0 | 1 | 0 | 0 | 0 | 3 | 4 | 0.97 | - |

## Entity extraction (C only)
- TP=4, FP=0, FN=4; precision=1.00, recall=0.50, F1=0.67
- Captured: ['京都', '山田さん', '東京大学', '田中先生']
- Missed:   ['ソニー', '新宿駅', '渋谷', '鈴木部長']
- Spurious: []

## Per-session mock log
### A
- session 0: handle_in=None, msgs_emitted=4, audio_chunks_sent=0, no seed
### B
- session 0: handle_in=None, msgs_emitted=4, audio_chunks_sent=0, 1 seed(s), turns=9, text_bytes=373
### C
- session 0: handle_in=None, msgs_emitted=4, audio_chunks_sent=0, no seed