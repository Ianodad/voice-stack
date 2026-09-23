# Voice Stack Latency Benchmark

**Generated:** 2026-09-23T18:51:24+00:00

## Machine

- Platform: macOS-26.6.2-arm64-arm-64bit
- Chip: Apple M5 Pro
- Memory: 64 GB
- Python: 3.12.13

## Models

- STT: `mlx-community/parakeet-tdt-0.6b-v3`
- LLM: `mlx-community/Qwen3.6-35B-A3B-4bit`
- TTS: `mlx-community/Kokoro-82M-bf16`

## Package versions (exact, as installed in .venv)

- mlx: 0.32.2
- mlx-lm: 0.31.3
- mlx-audio: 0.5.5
- parakeet-mlx: 0.5.2
- torch: 2.14.0
- psutil: 7.2.2
- misaki: 0.9.4
- silero-vad: 6.2.3
- huggingface-hub: 1.32.0

## Cold load time (once each, seconds)

| Model | Load time (s) |
|---|---|
| STT | 0.84 |
| LLM | 2.99 |
| TTS | 0.34 |

## Per-stage latency, N=5 runs (seconds)

Successful runs: 5/5

| Stage | Median | Min | Max |
|---|---|---|---|
| STT (audio -> transcript) | 0.093 | 0.090 | 0.775 |
| LLM TTFT | 0.200 | 0.197 | 1.943 |
| LLM total generation | 0.416 | 0.415 | 2.161 |
| TTS TTFA | 0.210 | 0.209 | 4.070 |
| TTS total synthesis | 0.220 | 0.218 | 4.115 |
| PIPELINE total | 0.729 | 0.725 | 6.853 |

**TTS streaming note:** `mlx-audio`'s `model.generate()` is a generator that yields one chunk per text segment (split on sentence boundaries). Our test sentence is a single short sentence, so it yielded exactly one chunk -- TTFA and total synthesis time are the same measurement here. This is genuine streaming for multi-sentence text, but for this specific input it degenerates to full-synthesis timing.

**First-call warmup note [measured during dry-run smoke tests, not hidden]:** the FIRST call to a model after cold load pays a one-time MLX lazy-compilation / pipeline-init cost separate from the load time itself. Observed in isolation on this machine: Kokoro TTS first `generate()` call took ~3.6s vs ~0.09s on the second call (same process, same text); parakeet STT first `transcribe()` call took ~1.7s vs ~0.07s warm. Run 0 of the N=5 loop below pays this cost for every stage, so it will show up as an outlier in the `max` column -- this is real per-process-lifetime cost, not noise, but a long-running voice assistant only pays it once at startup, not per utterance.

## GPU contention A/B (STT isolated vs STT immediately after LLM)

| Condition | Median (ms) | Min (ms) | Max (ms) |
|---|---|---|---|
| STT isolated | 74.8 | 73.9 | 90.4 |
| STT right after LLM generation | 87.3 | 86.1 | 87.7 |

Delta: +16.7% median STT latency when run right after LLM generation.

## RAM

- Process RSS before model load: 0.4 GB
- Process RSS after model load: 14.28 GB
- Peak process RSS during runs: 14.28 GB
- Peak MLX GPU memory (mx.get_peak_memory): 23.08 GB

## Sample output (last run)

- Transcript: 'What is the weather like in Nairobi today?'
- LLM response: 'I cannot provide real-time weather data, so please check a local weather service for current conditions in Nairobi.'
- TTS info: {'n_chunks': 1, 'total_samples': 170400, 'sample_rate': 24000, 'ttfa': 0.20879254100145772, 'total': 0.21823304099962115}
