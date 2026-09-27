# PLAN — Live voice loop MVP (rev 2, 2026-09-27 — Fable plan-lock applied)

Goal: `uv run voice-stack` → talk into mic → assistant answers aloud, fully local, all on M5 Pro.
Evidence base: `docs/BENCHMARK.md` (warm 729ms), `.planning/pipecat-research.md` (pipecat-ai 1.12.0 APIs, file:line cited).

## Architecture

```
LocalAudioTransport.input (16k mic)
  → ParakeetSTTService (custom SegmentedSTTService; parakeet-mlx, in-process)
  → user aggregator (SileroVADAnalyzer + default LocalSmartTurnAnalyzerV3)
  → OpenAILLMService → mlx_lm.server subprocess (127.0.0.1:8080, Qwen3.6-35B-A3B-4bit, enable_thinking=False)
  → MLXKokoroTTSService (custom TTSService; mlx-audio, in-process; base class sentence-aggregates)
  → LocalAudioTransport.output (24k speaker)
  → assistant aggregator
```

Key design calls:
- **LLM out of process** (mlx_lm.server). Reuses OpenAI service, streaming, context handling for free; localhost HTTP costs single-digit ms; prefix cache helps multi-turn. App launches it with `start_new_session=True`, polls `/v1/models` (timeout ≥60s), terminates it (then kills after grace period) in `finally` — including Ctrl-C and crashes. Orphan = 20GB GPU process left alive.
- **One dedicated MLX thread** for STT + TTS (`ThreadPoolExecutor(max_workers=1)`, models loaded on that thread). Reason [evidence, Fable-verified mlx 0.32.2]: default stream is per-thread; the hazard is two threads issuing Metal work on different streams concurrently during barge-in. One worker prevents it. Known limit: cancelling asyncio does not stop a running Kokoro call — next STT segment queues behind it (~0.2s warm, bounded by one-sentence chunks). Accepted for MVP; comment it in code.
- **Warmup before "ready"**: one dummy transcribe (scripts/spike_input.wav), one dummy TTS sentence, one LLM request. Kills the 6.85s first-turn penalty.
- **Metrics**: `PipelineParams(enable_metrics=True)` → Pipecat logs per-service TTFB. No custom timing code.
- **Echo**: no AEC in Pipecat. MVP assumes headphones; README says so. AEC = later phase.

## Tasks
One Sonnet executor runs T1→T4 serially; each check script gates the next task.

T1 — deps (serial, first)
- pyproject: `soundfile>=0.13.1,<0.14`; add `pipecat-ai[local]==1.12.0` (NOT `local-smart-turn` extra). `uv sync`.
- Accept: `uv run python -c "import pipecat, pyaudio, parakeet_mlx, mlx_audio, mlx_lm; from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3"` succeeds (fix import path per research §2 if different); `uv pip list` shows torch/mlx/mlx-lm/mlx-audio versions unchanged vs BENCHMARK.md; `uv pip show pipecat-ai` = 1.12.0; `ffmpeg -version` works (parakeet-mlx shells out to it).

T2 — `src/voice_stack/stt.py`: `ParakeetSTTService(SegmentedSTTService)`
- Keep WAV segments; write to NamedTemporaryFile; `model.transcribe(path)` on the shared MLX executor; yield `TranscriptionFrame(text, self._user_id, time_now_iso8601(), language)`; skip empty text.
- Constructor takes model id + executor. Pattern reference: `WhisperSTTServiceMLX` (research §3).
- Accept: `scripts/check_stt.py` feeds spike_input.wav bytes (as WAV) through `run_stt` directly, asserts ≥1 TranscriptionFrame and joined text contains "Nairobi" (live turns may emit several frames; aggregator merges them).

T3 — `src/voice_stack/tts.py`: `MLXKokoroTTSService(TTSService)`
- `run_tts(text)` → Kokoro generate on shared MLX executor → yield TTSStartedFrame / TTSAudioRawFrame(int16 PCM bytes, 24000, 1) / TTSStoppedFrame per research §5; sample_rate 24000; keep default SENTENCE aggregation. Voice + model id via constructor (defaults match latency_spike.py).
- Accept: `scripts/check_tts.py` calls `run_tts` on 2 sentences separately, asserts non-empty int16 audio frames at 24k, prints per-sentence TTFA.

T4 — `src/voice_stack/__init__.py` `main()` + `src/voice_stack/llm_server.py` (after T2+T3)
- llm_server: start/stop `mlx_lm.server` subprocess, readiness poll with timeout, stderr to log file.
- main: create MLX executor, load STT/TTS models on it, warmup all three, build pipeline per research skeleton (PipelineWorker/WorkerRunner, not deprecated PipelineTask), system prompt "concise voice assistant, 1–3 short sentences, no markdown", enable_metrics. (No `allow_interruptions` flag in 1.12.0 — interruptions ride the user-turn strategy, research §6.) Print "Ready — speak (use headphones)".
- Accept: `uv run voice-stack --check` builds everything, runs warmup, sends one text turn through the LLM server (streaming), exits 0 without opening the mic, and ASSERTS: warm LLM TTFT < 400ms; reply has no `<think>`; STT+TTS warm calls succeed; after exit no `mlx_lm.server` process remains (`pgrep -f mlx_lm.server` empty). Print all timings. Plain `uv run voice-stack` runs live.

T5 — live test (user, headphones). Record observed turn latency from metrics into docs/BENCHMARK.md "Live loop" section.

## Review
Routine-tier except threading/subprocess lifecycle → Sonnet 5 xhigh review of T2–T4 diff. (Not auth/schema/payments → no Codex.)

## Out of scope (MVP)
AEC, wake word, tool calling, voice selection UI, WhisperKit, Moshi, multi-user.

## Fable divergence log
Accepted all 7 points. Note: point 7 (one executor, no T2/T3 fan-out) overrides the global parallel-fan-out default — two ~50-line files don't justify worktree+merge overhead. No rejections.
