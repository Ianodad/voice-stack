## Handoff — Opus — voice-stack / Phase: Live voice loop MVP

Current State: Research (docs/RESEARCH.md) + latency spike (docs/BENCHMARK.md, 729ms warm) committed on master.
User picked next phase 2026-09-27: live voice loop MVP on Pipecat.

Decisions Made:
- Branch `feature/voice-loop-mvp` from master.
- Stack stays: parakeet-mlx STT, Qwen3.6-35B-A3B-4bit via mlx-lm, Kokoro-82M via mlx-audio, Silero VAD, Pipecat.
- MVP must include: startup warmup through all 3 models, sentence-chunked TTS.

Done:
- Pipecat research → `.planning/pipecat-research.md` (pipecat-ai 1.12.0; custom STT+TTS services; mlx_lm.server + OpenAILLMService; soundfile pin must relax to <0.14; no AEC → headphones).
- `.planning/PLAN.md` rev 1 written (T1 deps → T2 stt ∥ T3 tts → T4 main → T5 live test).

In progress:
- T1–T4 built by Sonnet executor; `uv run voice-stack --check` PASS re-verified by Opus 14:59 (LLM TTFT warm 187ms, no orphan server).
- Sonnet xhigh review: SIGTERM-orphan blocker + stale-server message + TTS cancel comment fixed; SIGTERM cleanup re-verified by Opus (server 1→0).
- Committed; ready for T5 live test.

Remaining:
1. T5: user runs `uv run voice-stack` with headphones; record live latency in docs/BENCHMARK.md.
2. Synthesis in PLAN.md; merge decision (feature branch → master only on user go).

Open Flags:
- Echo/self-hearing: may need headphones unless AEC exists.
- Known limits: SIGKILL/crash can still orphan mlx_lm.server (`pkill -f mlx_lm.server`); interrupted Kokoro sentence finishes on executor (~0.2s).

Git State: branch feature/voice-loop-mvp; see `git log -1` (MVP commit).

Resume instruction: ask user for live-test results (latency feel, barge-in, false cutoffs); fix what breaks; then synthesis.
