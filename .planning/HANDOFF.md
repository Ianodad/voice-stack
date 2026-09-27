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
- PLAN rev 2 locked (Fable applied). Sonnet executor running T1→T4.

Remaining:
1. Wait for executor report; verify checks myself.
3. Dispatch Sonnet executors; Sonnet xhigh review; synthesis.
4. Live test with mic (user must talk to it).

Open Flags:
- Echo/self-hearing: may need headphones unless AEC exists.
- No uncommitted code yet.

Git State: branch feature/voice-loop-mvp; HEAD 457bda8 (auto-backup checkpoint); .planning/ files uncommitted.

Resume instruction: check git log + which check scripts exist; resume T1–T4 from first failing gate per PLAN.md rev 2, then Sonnet xhigh review.
