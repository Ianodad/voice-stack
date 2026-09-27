## Handoff — Opus — voice-stack / Phase: Live voice loop MVP

Current State: Research (docs/RESEARCH.md) + latency spike (docs/BENCHMARK.md, 729ms warm) committed on master.
User picked next phase 2026-09-27: live voice loop MVP on Pipecat.

Decisions Made:
- Branch `feature/voice-loop-mvp` from master.
- Stack stays: parakeet-mlx STT, Qwen3.6-35B-A3B-4bit via mlx-lm, Kokoro-82M via mlx-audio, Silero VAD, Pipecat.
- MVP must include: startup warmup through all 3 models, sentence-chunked TTS.

In progress:
- Sonnet research agent verifying Pipecat APIs from installed source → `.planning/pipecat-research.md`.

Remaining:
1. Read pipecat-research.md, write `.planning/PLAN.md` (tasks, acceptance criteria).
2. Fable plan-lock pass (ledger line) → Sonnet review of plan.
3. Dispatch Sonnet executors; Sonnet xhigh review; synthesis.
4. Live test with mic (user must talk to it).

Open Flags:
- Echo/self-hearing: may need headphones unless AEC exists.
- No uncommitted code yet.

Git State: branch feature/voice-loop-mvp, clean, HEAD 97afa68.

Resume instruction: if pipecat-research.md exists, start step 1; else re-dispatch the research agent (prompt in this session's history: verify Pipecat transport/VAD/turn/STT/LLM/TTS APIs from installed source).
