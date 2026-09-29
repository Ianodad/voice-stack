## Handoff — Opus — voice-stack / Web UI built, awaiting live test

Current State: web UI implemented (Tasks 1-4 + final fix wave) on branch phase/web-ui, tracking origin/main (public repo Ianodad/voice-stack). HEAD ecea8ac-equivalent (hashes rewritten by filter-repo 2026-09-29; live-run logs scrubbed from history; backup ~/voice-stack-backup-2026-09-29.bundle). Latest commit local only, NOT pushed.

Done: history.py (SQLite), runtime.py + bot.py (extracted), server.py (FastAPI, WebRTC, Origin/Host guard), web/ (Vite TS orb UI), bot audio playback fix, README. Reviews: Sonnet xhigh per task + Opus final (Codex out of quota until Oct 3). Checks: scripts/check_{history,web,stt,tts}.py, `uv run voice-stack --check` PASS.

Remaining (user, on speakers): 1) `(cd web && npm ci && npm run build)`; `uv run voice-stack web`; open http://localhost:7860, hear bot, interrupt by voice, check no self-hearing. 2) hold Space while muted. 3) `pkill -f mlx_lm.server` mid-chat -> Restart LLM. Then record STT/LLM/TTS TTFB in docs/BENCHMARK.md.
Then: push, add LICENSE (MIT recommended), merge phase/web-ui -> feature/voice-loop-mvp only on user go.

Open Flags: Safari play() rejection not retried; botTtsText-before-botStarted could split a line; Space on other focused buttons; scripts/spike_input.wav (user's voice?) is in the public repo; no LICENSE.
Resume instruction: ask user for live-test results, fix what breaks, then push.
