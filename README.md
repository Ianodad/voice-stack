# voice-stack

Fully local voice assistant for Apple Silicon: parakeet-mlx (STT) → Qwen3.6 via mlx-lm (LLM) → Kokoro via mlx-audio (TTS), orchestrated by Pipecat. Nothing leaves the machine after setup.

## Requirements

Apple Silicon Mac with ~24 GB+ free unified memory (Qwen3.6-35B-A3B-4bit peaks near 23 GB GPU), Python 3.12+, [uv](https://docs.astral.sh/uv/), Node 20+ (web UI build). Models download from Hugging Face on first run, then everything runs offline.

## Run

```bash
uv sync
uv run voice-stack --check            # warm up all models, one timed LLM turn, exit
uv run voice-stack --barge-in         # local mic/speakers (use headphones; no echo cancellation)
uv run voice-stack                    # local mic, mic muted while the bot speaks

# Web UI (works on laptop speakers; browser echo cancellation enables barge-in)
(cd web && npm ci && npm run build)   # once, and after frontend changes
uv run voice-stack web                # http://localhost:7860 (127.0.0.1 only)
```

Web UI: animated orb (listening / thinking / speaking / muted), live transcript, mute, hold **Space** to talk while muted, **Esc** to interrupt the bot, saved conversations you can reopen and continue (SQLite at `~/.voice-stack/history.db`; the model sees the last 20 messages).

Frontend dev: `VOICE_STACK_DEV=1 uv run voice-stack web`, then `cd web && npm run dev` (proxy on :5173).

## Notes

- If a run is killed with SIGKILL, clean up with `pkill -f mlx_lm.server`.
- Checks: `scripts/check_history.py`, `scripts/check_web.py`, `scripts/check_stt.py`, `scripts/check_tts.py`.
- Design: `docs/superpowers/specs/2026-09-27-web-ui-design.md`; plan: `docs/superpowers/plans/2026-09-29-web-ui.md`; benchmarks: `docs/BENCHMARK.md`.

## Status

Live voice loop and web UI are working; see `docs/BENCHMARK.md` for latency numbers (warm pipeline ~0.73s in isolation, live LLM TTFB ~0.43s). Not yet covered: long-term memory across conversations, tool use, mobile layout, auth (localhost only by design).
