# Local Voice AI Stack — Model Research

**Date:** 2026-09-23
**Target hardware:** MacBook Pro, Apple M5 Pro, 18 cores (6 Super / 12 Performance), 64 GB unified memory
**Goal:** fully local (no cloud API) conversational voice assistant, sub-800ms voice-to-voice
**Method:** three parallel web-research passes (STT/VAD, TTS, orchestration+LLM), Sept 2026 sources

---

## 1. Verdict summary

| Layer | Pick | License | Why |
|---|---|---|---|
| Architecture | **Cascaded** (STT → LLM → TTS) | — | End-to-end speech models closed the latency gap but still lose on reasoning + tool use |
| Orchestration | **Pipecat** | BSD-2-Clause | Local-first services built in, active (release 2026-09-18), documented <800ms all-local Mac demos |
| VAD | **Silero VAD v5** | MIT | ~2 MB, RTF ≈0.004, negligible cost |
| Turn detection | **LiveKit Turn Detector v1.0** or Pipecat **Smart Turn v2** | Apache 2.0 | Semantic + acoustic; 9.9% false-cutoff at 300ms budget |
| STT | **parakeet-mlx** (primary) / **WhisperKit** (fallback) | Apache-2.0 + CC-BY-4.0 / MIT | See conflict note §3 |
| LLM | **Qwen3.6-35B-A3B** (MoE, 3B active) | Apache 2.0 | ~24.7 GB at Q5_K_M, low TTFT despite size; fits 64 GB with room for STT+TTS |
| Runtime | **mlx-lm** (or Ollama MLX backend) | MIT / MIT | MLX up to 3x decode, 4x TTFT vs llama.cpp on M5 Neural Accelerators |
| TTS | **Kokoro-82M** via `mlx-audio` | Apache 2.0 | TTFA ~90ms, RTF 0.019–0.08 on M4/M5, most mature Apple Silicon path |
| TTS fallback | **Kyutai Pocket TTS** (100M) | MIT | CPU-only (no Metal contention), voice cloning, 6+ languages |

Latency budget: VAD 50–100ms → STT 100–150ms → LLM TTFT 250–400ms → TTS TTFA 100–200ms = **~650–800ms**.

---

## 2. Architecture decision: cascaded, not end-to-end

Evidence for cascaded:
- Qwen3-Omni Talker: 702ms TTFA vs cascade 755ms — latency advantage is now marginal, not decisive.
- End-to-end models handle turn-taking and backchannel more naturally but produce **semantically weaker** responses.
- Cascades win on tool use, function calling, structured output — all solved in the text layer.
- Industry source (Sept 2026): "cascade remains the dominant production architecture in 2026" for fully self-hosted realtime agents.

Keep as a **side experiment, not the production brain:** Kyutai **Moshi** (`moshika-mlx-q4`) — genuinely self-hostable full-duplex on Mac via MLX, ~160-240ms, but its speech-LM backbone is small and not competitive for reasoning against a dedicated 27–35B text LLM.

---

## 3. Open conflict — STT choice (must resolve by benchmark)

The two STT candidates pull in opposite directions, and this is the one decision the research could **not** settle:

**WhisperKit + large-v3-turbo**
- Pro: most proven (ICML 2025 paper, MIT, widely shipped). 99 languages. Runs on **Neural Engine**, so it does not contend with the LLM for the Metal GPU.
- Con: Swift/Core ML package. Awkward to drive from a Python Pipecat pipeline — needs `whisperkit-cli` subprocess or an IPC shim. Chunked 30s windows; streaming is an incremental mode bolted on.

**parakeet-mlx + Parakeet TDT 0.6B**
- Pro: Python-native, true transducer streaming architecture, competitive WER (~6–7%), permissive license. Drops straight into Pipecat.
- Con: English-only. Runs on **Metal GPU** — competes with the LLM for the same silicon.

**Also watching:** Kyutai STT 2.6B (MLX-native, built-in *semantic* VAD, ~0.5s turn delay). Architecturally closest to ideal for an assistant, but newer and less battle-tested — fails the "proven" bar today.

> **Benchmark caveat [important].** Public Mac RTF numbers for Whisper vs Parakeet **directly contradict each other** across 2026 blog posts (one claims Parakeet 2.6x slower on Core ML, another claims 100x+ real-time via MLX). These read like uneven SEO content. Open ASR Leaderboard RTFx figures are measured on **NVIDIA GPUs, not Apple Silicon**. Treat every Mac latency number in this document as unverified until measured on this machine.

**Plan:** start with parakeet-mlx (integration cost is near zero), measure real end-to-end latency, and only pay the WhisperKit IPC cost if Parakeet's GPU contention with the LLM proves to be the bottleneck.

---

## 4. Licensing — commercial safety

**Safe (permissive):** Kokoro (Apache 2.0), Kyutai Pocket TTS (MIT), Kyutai TTS 1.6B + STT (CC-BY-4.0, attribution), Qwen3-TTS (Apache 2.0), Chatterbox (MIT), Fish Speech S1/S2 *current* weights (MIT), Silero VAD (MIT), WhisperKit (MIT), Parakeet (Apache-2.0 code / CC-BY-4.0 weights), Pipecat (BSD-2), mlx-audio (MIT).

**Avoid in the live path:**
- **XTTS-v2** — CPML, non-commercial, and Coqui Inc. is defunct so no commercial license is purchasable. Also 600ms TTFA, too slow regardless.
- **F5-TTS** — CC-BY-NC-4.0, non-commercial, no streaming path found.
- **Piper** — the maintained fork (`OHF-Voice/piper1-gpl`) is **GPL-3.0**; the old MIT `rhasspy/piper` repo is archived. Fine for personal/OSS, reciprocal obligations if embedded in closed source.
- **Sesame CSM** — Apache 2.0 code but gated, terms-bound weights.
- **Orpheus** — Llama-3.2 license applies on top of Apache 2.0 code.

---

## 5. Rejected options and why

| Option | Reason |
|---|---|
| Vocode | Stalled — last verified commit Nov 2024 |
| Bolna | Telephony-first, leans on hosted providers, "looking for maintainers" |
| LiveKit Agents (as primary) | Capable and Apache 2.0, but full self-host is ~3–4 weeks of infra work for single-user parity. Revisit if multi-user/rooms are ever needed. Its *turn detector* is still worth using standalone. |
| NVIDIA Canary / Canary-Qwen-2.5B | Tops ASR leaderboard (~5.6% WER) but CUDA-first, batch-oriented, no MLX/Core ML port |
| Distil-Whisper large-v3.5 | Batch/chunked only — no true streaming |
| Moonshine | Good for short commands; weaker on complex/accented speech. Vendor latency claims look cherry-picked. |
| Roll-your-own asyncio loop | Barge-in is the hard part — self-hearing suppression + frame-accurate task cancellation. Not worth rebuilding what Pipecat ships. |
| vllm-metal | Released 2026-09-22 (v0.28.0), lowest TTFT at concurrency 2–4 — but community-maintained and aimed at multi-session serving. Overkill for single user. |

---

## 6. Environment notes

- System `python3` is **3.14.6** — too new for much of the ML ecosystem (torch, onnxruntime, MLX ports). Pin the project to **Python 3.12** via `uv`.
- `ffmpeg` 8.1.2 present. `uv` 0.12.0 present. **Ollama not installed** (not required if going mlx-lm direct).
- 64 GB unified memory means the full stack (35B-A3B LLM ~25 GB + STT + TTS) fits with large headroom. Model size is not a binding constraint here; **GPU contention between STT and LLM is** — see §3.

---

## 7. Confidence

| Claim | Confidence |
|---|---|
| Cascaded is the right architecture today | **High** — multiple independent 2026 sources agree |
| Pipecat is the right orchestrator | **High** — active, BSD, local-first, working Mac reference build |
| Kokoro is the right TTS | **High** — Apache 2.0, consistent sub-100ms figures across sources |
| Qwen3.6-35B-A3B + MLX is the right brain | **Medium-High** — MoE/TTFT logic is sound; exact model choice worth re-checking at build time |
| Any specific Mac latency number in this doc | **Low** — sources contradict; must benchmark locally |
| STT pick (Parakeet vs WhisperKit) | **Low-Medium** — genuine unresolved tradeoff, decide by measurement |

Single piece of evidence that would most change this: a real end-to-end latency measurement on this M5 Pro with Parakeet and the LLM sharing the GPU.
