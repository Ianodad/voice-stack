# Web UI for the local voice assistant — design spec

**Date:** 2026-09-27
**Status:** approved in conversation (sections 1–3); awaiting written-spec review
**Branch:** `phase/web-ui` (parent `feature/voice-loop-mvp`)
**Builds on:** `.planning/PLAN.md` (live voice loop MVP), `.planning/pipecat-research.md` (pipecat-ai 1.12.0)

## 1. Intent

**What the user said:** a custom-designed web interface for daily personal use on this Mac. Must have: live transcript, animated orb with listening/thinking/speaking states, mute + push-to-talk, saved history where reopening a conversation continues it (context reloaded).

**Assumptions (confirmed by approval of design sections):** fully local — served on `127.0.0.1`, no internet after setup; single user; desktop browser only.

**Success looks like:**
1. `uv run voice-stack web` → open `http://localhost:7860` → talk on laptop speakers, interrupt the bot mid-sentence, no self-hearing.
2. Close the tab, reopen, continue yesterday's conversation and the bot remembers it.
3. Latency no worse than the current local-mic mode (live LLM TTFB mean 0.43s, n=22, `.planning/live-run-2026-09-27.log`).

## 2. Architecture

```
Browser (127.0.0.1:7860)                      Python process
┌──────────────────────────┐   WebRTC audio  ┌─────────────────────────────────┐
│ web/ (Vite, vanilla TS)  │◄──────────────►│ server.py  (FastAPI)            │
│  orb · transcript ·      │  RTVI messages  │   POST /api/offer  → new session│
│  history list · mute/PTT │                 │   /api/conversations (REST)     │
└──────────────────────────┘                 │ bot.py     build_pipeline(...)  │
                                             │ runtime.py loaded once at start │
                                             │ history.py SQLite (stdlib)      │
                                             └─────────────────────────────────┘
```

Key decision: **browser WebRTC via Pipecat `SmallWebRTCTransport`**. Browser `getUserMedia` echo cancellation removes self-hearing → barge-in works on speakers. This is the main reason for the approach; in web mode the `AlwaysUserMuteStrategy` is NOT used.

### Units

| Unit | Responsibility | Interface | Depends on |
|---|---|---|---|
| `src/voice_stack/runtime.py` | Load once per process: MLX executor, parakeet, Kokoro, `mlx_lm.server` lifecycle, warmup. Extracted from current `__init__.py`; lifecycle guarantees unchanged (SIGINT/SIGTERM → server stopped). | `Runtime.start()`, `Runtime.stop()`, attrs: `executor`, `stt_model`, `tts_model`, `llm_base_url`; `Runtime.restart_llm()` | `llm_server.py`, `stt.py`, `tts.py` |
| `src/voice_stack/bot.py` | Build the pipeline for any transport. | `build_pipeline(transport, runtime, messages, *, mute_while_bot_speaks: bool) -> (Pipeline, LLMContext)` | runtime, pipecat |
| `src/voice_stack/history.py` | SQLite persistence. | `create() -> id`, `list() -> [{id,title,updated_at}]`, `get(id) -> messages`, `append(id, role, content)`, `delete(id)`, `context_window(id, n=20) -> messages` | stdlib `sqlite3` |
| `src/voice_stack/server.py` | FastAPI app: static UI, WebRTC signaling, history REST, single active session. | routes below | bot, history, runtime |
| `src/voice_stack/__init__.py` | CLI: `voice-stack` (local mic, existing), `--barge-in`, `--check`, new `web` subcommand. | argparse | all |
| `web/` | Frontend, built to `web/dist/`, served by FastAPI. | talks to routes + RTVI | `@pipecat-ai/client-js`, `@pipecat-ai/small-webrtc-transport` (versions pinned to what matches pipecat 1.12.0) |

### Routes

| Route | Purpose |
|---|---|
| `GET /` | `web/dist/index.html` + assets |
| `POST /api/offer?conversation_id=<id>` | WebRTC SDP offer/answer via `SmallWebRTCRequestHandler`; starts a pipeline session loaded with that conversation's context window. No id → new conversation. |
| `GET /api/conversations` | list, newest first |
| `GET /api/conversations/{id}` | full messages (transcript display) |
| `DELETE /api/conversations/{id}` | delete |
| `POST /api/llm/restart` | relaunch `mlx_lm.server` + warmup after failure |

Bind `127.0.0.1` only.

### Data (SQLite, `~/.voice-stack/history.db`)

```
conversations(id TEXT PK, title TEXT, created_at TEXT, updated_at TEXT)
messages(id INTEGER PK, conversation_id TEXT FK ON DELETE CASCADE, role TEXT CHECK(role IN ('user','assistant')), content TEXT, created_at TEXT)
```
- Title = first user utterance truncated to 60 chars.
- Persist on turn completion: user message when the user turn is aggregated; assistant message when the assistant aggregator finalizes (interrupted → only the spoken part, which is what Pipecat's assistant aggregator already records).
- Only `user`/`assistant` roles stored. Pipecat's injected `developer` messages are never persisted.
- LLM receives system instruction + last **20** stored messages. Full history stored and shown.

## 3. Behavior

### Session model
- One active session. A new `/api/offer` cancels the existing pipeline worker; old tab shows "session moved to another tab".
- Switching conversation in the sidebar = disconnect + reconnect with new `conversation_id` (~1s, estimate).
- Models and `mlx_lm.server` persist across sessions (loaded once in `Runtime`).

### Orb states (driven by RTVI client events — exact event names to be verified against client-js matching pipecat 1.12.0)

| State | Trigger | Visual |
|---|---|---|
| connecting | page load / reconnect | faint slow pulse |
| listening | connected, mic on, nobody speaking | soft breathing, scales with mic level |
| user speaking | user-started-speaking | bright, follows mic level |
| thinking | user-stopped-speaking → before bot-started-speaking | slow swirl |
| speaking | bot-started-speaking | follows bot audio level |
| muted | mic disabled | greyed, "muted" caption |
| error | disconnect / server error | red ring + reconnect/restart button |

Canvas 2D, no animation library. Dark default, follows `prefers-color-scheme`. Desktop width only.

### Controls
- Mute button toggles mic (`enableMic`).
- Space: while muted, hold to talk (mic on while held; release = mic off → turn ends via VAD/turn detector). While unmuted, space does nothing.
- Esc: interrupt bot (client interrupt message — to be verified in client-js/RTVI 1.12.0).

### Transcript
- User line appears when parakeet finalizes the segment (segmented STT — no partial words).
- Bot line streams per sentence as TTS text events arrive.
- Reopened conversation shows full stored history above the live turns.

### Identity fix
System instruction states it is a local assistant running on the user's Mac (Qwen via MLX), not a cloud service. Fixes observed "I run on Google's servers" hallucination.

## 4. Errors

| Failure | Behavior |
|---|---|
| `mlx_lm.server` dies | Detected on next LLM error; UI error state + "Restart" → `POST /api/llm/restart` → reconnect. That turn not persisted. |
| Tab closed / reload | Session ends; completed turns already persisted; page offers "continue last conversation". |
| Second tab | Replaces first (see session model). |
| Mic permission denied | Message with Chrome/Safari fix steps. |
| Ctrl-C / SIGTERM on server | Existing lifecycle: `mlx_lm.server` always stopped; SQLite committed per turn. |
| Pipecat `developer` role | Existing fix (`supports_developer_role=False`) carried into `bot.py`. |

## 5. Testing

- `voice-stack --check` (existing) + history round-trip: create, append 2 turns, reload, assert 20-message window cap with 25 messages, delete; uses a temp DB path.
- STT/TTS services emit TTFB metrics (`start_ttfb_metrics`/`stop_ttfb_metrics`) so live logs show all three stages.
- `scripts/check_web.py`: start the real server (real runtime; models are cached, ~10s startup) on a random port, assert `GET /` 200, conversations CRUD, and `/api/offer` completes an SDP exchange using aiortc as a headless client. If headless WebRTC proves unworkable, downgrade to asserting the endpoint returns a valid SDP answer and note it.
- Live checklist (user): orb states cycle correctly; interrupt the bot on speakers without self-hearing; space-to-talk; reopen yesterday's conversation and ask "what did we talk about?".
- Review tier: Sonnet 5 xhigh (routine, single-user local app; not auth/payments/schema-migration). Session replacement + subprocess restart get explicit review attention.

## 6. Gate before building (task 0)

Verify from installed pipecat 1.12.0 source + npm client packages, before any other task:
1. `SmallWebRTCTransport` + `SmallWebRTCRequestHandler` import paths, required extra (`pipecat-ai[webrtc]` → aiortc), dependency resolve against current pins.
2. Compatible `@pipecat-ai/client-js` / `small-webrtc-transport` versions and RTVI event names for the orb states, transcript, and interrupt.
3. How the RTVIProcessor/observer is wired in 1.12.0 (log shows `RTVIProcessor#0` auto-linked).

If WebRTC is not viable: fall back to Pipecat WebSocket transport; this loses browser AEC → web mode reverts to mute-while-bot-speaks. Requires user sign-off (changes success criterion 1).

## 7. Out of scope (v1)

Long-term memory across conversations, search, voice/model settings UI, mobile layout, tool use (opening apps), multi-user, auth, LAN access.
