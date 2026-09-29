"""
Live local voice loop MVP: mic -> parakeet-mlx STT -> mlx_lm.server (Qwen,
OpenAILLMService) -> mlx-audio Kokoro TTS -> speaker, all on-device.

See .planning/PLAN.md for the architecture and design rationale.
"""

import argparse
import asyncio
import signal

from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

from voice_stack.bot import build_worker
from voice_stack.runtime import (  # noqa: F401  (re-exported)
    CHECK_USER_TEXT,
    ENABLE_THINKING_EXTRA_BODY,
    LLM_HOST,
    LLM_MODEL_ID,
    LLM_PORT,
    MAX_TOKENS,
    REPO_ROOT,
    SPIKE_AUDIO_PATH,
    STT_MODEL_ID,
    SYSTEM_PROMPT,
    TTS_LANG_CODE,
    TTS_MODEL_ID,
    TTS_VOICE,
    WARMUP_USER_TEXT,
    Runtime,
    _llm_developer_role_turn,
    _llm_text_turn,
)


async def async_main(check: bool, barge_in: bool) -> None:
    rt = Runtime()
    try:
        await rt.start()

        if check:
            llm = rt.make_llm()
            print(f"\nCheck: sending one timed text turn ({CHECK_USER_TEXT!r}) ...")
            reply, ttft = await _llm_text_turn(llm, CHECK_USER_TEXT)
            ttft_ms = ttft * 1000 if ttft is not None else None
            print(f"  reply={reply!r}")
            print(f"  LLM TTFT (warm) = {ttft_ms:.1f}ms" if ttft_ms is not None else "  LLM TTFT: N/A")

            print("Check: sending a developer-role turn through OpenAILLMService.run_inference() ...")
            dev_role_reply = await _llm_developer_role_turn(llm, CHECK_USER_TEXT)
            print(f"  reply={dev_role_reply!r}")

            assert rt.stt_frames, "STT warmup produced no TranscriptionFrame"
            assert rt.tts_frames, "TTS warmup produced no TTSAudioRawFrame"
            assert ttft_ms is not None and ttft_ms < 400, f"LLM TTFT {ttft_ms}ms >= 400ms"
            assert "<think>" not in reply, f"reply contained <think>: {reply!r}"
            assert dev_role_reply, "developer-role run_inference() turn produced no reply"
            assert "<think>" not in dev_role_reply, (
                f"developer-role reply contained <think>: {dev_role_reply!r}"
            )

            print("\ncheck: PASS")
            return

        transport = LocalAudioTransport(
            LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
        )
        worker, _ = build_worker(transport, rt, [], mute_while_bot_speaks=not barge_in)

        mode = "barge-in enabled, use headphones" if barge_in else "mic muted while bot speaks"
        print(f"Ready ({mode}) — speak")
        # handle_sigterm mirrors handle_sigint so `kill <pid>` during the live
        # pipeline run takes the same graceful WorkerRunner shutdown path as
        # Ctrl-C (see the module-level SIGTERM handler for the phases before
        # this point, i.e. server startup/warmup/--check).
        runner = WorkerRunner(handle_sigint=True, handle_sigterm=True)
        await runner.add_workers(worker)
        await runner.run()
    finally:
        rt.stop()


async def web_main(port: int) -> None:
    import uvicorn

    from voice_stack.history import History
    from voice_stack.server import create_app

    rt = Runtime()
    try:
        await rt.start()
        app = create_app(rt, History(), REPO_ROOT / "web" / "dist")
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
        print(f"Ready — open http://127.0.0.1:{port}")
        await uvicorn.Server(config).serve()
    finally:
        rt.stop()


def _raise_keyboard_interrupt(signum, frame) -> None:
    """Make SIGTERM take the same cleanup path as Ctrl-C (SIGINT).

    Installed before anything starts so `kill <pid>` during server startup,
    warmup, or --check still reaches async_main()'s `finally` (which stops
    mlx_lm.server) instead of leaving the 20GB subprocess orphaned --
    without this, SIGTERM has no handler here and skips that cleanup
    entirely (start_new_session=True already takes it out of SIGINT's/our
    process group's reach, so this is the only thing standing in for it).
    """
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description="Live local voice loop (MVP).")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Build and warm up the pipeline, run one timed LLM turn, and exit without opening the mic.",
    )
    parser.add_argument(
        "--barge-in",
        action="store_true",
        help=(
            "Keep the mic live while the bot speaks, for headphone use. Without this flag "
            "(default, for speakers with no AEC) the mic is muted while the bot speaks."
        ),
    )
    sub = parser.add_subparsers(dest="command")
    web = sub.add_parser("web", help="Serve the browser UI (WebRTC) on 127.0.0.1.")
    web.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        if args.command == "web":
            asyncio.run(web_main(args.port))
        else:
            asyncio.run(async_main(check=args.check, barge_in=args.barge_in))
    except KeyboardInterrupt:
        pass
