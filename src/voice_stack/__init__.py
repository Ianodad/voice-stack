"""
Live local voice loop MVP: mic -> parakeet-mlx STT -> mlx_lm.server (Qwen,
OpenAILLMService) -> mlx-audio Kokoro TTS -> speaker, all on-device.

See .planning/PLAN.md for the architecture and design rationale.
"""

import argparse
import asyncio
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.empty_user_turn import (
    DEFAULT_EMPTY_USER_TURN_INTERRUPTED_PROMPT as DEVELOPER_ROLE_RECOVERY_PROMPT,
)
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.workers.runner import WorkerRunner

from voice_stack.llm_server import MLXLMServer
from voice_stack.stt import ParakeetSTTService
from voice_stack.tts import MLXKokoroTTSService

REPO_ROOT = Path(__file__).resolve().parents[2]
SPIKE_AUDIO_PATH = REPO_ROOT / "scripts" / "spike_input.wav"

STT_MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"
LLM_MODEL_ID = "mlx-community/Qwen3.6-35B-A3B-4bit"
TTS_MODEL_ID = "mlx-community/Kokoro-82M-bf16"
TTS_VOICE = "af_heart"
TTS_LANG_CODE = "a"

LLM_HOST = "127.0.0.1"
LLM_PORT = 8080
# Qwen3.6's chat template defaults to "thinking" mode; a voice assistant has
# no latency budget for chain-of-thought, so this is disabled explicitly
# (same reasoning as scripts/latency_spike.py's run_llm).
ENABLE_THINKING_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

SYSTEM_PROMPT = "You are a concise voice assistant. Reply in 1-3 short sentences, no markdown."
WARMUP_USER_TEXT = "Say hello in one short sentence."
CHECK_USER_TEXT = "What is two plus two?"
MAX_TOKENS = 60


async def _llm_text_turn(llm: OpenAILLMService, user_text: str) -> tuple[str, float | None]:
    """One streaming text turn through the actual configured LLM client.

    Reuses the OpenAILLMService instance's own client/settings (single source
    of truth for base_url/model/extra_body) rather than building a second,
    parallel client -- used for warmup and for the --check TTFT/no-<think>
    assertions, since driving OpenAILLMService itself requires the full
    frame-based pipeline machinery this diagnostic call intentionally skips.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    stream = await llm._client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=messages,
        stream=True,
        max_tokens=MAX_TOKENS,
        extra_body=ENABLE_THINKING_EXTRA_BODY,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks.append(delta)
    return "".join(chunks), ttft


async def _llm_developer_role_turn(llm: OpenAILLMService, user_text: str) -> str | None:
    """One turn through OpenAILLMService.run_inference() with a developer-role
    message in context -- the same message shape Pipecat's own user
    aggregator injects when the user interrupts the bot with no transcript
    (see llm_response_universal.py's _maybe_recover_empty_user_turn).

    Regression check for the developer-role bug: without
    `llm.supports_developer_role = False`, the adapter sends role="developer"
    verbatim, Qwen3.6's chat template raises "Unexpected message role.", and
    mlx_lm.server 404s -- this drives the real OpenAILLMService + adapter
    path (unlike _llm_text_turn's raw client call) so --check actually
    exercises it.
    """
    context = LLMContext(
        messages=[
            {"role": "developer", "content": DEVELOPER_ROLE_RECOVERY_PROMPT},
            {"role": "user", "content": user_text},
        ]
    )
    return await llm.run_inference(context, max_tokens=MAX_TOKENS)


async def async_main(check: bool, barge_in: bool) -> None:
    # One dedicated MLX thread for STT + TTS (see PLAN.md "Key design calls":
    # avoids two threads issuing Metal work on different streams).
    executor = ThreadPoolExecutor(max_workers=1)
    llm_server = MLXLMServer(model_id=LLM_MODEL_ID, host=LLM_HOST, port=LLM_PORT)

    try:
        print(f"Starting mlx_lm.server ({LLM_MODEL_ID}) ...")
        t0 = time.perf_counter()
        llm_server.start(timeout=60.0)
        print(f"  ready in {time.perf_counter() - t0:.2f}s")

        print(f"Loading STT ({STT_MODEL_ID}) ...")
        t0 = time.perf_counter()
        stt = ParakeetSTTService(model_id=STT_MODEL_ID, executor=executor)
        print(f"  loaded in {time.perf_counter() - t0:.2f}s")

        print(f"Loading TTS ({TTS_MODEL_ID}) ...")
        t0 = time.perf_counter()
        tts = MLXKokoroTTSService(
            executor=executor, model_id=TTS_MODEL_ID, voice=TTS_VOICE, lang_code=TTS_LANG_CODE
        )
        print(f"  loaded in {time.perf_counter() - t0:.2f}s")

        llm = OpenAILLMService(
            base_url=llm_server.base_url,
            api_key="not-needed",
            settings=OpenAILLMService.Settings(
                model=LLM_MODEL_ID,
                system_instruction=SYSTEM_PROMPT,
                extra={"extra_body": ENABLE_THINKING_EXTRA_BODY},
            ),
        )
        # mlx_lm.server's Qwen3.6 chat template raises "Unexpected message
        # role." on role="developer" (see chat_template.jinja); Pipecat's own
        # user aggregator injects one on an empty interrupted turn. The base
        # class defaults to assuming native "developer" role support, so
        # override per-instance to make the adapter convert it to "user"
        # before sending (base_llm.py:343).
        llm.supports_developer_role = False

        # system_instruction above replaces the old initial "system" message
        # in context (base_llm.py's system_instruction path was deprecated in
        # 1.9.0 for the latter).
        context = LLMContext()
        user_agg, assistant_agg = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(),
                # No AEC (see PLAN.md): on speakers the bot hears itself and
                # transcribes fragments of its own reply as user speech. Mute
                # the mic while the bot is speaking by default; --barge-in
                # (headphone use) drops this to keep interruptions live.
                user_mute_strategies=[] if barge_in else [AlwaysUserMuteStrategy()],
            ),
            # user_turn_strategies left at default -> LocalSmartTurnAnalyzerV3, on-device.
        )

        transport = LocalAudioTransport(
            LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
        )

        pipeline = Pipeline(
            [transport.input(), stt, user_agg, llm, tts, transport.output(), assistant_agg]
        )
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000, audio_out_sample_rate=24000, enable_metrics=True
            ),
            # Local assistant waits indefinitely; default 300s idle timeout exits it.
            idle_timeout_secs=None,
        )

        # --- Warmup: pay MLX's one-time first-call lazy-compile cost here,
        # not on the user's first turn (see PLAN.md, ~6.85s first-turn penalty). ---
        print("Warming up ...")

        t0 = time.perf_counter()
        audio_bytes = SPIKE_AUDIO_PATH.read_bytes()
        stt_frames = [f async for f in stt.run_stt(audio_bytes) if f is not None]
        stt_warmup_s = time.perf_counter() - t0
        if stt_frames:
            print(f"  STT warmup: {stt_warmup_s:.2f}s, text={stt_frames[0].text!r}")
        else:
            print(f"  STT warmup: {stt_warmup_s:.2f}s, no transcript")

        t0 = time.perf_counter()
        tts_frames = [
            f async for f in tts.run_tts(WARMUP_USER_TEXT, context_id="warmup") if f is not None
        ]
        print(f"  TTS warmup: {time.perf_counter() - t0:.2f}s, frames={len(tts_frames)}")

        t0 = time.perf_counter()
        warmup_reply, warmup_ttft = await _llm_text_turn(llm, WARMUP_USER_TEXT)
        warmup_ttft_ms = f"{warmup_ttft * 1000:.1f}ms" if warmup_ttft is not None else "N/A"
        print(
            f"  LLM warmup: total={time.perf_counter() - t0:.2f}s ttft={warmup_ttft_ms} "
            f"reply={warmup_reply!r}"
        )

        if check:
            print(f"\nCheck: sending one timed text turn ({CHECK_USER_TEXT!r}) ...")
            reply, ttft = await _llm_text_turn(llm, CHECK_USER_TEXT)
            ttft_ms = ttft * 1000 if ttft is not None else None
            print(f"  reply={reply!r}")
            print(f"  LLM TTFT (warm) = {ttft_ms:.1f}ms" if ttft_ms is not None else "  LLM TTFT: N/A")

            print("Check: sending a developer-role turn through OpenAILLMService.run_inference() ...")
            dev_role_reply = await _llm_developer_role_turn(llm, CHECK_USER_TEXT)
            print(f"  reply={dev_role_reply!r}")

            assert stt_frames, "STT warmup produced no TranscriptionFrame"
            assert tts_frames, "TTS warmup produced no TTSAudioRawFrame"
            assert ttft_ms is not None and ttft_ms < 400, f"LLM TTFT {ttft_ms}ms >= 400ms"
            assert "<think>" not in reply, f"reply contained <think>: {reply!r}"
            assert dev_role_reply, "developer-role run_inference() turn produced no reply"
            assert "<think>" not in dev_role_reply, (
                f"developer-role reply contained <think>: {dev_role_reply!r}"
            )

            print("\ncheck: PASS")
            return

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
        llm_server.stop()
        executor.shutdown(wait=True)


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
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        asyncio.run(async_main(check=args.check, barge_in=args.barge_in))
    except KeyboardInterrupt:
        pass
