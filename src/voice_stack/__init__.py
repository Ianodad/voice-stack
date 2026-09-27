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


async def async_main(check: bool) -> None:
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
                model=LLM_MODEL_ID, extra={"extra_body": ENABLE_THINKING_EXTRA_BODY}
            ),
        )

        context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
        user_agg, assistant_agg = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
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

            assert stt_frames, "STT warmup produced no TranscriptionFrame"
            assert tts_frames, "TTS warmup produced no TTSAudioRawFrame"
            assert ttft_ms is not None and ttft_ms < 400, f"LLM TTFT {ttft_ms}ms >= 400ms"
            assert "<think>" not in reply, f"reply contained <think>: {reply!r}"

            print("\ncheck: PASS")
            return

        print("Ready — speak (use headphones)")
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
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        asyncio.run(async_main(check=args.check))
    except KeyboardInterrupt:
        pass
