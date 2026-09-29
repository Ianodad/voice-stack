"""Pipeline factory: builds one per-session PipelineWorker from a Runtime."""

from collections.abc import Callable

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

from voice_stack.runtime import (
    STT_MODEL_ID,
    TTS_LANG_CODE,
    TTS_MODEL_ID,
    TTS_VOICE,
    Runtime,
)
from voice_stack.stt import ParakeetSTTService
from voice_stack.tts import MLXKokoroTTSService


def build_worker(
    transport,
    runtime: Runtime,
    messages: list[dict],
    *,
    mute_while_bot_speaks: bool,
    on_turn: Callable[[str, str], None] | None = None,
) -> tuple[PipelineWorker, LLMContext]:
    """Build a fresh STT/TTS/LLM pipeline around `transport`, reusing the
    runtime's preloaded models. `messages` is prior {role, content} history.
    `on_turn(role, content)` fires for each non-empty finished user/assistant
    turn (interrupted assistant turns carry only the text actually spoken).
    """
    stt = ParakeetSTTService(
        model_id=STT_MODEL_ID, executor=runtime.executor, model=runtime.stt_model
    )
    tts = MLXKokoroTTSService(
        executor=runtime.executor,
        model_id=TTS_MODEL_ID,
        voice=TTS_VOICE,
        lang_code=TTS_LANG_CODE,
        model=runtime.tts_model,
    )
    llm = runtime.make_llm()

    # system_instruction (in make_llm) replaces the old initial "system"
    # message in context (base_llm.py's system_instruction path was deprecated
    # in 1.9.0 for the latter).
    context = LLMContext()
    if messages:
        context.set_messages(messages)
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            # No AEC (see PLAN.md): on speakers the bot hears itself and
            # transcribes fragments of its own reply as user speech. Mute
            # the mic while the bot is speaking by default; --barge-in
            # (headphone use) drops this to keep interruptions live.
            user_mute_strategies=[AlwaysUserMuteStrategy()] if mute_while_bot_speaks else [],
        ),
        # user_turn_strategies left at default -> LocalSmartTurnAnalyzerV3, on-device.
    )

    if on_turn is not None:

        @user_agg.event_handler("on_user_turn_stopped")
        async def _on_user_turn_stopped(aggregator, strategy, message):
            if message.content and message.content.strip():
                on_turn("user", message.content)

        @assistant_agg.event_handler("on_assistant_turn_stopped")
        async def _on_assistant_turn_stopped(aggregator, message):
            if message.content and message.content.strip():
                on_turn("assistant", message.content)

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
    return worker, context
