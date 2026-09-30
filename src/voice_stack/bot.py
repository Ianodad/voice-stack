"""Pipeline factory: builds one per-session PipelineWorker from a Runtime."""

from collections.abc import Callable
from datetime import date

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMFullResponseStartFrame, LLMTextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserverParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

from voice_stack import toolset
from voice_stack.runtime import (
    STT_MODEL_ID,
    TTS_LANG_CODE,
    TTS_MODEL_ID,
    TTS_VOICE,
    Runtime,
)
from voice_stack.stt import ParakeetSTTService
from voice_stack.tts import MLXKokoroTTSService


class ReplyTap(FrameProcessor):
    """Pass-through between llm and tts: remembers the exact LLM text of the
    current reply (code in its true position, no spoken cue)."""

    def __init__(self):
        super().__init__()
        self.full = ""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self.full = ""
        elif isinstance(frame, LLMTextFrame):
            self.full += frame.text
        await self.push_frame(frame, direction)


def reply_content(tap: ReplyTap, context: LLMContext, message) -> str:
    """Content to store for a finished assistant turn. When the reply had a
    code fence and was not interrupted, the aggregator context holds the code
    out of order and the spoken cue; rewrite the last assistant message to the
    exact full reply. Otherwise return message.content unchanged."""
    if "```" not in tap.full or message.interrupted:
        return message.content
    full = tap.full.strip()
    # Match the message holding THIS turn's spoken text. With tool calls the
    # last assistant message can be a tool_calls message; never touch those.
    spoken = (message.content or "").strip()
    for msg in reversed(context.messages):
        if (
            msg.get("role") == "assistant"
            and not msg.get("tool_calls")
            and isinstance(msg.get("content"), str)
            and msg["content"].strip() == spoken
        ):
            msg["content"] = full
            break
    return full


def make_assistant_turn_handler(
    tap: ReplyTap, context: LLMContext, on_turn: Callable[[str, str], None] | None
):
    async def _on_assistant_turn_stopped(aggregator, message):
        content = reply_content(tap, context, message)
        if on_turn is not None and content and content.strip():
            on_turn("assistant", content)

    return _on_assistant_turn_stopped


def build_worker(
    transport,
    runtime: Runtime,
    messages: list[dict],
    *,
    mute_while_bot_speaks: bool,
    on_turn: Callable[[str, str], None] | None = None,
    tools: "toolset.ToolSession | None" = None,
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
    if tools is None:
        llm = runtime.make_llm()
        context = LLMContext()
    else:
        llm = runtime.make_llm(
            system_instruction=toolset.system_prompt(date.today(), tools.root)
        )
        schema, handlers = toolset.build(tools)
        toolset.register(llm, handlers)
        context = LLMContext(tools=schema)

    # system_instruction (in make_llm) replaces the old initial "system"
    # message in context (base_llm.py's system_instruction path was deprecated
    # in 1.9.0 for the latter).
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

    reply_tap = ReplyTap()
    assistant_agg.event_handler("on_assistant_turn_stopped")(
        make_assistant_turn_handler(reply_tap, context, on_turn)
    )

    if tools is not None:

        @user_agg.event_handler("on_user_turn_started")
        async def _on_user_turn_started(aggregator, strategy):
            tools.reset_turn()

    if on_turn is not None:

        @user_agg.event_handler("on_user_turn_stopped")
        async def _on_user_turn_stopped(aggregator, strategy, message):
            if message.content and message.content.strip():
                on_turn("user", message.content)

    pipeline = Pipeline(
        [transport.input(), stt, user_agg, llm, reply_tap, tts, transport.output(), assistant_agg]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000, audio_out_sample_rate=24000, enable_metrics=True
        ),
        # Local assistant waits indefinitely; default 300s idle timeout exits it.
        idle_timeout_secs=None,
        # The spoken "code is on screen" cue is hidden from the client.
        rtvi_observer_params=RTVIObserverParams(skip_aggregator_types=["cue"]),
        app_resources=tools,
    )
    return worker, context
