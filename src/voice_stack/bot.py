"""Pipeline factory: builds one per-session PipelineWorker from a Runtime."""

import asyncio
from collections import Counter
from collections.abc import Callable
from datetime import date

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMFullResponseStartFrame, LLMTextFrame, TTSSpeakFrame
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
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

from loguru import logger

from voice_stack import toolset
from voice_stack.fence import CUES, LOCAL_CODE_CUE, print_code_block
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


def _derived_from(spoken: str, full: str) -> bool:
    """True when the turn's stored text is plausibly made from `full`. The context
    message holds the code out of order plus the spoken cue, so compare words, not
    order: with the cues removed, every word must be available in the full reply
    (multiset), and something must remain."""
    for cue in CUES:
        spoken = spoken.replace(cue, " ")
    words = Counter(spoken.split())
    return bool(words) and not (words - Counter(full.split()))


def reply_content(tap: ReplyTap, context: LLMContext, message) -> str:
    """Content to store for a finished assistant turn. When the reply had a
    code fence and was not interrupted, the aggregator context holds the code
    out of order and the spoken cue; rewrite the last assistant message to the
    exact full reply. Otherwise return message.content unchanged."""
    if "```" not in tap.full or message.interrupted:
        return message.content
    full = tap.full.strip()
    if not _derived_from(message.content or "", full):
        return message.content  # stale tap (e.g. a fixed TTSSpeakFrame turn): no rewrite
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


async def false_card_backstop(tools, text: str | None, queue) -> bool:
    """Tools mode: the finished assistant turn claims a confirmation card, yet no card is live
    (pending AND shown) and none is being or was proposed this turn -> the model repeated an old
    reply. Queue a fixed correction (it enters context as the assistant's words) and clear the UI's
    cards. At most once per turn; never while a proposal is in flight or a card is live.
    `queue(frame)` awaits worker.queue_frame. Never raises. Returns True when it fired."""
    try:
        if tools is None or tools.corrected_turn or tools.proposed_turn:
            return False
        if text == toolset.SPEAK_CORRECTION or not toolset.claims_card(text):
            return False
        await toolset.refresh_shown(tools)      # a timed-out card is not live: clears shown_id
        if tools.shown_id is not None:
            return False
        if await asyncio.to_thread(tools.pending.list, tools.session_id):
            return False        # a real card is pending: the claim is true
        if tools.proposing > 0:
            return False
        tools.corrected_turn = True
        toolset.retire_cards(tools.context, tools.last_outcome or "cleared")
        await queue(RTVIServerMessageFrame(data={"type": "actions_cleared"}))
        await queue(TTSSpeakFrame(toolset.SPEAK_CORRECTION, append_to_context=True))
        return True
    except Exception:
        logger.exception("false-card backstop failed")
        return False


async def false_done_backstop(tools, text: str | None, queue) -> bool:
    """Tools mode: a card was proposed this turn (and is pending) but the reply says 'I've updated /
    moved ...' -- a false claim the card contradicts. Queue ONE fixed clarification (in context).
    At most once per turn. Never raises. Returns True when it fired."""
    try:
        if tools is None or tools.clarified_turn or not tools.proposed_turn or not toolset.claims_done(text):
            return False
        if not await asyncio.to_thread(tools.pending.list, tools.session_id):
            return False        # the card is already resolved: the claim may be true
        tools.clarified_turn = True
        await queue(TTSSpeakFrame(toolset.SPEAK_CLARIFY, append_to_context=True))
        return True
    except Exception:
        logger.exception("false-done backstop failed")
        return False


async def user_turn_started(tools, context) -> None:
    """Tools mode, a new user turn begins: reset per-turn flags; a timed-out card is dead (clear it,
    scrub 'expired'); with no live card, scrub stale 'a card is on screen' traces from the context
    using how the last card ended."""
    tools.reset_turn()
    await toolset.refresh_shown(tools)
    if tools.shown_id is None:
        toolset.retire_cards(context, tools.last_outcome or "cleared")


def make_assistant_turn_handler(
    tap: ReplyTap, context: LLMContext, on_turn: Callable[[str, str], None] | None,
    tools: "toolset.ToolSession | None" = None, queue=None,
):
    async def _on_assistant_turn_stopped(aggregator, message):
        content = reply_content(tap, context, message)
        tap.full = ""  # consumed: never let a later turn without an LLM reply reuse it
        if on_turn is not None and content and content.strip():
            on_turn("assistant", content)
        if tools is not None and queue is not None:
            if not await false_card_backstop(tools, content, queue):
                await false_done_backstop(tools, content, queue)

    return _on_assistant_turn_stopped


def build_worker(
    transport,
    runtime: Runtime,
    messages: list[dict],
    *,
    mute_while_bot_speaks: bool,
    on_turn: Callable[[str, str], None] | None = None,
    tools: "toolset.ToolSession | None" = None,
    terminal_code: bool = False,
) -> tuple[PipelineWorker, LLMContext]:
    """Build a fresh STT/TTS/LLM pipeline around `transport`, reusing the
    runtime's preloaded models. `messages` is prior {role, content} history.
    `terminal_code`: no browser UI, so print code blocks to stdout and say so in the cue.
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
        **({"code_cue": LOCAL_CODE_CUE, "on_code": print_code_block} if terminal_code else {}),
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
    holder: dict = {}   # the worker exists only after the pipeline is built

    async def _queue(frame):
        await holder["worker"].queue_frame(frame)

    assistant_agg.event_handler("on_assistant_turn_stopped")(
        make_assistant_turn_handler(reply_tap, context, on_turn, tools, _queue)
    )

    if tools is not None:
        tools.context = context

        @user_agg.event_handler("on_user_turn_started")
        async def _on_user_turn_started(aggregator, strategy):
            await user_turn_started(tools, context)

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
    holder["worker"] = worker
    return worker, context
