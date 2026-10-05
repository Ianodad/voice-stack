"""
Voice sleep/wake gate check: voice_stack.wake (classify() + WakeGate).

Offline, no models, no network. Run: uv run python scripts/check_wake.py
"""
import asyncio

from pipecat.frames.frames import EndFrame, Frame, TranscriptionFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from voice_stack.wake import MUTE_PHRASES, NAME_ALIASES, SLEEP_REPLY, WAKE_REPLY, WakeGate, classify

TS = "2024-01-01T00:00:00Z"


def _tf(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "u1", TS, None)


def check_classify():
    # --- awake: each whole-utterance mute phrase ---
    for phrase in MUTE_PHRASES:
        assert classify(phrase, awake=True) == ("sleep", ""), phrase
        assert classify(phrase.capitalize(), awake=True) == ("sleep", ""), phrase
        assert classify(phrase + ".", awake=True) == ("sleep", ""), phrase

    # --- awake: name-prefixed mute ---
    assert classify("zuri mute", awake=True) == ("sleep", "")
    assert classify("Zuri, stop listening", awake=True) == ("sleep", "")
    assert classify("zoori mute yourself", awake=True) == ("sleep", "")
    assert classify("zury go to sleep", awake=True) == ("sleep", "")

    # --- awake: mute phrase inside a longer sentence must be "pass" ---
    for text in (
        "don't mute the file",
        "how do I mute a mic",
        "please mute yourself later",
        "can you stop listening to music",
    ):
        action, text_out = classify(text, awake=True)
        assert action == "pass" and text_out == text, (text, action, text_out)

    # --- awake: ordinary speech -> pass, unchanged ---
    action, text_out = classify("What's the weather today?", awake=True)
    assert action == "pass" and text_out == "What's the weather today?"

    # --- asleep: random speech -> drop ---
    action, text_out = classify("I think it's going to rain later", awake=False)
    assert (action, text_out) == ("drop", "")

    # --- asleep: name alias + remainder -> wake ---
    action, text_out = classify("Zuri what's the weather", awake=False)
    assert (action, text_out) == ("wake", "what's the weather"), (action, text_out)

    action, text_out = classify("Zuri, what's the weather", awake=False)
    assert (action, text_out) == ("wake", "what's the weather"), (action, text_out)

    # --- asleep: alias alone -> wake, empty remainder ---
    action, text_out = classify("zoori", awake=False)
    assert (action, text_out) == ("wake", ""), (action, text_out)

    # --- asleep: each alias on its own wakes with an empty remainder ---
    for alias in NAME_ALIASES:
        assert classify(alias, awake=False) == ("wake", ""), alias
        assert classify(alias.capitalize() + "!", awake=False) == ("wake", ""), alias

    # --- asleep: alias followed by a question keeps original casing/punctuation ---
    action, text_out = classify("zury turn on the lights please", awake=False)
    assert (action, text_out) == ("wake", "turn on the lights please"), (action, text_out)

    print("classify(): PASS")


class Collector(FrameProcessor):
    """Sink: records every frame it sees (besides lifecycle frames)."""

    def __init__(self):
        super().__init__()
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


async def check_wake_gate():
    gate = WakeGate()
    collector = Collector()
    worker = PipelineWorker(
        Pipeline([gate, collector]),
        params=PipelineParams(),
        idle_timeout_secs=None,
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    async def feed():
        assert gate.awake is True
        await worker.queue_frames([_tf("What's the weather today?")])
        await worker.queue_frames([_tf("mute")])
        await worker.queue_frames([_tf("some random background noise")])
        await worker.queue_frames([_tf("Zuri, what's the weather")])
        await worker.queue_frames([_tf("stop listening")])
        await worker.queue_frames([_tf("zoori")])
        await asyncio.sleep(0.3)
        await worker.queue_frames([EndFrame()])

    asyncio.create_task(feed())
    await asyncio.wait_for(runner.run(), 30)

    transcriptions = [f.text for f in collector.frames if isinstance(f, TranscriptionFrame)]
    spoken = [f.text for f in collector.frames if isinstance(f, TTSSpeakFrame)]

    # Only the two real-content utterances reach downstream as transcripts:
    # the opening question (untouched) and the de-prefixed wake remainder.
    # "mute", the asleep-noise, "stop listening" and the bare alias never do.
    assert transcriptions == ["What's the weather today?", "what's the weather"], transcriptions

    # Each mute/wake transition speaks its fixed line, in order, and nothing else.
    assert spoken == [SLEEP_REPLY, SLEEP_REPLY, WAKE_REPLY], spoken

    # Gate ends awake (woke on the bare alias last).
    assert gate.awake is True

    print("WakeGate: PASS")


async def main():
    check_classify()
    await check_wake_gate()
    print("check_wake.py: PASS")


asyncio.run(main())
