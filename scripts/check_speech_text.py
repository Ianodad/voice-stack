"""
Speech-layer markdown stripping: the text passed to run_tts never carries markdown marks, while
code skipping, the on-screen transcript (bot-output) and the stored reply keep their old behaviour.

Runs the real MLXKokoroTTSService (stub model) + FenceAggregator in a real PipelineWorker.
Run: uv run python scripts/check_speech_text.py
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pipecat.frames.frames import EndFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIObserverParams
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

from voice_stack.bot import ReplyTap, make_assistant_turn_handler
from voice_stack.speech_text import strip_markdown
from voice_stack.tts import MLXKokoroTTSService


class FakeModel:
    def generate(self, **kw):
        class R:
            audio = np.zeros(2400, dtype=np.float32)
            sample_rate = 24000

        yield R()


class FakeOut(BaseOutputTransport):
    def __init__(self):
        super().__init__(TransportParams(audio_out_enabled=True, audio_out_sample_rate=24000))
        self.msgs = []

    async def start(self, frame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        return True

    async def send_message(self, frame):
        self.msgs.append(frame.message)


async def run(reply: str, chunk=7):
    """Stream `reply` in small chunks (so marks straddle token boundaries) through the real service."""
    tts = MLXKokoroTTSService(executor=ThreadPoolExecutor(max_workers=1), model=FakeModel())
    spoken = []
    orig = tts.run_tts

    async def spy(text, cid):
        spoken.append(text)
        async for f in orig(text, cid):
            yield f

    tts.run_tts = spy
    ctx = LLMContext()
    _, asst = LLMContextAggregatorPair(ctx)
    tap, turns = ReplyTap(), []
    asst.event_handler("on_assistant_turn_stopped")(
        make_assistant_turn_handler(tap, ctx, lambda role, c: turns.append((role, c))))
    out = FakeOut()
    worker = PipelineWorker(
        Pipeline([tap, tts, out, asst]), params=PipelineParams(audio_out_sample_rate=24000),
        idle_timeout_secs=None, rtvi_observer_params=RTVIObserverParams(skip_aggregator_types=["cue"]))
    worker.rtvi._client_version = [2, 1, 0]
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    async def feed():
        await worker.queue_frames([LLMFullResponseStartFrame()])
        for i in range(0, len(reply), chunk):
            await worker.queue_frames([LLMTextFrame(reply[i:i + chunk])])
        await worker.queue_frames([LLMFullResponseEndFrame()])
        await asyncio.sleep(1.2)
        await worker.queue_frames([EndFrame()])

    asyncio.create_task(feed())
    await asyncio.wait_for(runner.run(), 30)
    shown = [m["data"]["text"] for m in out.msgs if m.get("type") == "bot-output"]
    return spoken, shown, ctx, turns


def joined(spoken):
    return " ".join(spoken)


async def main():
    # ---- pure function ----
    cases = {
        "**Nation.": "Nation.",
        "Africa** - Offers news.": "Africa - Offers news.",
        "See `e2e-config.txt` and my_file_name.txt": "See e2e-config.txt and my_file_name.txt",
        "[link text](http://x)": "link text",
        "# Title": "Title",
        "Costs 3*4 dollars": "Costs 3*4 dollars",
        "2 * 3 = 6": "2 * 3 = 6",
        "It is 3.5 percent, really! \U0001F600": "It is 3.5 percent, really! \U0001F600",
        "e2e_config and x_1 and __init__ ok": "e2e_config and x_1 and __init__ ok",
        "_gently_ done": "gently done",
        "- one": "one",
        "- ": "",
        "5 * 6 * 7": "5 * 6 * 7",
        "*hi* **there** ~~you~~": "hi there you",
        # meanings of numbers/comparisons survive; markers before letters still go
        "> 5 servers": "> 5 servers", "- 5 degrees": "- 5 degrees", "3 *4": "3 *4",
        "3 *4 and 5* x": "3 *4 and 5* x", "> Note this": "Note this", "* item": "item",
        "- **Nation. Africa** x": "Nation. Africa x", "1. **Bold** y": "1. Bold y",
    }
    for src, want in cases.items():
        got = strip_markdown(src)
        assert got == want, (src, got, want)
    import time
    for big in ("*a " * 60000, "**a _b " * 20000, "[x " * 60000, "`" + "y " * 90000):
        t0 = time.time(); out = strip_markdown(big); dt = time.time() - t0
        assert dt < 1.0 and "`" not in out and "**" not in out, (big[:12], dt)   # capped: no quadratic blow-up
    print("strip_markdown unit cases ok")

    # ---- real service: what run_tts actually receives ----
    reply = "**Nation.** Here are headlines:\n- one\n- two\n1. three"
    spoken, shown, ctx, turns = await run(reply)
    s = joined(spoken)
    assert "*" not in s and "- " not in s, spoken
    assert "Nation." in s and "one" in s and "two" in s and "three" in s, spoken
    assert turns and "**" in turns[-1][1] and "- one" in turns[-1][1], turns   # stored reply keeps the markdown
    assert any("**" in t for t in shown), shown                       # so does the on-screen transcript
    print("markdown reply: spoken =", spoken)

    spoken, shown, _, turns = await run("See `e2e-config.txt` and my_file_name.txt for details.")
    s = joined(spoken)
    assert "`" not in s and "e2e-config.txt" in s and "my_file_name.txt" in s, spoken
    assert "`e2e-config.txt`" in turns[-1][1], turns
    assert any("`e2e-config.txt`" in t for t in shown), shown        # on-screen transcript keeps markdown text
    print("backticks: spoken =", spoken)

    spoken, *_ = await run("Read [the docs](http://example.com/x) now.")
    assert "http" not in joined(spoken) and "the docs" in joined(spoken), spoken
    spoken, *_ = await run("# Title\nBody text here.")
    assert "#" not in joined(spoken) and "Title" in joined(spoken), spoken
    spoken, *_ = await run("Costs 3*4 dollars, and 2 * 3 = 6 is true. Pi is 3.5. \U0001F600")
    assert "3*4" in joined(spoken) and "2 * 3 = 6" in joined(spoken) and "3.5" in joined(spoken) \
        and "\U0001F600" in joined(spoken), spoken
    spoken, *_ = await run("The file is e2e_config and my_file_name.txt here.")
    assert "e2e_config" in joined(spoken) and "my_file_name.txt" in joined(spoken), spoken
    print("links / headers / arithmetic / emoji / underscores ok")

    # bold straddling a sentence split, streamed in tiny chunks
    spoken, *_ = await run("Top story: **Nation. Africa** reports a result today.", chunk=3)
    assert "*" not in joined(spoken) and "Nation." in joined(spoken) and "Africa" in joined(spoken), spoken
    print("bold across sentence split: spoken =", spoken)

    # code fence: code is still never spoken (skipped before filters), and not mangled in the stored reply
    code = "def add(a, b):\n    return a * b  # **x** `y`\n"
    reply = f"Here is a function.\n```python\n{code}```\nIt multiplies **two** numbers."
    spoken, shown, _, turns = await run(reply)
    s = joined(spoken)
    assert "def add" not in s and "return a" not in s and "#" not in s, spoken
    assert "Here is a function." in s and "multiplies two numbers" in s, spoken
    assert code in turns[-1][1], turns                                # stored reply keeps the exact code
    assert any(code.strip() in t for t in shown), shown               # code box text reaches the client intact
    print("code fence: spoken =", spoken)
    print("PASS")


asyncio.run(main())
