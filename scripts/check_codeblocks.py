"""
B1 check: fenced code is shown (code bot-output), never spoken; the stored
reply contains the exact code in its true position.

Runs the real MLXKokoroTTSService (stub model), real FenceAggregator, real
ReplyTap + assistant-turn handler from voice_stack.bot, through a real
PipelineWorker with an RTVI observer, and collects TTS input / bot-output.

Run: uv run python scripts/check_codeblocks.py
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pipecat.frames.frames import (
    EndFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIObserverParams
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

from voice_stack.bot import ReplyTap, make_assistant_turn_handler
from voice_stack.fence import CODE_CUE
from voice_stack.runtime import CODE_RULES, SYSTEM_PROMPT
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


async def run(parts):
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
    tap = ReplyTap()
    turns = []
    asst.event_handler("on_assistant_turn_stopped")(
        make_assistant_turn_handler(tap, ctx, lambda role, c: turns.append((role, c)))
    )
    out = FakeOut()
    worker = PipelineWorker(
        Pipeline([tap, tts, out, asst]),
        params=PipelineParams(audio_out_sample_rate=24000),
        idle_timeout_secs=None,
        rtvi_observer_params=RTVIObserverParams(skip_aggregator_types=["cue"]),
    )
    worker.rtvi._client_version = [2, 1, 0]  # pretend a v2 client finished the handshake
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await worker.queue_frames(
        [LLMFullResponseStartFrame()]
        + [LLMTextFrame(p) for p in parts]
        + [LLMFullResponseEndFrame()]
    )

    async def stopper():
        await asyncio.sleep(2)
        await worker.queue_frames([EndFrame()])

    asyncio.create_task(stopper())
    await asyncio.wait_for(runner.run(), 30)
    outputs = [m["data"] for m in out.msgs if m.get("type") == "bot-output"]
    return spoken, outputs, ctx, turns


def codes(outputs):
    return [d["text"] for d in outputs if str(d["aggregated_by"]) == "code"]


def prose(outputs):
    return [d["text"] for d in outputs if str(d["aggregated_by"]) == "sentence" and not d["spoken"]]


async def main():
    # 1. basic, fence split across chunks
    parts = ["Here is how. ", "```py", "thon\nimport x\n", "x.run()\n", "```", " That imports x. ", "It then runs it."]
    full = "".join(parts)
    spoken, outputs, ctx, turns = await run(parts)
    assert spoken == ["Here is how.", CODE_CUE, "That imports x. It then runs it."], spoken
    assert not any("import x\n" in s or "x.run()" in s or "```" in s for s in spoken), spoken
    assert codes(outputs) == ["```python\nimport x\nx.run()\n```"], codes(outputs)
    assert all(CODE_CUE not in d["text"] for d in outputs), "cue must be hidden from the client"
    assert prose(outputs) == ["Here is how.", "That imports x. It then runs it."], prose(outputs)
    assert ctx.messages == [{"role": "assistant", "content": full.strip()}], ctx.messages
    assert turns == [("assistant", full.strip())], turns
    print("basic ok")

    # 2. two blocks, one chunk
    full = "First one. ```js\nlet a=1;\n``` Then second. ```sh\nls -la\n``` Done now."
    spoken, outputs, ctx, turns = await run([full])
    assert codes(outputs) == ["```js\nlet a=1;\n```", "```sh\nls -la\n```"], codes(outputs)
    assert spoken == ["First one.", CODE_CUE, "Then second.", CODE_CUE, "Done now."], spoken
    assert ctx.messages[-1]["content"] == full and turns == [("assistant", full)]
    print("two blocks ok")

    # 3. unterminated fence is still shown as code
    spoken, outputs, ctx, turns = await run(["Look at this. ```python\nimport x\nprint(", "x)"])
    assert codes(outputs) == ["```python\nimport x\nprint(x)"], codes(outputs)
    assert spoken == ["Look at this."], spoken
    assert turns[0][1] == "Look at this. ```python\nimport x\nprint(x)", turns
    print("unterminated ok")

    # 4. inline backticks and indented text stay spoken prose
    spoken, outputs, ctx, turns = await run(
        ["Use the `print` function. Then call `x.run()` okay. ", "    indented line is prose. Done."]
    )
    assert codes(outputs) == [], codes(outputs)
    joined = " ".join(spoken)
    assert "`print`" in joined and "`x.run()`" in joined and "indented line is prose" in joined, spoken
    assert CODE_CUE not in spoken
    print("inline/indented ok")

    # 5. char-by-char streaming
    full = "Hi there. ```py\na = 1\n``` Bye now."
    spoken, outputs, ctx, turns = await run(list(full))
    assert codes(outputs) == ["```py\na = 1\n```"], codes(outputs)
    assert spoken == ["Hi there.", CODE_CUE, "Bye now."], spoken
    assert turns == [("assistant", full)], turns
    print("char-by-char ok")

    # 6. plain reply: nothing rewritten, on_turn content untouched
    spoken, outputs, ctx, turns = await run(["Two plus two is four."])
    assert codes(outputs) == [] and turns == [("assistant", "Two plus two is four.")], turns
    print("plain reply ok")

    # 7. prompt carries the coding rules
    assert CODE_RULES in SYSTEM_PROMPT and "fenced block" in CODE_RULES
    print("prompt ok")
    print("PASS")


asyncio.run(main())
