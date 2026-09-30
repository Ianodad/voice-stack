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
    InterruptionFrame,
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
from voice_stack.fence import CODE_CUE, FenceAggregator
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


INTERRUPT = object()  # in a parts list: interrupt the bot, then a new reply starts


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

    async def feed():
        await worker.queue_frames([LLMFullResponseStartFrame()])
        for p in parts:
            if p is INTERRUPT:
                await asyncio.sleep(0.4)
                await worker.queue_frames([InterruptionFrame(), LLMFullResponseStartFrame()])
                await asyncio.sleep(0.2)
            else:
                await worker.queue_frames([LLMTextFrame(p)])
        await worker.queue_frames([LLMFullResponseEndFrame()])
        await asyncio.sleep(1.5)
        await worker.queue_frames([EndFrame()])

    asyncio.create_task(feed())
    await asyncio.wait_for(runner.run(), 30)
    outputs = [m["data"] for m in out.msgs if m.get("type") == "bot-output"]
    return spoken, outputs, ctx, turns


def codes(outputs):
    return [d["text"] for d in outputs if str(d["aggregated_by"]) == "code"]


def prose(outputs):
    return [d["text"] for d in outputs if str(d["aggregated_by"]) == "sentence" and not d["spoken"]]


async def agg_run(chunks, flush=True):
    """Drive the aggregator alone: [(type, text)] of everything it yields."""
    agg = FenceAggregator(cue=CODE_CUE)
    got = []
    for c in chunks:
        async for a in agg.aggregate(c):
            got.append((str(a.type), a.text))
    if flush:
        a = await agg.flush()
        if a:
            got.append((str(a.type), a.text))
    return got


def only(got, kind):
    return [t for k, t in got if k == kind]


def assert_no_code_spoken(got, *needles):
    spoken = " ".join(only(got, "sentence"))
    for n in needles:
        assert n not in spoken, (n, got)


async def main():
    # 1. basic, fence split across chunks
    parts = ["Here is how.\n", "```py", "thon\nimport x\n", "x.run()\n", "```", "\nThat imports x. ", "It then runs it."]
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
    full = "First one.\n```js\nlet a=1;\n```\nThen second.\n```sh\nls -la\n```\nDone now."
    spoken, outputs, ctx, turns = await run([full])
    assert codes(outputs) == ["```js\nlet a=1;\n```", "```sh\nls -la\n```"], codes(outputs)
    assert spoken == ["First one.", CODE_CUE, "Then second.", CODE_CUE, "Done now."], spoken
    assert ctx.messages[-1]["content"] == full and turns == [("assistant", full)]
    print("two blocks ok")

    # 3. unterminated fence is still shown as code
    spoken, outputs, ctx, turns = await run(["Look at this.\n```python\nimport x\nprint(", "x)"])
    assert codes(outputs) == ["```python\nimport x\nprint(x)"], codes(outputs)
    assert spoken == ["Look at this."], spoken
    assert turns[0][1] == "Look at this.\n```python\nimport x\nprint(x)", turns
    print("unterminated ok")

    # 4. inline backticks, indented text, and mid-line ``` stay spoken prose
    spoken, outputs, ctx, turns = await run(
        ["Use the `print` function. Then call `x.run()` okay. ", "    indented line is prose. ",
         "Also ```triple``` inline is prose. Done."]
    )
    assert codes(outputs) == [], codes(outputs)
    joined = " ".join(spoken)
    assert "`print`" in joined and "`x.run()`" in joined and "indented line is prose" in joined, spoken
    assert "```triple```" in joined and "Done." in joined, spoken
    assert CODE_CUE not in spoken
    print("inline/indented ok")

    # 5. char-by-char streaming
    full = "Hi there.\n```py\na = 1\n```\nBye now."
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

    # 8. tilde fence (the original failure mode: code was spoken)
    for chunks in (["Hi.\n~~~python\nx=1\n~~~\nDone."], list("Hi.\n~~~python\nx=1\n~~~\nDone.")):
        got = await agg_run(chunks)
        assert only(got, "code") == ["~~~python\nx=1\n~~~"], got
        assert_no_code_spoken(got, "x=1", "~~~")
        assert only(got, "sentence") == ["Hi.", "Done."], got
    spoken, outputs, ctx, turns = await run(["Hi.\n~~~python\nx=1\n~~~\nDone."])
    assert codes(outputs) == ["~~~python\nx=1\n~~~"] and "x=1" not in " ".join(spoken), (spoken, outputs)
    print("tilde ok")

    # 9. 4+ backtick fence; shorter/other-char runs inside do not close it
    got = await agg_run(["Hi.\n````python\nx=1\n```\ny=2\n~~~~\n````\nDone."])
    assert only(got, "code") == ["````python\nx=1\n```\ny=2\n~~~~\n````"], got
    assert only(got, "sentence") == ["Hi.", "Done."], got
    got = await agg_run(list("Hi.\n````python\nx=1\n````\nDone."))
    assert only(got, "code") == ["````python\nx=1\n````"] and only(got, "sentence") == ["Hi.", "Done."], got
    print("long fence ok")

    # 10. odd/embedded fences do not break state for the rest of the reply
    py = "Here.\n```python\ns = \'\'\' ``` \'\'\'\nprint(s)\n```\nThat prints a string. Done."
    got = await agg_run([py])
    assert only(got, "code") == ["```python\ns = \'\'\' ``` \'\'\'\nprint(s)\n```"], got
    assert only(got, "sentence") == ["Here.", "That prints a string.", "Done."], got
    assert_no_code_spoken(got, "print(s)", "```")
    heredoc = "Run it.\n```bash\ncat <<EOF\n  ```\necho \"``` text\"\nEOF\n```\nThat writes a file. Done."
    for chunks in ([heredoc], list(heredoc)):
        got = await agg_run(chunks)
        assert len(only(got, "code")) == 1 and "EOF" in only(got, "code")[0], got
        assert only(got, "sentence") == ["Run it.", "That writes a file.", "Done."], got
    spoken, outputs, ctx, turns = await run([py])
    assert spoken[:2] == ["Here.", CODE_CUE] and " ".join(spoken[2:]) == "That prints a string. Done.", spoken
    assert codes(outputs) == ["```python\ns = \'\'\' ``` \'\'\'\nprint(s)\n```"], codes(outputs)
    # a fence opener mid-line is prose, not code
    got = await agg_run(["Say ```python then stop. Fine."])
    assert only(got, "code") == [] and "```python" in " ".join(only(got, "sentence")), got
    print("embedded fences ok")

    # 11. CRLF: no \r in code text
    got = await agg_run(["Hi.\r\n```py\r\nx=1\r\n```\r\nDone."])
    assert only(got, "code") == ["```py\nx=1\n```"], got
    print("crlf ok")

    # 12. interruption: state is cleared, the next reply's prose is spoken
    spoken, outputs, ctx, turns = await run(["Look.\n```py\nx = 1\n", INTERRUPT, "Next reply is prose. Fine."])
    assert "x = 1" not in " ".join(spoken), spoken
    assert "Next reply is prose. Fine." in " ".join(spoken) and codes(outputs) == [], (spoken, outputs)
    print("interrupt mid-code ok")
    spoken, outputs, ctx, turns = await run(
        ["Look.\n```py\nx = 1\n```\nFirst step, ", INTERRUPT, "Next reply is prose. Fine."]
    )
    assert "Next reply is prose. Fine." in " ".join(spoken) and len(codes(outputs)) == 1, (spoken, outputs)
    assert "x = 1" not in " ".join(spoken), spoken
    print("interrupt after code ok")
    spoken, outputs, ctx, turns = await run(
        ["Look.\n```py\nx = 1\n```\nFirst step. Second ", INTERRUPT, "Next reply is prose. Fine."]
    )
    assert "Next reply is prose. Fine." in " ".join(spoken) and len(codes(outputs)) == 1, (spoken, outputs)
    print("interrupt mid-explanation ok")
    # aggregator-level: interruption while a closing run or fence run is pending
    for tail in ("```", "``", "x\n``"):
        agg = FenceAggregator(cue=CODE_CUE)
        for c in f"Hi.\n```py\nx\n{tail}" if tail != "x\n``" else "Hi.\n```py\nx\n``":
            async for _ in agg.aggregate(c):
                pass
        await agg.handle_interruption()
        got = [a async for a in agg.aggregate("Fresh prose here. More.")]
        assert [a.text for a in got] == ["Fresh prose here."], got
    print("interrupt state reset ok")
    print("PASS")


asyncio.run(main())
