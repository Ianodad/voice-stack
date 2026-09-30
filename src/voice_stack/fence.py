"""Fence-aware text aggregator for TTS.

Splits streamed LLM text into spoken prose sentences and fenced ``` code
blocks. Code blocks come out as one PatternMatch of type "code" whose text is
the raw fence (```lang ... ```); TTS skips that type (skip_aggregator_types),
the client renders it in a code box. An optional cue sentence (type "cue") is
spoken after each block so the user knows the code is on screen.
"""

from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.pattern_pair_aggregator import PatternMatch
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

FENCE = "```"
CODE_CUE = "I've put the code on screen."


class FenceAggregator(SimpleTextAggregator):
    """Sentence aggregator that yields ``` fenced blocks as one PatternMatch(type='code').
    A fence may be split across streamed chunks (even char by char). Inline
    `code` and indented text stay prose. An unterminated fence at flush is
    still yielded as code. Reset on interruption."""

    def __init__(self, cue: str | None = None, **kw):
        super().__init__(**kw)
        self._cue = cue
        self._code: str | None = None  # None = prose mode

    async def aggregate(self, text: str):
        for ch in text:
            if self._code is not None:  # inside a fence
                self._code += ch
                if self._code.endswith(FENCE):
                    body = self._code[: -len(FENCE)]
                    self._code = None
                    raw = f"{FENCE}{body}{FENCE}"
                    yield PatternMatch(content=raw, type="code", full_match=raw)
                    if self._cue:
                        yield Aggregation(text=self._cue, type="cue")
                continue
            self._text += ch
            if self._text.endswith(FENCE):  # fence opens
                pre = self._text[: -len(FENCE)]
                await super().reset()
                self._code = ""
                if pre.strip():
                    yield Aggregation(text=pre.strip(), type=AggregationType.SENTENCE)
                continue
            agg = await self._check_sentence_with_lookahead(ch)
            if agg:
                yield agg

    async def flush(self):
        if self._code is not None:  # unterminated fence: keep it as code
            body, self._code = self._code, None
            await super().reset()
            raw = f"{FENCE}{body}"
            return PatternMatch(content=raw, type="code", full_match=raw)
        return await super().flush()

    async def handle_interruption(self):
        self._code = None
        await super().handle_interruption()

    async def reset(self):
        self._code = None
        await super().reset()
