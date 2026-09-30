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

CODE_CUE = "I've put the code on screen."
_FENCE_CHARS = "`~"


class FenceAggregator(SimpleTextAggregator):
    """Sentence aggregator that yields fenced blocks as one PatternMatch(type='code')
    whose text is the raw fence. A fence may be split across streamed chunks
    (even char by char). An unterminated fence at flush is still yielded as
    code. Everything resets on interruption."""

    def __init__(self, cue: str | None = None, **kw):
        super().__init__(**kw)
        self._cue = cue
        self._reset_fence_state()

    def _reset_fence_state(self):
        self._code: str | None = None  # None = prose mode, else text after the opening run
        self._open = ""  # opening run, e.g. "```" or "~~~~"
        self._bol = True  # prose mode: next char starts a line
        self._run = ""  # prose mode: pending fence-char run at line start
        self._cpend = ""  # code mode: pending closing-run candidate (+ trailing blanks)

    async def aggregate(self, text: str):
        for ch in text:
            if ch == "\r":
                continue
            gen = self._code_char(ch) if self._code is not None else self._prose_char(ch)
            async for agg in gen:
                yield agg

    async def _plain(self, ch: str):
        self._text += ch
        agg = await self._check_sentence_with_lookahead(ch)
        if agg:
            yield agg

    async def _prose_char(self, ch: str):
        if self._run:
            if ch == self._run[0]:
                self._run += ch
                return
            run, self._run = self._run, ""
            if len(run) >= 3:  # fence opens; ch is the first char of the info line
                pre = self._text
                await SimpleTextAggregator.reset(self)
                self._open, self._code = run, ch
                if pre.strip():
                    yield Aggregation(text=pre.strip(), type=AggregationType.SENTENCE)
                return
            for c in run:  # not a fence: it was ordinary prose
                async for agg in self._plain(c):
                    yield agg
        if ch in _FENCE_CHARS and self._bol:
            self._run = ch
            self._bol = False
            return
        self._bol = ch == "\n"
        async for agg in self._plain(ch):
            yield agg

    def _close_len(self) -> int:
        return len(self._cpend.rstrip(" \t"))

    def _finish(self, closer: str) -> PatternMatch:
        raw = f"{self._open}{self._code}{closer}"
        self._reset_fence_state()
        return PatternMatch(content=raw, type="code", full_match=raw)

    async def _code_char(self, ch: str):
        if self._cpend:
            run_len = self._close_len()
            blanks = len(self._cpend) > run_len
            if ch == self._cpend[0] and not blanks:
                self._cpend += ch
                return
            if ch in " \t" and run_len >= len(self._open):
                self._cpend += ch
                return
            if ch == "\n" and run_len >= len(self._open):
                yield self._finish(self._cpend[:run_len])
                if self._cue:
                    yield Aggregation(text=self._cue, type="cue")
                return
            self._code += self._cpend + ch  # not a closing fence after all
            self._cpend = ""
            return
        if ch == self._open[0] and self._code.endswith("\n"):
            self._cpend = ch
            return
        self._code += ch

    async def flush(self):
        if self._run:  # pending run at end of text
            run, self._run = self._run, ""
            if len(run) >= 3:
                self._open, self._code = run, ""
            else:
                self._text += run
        if self._code is not None:
            if self._cpend and self._close_len() >= len(self._open):
                return self._finish(self._cpend.rstrip(" \t"))  # closer at end of text
            self._code += self._cpend
            self._cpend = ""
            return self._finish("")  # unterminated fence: keep it as code
        self._bol = True
        return await super().flush()

    async def handle_interruption(self):
        self._reset_fence_state()
        await super().handle_interruption()

    async def reset(self):
        self._reset_fence_state()
        await super().reset()
