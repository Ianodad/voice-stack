"""Fence-aware text aggregator for TTS.

Splits streamed LLM text into spoken prose sentences and fenced ``` code
blocks. Code blocks come out as one PatternMatch of type "code" whose text is
the raw fence (```lang ... ```); TTS skips that type (skip_aggregator_types),
the client renders it in a code box. An optional cue sentence (type "cue") is
spoken after each block so the user knows the code is on screen.

Fence rules (mirrored by parseFences in web/src/transcript.ts):
- an opening fence is a run of 3+ backticks or tildes. At the start of a line
  the rest of the line is the info string (language). Mid-line it counts only
  when the run is followed by an optional short info string (letters, digits,
  + # - _ . /, max 20 chars) and then a newline; the text before it on that
  line is ordinary prose. Any other mid-line run (``` in prose, in a string)
  stays prose;
- it closes on a run of the SAME character, at least as long, at the start of
  a line and followed by end of line, end of text, or blanks and then more
  text (that text is prose). A run directly followed by letters (a nested
  "```bash") is not a closer. Fence marks in indented lines never toggle;
- carriage returns are dropped.
"""

from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.pattern_pair_aggregator import PatternMatch
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

CODE_CUE = "I've put the code on screen."
LOCAL_CODE_CUE = "I've printed the code in the terminal."
CUES = (CODE_CUE, LOCAL_CODE_CUE)
_FENCE_CHARS = "`~"
_INFO_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+#-_./"
)
_INFO_MAX = 20


def print_code_block(raw: str) -> None:
    """Terminal display for a raw fence: delimited, language label, the code."""
    lines = raw.split("\n")
    first = lines[0].lstrip(_FENCE_CHARS)
    lang = first.strip()
    body = lines[1:]
    if body and body[-1].strip() and set(body[-1].strip()) <= set(_FENCE_CHARS):
        body = body[:-1]
    elif not lines[1:]:  # single line: ```code``` with no newline
        body = []
    bar = "-" * 40
    print(f"\n{bar}\ncode ({lang or 'text'})\n" + "\n".join(body) + f"\n{bar}", flush=True)


class FenceAggregator(SimpleTextAggregator):
    """Sentence aggregator that yields fenced blocks as one PatternMatch(type='code')
    whose text is the raw fence. A fence may be split across streamed chunks
    (even char by char). An unterminated fence at flush is still yielded as
    code. Everything resets on interruption."""

    def __init__(self, cue: str | None = None, on_code=None, **kw):
        super().__init__(**kw)
        self._cue = cue
        self._on_code = on_code  # called with the raw fence of each finished code block
        self._reset_fence_state()

    def _reset_fence_state(self):
        self._code: str | None = None  # None = prose mode, else text after the opening run
        self._open = ""  # opening run, e.g. "```" or "~~~~"
        self._bol = True  # prose mode: next char starts a line
        self._run = ""  # prose mode: pending fence-char run
        self._run_bol = True  # ... and whether it started a line
        self._info = ""  # prose mode: mid-line run's info string so far
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

    async def _open_fence(self, run: str, first: str):
        pre = self._text
        await SimpleTextAggregator.reset(self)
        self._open, self._code = run, first
        self._run = self._info = ""
        if pre.strip():
            yield Aggregation(text=pre.strip(), type=AggregationType.SENTENCE)

    async def _prose_char(self, ch: str):
        if self._run:
            if ch == self._run[0] and not self._info:
                self._run += ch
                return
            if len(self._run) >= 3:
                if self._run_bol:  # fence opens; ch is the first char of the info line
                    async for agg in self._open_fence(self._run, ch):
                        yield agg
                    return
                if ch == "\n":  # mid-line opener: code starts on the next line
                    async for agg in self._open_fence(self._run, self._info + "\n"):
                        yield agg
                    return
                if ch in _INFO_CHARS and len(self._info) < _INFO_MAX:
                    self._info += ch
                    return
            text, self._run, self._info = self._run + self._info, "", ""
            for c in text:  # not a fence: it was ordinary prose
                async for agg in self._plain(c):
                    yield agg
        if ch in _FENCE_CHARS:
            self._run, self._run_bol, self._bol = ch, self._bol, False
            return
        self._bol = ch == "\n"
        async for agg in self._plain(ch):
            yield agg

    def _close_len(self) -> int:
        return len(self._cpend.rstrip(" \t"))

    def _finish(self, closer: str) -> PatternMatch:
        raw = f"{self._open}{self._code}{closer}"
        self._reset_fence_state()
        if self._on_code is not None:
            try:
                self._on_code(raw)
            except Exception:
                pass  # a display hook must never break speech
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
            if run_len >= len(self._open) and (ch == "\n" or blanks):
                yield self._finish(self._cpend[:run_len])
                if self._cue:
                    yield Aggregation(text=self._cue, type="cue")
                if ch != "\n":  # text after the closing fence is prose
                    self._bol = False
                    async for agg in self._prose_char(ch):
                        yield agg
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
            run, info, bol = self._run, self._info, self._run_bol
            self._run = self._info = ""
            if len(run) >= 3 and bol and not self._text.strip():
                self._open, self._code = run, ""
            elif len(run) >= 3 and bol:
                pass  # bare fence after unspoken prose: speak the prose, drop the empty fence
            else:
                self._text += run + info
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
