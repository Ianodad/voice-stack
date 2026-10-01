"""Speech-layer markdown stripping: what the TTS speaks must never contain markdown marks.

Applied by TTSService.text_transforms to each aggregated chunk (a sentence) AFTER the fence
aggregator has split off code (code chunks are skipped before transforms run) and only to the text
sent to run_tts. The on-screen transcript and the LLM context keep the original text.

Stateless per chunk on purpose: a bold span can straddle a sentence split ("**Nation. Africa**"),
so stray marks are removed even when their partner is in another chunk. Underscores (e2e_config, my_file_name, __init__) and
arithmetic stars (3*4, 2 * 3) are left alone.
"""
import re

_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)\s]*(?:\s+\"[^\"]*\")?\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)\s]*(?:\s+\"[^\"]*\")?\)")
_FENCE = re.compile(r"^[ \t]*`{3,}[\w+-]*[ \t]*$", re.M)   # a fence line only; ```x``` inline keeps x
_INLINE_CODE = re.compile(r"`([^`\n]*)`")
_HEADER = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.*?)(?:[ \t]+#+[ \t]*)?$", re.M)
_RULE = re.compile(r"^[ \t]*([-*_=])(?:[ \t]*\1){2,}[ \t]*$", re.M)
_QUOTE = re.compile(r"^[ \t]*>+[ \t]?", re.M)
_BULLET = re.compile(r"^[ \t]*[-*+•●◦][ \t]+", re.M)
_TABLE_SEP = re.compile(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$", re.M)
_TABLE_ROW = re.compile(r"^[ \t]*\|(.*)\|[ \t]*$", re.M)
_BOLD = re.compile(r"(?<![\w*])\*\*(?=[^\s*])(.+?)(?<=[^\s*])\*\*(?![\w*])", re.S)  # __x__ left alone: __init__
_ITAL_STAR = re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])", re.S)
_ITAL_UND = re.compile(r"(?<![\w_])_(?=[^\s_])(.+?)(?<=[^\s_])_(?![\w_])", re.S)
_STRIKE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~", re.S)
_STRAY_STAR2 = re.compile(r"\*{2,}")
# a lone star glued to a word on one side only: an emphasis mark whose partner is in another chunk
_STRAY_STAR = re.compile(r"(?<![\w*])\*(?=\w)|(?<=\w)\*(?![\w*])")
_BLANKS = re.compile(r"\n{2,}")
_ONLY_MARKS = re.compile(r"^[\s\-*_#>|~`=+.:•]*$")


def strip_markdown(text: str) -> str:
    t = _IMAGE.sub(r"\1", text)
    t = _LINK.sub(r"\1", t)
    t = _FENCE.sub("", t)
    t = _INLINE_CODE.sub(r"\1", t)
    t = t.replace("`", "")
    t = _HEADER.sub(r"\1", t)
    t = _RULE.sub("", t)
    t = _QUOTE.sub("", t)
    t = _BULLET.sub("", t)
    t = _TABLE_SEP.sub("", t)
    t = _TABLE_ROW.sub(lambda m: ", ".join(c.strip() for c in m.group(1).split("|") if c.strip()), t)
    t = _BOLD.sub(r"\1", t)
    t = _STRIKE.sub(r"\1", t)
    t = _ITAL_STAR.sub(r"\1", t)
    t = _ITAL_UND.sub(r"\1", t)
    t = _STRAY_STAR2.sub("", t)
    t = _STRAY_STAR.sub("", t)
    t = _BLANKS.sub("\n", t)
    t = "\n".join(line.strip() for line in t.split("\n")).strip()
    return "" if _ONLY_MARKS.match(t) else t


async def speech_transform(text: str, aggregation_type: str) -> str:
    """Pipecat TTS text transform (register as ("*", speech_transform)): strips markdown from the
    text sent to run_tts ONLY. Unlike text_filters, transforms leave the text that reaches the
    transcript and the LLM context untouched."""
    return strip_markdown(text)
