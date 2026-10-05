"""Voice-controlled sleep/wake gate for the mic, enforced in CODE.

The model is never asked to honor "mute yourself" -- and can never falsely
claim it did, since it never gets a say. `classify()` is a pure text
classifier; `WakeGate` is the pipeline processor that applies it to the live
transcript stream.

Placement (see bot.build_worker): between `stt` and `user_agg`. WakeGate
looks at TranscriptionFrame only -- every other frame (raw audio, VAD
start/stop, control frames) passes through completely unchanged, in
whichever direction it arrives. This matters: VAD detection and the turn
analyzer both live inside `user_agg` (LLMUserAggregator) and work from raw
`InputAudioRawFrame`s plus the `VADUserStartedSpeakingFrame` /
`VADUserStoppedSpeakingFrame` pair it then broadcasts back *upstream* to
`stt` -- `ParakeetSTTService` is a `SegmentedSTTService`, which only
buffers/flushes audio on those VAD frames. If WakeGate touched anything but
TranscriptionFrame, STT's own segmentation (and therefore its ability to ever
hear the wake word again) would break while "asleep". So WakeGate must stay
narrow, exactly as specified.

Wedge investigation (per spec): when WakeGate swallows a TranscriptionFrame
(sleep / drop / empty-remainder wake), `user_agg`'s internal
`UserTurnController` never sees it. Read against the installed pipecat
source (`turns/user_turn_controller.py`,
`turns/user_stop/turn_analyzer_user_turn_stop_strategy.py`,
`processors/aggregators/llm_response_universal.py`):

- The turn is NOT stuck forever. `UserTurnController` runs its own
  `user_turn_stop_timeout` watchdog (default 5s) independent of any stop
  strategy; it force-closes an open turn once `user_turn_stop_timeout`
  seconds pass with no VAD/transcription activity, regardless of whether a
  transcript ever arrived. A swallowed turn always closes within that bound.
- `TurnAnalyzerUserTurnStopStrategy` re-arms directly off VAD frames (its own
  `process_frame`, not just the controller's start/stop event bubble), so a
  real follow-up utterance right after a swallowed one is still segmented
  and transcribed correctly -- content never leaks or merges across turns.
- Narrower edge case (documented, not fixed here): the controller's
  `_user_turn` flag itself is only cleared by `_trigger_user_turn_stop`, so
  if a second utterance's VAD-start lands before the stale turn's watchdog
  fires, the controller's own `on_user_turn_started` event for that second
  utterance is suppressed (it thinks the turn is still open). In tools mode
  this means `ToolSession.reset_turn()` may not fire for that one turn,
  which could let a per-turn flag (e.g. `proposed_turn`) survive one extra
  turn. This needs a pipecat-level turn API change to close cleanly (there
  is no public "force-finalize" hook); flagged for Opus rather than patched
  here, since it would mean reaching into pipecat's private turn internals
  from this file. In practice it is bounded (never longer than
  `user_turn_stop_timeout`, 5s) and requires two utterances in quick
  succession across a swallow, so it is a narrow risk rather than a hang.
"""

import re

from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# STT mishearings of "Zuri". "zuri's" is explicitly not needed (no possessive
# variant).
NAME_ALIASES = {"zuri", "zoori", "zury", "zoory", "zuree"}

# Whole-utterance mute commands only (see classify()).
MUTE_PHRASES = {"mute", "mute yourself", "stop listening", "go to sleep"}

SLEEP_REPLY = "Okay, I'm muted. Say Zuri to wake me."
WAKE_REPLY = "I'm here."

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")
# First run of letters in the raw text, then any immediately-following
# non-alphanumeric run (punctuation/whitespace separator), then whatever is
# left -- used only in the asleep branch, where we need to slice the
# *original* text (to preserve casing) rather than the normalized one.
_LEADING_WORD_RE = re.compile(r"^\s*([A-Za-z]+)([^A-Za-z0-9]*)(.*)$", re.DOTALL)


def _normalize(text: str) -> str:
    """lowercase, strip punctuation, collapse spaces."""
    t = _PUNCT_RE.sub("", text.lower())
    return _SPACE_RE.sub(" ", t).strip()


def classify(text: str, awake: bool) -> tuple[str, str]:
    """Classify one transcript against the wake/sleep gate.

    Returns (action, text_out):
      - "pass":  forward the ORIGINAL text unchanged (awake, not a mute command).
      - "sleep": go to sleep; text_out is "" (caller speaks the fixed reply).
      - "wake":  wake up; text_out is the remainder after the name alias
                 (original casing, leading punctuation stripped), or "" if
                 nothing followed the name.
      - "drop":  swallow silently; text_out is "".
    """
    if awake:
        norm = _normalize(text)
        words = norm.split()
        if words and words[0] in NAME_ALIASES:
            words = words[1:]
        if " ".join(words) in MUTE_PHRASES:
            return "sleep", ""
        return "pass", text

    # Asleep: only a name-alias-led utterance does anything.
    m = _LEADING_WORD_RE.match(text)
    if m and m.group(1).lower() in NAME_ALIASES:
        remainder = m.group(3).lstrip()
        return "wake", remainder
    return "drop", ""


class WakeGate(FrameProcessor):
    """Code-enforced sleep/wake gate. See module docstring for placement and
    the wedge investigation.

    Holds `awake` (starts True). Only TranscriptionFrame is inspected; every
    other frame is passed through unchanged, in whichever direction it
    arrives -- no-op-safe by default.
    """

    def __init__(self):
        super().__init__()
        self.awake = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, TranscriptionFrame):
            await self.push_frame(frame, direction)
            return

        action, text_out = classify(frame.text, self.awake)

        if action == "pass":
            await self.push_frame(frame, direction)
            return

        if action == "sleep":
            self.awake = False
            print(f"[wake] asleep (said: {frame.text!r})")
            await self.push_frame(TTSSpeakFrame(SLEEP_REPLY), FrameDirection.DOWNSTREAM)
            return

        if action == "wake":
            self.awake = True
            print(f"[wake] awake (said: {frame.text!r})")
            if text_out:
                frame.text = text_out
                await self.push_frame(frame, direction)
            else:
                await self.push_frame(TTSSpeakFrame(WAKE_REPLY), FrameDirection.DOWNSTREAM)
            return

        if action == "drop":
            return

        logger.warning(f"{self}: unknown wake-gate action {action!r}; dropping")
