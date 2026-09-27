"""
Custom Pipecat STT service: parakeet-mlx.

No built-in local/MLX Parakeet STT service ships with pipecat-ai (see
.planning/pipecat-research.md §3), so this subclasses SegmentedSTTService --
the same base WhisperSTTServiceMLX uses -- and adapts it for parakeet-mlx's
BaseParakeet.transcribe(), which takes a real file path on disk (it shells
out to ffmpeg) rather than an in-memory array like mlx_whisper.transcribe().

STT and TTS share one ThreadPoolExecutor(max_workers=1) so all MLX/Metal
work for both stays on a single thread (see PLAN.md "Key design calls").
The model is loaded on that same executor, not the caller's thread.
"""

import asyncio
import tempfile
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from parakeet_mlx import from_pretrained
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601


class ParakeetSTTService(SegmentedSTTService):
    """Local STT via parakeet-mlx, run on a shared MLX executor thread.

    wants_wav_segments stays at its default (True): parakeet-mlx needs a
    real file on disk, so run_stt writes the WAV bytes Pipecat hands it to a
    NamedTemporaryFile before calling model.transcribe().
    """

    def __init__(self, *, model_id: str, executor: ThreadPoolExecutor, **kwargs):
        super().__init__(**kwargs)
        self._executor = executor
        # Loaded on the shared MLX executor thread, not the caller's thread.
        self._model = executor.submit(from_pretrained, model_id).result()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        loop = asyncio.get_running_loop()
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(audio)
            f.flush()
            result = await loop.run_in_executor(self._executor, self._model.transcribe, f.name)

        text = result.text.strip()
        if text:
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), None)
