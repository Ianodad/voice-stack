"""
Custom Pipecat TTS service: mlx-audio Kokoro.

The built-in pipecat KokoroTTSService uses kokoro-onnx (a different backend
from mlx-audio's mlx-community/Kokoro-82M-bf16), so this is a small custom
TTSService subclass instead (see .planning/pipecat-research.md §5).

Base TTSService already sentence-aggregates streamed LLM text before calling
run_tts (TextAggregationMode.SENTENCE, the default) and handles
TTSStartedFrame/TTSStoppedFrame + TTFB metrics automatically when
push_start_frame=True, push_stop_frames=True are passed to __init__ -- no
custom timing/aggregation code needed here.

STT and TTS share one ThreadPoolExecutor(max_workers=1) so all MLX/Metal
work for both stays on a single thread (see PLAN.md "Key design calls").
"""

import asyncio
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from mlx_audio.tts.utils import load_model
from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService


class MLXKokoroTTSService(TTSService):
    """Local TTS via mlx-audio's Kokoro, run on a shared MLX executor thread."""

    def __init__(
        self,
        *,
        executor: ThreadPoolExecutor,
        model_id: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "af_heart",
        lang_code: str = "a",
        speed: float = 1.0,
        **kwargs,
    ):
        # language=None: multi-language switching goes through lang_code
        # above, not the Language-enum-based settings field (see
        # pipecat/services/settings.py -- None marks a store-mode field as
        # unsupported rather than leaving it NOT_GIVEN).
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=24000,
            settings=TTSSettings(model=model_id, voice=voice, language=None),
            **kwargs,
        )
        self._executor = executor
        self._voice = voice
        self._lang_code = lang_code
        self._speed = speed
        # Loaded on the shared MLX executor thread, not the caller's thread.
        self._model = executor.submit(load_model, model_id).result()

    def can_generate_metrics(self) -> bool:
        # Base class gates TTFB/processing metrics on this; default is False.
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        loop = asyncio.get_running_loop()
        # model.generate() is a lazy generator (one GenerationResult per
        # sentence-internal chunk). Step it one chunk at a time on the shared
        # MLX executor thread so audio streams out as each chunk is ready,
        # instead of blocking until the whole sentence is synthesized.
        chunks = self._model.generate(
            text=text, voice=self._voice, speed=self._speed, lang_code=self._lang_code
        )
        sentinel = object()

        def _next_chunk():
            result = next(chunks, sentinel)
            if result is sentinel:
                return None
            # mx.array -> numpy must happen HERE, on the same MLX executor
            # thread that built the array's lazy compute graph. Converting it
            # back on the asyncio event-loop thread crashes MLX ("There is no
            # Stream(gpu, 0) in current thread") -- the per-thread-stream
            # hazard PLAN.md calls out, hit via cross-thread lazy eval rather
            # than concurrent access. Confirmed by direct repro.
            audio_np = np.array(result.audio)
            audio_int16 = (audio_np * 32767).astype(np.int16).tobytes()
            return audio_int16, result.sample_rate

        # NOTE: cancelling this run_tts (e.g. on barge-in) does not stop a
        # Kokoro call already running on the shared MLX executor thread --
        # it runs to completion and the next job queues behind it (~0.2s
        # warm, bounded by sentence-sized chunks). Accepted for MVP per
        # PLAN.md "Key design calls".
        while True:
            chunk = await loop.run_in_executor(self._executor, _next_chunk)
            if chunk is None:
                break
            audio_int16, sample_rate = chunk
            yield TTSAudioRawFrame(
                audio=audio_int16,
                sample_rate=sample_rate,
                num_channels=1,
                context_id=context_id,
            )
