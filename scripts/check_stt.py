"""
T2 acceptance check (PLAN.md): feeds scripts/spike_input.wav bytes through
ParakeetSTTService.run_stt directly (no Pipecat pipeline) and asserts at
least one TranscriptionFrame comes back with "Nairobi" in the joined text.

Run: uv run python scripts/check_stt.py
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipecat.frames.frames import TranscriptionFrame

from voice_stack.stt import ParakeetSTTService

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_PATH = SCRIPT_DIR / "spike_input.wav"
STT_MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"


async def main():
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        t0 = time.perf_counter()
        stt = ParakeetSTTService(model_id=STT_MODEL_ID, executor=executor)
        print(f"Model load: {time.perf_counter() - t0:.2f}s")

        audio_bytes = AUDIO_PATH.read_bytes()

        t0 = time.perf_counter()
        frames = [f async for f in stt.run_stt(audio_bytes) if f is not None]
        print(f"run_stt: {time.perf_counter() - t0:.2f}s")

        assert len(frames) >= 1, f"expected >=1 TranscriptionFrame, got {len(frames)}"
        joined = " ".join(f.text for f in frames if isinstance(f, TranscriptionFrame))
        print(f"Frames: {len(frames)}")
        print(f"Joined text: {joined!r}")
        assert "Nairobi" in joined, f"expected 'Nairobi' in joined text, got {joined!r}"
        print("check_stt.py: PASS")
    finally:
        executor.shutdown(wait=True)


if __name__ == "__main__":
    asyncio.run(main())
