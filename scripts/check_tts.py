"""
T3 acceptance check (PLAN.md): calls MLXKokoroTTSService.run_tts on 2
sentences separately, asserts non-empty int16 audio frames at 24k, prints
per-sentence TTFA (time to first audio frame).

Run: uv run python scripts/check_tts.py
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from pipecat.frames.frames import TTSAudioRawFrame

from voice_stack.tts import MLXKokoroTTSService

SENTENCES = [
    "The weather in Nairobi today is warm and mostly sunny.",
    "Let me know if you would like anything else.",
]


async def main():
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        t0 = time.perf_counter()
        tts = MLXKokoroTTSService(executor=executor)
        print(f"Model load: {time.perf_counter() - t0:.2f}s")

        for i, text in enumerate(SENTENCES):
            t0 = time.perf_counter()
            ttfa = None
            frames = []
            async for frame in tts.run_tts(text, context_id=f"check-{i}"):
                if frame is None:
                    continue
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                frames.append(frame)

            audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
            assert audio_frames, f"sentence {i}: expected >=1 TTSAudioRawFrame, got 0"
            for f in audio_frames:
                assert f.sample_rate == 24000, f"expected 24000Hz, got {f.sample_rate}"
                assert len(f.audio) > 0, "expected non-empty audio bytes"

            total_bytes = sum(len(f.audio) for f in audio_frames)
            print(
                f"Sentence {i}: {len(audio_frames)} frame(s), "
                f"TTFA={ttfa * 1000:.1f}ms, total_bytes={total_bytes}"
            )

        print("check_tts.py: PASS")
    finally:
        executor.shutdown(wait=True)


if __name__ == "__main__":
    asyncio.run(main())
