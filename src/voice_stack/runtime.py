"""
Shared runtime for the voice stack: owns the mlx_lm.server subprocess and the
preloaded STT/TTS models, and pays MLX's one-time warmup cost.

Both the local CLI loop and the web server build per-session pipelines from
one Runtime (see bot.build_worker) so models load once per process.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.turns.empty_user_turn import (
    DEFAULT_EMPTY_USER_TURN_INTERRUPTED_PROMPT as DEVELOPER_ROLE_RECOVERY_PROMPT,
)

from voice_stack.llm_server import MLXLMServer
from voice_stack.stt import ParakeetSTTService
from voice_stack.tts import MLXKokoroTTSService

REPO_ROOT = Path(__file__).resolve().parents[2]
SPIKE_AUDIO_PATH = REPO_ROOT / "scripts" / "spike_input.wav"

STT_MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"
LLM_MODEL_ID = "mlx-community/Qwen3.6-35B-A3B-4bit"
TTS_MODEL_ID = "mlx-community/Kokoro-82M-bf16"
TTS_VOICE = "af_heart"
TTS_LANG_CODE = "a"

LLM_HOST = "127.0.0.1"
LLM_PORT = 8080
# Qwen3.6's chat template defaults to "thinking" mode; a voice assistant has
# no latency budget for chain-of-thought, so this is disabled explicitly
# (same reasoning as scripts/latency_spike.py's run_llm).
ENABLE_THINKING_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

SYSTEM_PROMPT = (
    "You are a concise voice assistant running entirely locally on the user's Mac "
    "(Apple Silicon): speech-to-text, this language model (Qwen3.6), and text-to-speech "
    "all run on-device, with no internet or cloud access. You cannot open apps or browse. "
    "Speech transcripts may contain mishearings; if a request is unclear, ask briefly. "
    "Reply in 1-3 short sentences, no markdown."
)
WARMUP_USER_TEXT = "Say hello in one short sentence."
CHECK_USER_TEXT = "What is two plus two?"
MAX_TOKENS = 60


async def _llm_text_turn(llm: OpenAILLMService, user_text: str) -> tuple[str, float | None]:
    """One streaming text turn through the actual configured LLM client.

    Reuses the OpenAILLMService instance's own client/settings (single source
    of truth for base_url/model/extra_body) rather than building a second,
    parallel client -- used for warmup and for the --check TTFT/no-<think>
    assertions, since driving OpenAILLMService itself requires the full
    frame-based pipeline machinery this diagnostic call intentionally skips.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    stream = await llm._client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=messages,
        stream=True,
        max_tokens=MAX_TOKENS,
        extra_body=ENABLE_THINKING_EXTRA_BODY,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks.append(delta)
    return "".join(chunks), ttft


async def _llm_developer_role_turn(llm: OpenAILLMService, user_text: str) -> str | None:
    """One turn through OpenAILLMService.run_inference() with a developer-role
    message in context -- the same message shape Pipecat's own user
    aggregator injects when the user interrupts the bot with no transcript
    (see llm_response_universal.py's _maybe_recover_empty_user_turn).

    Regression check for the developer-role bug: without
    `llm.supports_developer_role = False`, the adapter sends role="developer"
    verbatim, Qwen3.6's chat template raises "Unexpected message role.", and
    mlx_lm.server 404s -- this drives the real OpenAILLMService + adapter
    path (unlike _llm_text_turn's raw client call) so --check actually
    exercises it.
    """
    context = LLMContext(
        messages=[
            {"role": "developer", "content": DEVELOPER_ROLE_RECOVERY_PROMPT},
            {"role": "user", "content": user_text},
        ]
    )
    return await llm.run_inference(context, max_tokens=MAX_TOKENS)


class Runtime:
    """Owns the LLM server, the shared MLX executor, and the loaded models."""

    def __init__(self) -> None:
        # One dedicated MLX thread for STT + TTS (see PLAN.md "Key design calls":
        # avoids two threads issuing Metal work on different streams).
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.llm_server = MLXLMServer(model_id=LLM_MODEL_ID, host=LLM_HOST, port=LLM_PORT)
        self.stt_model = None
        self.tts_model = None
        # Warmup outputs, kept so --check can assert on them.
        self.stt_frames: list = []
        self.tts_frames: list = []

    @property
    def llm_base_url(self) -> str:
        return self.llm_server.base_url

    def make_llm(self) -> OpenAILLMService:
        llm = OpenAILLMService(
            base_url=self.llm_base_url,
            api_key="not-needed",
            settings=OpenAILLMService.Settings(
                model=LLM_MODEL_ID,
                system_instruction=SYSTEM_PROMPT,
                extra={"extra_body": ENABLE_THINKING_EXTRA_BODY},
            ),
        )
        # mlx_lm.server's Qwen3.6 chat template raises "Unexpected message
        # role." on role="developer" (see chat_template.jinja); Pipecat's own
        # user aggregator injects one on an empty interrupted turn. The base
        # class defaults to assuming native "developer" role support, so
        # override per-instance to make the adapter convert it to "user"
        # before sending (base_llm.py:343).
        llm.supports_developer_role = False
        return llm

    def _start_llm_server(self) -> None:
        print(f"Starting mlx_lm.server ({LLM_MODEL_ID}) ...")
        t0 = time.perf_counter()
        self.llm_server.start(timeout=60.0)
        print(f"  ready in {time.perf_counter() - t0:.2f}s")

    async def _warmup_llm(self) -> None:
        llm = self.make_llm()
        t0 = time.perf_counter()
        warmup_reply, warmup_ttft = await _llm_text_turn(llm, WARMUP_USER_TEXT)
        warmup_ttft_ms = f"{warmup_ttft * 1000:.1f}ms" if warmup_ttft is not None else "N/A"
        print(
            f"  LLM warmup: total={time.perf_counter() - t0:.2f}s ttft={warmup_ttft_ms} "
            f"reply={warmup_reply!r}"
        )

    async def start(self) -> None:
        self._start_llm_server()

        print(f"Loading STT ({STT_MODEL_ID}) ...")
        t0 = time.perf_counter()
        stt = ParakeetSTTService(model_id=STT_MODEL_ID, executor=self.executor)
        print(f"  loaded in {time.perf_counter() - t0:.2f}s")

        print(f"Loading TTS ({TTS_MODEL_ID}) ...")
        t0 = time.perf_counter()
        tts = MLXKokoroTTSService(
            executor=self.executor,
            model_id=TTS_MODEL_ID,
            voice=TTS_VOICE,
            lang_code=TTS_LANG_CODE,
        )
        print(f"  loaded in {time.perf_counter() - t0:.2f}s")
        self.stt_model = stt._model
        self.tts_model = tts._model

        # --- Warmup: pay MLX's one-time first-call lazy-compile cost here,
        # not on the user's first turn (see PLAN.md, ~6.85s first-turn penalty). ---
        print("Warming up ...")

        t0 = time.perf_counter()
        audio_bytes = SPIKE_AUDIO_PATH.read_bytes()
        self.stt_frames = [f async for f in stt.run_stt(audio_bytes) if f is not None]
        stt_warmup_s = time.perf_counter() - t0
        if self.stt_frames:
            print(f"  STT warmup: {stt_warmup_s:.2f}s, text={self.stt_frames[0].text!r}")
        else:
            print(f"  STT warmup: {stt_warmup_s:.2f}s, no transcript")

        t0 = time.perf_counter()
        self.tts_frames = [
            f async for f in tts.run_tts(WARMUP_USER_TEXT, context_id="warmup") if f is not None
        ]
        print(f"  TTS warmup: {time.perf_counter() - t0:.2f}s, frames={len(self.tts_frames)}")

        await self._warmup_llm()

    def stop(self) -> None:
        self.llm_server.stop()
        self.executor.shutdown(wait=True)

    async def restart_llm(self) -> None:
        self.llm_server.stop()
        self._start_llm_server()
        await self._warmup_llm()
