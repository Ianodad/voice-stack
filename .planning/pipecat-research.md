# Pipecat wiring research — for local MLX voice stack (parakeet-mlx / mlx-lm / mlx-audio Kokoro)

**Method:** installed `pipecat-ai` fresh into a throwaway venv (`uv venv --python 3.12` +
`uv pip install`) and read the *installed* source (not blog posts / not GitHub main branch —
whatever `pip` actually resolved). Package root for all file:line citations below:

```
<pcenv>/lib/python3.12/site-packages/pipecat/...
```

**Installed version: `pipecat-ai==1.12.0`** (latest on PyPI as of 2026-09-27; `pipecat.__init__:54`
prints `ᓚᘏᗢ Pipecat 1.12.0`).

**Extras used:** `pipecat-ai[local,local-smart-turn]` initially. Finding below (§8) shows
`local-smart-turn` was unnecessary — see recommendation.

**Portaudio:** already installed via Homebrew — `brew list --versions portaudio` → `portaudio 19.7.0`.
`pyaudio==0.2.14` (pulled by the `local` extra) links against it; no extra setup needed.

---

## Big-picture warning before the details

Pipecat's architecture changed substantially between the version most blog posts / tutorials
describe (~0.0.x / 1.0–1.3) and 1.12.0. Three load-bearing changes that will make older
tutorials actively wrong:

1. **`PipelineTask` / `PipelineRunner` are deprecated** (since 1.3.0, removed in 2.0.0). Use
   `PipelineWorker` (`pipecat/pipeline/worker.py:199`) + `WorkerRunner`
   (`pipecat/workers/runner.py:83`) instead. `pipecat/pipeline/task.py` and
   `pipecat/pipeline/runner.py` are now just deprecation shims re-exporting the new names.
2. **VAD is no longer a transport param.** `TransportParams` (`pipecat/transports/base_transport.py:25-88`)
   has *no* `vad_analyzer` or `turn_analyzer` field at all in this version. VAD/turn detection now
   lives on the **LLM user context aggregator**, via `LLMUserAggregatorParams(vad_analyzer=...)`.
3. **Smart Turn v3 (local, ONNX) is the *default* turn-stop strategy** if you don't configure
   anything (`pipecat/turns/user_turn_strategies.py:45-53`). You get local, on-device turn
   detection for free just by using `LLMContextAggregatorPair` — no explicit wiring needed unless
   you want to override it.

---

## 1. Local mic/speaker transport

- **Class:** `LocalAudioTransport` — `pipecat/transports/local/audio.py:203`
- **Import:** `from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams`
- **Params class:** `LocalAudioTransportParams(TransportParams)` — `pipecat/transports/local/audio.py:34-43`
  - Adds only `input_device_index: int | None` and `output_device_index: int | None` on top of
    the base `TransportParams` fields (`audio_in_enabled`, `audio_in_sample_rate`,
    `audio_out_enabled`, `audio_out_sample_rate`, `audio_in_channels`, `audio_out_channels`, etc. —
    `pipecat/transports/base_transport.py:25-88`).
  - No `vad_analyzer` / `turn_analyzer` field here (see §2 — that's wired elsewhere now).
- **Needs PyAudio:** yes, hard dependency, imported at module load —
  `pipecat/transports/local/audio.py:24-31`:
  ```python
  try:
      import pyaudio
  except ModuleNotFoundError as e:
      logger.error('In order to use local audio, you need to `uv add "pipecat-ai[local]"`. '
                   'On MacOS, you also need to `brew install portaudio`.')
      raise ImportError(...) from e
  ```
  Install with the `local` extra: `pipecat-ai[local]` → pulls `pyaudio~=0.2.14` (confirmed from
  PyPI metadata: `pyaudio~=0.2.14; extra == "local"`). **Portaudio is already installed on this
  machine** (`brew list --versions portaudio` → `19.7.0`), so no action needed there.
- **Mechanics:** `LocalAudioInputTransport`/`LocalAudioOutputTransport`
  (`pipecat/transports/local/audio.py:46`, `:121`) open blocking PyAudio streams; input uses a
  `stream_callback` that hands 20ms chunks to `push_audio_frame` via
  `asyncio.run_coroutine_threadsafe`; output writes via a 1-worker `ThreadPoolExecutor`. Simple
  and synchronous under the hood — fine for a single local session.
- **Usage:**
  ```python
  transport = LocalAudioTransport(
      LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
  )
  # transport.input() / transport.output() return FrameProcessors for the Pipeline list
  ```

---

## 2. VAD + turn detection — biggest departure from older docs

### Where it actually gets wired (not transport params)

VAD and turn-stop strategies attach to the **`LLMUserAggregatorParams`** you pass into
`LLMContextAggregatorPair`, not to the transport. Confirmed straight from Pipecat's own CLI
bot-generator template (`pipecat/cli/templates/server/_macros/helper_functions.jinja2:23-28`,
the actual code Pipecat's own `pipecat init` scaffolds):

```python
context = LLMContext()
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
)
```

Internally, `LLMUserAggregator.__init__` (`pipecat/processors/aggregators/llm_response_universal.py:764-777`)
does:
```python
self._vad_controller: VADController | None = None
if self._params.vad_analyzer:
    self._vad_controller = VADController(self._params.vad_analyzer, ...)
    self._vad_controller.add_event_handler("on_speech_started", self._on_vad_speech_started)
    ...
```
i.e. the aggregator itself runs a `VADController` over the raw audio frames flowing through it
from `transport.input()`, and emits `VADUserStartedSpeakingFrame` / `VADUserStoppedSpeakingFrame`
downstream. There is also a standalone `VADProcessor` pipeline processor
(`pipecat/processors/audio/vad_processor.py:41`, `vad_processor = VADProcessor(vad_analyzer=SileroVADAnalyzer())`)
for cases where you want VAD state elsewhere in the pipeline, but the aggregator path above is
the one that drives turn detection.

### SileroVADAnalyzer

- **Import:** `from pipecat.audio.vad.silero import SileroVADAnalyzer`
- **Class:** `pipecat/audio/vad/silero.py:130`
- Bundles its own ONNX model at `pipecat/audio/vad/data/silero_vad.onnx` and loads it via
  `onnxruntime` (a **base** pipecat-ai dependency, not an extra — confirmed from PyPI metadata,
  `onnxruntime~=1.24.3` has no `extra ==` marker). **No `silero-vad` pip package needed** — the
  requested `[...,silero,...]` extra does not exist on PyPI for `pipecat-ai` (see §8).
- Only accepts 8000 or 16000 Hz (`silero.py:184-187`, `set_sample_rate` raises otherwise).
- `VADParams` (`pipecat/audio/vad/vad_analyzer.py:47-60`): `confidence=0.7`, `start_secs=0.2`,
  `stop_secs=0.2`, `min_volume=0.6` are the defaults.

### Smart Turn (local, v3) — this is the default

- **Import:** `from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3`
- **Class:** `pipecat/audio/turn/smart_turn/local_smart_turn_v3.py:28`
- **Default:** `default_user_turn_stop_strategies()` (`pipecat/turns/user_turn_strategies.py:45-53`)
  returns `[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]` — this is
  what you get automatically if you don't set `user_turn_strategies` on `LLMUserAggregatorParams`
  at all. `UserTurnStrategies.__post_init__` (`user_turn_strategies.py:76-80`) fills in this
  default plus `default_user_turn_start_strategies()` = `[VADUserTurnStartStrategy(),
  TranscriptionUserTurnStartStrategy()]`.
- **Dependencies — important cost-saving finding:** `LocalSmartTurnAnalyzerV3` uses only
  `onnxruntime` + `soxr` (both base pipecat-ai deps) and a bundled model file
  (`pipecat/audio/turn/smart_turn/data/smart-turn-v3.2-cpu.onnx`). It does **not** need
  torch/transformers/coremltools. Those heavy deps belong to the *older*, non-default variants:
  `LocalSmartTurnAnalyzer` (PyTorch, v2, deprecated — `local_smart_turn_v2.py:20-28` imports
  `torch`, `transformers.Wav2Vec2*`) and `LocalCoreMLSmartTurnAnalyzer`
  (`local_coreml_smart_turn.py:20-23` imports `coremltools`, `torch`,
  `transformers.AutoFeatureExtractor`). **Recommendation: don't install the `local-smart-turn`
  extra at all** — you get v3 for free from base `pipecat-ai`, and skip pulling in
  torch/transformers/coremltools a second time next to your own MLX torch pin (see §8's dry-run).
- Model expects 16kHz input; resamples automatically via `soxr` if your pipeline runs at a
  different rate (`local_smart_turn_v3.py:123-136`, `_resample_to_model_rate`).
- Audio is truncated/padded to the last 8 seconds and run through a vendored Whisper log-mel
  feature extractor (`local_smart_turn_v3.py:141-171`) → sigmoid probability → complete/incomplete.

### Interruptions ride on the same mechanism

`enable_interruptions` defaults to `True` throughout the turn-strategy classes (e.g.
`pipecat/turns/user_start/base_user_turn_start_strategy.py:35,56`). There is no more
`PipelineParams(allow_interruptions=True)` flag (grepped the whole package — no hits) — barge-in
is on by default as soon as VAD is wired in via `LLMUserAggregatorParams(vad_analyzer=...)`: a
`VADUserStartedSpeakingFrame` while the bot is talking broadcasts an interruption.

---

## 3. STT — no built-in parakeet/MLX service; use `SegmentedSTTService`

- Grepped the whole installed tree for "parakeet" and "mlx" under `services/`: the only hit is
  `pipecat/services/nvidia/sagemaker/stt.py:101`, a default *model name string* for an
  NVIDIA SageMaker-hosted Parakeet endpoint (cloud, unrelated to `parakeet-mlx`). **There is no
  built-in local/MLX Parakeet STT service.**
- There *is* a built-in MLX Whisper service, `WhisperSTTServiceMLX`
  (`pipecat/services/whisper/stt.py:461`), which is the closest real reference implementation for
  a local MLX STT and is worth copying the pattern from.
- **Base class to subclass:** `SegmentedSTTService` — `pipecat/services/stt_service.py:798`
  - Import: `from pipecat.services.stt_service import SegmentedSTTService`
  - **Method to override:** 
    ```python
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]
    ```
    (`stt_service.py:336` on the base `STTService`, inherited unchanged by `SegmentedSTTService`).
  - **What audio it receives:** one complete utterance's audio, buffered by VAD start/stop, with
    `trailing_silence_secs` (default 0.5s) of silence appended
    (`stt_service.py:940`, `_handle_user_stopped_speaking`). By default this is wrapped in a
    **WAV container** (`pcm_to_wav`) at the pipeline's input sample rate
    (`stt_service.py:949-950`) — controlled by the `wants_wav_segments` property
    (`stt_service.py:895-904`), which defaults to `True`. Set it `False` only if your model wants
    raw 16-bit PCM directly.
  - **What frames to yield:** typically one `TranscriptionFrame(text, user_id, timestamp, language)`
    per segment — exact call signature copied from `WhisperSTTServiceMLX.run_stt`
    (`whisper/stt.py:610-615`):
    ```python
    yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), language)
    ```
    (`time_now_iso8601` from `pipecat.utils.time`). `SegmentedSTTService` also auto-marks these
    `finalized = True` in its own `push_frame` override (`stt_service.py:906-918`).
- **Important constraint specific to `parakeet-mlx` (verified against the project's own
  `.venv`, not the throwaway one):** `BaseParakeet.transcribe()`
  (`.venv/lib/python3.12/site-packages/parakeet_mlx/parakeet.py:133`) takes a **file path**, not
  raw audio: `def transcribe(self, path: Path | str, ...)`. It calls
  `load_audio(audio_path, ...)`
  (`parakeet_mlx/audio.py:51-53`), which **shells out to `ffmpeg`** with `str(filename)` — it is
  not file-like-object compatible, needs a real path on disk, and requires `ffmpeg` on `PATH`
  (raises `RuntimeError` otherwise). This is *different* from `WhisperSTTServiceMLX`, whose
  `mlx_whisper.transcribe()` accepts an in-memory float32 numpy array directly
  (`whisper/stt.py:564-580`) — you cannot copy that part of the Whisper pattern for Parakeet.
  **Practical implication:** keep `wants_wav_segments = True` (the default), write the WAV bytes
  Pipecat hands you to a `tempfile.NamedTemporaryFile(suffix=".wav")`, and call
  `model.transcribe(path)` on that. Also run it via `asyncio.to_thread` (as `WhisperSTTServiceMLX`
  does at `whisper/stt.py:571-577`) since it's a blocking call (subprocess + MLX compute) that
  would otherwise stall the pipeline's event loop.

---

## 4. LLM — run `mlx_lm.server`, use `OpenAILLMService`

**Recommendation: (a) — run `mlx_lm.server` and point `OpenAILLMService` at it via `base_url`.**
No custom `LLMService` needed. Evidence:

- `BaseOpenAILLMService.__init__` (`pipecat/services/openai/base_llm.py:161-186`) accepts
  `base_url` directly and passes it straight to the `openai` Python client's constructor
  (`create_client`, `base_llm.py:256`). `OpenAILLMService` (`services/openai/llm.py:15`) is a thin
  subclass of it — this is the standard "point at any OpenAI-compatible server" path.
- Streaming is unconditional: `build_chat_completion_params`
  (`base_llm.py:362-393`) always sets `"stream": True, "stream_options": {"include_usage": True}`.
- **Passing `enable_thinking=False` / `chat_template_kwargs` through to mlx_lm.server:**
  `build_chat_completion_params` does `params.update(self._settings.extra)` at the end
  (`base_llm.py:395`), and those merged params go straight into
  `self._client.chat.completions.create(**params)` (`base_llm.py:359`). The openai-python SDK's
  `Completions.create()` (checked in the throwaway venv,
  `openai/resources/chat/completions/completions.py:253-299`) has a first-class
  `extra_body: Body | None = None` parameter that gets merged into the raw JSON request body sent
  over the wire. **Verified against the project's own `mlx_lm` 0.31.3** (not the throwaway venv):
  `mlx_lm/server.py:1192` reads `self.chat_template_kwargs = self.body.get("chat_template_kwargs")`
  straight off the top-level request JSON, and passes it through to generation at
  `mlx_lm/server.py:1405`. So the wiring is:
  ```python
  llm = OpenAILLMService(
      base_url="http://127.0.0.1:8080/v1",
      api_key="not-needed",
      model="mlx-community/Qwen3.6-35B-A3B-4bit",
      settings=OpenAILLMService.Settings(
          extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
      ),
  )
  ```
  `[inference]`: I confirmed each half (Pipecat's `extra_body` passthrough, and mlx_lm.server's
  top-level `chat_template_kwargs` read) from source, but did **not** run an end-to-end request
  against a live `mlx_lm.server` in this session — worth a quick smoke test before relying on it
  (`curl -d '{"chat_template_kwargs":{"enable_thinking":false}, ...}'` against
  `mlx_lm.server` directly is the fastest sanity check).
- Start the server (from the project's own venv, confirmed via
  `mlx_lm/server.py` argparse help text at line ~1850): 
  ```
  uv run mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --port 8080
  ```

---

## 5. TTS — no built-in Kokoro-MLX service; Kokoro built-in uses `kokoro-onnx`, not MLX

- **Built-in `KokoroTTSService`** (`pipecat/services/kokoro/tts.py:118`) exists but is backed by
  **`kokoro-onnx`** (`from kokoro_onnx import Kokoro`, `tts.py:33`), auto-downloading its own ONNX
  model + voices file from GitHub releases (`tts.py:39-63`). This is a *different* Kokoro backend
  from `mlx-audio`'s `mlx-community/Kokoro-82M-bf16` — **not usable as-is for our stack.** You need
  a custom `TTSService` subclass.
- **Base class:** `TTSService` — `pipecat/services/tts_service.py:115`
  - Import: `from pipecat.services.tts_service import TTSService`
  - **Method to override:**
    ```python
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]
    ```
    (`tts_service.py:556`, abstract).
  - **Frames to yield:** `TTSAudioRawFrame(audio=..., sample_rate=..., num_channels=1,
    context_id=context_id)` per chunk — exact pattern copied from `KokoroTTSService.run_tts`
    (`kokoro/tts.py:236-267`), which is a synchronous local-model TTS and the closest real
    template:
    ```python
    yield TTSAudioRawFrame(audio=audio_data, sample_rate=self.sample_rate,
                            num_channels=1, context_id=context_id)
    ```
  - Pass `push_start_frame=True, push_stop_frames=True` to the base `__init__`
    (`tts_service.py:159-162`) and the base class handles `TTSStartedFrame`/`TTSStoppedFrame` and
    audio-context bookkeeping for you — your `run_tts` only needs to yield audio frames (see
    `KokoroTTSService.__init__`, `kokoro/tts.py:200-205`).
  - **Sample rate handling:** `self.sample_rate` is the *pipeline's* output rate
    (`PipelineParams.audio_out_sample_rate`, default `24000` —
    `pipecat/pipeline/worker.py:186`), which conveniently already matches mlx-audio's Kokoro
    native 24kHz output — so as long as you set `PipelineParams(audio_out_sample_rate=24000)` (or
    leave it at the default), **no resampling is needed**. `KokoroTTSService` resamples
    defensively with `create_stream_resampler()` (`kokoro/tts.py:218,258`) in case the two rates
    differ — worth doing the same defensively even though they should match here.
  - **mlx-audio's `GenerationResult.audio` is an `mx.array`, not a numpy array**
    (confirmed in the project's own `.venv`,
    `mlx_audio/tts/models/base.py:72-86`) — convert with `np.array(result.audio)` before
    scaling to int16 PCM bytes.

### Does Pipecat already aggregate LLM text into sentences before TTS? — **Yes, by default**

This directly answers the "TTS didn't stream" caveat from the latency spike. `TTSService.__init__`
(`tts_service.py:151-158`) has:
```python
text_aggregation_mode: TextAggregationMode | None = None,
```
and the docstring is explicit (`tts_service.py:197-199`): *"TextAggregationMode.SENTENCE
(default) buffers until sentence boundaries, TextAggregationMode.TOKEN streams tokens directly for
lower latency."* `TextAggregationMode` itself (`tts_service.py:88-101`) documents `SENTENCE` as
the mode that "Produces more natural speech but adds latency (~200-300ms per sentence)." So: the
base `TTSService` buffers the LLM's streamed tokens into complete sentences (using `sentencex`, a
**base** pipecat-ai dependency — confirmed from PyPI metadata, no extra needed) and calls your
`run_tts()` once per sentence. This *is* the "streaming" that the latency spike's docstring
anticipated for multi-sentence responses — your custom TTS service doesn't need to do any of its
own sentence-splitting; it just needs to handle being called multiple times per LLM turn.

---

## 6. Interruptions / barge-in / echo

- Already covered in §2: `enable_interruptions=True` is the default across the turn-start
  strategies, and it "just works" once VAD is wired into `LLMUserAggregatorParams`. No separate
  `PipelineParams` flag exists in this version.
- **Echo cancellation (AEC): not provided by Pipecat for local audio.** Grepped the whole package
  for "echo cancel" / "AEC" — zero hits. The only audio filters shipped
  (`pipecat/audio/filters/`: `krisp_viva_filter.py`, `aic_filter.py`, `koala_filter.py`,
  `rnnoise_filter.py`) are noise-suppression filters, not echo cancellers, and Krisp/AIC/Koala are
  third-party proprietary SDKs (Krisp specifically: "Krisp is available when deployed to Pipecat
  Cloud" per the CLI template, `cli/templates/server/bot_cascade.py.jinja2:76-84` — i.e. not
  something you get for a plain local run).
  **Conclusion: with `LocalAudioTransport` playing TTS out of speakers and picking it back up on
  the same mic, you will get self-triggered VAD/turn-detection on the bot's own voice unless you
  either (a) use headphones (breaks the loop physically — the correct fix for this setup), or
  (b) rely on the OS's own AEC if your input device exposes one (PortAudio/PyAudio does not do
  AEC itself — it's a thin device I/O layer).** This is a real gap in the plan for a
  speaker-based local setup; headphones are the pragmatic answer here, not a Pipecat feature.

---

## 7. Context aggregator — current API

```python
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
```
- **`LLMContext.__init__`** (`pipecat/processors/aggregators/llm_context.py:91-96`):
  ```python
  def __init__(self, messages=None, tools=NOT_GIVEN, tool_choice=NOT_GIVEN)
  ```
  System prompt is just the first message with `role="system"`, OpenAI-dict style:
  ```python
  context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
  ```
  (Pipecat's own CLI template instead constructs an empty `LLMContext()` and presumably relies on
  `OpenAILLMService.Settings(system_instruction=...)` — both work; `base_llm.py:212,263,342`
  shows `system_instruction` as an alternate, provider-specific path. The `messages=[...]`
  approach is more portable across LLM services.)
- **`LLMContextAggregatorPair.__init__`** (`llm_response_universal.py:2494-2502`):
  ```python
  def __init__(self, context, *, user_params=None, assistant_params=None,
               add_tool_change_messages=None, realtime_service_mode=None)
  ```
  Returns `(user_aggregator, assistant_aggregator)` — put `user_aggregator` right after your STT
  in the pipeline list, and `assistant_aggregator` after the TTS/output stage, per the cascade
  pipeline template (`cli/templates/server/_macros/pipeline_components.jinja2:60-70`).
- **`LLMUserAggregatorParams`** (`llm_response_universal.py:124-186`, `@dataclass`) — relevant
  fields: `vad_analyzer: VADAnalyzer | None = None`, `user_turn_strategies: UserTurnStrategies |
  None = None` (defaults described in §2), `user_idle_timeout: float = 0`,
  `audio_idle_timeout: float = 1.0`.

---

## 8. Dependency conflicts — real dry-run against the project's exact pins

Did **not** touch the project's real `.venv`. Copied the project's dependency set into a scratch
`pyproject.toml` and ran `uv lock` there (full resolver run, not `--dry-run` text-only — `uv lock`
against a throwaway project is the equivalent and gives exact resolved versions).

**Finding — real, hard conflict on `soundfile`:**
```
Because pipecat-ai>=1.12.0 depends on soundfile>=0.13.1,<0.14.dev0 and
your project depends on pipecat-ai[local]==1.12.0, we can conclude that
your project depends on soundfile>=0.13.1,<0.14.dev0.
And because your project depends on soundfile>=0.14.0, we can conclude
that your project's requirements are unsatisfiable.
```
`soundfile>=0.13.1,<0.14.dev0` is a **base** pipecat-ai dependency (confirmed from PyPI metadata:
`soundfile~=0.13.1` with no extra marker) — unrelated to which extras you pick, so this conflict
is unavoidable with pipecat-ai 1.12.0 as-is. The project's `pyproject.toml` currently pins
`soundfile>=0.14.0` (line 18).

**Silent-downgrade trap:** if you add `pipecat-ai[local,local-smart-turn]` to the project
*without* pinning a version, `uv lock` does **not** error — it silently resolves to
**`pipecat-ai==1.8.1`** instead of 1.12.0, because 1.8.1 apparently has a looser `soundfile`
constraint. This is dangerous: 1.8.1 may not have the `turns/` module architecture, the default
Smart Turn v3 behavior, or other APIs described throughout this doc — all discovered against
1.12.0. **Always pin `pipecat-ai==1.12.0` explicitly** so a bad resolve fails loudly instead of
quietly downgrading you onto a different, unverified API surface.

**Fix + clean resolve:** relaxing the project's pin to `soundfile>=0.13.1,<0.14` and pinning
`pipecat-ai[local]==1.12.0` (no `local-smart-turn` — see §2's cost-saving finding) resolves
**cleanly, 162 packages, no other conflicts**, and leaves every other pin untouched:

| package | before (project) | after (with pipecat-ai[local]==1.12.0) |
|---|---|---|
| torch | 2.14.0 | **2.14.0** (unchanged) |
| mlx | 0.32.2 | **0.32.2** (unchanged) |
| mlx-lm | 0.31.3 | **0.31.3** (unchanged) |
| mlx-audio | 0.5.5 | 0.5.6 (patch bump, harmless — project pin is `>=`) |
| huggingface-hub | 1.32.0 | 1.33.0 (minor bump, harmless — project pin is `>=`) |
| transformers | 5.17.0 | 5.17.0 (unchanged) |
| soundfile | 0.14.0 | **0.13.1 (needs the pyproject.toml pin relaxed)** |
| torchaudio | (not present) | not pulled in (confirms `local-smart-turn` extra unneeded) |
| coremltools | (not present) | not pulled in (same) |
| pyaudio | (not present) | 0.2.14 (new, for local transport) |
| onnxruntime | (not present) | 1.24.4 (new, for VAD + Smart Turn v3) |
| soxr | (not present, mlx-audio may vendor its own) | 1.0.0 (new) |

**Action needed in the real project:** change `"soundfile>=0.14.0"` to `"soundfile>=0.13.1,<0.14"`
in `pyproject.toml`, and add `"pipecat-ai[local]==1.12.0"`. Everything else resolves cleanly.

---

## Minimal skeleton pipeline (believed correct for pipecat-ai 1.12.0)

```python
import asyncio
import tempfile

import numpy as np
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TranscriptionFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair, LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.utils.time import time_now_iso8601
from pipecat.workers.runner import WorkerRunner

from parakeet_mlx import from_pretrained as load_stt_model
from mlx_audio.tts.utils import load_model as load_tts_model


class ParakeetSTTService(SegmentedSTTService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # wants_wav_segments defaults True — keep it
        self._model = load_stt_model("mlx-community/parakeet-tdt-0.6b-v3")

    async def run_stt(self, audio: bytes):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(audio)
            f.flush()
            result = await asyncio.to_thread(self._model.transcribe, f.name)
        if result.text.strip():
            yield TranscriptionFrame(result.text, self._user_id, time_now_iso8601(), None)


class KokoroMLXTTSService(TTSService):
    def __init__(self, **kwargs):
        super().__init__(push_start_frame=True, push_stop_frames=True, sample_rate=24000, **kwargs)
        self._model = load_tts_model("mlx-community/Kokoro-82M-bf16")

    async def run_tts(self, text: str, context_id: str):
        for result in self._model.generate(text=text, voice="af_heart", speed=1.0, lang_code="a"):
            audio_np = np.array(result.audio)  # mx.array -> numpy
            audio_int16 = (audio_np * 32767).astype(np.int16).tobytes()
            yield TTSAudioRawFrame(audio=audio_int16, sample_rate=result.sample_rate,
                                    num_channels=1, context_id=context_id)


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )
    stt = ParakeetSTTService()
    tts = KokoroMLXTTSService()
    llm = OpenAILLMService(
        base_url="http://127.0.0.1:8080/v1",
        api_key="not-needed",
        model="mlx-community/Qwen3.6-35B-A3B-4bit",
        settings=OpenAILLMService.Settings(
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        ),
    )

    context = LLMContext(messages=[
        {"role": "system", "content": "You are a concise voice assistant. Answer in one short sentence."}
    ])
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
        # user_turn_strategies left at default -> LocalSmartTurnAnalyzerV3, on-device
    )

    pipeline = Pipeline([
        transport.input(), stt, user_agg, llm, tts, transport.output(), assistant_agg,
    ])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=24000),
    )
    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    # Requires `mlx_lm.server` running separately:
    #   uv run mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --port 8080
    asyncio.run(main())
```

Notes on the skeleton:
- Untested end-to-end (no live mic session was run in this research pass) — `[unverified]` as a
  whole, though every individual API call is copied from real installed source as cited above.
- Headphones strongly recommended (§6, no AEC).
- `mlx_lm.server` must be started separately before running this script.
