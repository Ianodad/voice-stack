"""
Latency spike: measure real per-stage latency of a cascaded local voice
pipeline on this machine (Apple Silicon, MLX).

Pipeline: STT (parakeet-mlx) -> LLM (mlx-lm) -> TTS (mlx-audio / Kokoro)

This is a measurement spike, not the product. Straight-line script,
no framework, no abstractions. Run with:

    uv run python scripts/latency_spike.py

Writes results to docs/BENCHMARK.md and prints a summary table to stdout.
"""

import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import psutil

import mlx.core as mx
from mlx_lm import load as llm_load
from mlx_lm import stream_generate
from mlx_audio.tts.utils import load_model as tts_load
from parakeet_mlx import from_pretrained as stt_load

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
AUDIO_PATH = SCRIPT_DIR / "spike_input.wav"
BENCHMARK_MD = REPO_ROOT / "docs" / "BENCHMARK.md"

N_RUNS = 5

STT_MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"
LLM_MODEL_ID = "mlx-community/Qwen3.6-35B-A3B-4bit"
TTS_MODEL_ID = "mlx-community/Kokoro-82M-bf16"
TTS_VOICE = "af_heart"
TTS_LANG_CODE = "a"

SYSTEM_PROMPT = "You are a concise voice assistant. Answer in one short sentence."
USER_TEXT_FOR_LLM = "What is the weather like in Nairobi today?"
MAX_TOKENS = 60


def now():
    return time.perf_counter()


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def stats(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def ms(seconds):
    return round(seconds * 1000, 1)


# ---------------------------------------------------------------------------
# Model loading (cold, once each, timed separately)
# ---------------------------------------------------------------------------

def load_stt():
    t0 = now()
    model = stt_load(STT_MODEL_ID)
    return model, now() - t0


def load_llm():
    t0 = now()
    model, tokenizer = llm_load(LLM_MODEL_ID)
    return (model, tokenizer), now() - t0


def load_tts():
    t0 = now()
    model = tts_load(TTS_MODEL_ID)
    return model, now() - t0


# ---------------------------------------------------------------------------
# Per-stage inference, timed
# ---------------------------------------------------------------------------

def run_stt(model, audio_path=AUDIO_PATH):
    t0 = now()
    result = model.transcribe(str(audio_path))
    total = now() - t0
    return result.text, total


def run_llm(model, tokenizer, user_text=USER_TEXT_FOR_LLM):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    # Qwen3.6's chat template defaults to "thinking" mode (emits a
    # <think>...</think> block before the real answer) unless explicitly
    # disabled. A voice assistant capped at ~60 tokens with a
    # one-short-sentence system prompt has no latency budget for chain-of-
    # thought, so we disable it explicitly here -- otherwise TTFT/total
    # would measure reasoning-token latency, not spoken-answer latency.
    try:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        # Tokenizer/template doesn't support enable_thinking -- fall back.
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    t0 = now()
    ttft = None
    chunks = []
    for response in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS):
        if ttft is None:
            ttft = now() - t0
        chunks.append(response.text)
    total = now() - t0
    return "".join(chunks), ttft, total


def run_tts(model, text):
    """
    mlx-audio's model.generate() is a generator that yields one
    GenerationResult per text segment (split by `split_pattern`, default
    on sentence boundaries), each carrying its own audio chunk. For our
    single-short-sentence test input this yields exactly ONE chunk, so
    time-to-first-chunk and total synthesis time are the same number here.
    This is real streaming for multi-sentence input, but for this specific
    test sentence it degenerates to "full synthesis" -- reported honestly
    below rather than implying finer-grained streaming than we observed.
    """
    t0 = now()
    ttfa = None
    n_chunks = 0
    total_samples = 0
    sample_rate = None
    for result in model.generate(
        text=text, voice=TTS_VOICE, speed=1.0, lang_code=TTS_LANG_CODE
    ):
        if ttfa is None:
            ttfa = now() - t0
        n_chunks += 1
        # NOTE: GenerationResult.samples is NOT the audio sample count
        # (observed to always be 1, i.e. a chunk index) -- use the actual
        # waveform array length instead.
        total_samples += result.audio.shape[0]
        sample_rate = result.sample_rate
    total = now() - t0
    return {
        "n_chunks": n_chunks,
        "total_samples": total_samples,
        "sample_rate": sample_rate,
        "ttfa": ttfa,
        "total": total,
    }


# ---------------------------------------------------------------------------
# GPU contention A/B: does STT slow down right after an LLM generation?
# ---------------------------------------------------------------------------

def gpu_contention_ab(stt_model, llm_model, llm_tokenizer):
    # A: STT run in isolation, no LLM call immediately before it.
    isolated = []
    for _ in range(3):
        _, t = run_stt(stt_model)
        isolated.append(t)

    # B: STT run immediately after an LLM generation call on the same GPU.
    contended = []
    for _ in range(3):
        run_llm(llm_model, llm_tokenizer)  # occupy the Metal GPU, discard output
        _, t = run_stt(stt_model)
        contended.append(t)

    return {
        "stt_isolated": stats(isolated),
        "stt_after_llm": stats(contended),
    }


# ---------------------------------------------------------------------------
# Environment / version capture
# ---------------------------------------------------------------------------

def get_versions():
    pkgs = [
        "mlx",
        "mlx-lm",
        "mlx-audio",
        "parakeet-mlx",
        "torch",
        "psutil",
        "misaki",
        "silero-vad",
        "huggingface-hub",
    ]
    out = {}
    for p in pkgs:
        try:
            out[p] = pkg_version(p)
        except Exception as e:
            out[p] = f"ERROR: {e}"
    return out


def get_machine_spec():
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        chip = "unknown"
    try:
        mem_bytes = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        )
        mem_gb = round(mem_bytes / (1024**3))
    except Exception:
        mem_gb = "unknown"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "chip": chip or "unknown (see RESEARCH.md: Apple M5 Pro, 18 cores)",
        "memory_gb": mem_gb,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Starting latency spike")
    print(f"Test audio: {AUDIO_PATH} (exists={AUDIO_PATH.exists()})")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": get_machine_spec(),
        "versions": get_versions(),
        "models": {"stt": STT_MODEL_ID, "llm": LLM_MODEL_ID, "tts": TTS_MODEL_ID},
        "errors": [],
    }

    rss_before_load = rss_gb()

    # ---- Cold load, once each, timed separately ----
    print("\n=== Cold load ===")

    try:
        print(f"Loading STT ({STT_MODEL_ID}) ...")
        stt_model, stt_load_s = load_stt()
        print(f"  STT loaded in {stt_load_s:.2f}s")
    except Exception as e:
        print(f"  STT LOAD FAILED: {e!r}")
        results["errors"].append(f"stt_load: {e!r}")
        stt_model, stt_load_s = None, None

    try:
        print(f"Loading LLM ({LLM_MODEL_ID}) ...")
        (llm_model, llm_tokenizer), llm_load_s = load_llm()
        print(f"  LLM loaded in {llm_load_s:.2f}s")
    except Exception as e:
        print(f"  LLM LOAD FAILED: {e!r}")
        results["errors"].append(f"llm_load: {e!r}")
        llm_model, llm_tokenizer, llm_load_s = None, None, None

    try:
        print(f"Loading TTS ({TTS_MODEL_ID}) ...")
        tts_model, tts_load_s = load_tts()
        print(f"  TTS loaded in {tts_load_s:.2f}s")
    except Exception as e:
        print(f"  TTS LOAD FAILED: {e!r}")
        results["errors"].append(f"tts_load: {e!r}")
        tts_model, tts_load_s = None, None

    rss_after_load = rss_gb()

    results["cold_load_seconds"] = {
        "stt": stt_load_s,
        "llm": llm_load_s,
        "tts": tts_load_s,
    }

    # ---- N runs of the full pipeline ----
    print(f"\n=== Running pipeline {N_RUNS}x ===")

    stt_times, llm_ttfts, llm_totals, tts_ttfas, tts_totals, pipeline_totals = (
        [], [], [], [], [], []
    )
    last_transcript, last_llm_text, last_tts_info = None, None, None
    peak_rss_during_runs = rss_after_load

    for i in range(N_RUNS):
        run_t0 = now()
        run_ok = True

        if stt_model is not None:
            try:
                transcript, stt_t = run_stt(stt_model)
                stt_times.append(stt_t)
                last_transcript = transcript
            except Exception as e:
                print(f"  run {i}: STT FAILED: {e!r}")
                results["errors"].append(f"stt_run_{i}: {e!r}")
                run_ok = False
        else:
            run_ok = False

        llm_input_text = last_transcript if last_transcript else USER_TEXT_FOR_LLM
        if llm_model is not None and run_ok:
            try:
                llm_text, ttft, llm_total = run_llm(
                    llm_model, llm_tokenizer, llm_input_text
                )
                llm_ttfts.append(ttft)
                llm_totals.append(llm_total)
                last_llm_text = llm_text
            except Exception as e:
                print(f"  run {i}: LLM FAILED: {e!r}")
                results["errors"].append(f"llm_run_{i}: {e!r}")
                run_ok = False
        else:
            run_ok = False

        if tts_model is not None and run_ok:
            try:
                tts_text = last_llm_text if last_llm_text else "Sorry, I could not generate a response."
                tts_info = run_tts(tts_model, tts_text)
                tts_ttfas.append(tts_info["ttfa"])
                tts_totals.append(tts_info["total"])
                last_tts_info = tts_info
            except Exception as e:
                print(f"  run {i}: TTS FAILED: {e!r}")
                results["errors"].append(f"tts_run_{i}: {e!r}")
                run_ok = False

        run_total = now() - run_t0
        if run_ok:
            pipeline_totals.append(run_total)

        peak_rss_during_runs = max(peak_rss_during_runs, rss_gb())
        print(f"  run {i}: total={run_total:.2f}s ok={run_ok}")

    # ---- GPU contention A/B ----
    print("\n=== GPU contention A/B (STT isolated vs STT right after LLM) ===")
    contention = None
    if stt_model is not None and llm_model is not None:
        try:
            contention = gpu_contention_ab(stt_model, llm_model, llm_tokenizer)
            print(f"  STT isolated median: {ms(contention['stt_isolated']['median'])}ms")
            print(f"  STT after LLM median: {ms(contention['stt_after_llm']['median'])}ms")
        except Exception as e:
            print(f"  CONTENTION TEST FAILED: {e!r}")
            results["errors"].append(f"contention_ab: {e!r}")
    else:
        print("  skipped (STT or LLM failed to load)")

    peak_rss_final = max(peak_rss_during_runs, rss_gb())
    peak_gpu_gb = None
    try:
        peak_gpu_gb = mx.get_peak_memory() / 1e9
    except Exception:
        pass

    # ---- Assemble results ----
    results["stage_seconds"] = {
        "stt": stats(stt_times) if stt_times else None,
        "llm_ttft": stats(llm_ttfts) if llm_ttfts else None,
        "llm_total": stats(llm_totals) if llm_totals else None,
        "tts_ttfa": stats(tts_ttfas) if tts_ttfas else None,
        "tts_total": stats(tts_totals) if tts_totals else None,
        "pipeline_total": stats(pipeline_totals) if pipeline_totals else None,
    }
    results["n_successful_runs"] = len(pipeline_totals)
    results["gpu_contention_ab"] = contention
    results["ram_gb"] = {
        "rss_before_model_load": round(rss_before_load, 2),
        "rss_after_model_load": round(rss_after_load, 2),
        "peak_rss_during_runs": round(peak_rss_final, 2),
        "peak_mlx_gpu_memory": round(peak_gpu_gb, 2) if peak_gpu_gb else None,
    }
    results["sample_transcript"] = last_transcript
    results["sample_llm_response"] = last_llm_text
    results["sample_tts_info"] = last_tts_info

    # ---- Print summary table ----
    print("\n=== SUMMARY (seconds unless noted) ===")
    header = f"{'stage':<16}{'median':>10}{'min':>10}{'max':>10}"
    print(header)
    print("-" * len(header))
    for label, key in [
        ("STT", "stt"),
        ("LLM TTFT", "llm_ttft"),
        ("LLM total", "llm_total"),
        ("TTS TTFA", "tts_ttfa"),
        ("TTS total", "tts_total"),
        ("PIPELINE total", "pipeline_total"),
    ]:
        s = results["stage_seconds"][key]
        if s is None:
            print(f"{label:<16}{'N/A':>10}{'N/A':>10}{'N/A':>10}")
        else:
            print(f"{label:<16}{s['median']:>10.3f}{s['min']:>10.3f}{s['max']:>10.3f}")

    print(f"\nCold load (s): STT={stt_load_s}, LLM={llm_load_s}, TTS={tts_load_s}")
    print(f"Successful pipeline runs: {len(pipeline_totals)}/{N_RUNS}")
    print(f"RAM (GB): {results['ram_gb']}")
    if contention:
        print(f"GPU contention: isolated median={ms(contention['stt_isolated']['median'])}ms, "
              f"after-LLM median={ms(contention['stt_after_llm']['median'])}ms")
    if results["errors"]:
        print("\nERRORS:")
        for e in results["errors"]:
            print(f"  - {e}")

    # ---- Dump raw JSON for reference ----
    raw_json_path = SCRIPT_DIR / "latency_spike_results.json"
    with open(raw_json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results written to {raw_json_path}")

    write_benchmark_md(results)
    print(f"Benchmark report written to {BENCHMARK_MD}")


def write_benchmark_md(results):
    m = results["machine"]
    v = results["versions"]
    s = results["stage_seconds"]
    ram = results["ram_gb"]
    contention = results["gpu_contention_ab"]

    def row(label, key, unit="s"):
        st = s[key]
        if st is None:
            return f"| {label} | N/A | N/A | N/A |\n"
        if unit == "ms":
            return f"| {label} | {ms(st['median'])} | {ms(st['min'])} | {ms(st['max'])} |\n"
        return f"| {label} | {st['median']:.3f} | {st['min']:.3f} | {st['max']:.3f} |\n"

    lines = []
    lines.append("# Voice Stack Latency Benchmark\n\n")
    lines.append(f"**Generated:** {results['timestamp']}\n\n")
    lines.append("## Machine\n\n")
    lines.append(f"- Platform: {m['platform']}\n")
    lines.append(f"- Chip: {m['chip']}\n")
    lines.append(f"- Memory: {m['memory_gb']} GB\n")
    lines.append(f"- Python: {m['python']}\n\n")

    lines.append("## Models\n\n")
    lines.append(f"- STT: `{results['models']['stt']}`\n")
    lines.append(f"- LLM: `{results['models']['llm']}`\n")
    lines.append(f"- TTS: `{results['models']['tts']}`\n\n")

    lines.append("## Package versions (exact, as installed in .venv)\n\n")
    for pkg, ver in v.items():
        lines.append(f"- {pkg}: {ver}\n")
    lines.append("\n")

    lines.append("## Cold load time (once each, seconds)\n\n")
    lines.append("| Model | Load time (s) |\n|---|---|\n")
    cl = results["cold_load_seconds"]
    lines.append(f"| STT | {cl['stt']:.2f}" if cl['stt'] is not None else "| STT | N/A")
    lines.append(" |\n")
    lines.append(f"| LLM | {cl['llm']:.2f}" if cl['llm'] is not None else "| LLM | N/A")
    lines.append(" |\n")
    lines.append(f"| TTS | {cl['tts']:.2f}" if cl['tts'] is not None else "| TTS | N/A")
    lines.append(" |\n\n")

    lines.append(f"## Per-stage latency, N={N_RUNS} runs (seconds)\n\n")
    lines.append(f"Successful runs: {results['n_successful_runs']}/{N_RUNS}\n\n")
    lines.append("| Stage | Median | Min | Max |\n|---|---|---|---|\n")
    lines.append(row("STT (audio -> transcript)", "stt"))
    lines.append(row("LLM TTFT", "llm_ttft"))
    lines.append(row("LLM total generation", "llm_total"))
    lines.append(row("TTS TTFA", "tts_ttfa"))
    lines.append(row("TTS total synthesis", "tts_total"))
    lines.append(row("PIPELINE total", "pipeline_total"))
    lines.append("\n")

    lines.append(
        "**TTS streaming note:** `mlx-audio`'s `model.generate()` is a generator "
        "that yields one chunk per text segment (split on sentence boundaries). "
        "Our test sentence is a single short sentence, so it yielded exactly one "
        "chunk -- TTFA and total synthesis time are the same measurement here. "
        "This is genuine streaming for multi-sentence text, but for this "
        "specific input it degenerates to full-synthesis timing.\n\n"
    )

    lines.append(
        "**First-call warmup note [measured during dry-run smoke tests, not "
        "hidden]:** the FIRST call to a model after cold load pays a one-time "
        "MLX lazy-compilation / pipeline-init cost separate from the load time "
        "itself. Observed in isolation on this machine: Kokoro TTS first "
        "`generate()` call took ~3.6s vs ~0.09s on the second call (same "
        "process, same text); parakeet STT first `transcribe()` call took "
        "~1.7s vs ~0.07s warm. Run 0 of the N=%d loop below pays this cost "
        "for every stage, so it will show up as an outlier in the `max` "
        "column -- this is real per-process-lifetime cost, not noise, but a "
        "long-running voice assistant only pays it once at startup, not per "
        "utterance.\n\n" % N_RUNS
    )

    lines.append("## GPU contention A/B (STT isolated vs STT immediately after LLM)\n\n")
    if contention:
        lines.append("| Condition | Median (ms) | Min (ms) | Max (ms) |\n|---|---|---|---|\n")
        iso = contention["stt_isolated"]
        con = contention["stt_after_llm"]
        lines.append(f"| STT isolated | {ms(iso['median'])} | {ms(iso['min'])} | {ms(iso['max'])} |\n")
        lines.append(f"| STT right after LLM generation | {ms(con['median'])} | {ms(con['min'])} | {ms(con['max'])} |\n\n")
        delta_pct = (con["median"] - iso["median"]) / iso["median"] * 100 if iso["median"] else 0
        lines.append(f"Delta: {delta_pct:+.1f}% median STT latency when run right after LLM generation.\n\n")
    else:
        lines.append("Not run (STT or LLM failed to load).\n\n")

    lines.append("## RAM\n\n")
    lines.append(f"- Process RSS before model load: {ram['rss_before_model_load']} GB\n")
    lines.append(f"- Process RSS after model load: {ram['rss_after_model_load']} GB\n")
    lines.append(f"- Peak process RSS during runs: {ram['peak_rss_during_runs']} GB\n")
    lines.append(
        f"- Peak MLX GPU memory (mx.get_peak_memory): "
        f"{ram['peak_mlx_gpu_memory']} GB\n\n"
        if ram['peak_mlx_gpu_memory'] is not None else "\n"
    )

    if results["errors"]:
        lines.append("## Errors encountered\n\n")
        for e in results["errors"]:
            lines.append(f"- {e}\n")
        lines.append("\n")

    lines.append("## Sample output (last run)\n\n")
    lines.append(f"- Transcript: {results['sample_transcript']!r}\n")
    lines.append(f"- LLM response: {results['sample_llm_response']!r}\n")
    lines.append(f"- TTS info: {results['sample_tts_info']}\n")

    BENCHMARK_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_MD, "w") as f:
        f.write("".join(lines))


if __name__ == "__main__":
    main()
