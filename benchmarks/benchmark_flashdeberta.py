"""
Benchmark: DeBERTa attention backends for full GLiNER2 NER inference

Compares end-to-end NER extraction latency between the standard HuggingFace
DebertaV2 backend, FlashDeBERTa, and DisentangledFlash.

Test matrix:
  - Batch sizes: 1, 2, 4, 8
  - Sequence lengths: 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 tokens
  - Backends: standard (AutoModel), FlashDeBERTa, DisentangledFlash

Protocol:
  - Model loaded once per backend (separate processes via subprocess)
  - Configurable warmup and measured iterations (defaults: 3 and 10)
  - Reports mean, median, stdev, speedup, peak memory
  - Welch's t-test for significance (p < 0.05)
  - Peak memory via torch.cuda (GPU) or tracemalloc (CPU)
  - CUDA synchronize before all timing points on GPU

Usage:
  # Full benchmark (runs all backends):
  python benchmarks/benchmark_flashdeberta.py

  # Single backend (useful for debugging):
  python benchmarks/benchmark_flashdeberta.py --backend standard
  python benchmarks/benchmark_flashdeberta.py --backend flash
  python benchmarks/benchmark_flashdeberta.py --backend disentangled_flash

  # Custom settings:
  python benchmarks/benchmark_flashdeberta.py --model fastino/gliner2-base-v1 --dtype fp16 --warmup 10 --measure 30

  # Isolate encoder speed from preprocessing and decoding:
  python benchmarks/benchmark_flashdeberta.py --encoder-only

Environment variables:
  USE_FLASHDEBERTA  — set automatically by the script when running flash backend
"""

import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Any, Dict, Optional, Tuple

import torch

ENTITY_TYPES = ["company", "person", "product", "location", "date"]

# Extend long-context coverage through 2048 * 4 tokens.
SEQUENCE_LENGTHS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
BATCH_SIZES = [1, 2, 4, 8]

# Seed sentence pool — repeated/truncated to reach target token counts.
# Sentences contain diverse entity types for realistic NER workload.
_SEED_SENTENCES = [
    "Apple Inc. CEO Tim Cook announced the launch of the iPhone 15 Pro Max at a special event in Cupertino, California on September 12, 2023.",
    "Google CEO Sundar Pichai unveiled the Pixel 8 smartphone at a press conference in Mountain View.",
    "Microsoft CEO Satya Nadella presented Windows 11 at the Build developer conference in Seattle.",
    "Amazon's Andy Jassy revealed new Echo Show devices at an event in Arlington, Virginia.",
    "Tesla CEO Elon Musk announced record quarterly deliveries of 466,000 vehicles during the Q3 earnings call.",
    "Meta CEO Mark Zuckerberg demonstrated the Quest 3 mixed reality headset at the Connect conference in Menlo Park.",
    "Samsung Electronics President JH Han introduced the Galaxy S24 Ultra at the Unpacked event in San Jose.",
    "Intel CEO Pat Gelsinger announced the Core Ultra processor lineup at the Innovation event in San Jose.",
    "Nvidia CEO Jensen Huang revealed the RTX 5090 graphics card at the GTC conference in San Jose.",
    "Sony Interactive Entertainment CEO Jim Ryan presented the PlayStation 6 roadmap at a Tokyo press event.",
    "Adobe released major AI features for Photoshop at their MAX conference in Los Angeles.",
    "Spotify CEO Daniel Ek launched the audiobook subscription tier at a Stockholm press event.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def resolve_dtype(name: Optional[str], device: torch.device) -> Tuple[str, torch.dtype]:
    """Resolve the requested precision, defaulting to fp16 only on CUDA."""
    resolved = name or ("fp16" if device.type == "cuda" else "fp32")
    if device.type != "cuda" and resolved != "fp32":
        raise RuntimeError(f"{resolved} benchmarking requires a CUDA device")
    if resolved == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 benchmarking requires a CUDA device with bf16 support")
    return resolved, {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[resolved]


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def generate_text_for_token_length(tokenizer, target_tokens: int) -> str:
    """Generate a text that tokenizes to approximately target_tokens tokens.

    Repeats seed sentences until the target is reached, then truncates
    at a word boundary to hit the target token count.
    """
    # Build a long text by repeating seed sentences
    seed = " ".join(_SEED_SENTENCES)
    text = seed
    while len(tokenizer.encode(text)) < target_tokens + 50:
        text = text + " " + seed

    # Binary search for the right word-boundary truncation
    words = text.split()
    lo, hi = 1, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = " ".join(words[:mid])
        n_tok = len(tokenizer.encode(candidate))
        if n_tok <= target_tokens:
            lo = mid
        else:
            hi = mid - 1

    return " ".join(words[:lo])


# ---------------------------------------------------------------------------
# Single-backend benchmark (runs inside one process)
# ---------------------------------------------------------------------------

def run_single_backend(
    model_name: str,
    backend: str,
    n_warmup: int,
    n_measure: int,
    dtype_name: Optional[str],
    architecture: str,
    encoder_only: bool,
) -> Dict[str, Any]:
    """Run benchmark for a single backend. Returns JSON-serializable results."""
    # Keep the legacy environment switch aligned with the explicit API option.
    if backend == "flash":
        os.environ["USE_FLASHDEBERTA"] = "1"
    else:
        os.environ.pop("USE_FLASHDEBERTA", None)

    from gliner2 import AutoExtractor

    print(f"\nLoading model ({backend} backend)...")
    load_kwargs = {
        "attention_backend": "flashdeberta" if backend == "flash" else "standard"
    }
    if architecture != "auto":
        load_kwargs["architecture"] = architecture
    model = AutoExtractor.from_pretrained(model_name, **load_kwargs)
    model.eval()

    # Detect the construction-time backend.
    encoder_class = model.encoder.__class__.__name__
    print(f"  Encoder class: {encoder_class}")
    if backend == "flash" and encoder_class != "FlashDebertaV2Model":
        raise RuntimeError(
            "FlashDeBERTa was requested but did not activate "
            f"(loaded {encoder_class}). Verify that flashdeberta is installed "
            "and the checkpoint uses a DebertaV2 encoder."
        )
    if backend == "standard" and encoder_class == "FlashDebertaV2Model":
        raise RuntimeError("Standard backend unexpectedly loaded FlashDeBERTa")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_dtype, torch_dtype = resolve_dtype(dtype_name, device)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    model = model.to(device=device, dtype=torch_dtype)
    if backend == "disentangled_flash":
        model.enable_disentangled_flash()
        if not getattr(model, "_disentangled_flash_backend", None):
            raise RuntimeError("DisentangledFlash was requested but did not activate")
        encoder_class = (
            f"{encoder_class} -> {model.encoder.encoder.__class__.__name__}"
        )
    print(f"  Device: {device}")
    print(f"  Dtype: {resolved_dtype}")
    print(f"  Architecture: {model.architecture}")
    print(f"  Mode: {'encoder-only' if encoder_only else 'end-to-end'}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  FlashDeBERTa: {package_version('flashdeberta')}")
    print(f"  DisentangledFlash: {package_version('disentangled-flash')}")

    tokenizer = model.processor.tokenizer

    # Pre-generate texts for each target sequence length
    print("  Generating texts for target token lengths...")
    texts_by_seqlen = {}
    for seq_len in SEQUENCE_LENGTHS:
        text = generate_text_for_token_length(tokenizer, seq_len)
        actual_tokens = len(tokenizer.encode(text))
        texts_by_seqlen[seq_len] = text
        print(f"    {seq_len} tokens -> actual {actual_tokens} tokens, {len(text.split())} words")

    results = {}

    for seq_len in SEQUENCE_LENGTHS:
        base_text = texts_by_seqlen[seq_len]
        for bs in BATCH_SIZES:
            texts = [base_text] * bs
            cond = f"seq{seq_len}_bs{bs}"
            encoder_inputs = None
            if encoder_only:
                encoder_inputs = tokenizer(
                    texts,
                    padding=True,
                    return_tensors="pt",
                )
                encoder_inputs = {
                    key: value.to(device) for key, value in encoder_inputs.items()
                }

            def run_iteration():
                if encoder_only:
                    return model.encoder(**encoder_inputs)
                return model.batch_extract_entities(
                    texts, ENTITY_TYPES, batch_size=bs
                )

            # Warmup
            with torch.inference_mode():
                for _ in range(n_warmup):
                    run_iteration()

            # Measure
            timings = []
            # Peak memory tracking
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                mem_before = torch.cuda.memory_allocated(device)
            else:
                tracemalloc.start()

            with torch.inference_mode():
                for _ in range(n_measure):
                    sync(device)
                    t0 = time.perf_counter()
                    run_iteration()
                    sync(device)
                    timings.append(time.perf_counter() - t0)

            if device.type == "cuda":
                peak_mem = torch.cuda.max_memory_allocated(device)
                peak_mem_delta = peak_mem - mem_before
            else:
                _, peak_mem_delta = tracemalloc.get_traced_memory()
                peak_mem = peak_mem_delta  # tracemalloc reports peak from start
                tracemalloc.stop()

            mean_t = statistics.mean(timings)
            med_t = statistics.median(timings)
            std_t = statistics.stdev(timings) if len(timings) > 1 else 0.0

            results[cond] = {
                "timings": timings,
                "mean": mean_t,
                "median": med_t,
                "stdev": std_t,
                "peak_memory_mb": peak_mem / (1024 * 1024),
                "peak_memory_delta_mb": peak_mem_delta / (1024 * 1024),
            }

            print(f"  [{backend}] seq={seq_len:>4} bs={bs}: "
                  f"mean={mean_t*1000:.1f}ms  median={med_t*1000:.1f}ms  "
                  f"stdev={std_t*1000:.1f}ms  "
                  f"peak_mem={peak_mem / (1024 * 1024):.1f}MB")

    return {
        "backend": backend,
        "encoder_class": encoder_class,
        "device": str(device),
        "dtype": resolved_dtype,
        "architecture": model.architecture,
        "mode": "encoder-only" if encoder_only else "end-to-end",
        "torch_version": torch.__version__,
        "flashdeberta_version": package_version("flashdeberta"),
        "disentangled_flash_version": package_version("disentangled-flash"),
        "selected_attention_backend": (
            getattr(model, "_disentangled_flash_backend", None)
            if backend == "disentangled_flash"
            else backend
        ),
        "model_name": model_name,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "conditions": results,
    }


# ---------------------------------------------------------------------------
# Comparison & reporting
# ---------------------------------------------------------------------------

def _welch_ttest(a, b):
    """Welch's t-test (two-sided) without scipy.

    Returns (t_statistic, p_value). Uses the normal approximation for
    the p-value which is accurate for n >= 20; for smaller n it is
    conservative (slightly overestimates p).
    """
    import math

    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0

    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0

    t = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1.0

    # Approximate two-sided p-value using the t-distribution CDF.
    # For large df this converges to the normal; for small df it's
    # a reasonable approximation via the regularized incomplete beta function.
    x = df / (df + t * t)
    # Regularized incomplete beta via continued fraction (Lentz's method)
    a_param, b_param = df / 2.0, 0.5
    p_val = _betainc(a_param, b_param, x)

    return t, p_val


def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b) via continued fraction."""
    import math

    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use the continued fraction expansion (Lentz's algorithm)
    ln_prefix = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1 - x)
    )
    prefix = math.exp(ln_prefix)

    # If x < (a+1)/(a+b+2), use direct CF; otherwise use 1 - I_{1-x}(b, a)
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1.0 - x)

    # Lentz's continued fraction
    EPS = 1e-30
    TINY = 1e-30
    max_iter = 200

    f = TINY
    c = TINY
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < TINY:
        d = TINY
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + numerator / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        f *= c * d

        # Odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + numerator / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < EPS:
            break

    return prefix * f / a


def compare_results(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> None:
    """Print comparison table with speedups and significance tests."""
    for key in ("model_name", "device", "dtype", "architecture", "mode"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(
                f"Cannot compare results with different {key}: "
                f"{baseline.get(key)!r} != {candidate.get(key)!r}"
            )

    baseline_name = baseline["backend"]
    candidate_name = candidate["backend"]

    print(f"\n{'=' * 90}")
    print(f"  {candidate_name} vs {baseline_name} Benchmark Results")
    print(f"{'=' * 90}")
    print(f"  Model:   {baseline['model_name']}")
    print(f"  Device:  {baseline['device']}")
    print(f"  Dtype:   {baseline['dtype']}")
    print(f"  Architecture: {baseline['architecture']}")
    print(f"  Mode:    {baseline['mode']}")
    print(f"  Baseline encoder:  {baseline['encoder_class']}")
    print(f"  Candidate encoder: {candidate['encoder_class']}")
    print(f"  Warmup: {baseline['n_warmup']}  Measured: {baseline['n_measure']}")
    print(f"{'=' * 90}")

    header = (
        f"  {'Condition':<20} "
        f"{'Base mean':>10} {'Base med':>10} "
        f"{'Cand mean':>10} {'Cand med':>10} "
        f"{'Speedup':>9} "
        f"{'p-value':>9} "
        f"{'Base mem':>9} {'Cand mem':>9} {'Mem ratio':>9}"
    )
    print(header)
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} "
          f"{'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9}")

    all_speedups = []
    all_mem_ratios = []
    significant_conditions = 0

    for cond in baseline["conditions"]:
        std = baseline["conditions"][cond]
        fla = candidate["conditions"][cond]

        std_timings = std["timings"]
        fla_timings = fla["timings"]

        speedup_ratio = std["median"] / fla["median"] if fla["median"] > 0 else float("inf")
        _, p_value = _welch_ttest(std_timings, fla_timings)
        significant_conditions += int(p_value < 0.05)

        std_mem = std.get("peak_memory_mb", 0)
        fla_mem = fla.get("peak_memory_mb", 0)
        mem_ratio = std_mem / fla_mem if fla_mem > 0 else float("inf")

        all_speedups.append(speedup_ratio)
        all_mem_ratios.append(mem_ratio)

        print(
            f"  {cond:<20} "
            f"{std['mean']*1000:>9.1f}ms {std['median']*1000:>9.1f}ms "
            f"{fla['mean']*1000:>9.1f}ms {fla['median']*1000:>9.1f}ms "
            f"{speedup_ratio:>8.2f}x "
            f"{p_value:>9.3g} "
            f"{std_mem:>8.1f}M {fla_mem:>8.1f}M {mem_ratio:>8.2f}x"
        )

    # Summary
    print(f"\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'=' * 90}")

    total_conds = len(baseline["conditions"])

    if all_speedups:
        print(f"  Conditions tested: {total_conds}")
        print(f"  Speedup range: {min(all_speedups):.2f}x to {max(all_speedups):.2f}x")
        print(f"  Overall median speedup: {statistics.median(all_speedups):.2f}x")
        print(
            "  Statistically significant differences (p < 0.05): "
            f"{significant_conditions}/{total_conds}"
        )

    if all_mem_ratios:
        print("\n  Peak memory ratio (baseline / candidate, >1x = candidate uses less):")
        print(f"  Memory ratio range: {min(all_mem_ratios):.2f}x to {max(all_mem_ratios):.2f}x")
        print(f"  Overall median memory ratio: {statistics.median(all_mem_ratios):.2f}x")

    regressions = [s for s in all_speedups if s < 0.95]
    if regressions:
        print(f"  WARNING: {len(regressions)} conditions showed regressions")


def run_subprocess_backend(
    backend: str,
    model_name: str,
    n_warmup: int,
    n_measure: int,
    dtype_name: Optional[str],
    architecture: str,
    encoder_only: bool,
    output_file: str,
) -> Optional[Dict[str, Any]]:
    """Run a single backend in a subprocess to ensure clean env."""
    env = os.environ.copy()
    if backend == "flash":
        env["USE_FLASHDEBERTA"] = "1"
    else:
        env.pop("USE_FLASHDEBERTA", None)

    cmd = [
        sys.executable, __file__,
        "--backend", backend,
        "--model", model_name,
        "--warmup", str(n_warmup),
        "--measure", str(n_measure),
        "--architecture", architecture,
        "--output", output_file,
    ]
    if dtype_name is not None:
        cmd.extend(["--dtype", dtype_name])
    if encoder_only:
        cmd.append("--encoder-only")

    print(f"\n--- Running {backend} backend in subprocess ---")
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"ERROR: {backend} backend exited with code {result.returncode}")
        return None

    with open(output_file) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GLiNER2 DeBERTa attention backends"
    )
    parser.add_argument(
        "--backend",
        choices=["standard", "flash", "disentangled_flash", "both", "all"],
        default="all",
        help="Which backend to benchmark (default: all)"
    )
    parser.add_argument(
        "--model", default="fastino/gliner2-base-v1",
        help="Model name or path (default: fastino/gliner2-base-v1)"
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument("--measure", type=int, default=10, help="Measured iterations (default: 10)")
    parser.add_argument(
        "--dtype", choices=["fp32", "fp16", "bf16"], default=None,
        help="Model precision (default: fp16 on CUDA, fp32 otherwise)"
    )
    parser.add_argument(
        "--architecture", choices=["auto", "span", "boundary"], default="auto",
        help="Extractor architecture; must match the checkpoint (default: auto)"
    )
    parser.add_argument(
        "--encoder-only", action="store_true",
        help="Benchmark only the encoder, excluding preprocessing and decoding"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file (used internally for subprocess communication)"
    )
    args = parser.parse_args()

    # Single-backend mode (used by subprocess or direct invocation)
    if args.backend in ("standard", "flash", "disentangled_flash"):
        result = run_single_backend(
            args.model,
            args.backend,
            args.warmup,
            args.measure,
            args.dtype,
            args.architecture,
            args.encoder_only,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f)
            print(f"  Results written to {args.output}")
        else:
            # Pretty-print when run directly
            print(json.dumps(result, indent=2, default=str))
        return

    # Comparison mode — run each backend in a separate subprocess for clean state.
    print("=" * 90)
    print("  GLiNER2 DeBERTa Attention Backend Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Warmup: {args.warmup}  Measured: {args.measure}")
    print(f"  Dtype: {args.dtype or 'auto (fp16 on CUDA)'}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Mode: {'encoder-only' if args.encoder_only else 'end-to-end'}")
    print("=" * 90)

    std_file = "/tmp/flashdeberta_bench_standard.json"
    flash_file = "/tmp/flashdeberta_bench_flash.json"
    disentangled_file = "/tmp/flashdeberta_bench_disentangled.json"

    std_result = run_subprocess_backend(
        "standard",
        args.model,
        args.warmup,
        args.measure,
        args.dtype,
        args.architecture,
        args.encoder_only,
        std_file,
    )
    flash_result = run_subprocess_backend(
        "flash",
        args.model,
        args.warmup,
        args.measure,
        args.dtype,
        args.architecture,
        args.encoder_only,
        flash_file,
    )

    disentangled_result = None
    if args.backend == "all":
        disentangled_result = run_subprocess_backend(
            "disentangled_flash",
            args.model,
            args.warmup,
            args.measure,
            args.dtype,
            args.architecture,
            args.encoder_only,
            disentangled_file,
        )

    required_results = [std_result, flash_result]
    if args.backend == "all":
        required_results.append(disentangled_result)
    if any(result is None for result in required_results):
        print("\nERROR: One or more backends failed. Cannot compare.")
        if std_result is None:
            print("  Standard backend failed.")
        if flash_result is None:
            print("  Flash backend failed. Is the 'flashdeberta' package installed?")
        if args.backend == "all" and disentangled_result is None:
            print(
                "  DisentangledFlash backend failed. Is the "
                "'disentangled-flash' package installed?"
            )
        sys.exit(1)

    compare_results(std_result, flash_result)
    if disentangled_result is not None:
        compare_results(std_result, disentangled_result)
        compare_results(flash_result, disentangled_result)

    # Save combined results
    combined_file = "benchmarks/attention_backend_results.json"
    combined = {"standard": std_result, "flash": flash_result}
    if disentangled_result is not None:
        combined["disentangled_flash"] = disentangled_result
    with open(combined_file, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\n  Full results saved to {combined_file}")


if __name__ == "__main__":
    main()
