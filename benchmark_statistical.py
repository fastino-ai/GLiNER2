"""
Statistical benchmark with confidence intervals and p-values.

Micro-benchmarks: interleaved old/new in same process → paired t-test.
End-to-end: saves raw timings to JSON for cross-process Welch's t-test.

Usage:
  # Baseline
  git stash
  python benchmark_statistical.py --tag baseline --n 300
  git stash pop

  # Optimized
  python benchmark_statistical.py --tag optimized --n 300

  # Compare
  python benchmark_statistical.py --compare baseline optimized
"""

import argparse
import json
import math
import random
import time
import statistics
from collections import OrderedDict

import torch
from scipy import stats as sp_stats

# ─── Helpers ──────────────────────────────────────────────────────


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ci95(data):
    """95% CI half-width using t-distribution."""
    n = len(data)
    if n < 2:
        return 0.0
    se = statistics.stdev(data) / math.sqrt(n)
    t_crit = sp_stats.t.ppf(0.975, df=n - 1)
    return t_crit * se


def collect(fn, n_warmup, n_iter):
    """Run fn with warmup, return list of times in ms."""
    for _ in range(n_warmup):
        fn()
    sync()
    times = []
    for _ in range(n_iter):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def paired_test(old_times, new_times):
    """Paired t-test on matched samples. Returns (t_stat, p_value, mean_diff, ci95_diff)."""
    diffs = [o - n for o, n in zip(old_times, new_times)]
    n = len(diffs)
    mean_d = statistics.mean(diffs)
    se_d = statistics.stdev(diffs) / math.sqrt(n)
    t_stat = mean_d / se_d if se_d > 0 else 0
    p_val = 2 * sp_stats.t.sf(abs(t_stat), df=n - 1)
    hw = ci95(diffs)
    return t_stat, p_val, mean_d, hw


def welch_test(a, b):
    """Welch's t-test (unequal variance). Returns (t_stat, p_value)."""
    t_stat, p_val = sp_stats.ttest_ind(a, b, equal_var=False)
    return t_stat, p_val


def fmt_p(p):
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


# ─── End-to-end benchmark ────────────────────────────────────────


def run_e2e(n_iter, n_warmup):
    """Run end-to-end scenarios, return dict of {name: [times]}."""
    from gliner2 import GLiNER2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    model = model.to(device)
    model.eval()

    text1 = "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino on September 12, 2023."
    ents = ["company", "person", "product", "location", "date"]
    texts8 = [
        "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino.",
        "Google's Sundar Pichai spoke at the conference in Mountain View.",
        "Microsoft released Windows 11 in Redmond last year.",
        "Amazon founder Jeff Bezos invested in Blue Origin in Seattle.",
        "Tesla CEO Elon Musk unveiled the Cybertruck at the Fremont factory.",
        "Meta's Mark Zuckerberg presented Quest 3 in Menlo Park.",
        "NVIDIA's Jensen Huang showcased the H100 GPU at GTC in San Jose.",
        "OpenAI CEO Sam Altman launched GPT-4 in San Francisco.",
    ]
    long_text = (
        "Apple Inc., headquartered in Cupertino, California, is a multinational technology company "
        "founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. The company designs, "
        "develops, and sells consumer electronics, computer software, and online services. Tim Cook "
        "has served as CEO since August 2011. Apple's main products include the iPhone, iPad, Mac, "
        "Apple Watch, and AirPods. The company also operates services including the App Store, "
        "Apple Music, iCloud, and Apple TV Plus. In 2023, Apple reported annual revenue of $383 "
        "billion, making it the world's largest technology company by revenue. The company employs "
        "over 160,000 people worldwide."
    )
    ents6 = ["company", "person", "product", "location", "date", "monetary_value"]
    text_struct = (
        "John Smith, aged 35, is a software engineer at Google in Mountain View."
    )
    schema_struct = model.create_schema()
    schema_struct.structure("person").field("name").field("age").field(
        "job_title"
    ).field("company").field("location")
    text_rel = "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino on September 12."
    rels = ["CEO_of", "located_in", "announced_on"]

    results = OrderedDict()
    scenarios = [
        ("single_entity", lambda: model.extract_entities(text1, ents)),
        ("single_structure", lambda: model.extract(text_struct, schema_struct)),
        ("single_relation", lambda: model.extract_relations(text_rel, rels)),
        (
            "batch8_entity",
            lambda: model.batch_extract_entities(texts8, ents, batch_size=8),
        ),
        ("long_text_entity", lambda: model.extract_entities(long_text, ents6)),
    ]

    for name, fn in scenarios:
        print(f"  Running {name} (n={n_iter})...", end=" ", flush=True)
        times = collect(fn, n_warmup, n_iter)
        results[name] = times
        m, hw = statistics.mean(times), ci95(times)
        print(f"{m:.2f} ± {hw:.2f} ms")

    return results


# ─── Micro-benchmarks (interleaved old/new) ──────────────────────


def run_micro(n_iter, n_warmup):
    """Run micro-benchmarks with interleaved old/new for paired comparison."""
    import copy
    from gliner2 import GLiNER2
    from gliner2.training.trainer import ExtractorCollator
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    model = model.to(device)
    model.eval()
    tokenizer = model.processor.tokenizer

    results = OrderedDict()

    # --- OPT-1: Token ID lookup ---
    special_set_str = {"[P]", "[C]", "[E]", "[R]", "[L]"}
    special_ids = frozenset(tokenizer.convert_tokens_to_ids(t) for t in special_set_str)
    dummy_ids = list(range(200))

    def opt1_old():
        for tid in dummy_ids:
            tok = tokenizer.convert_ids_to_tokens(tid)
            _ = tok in special_set_str

    def opt1_new():
        for tid in dummy_ids:
            _ = tid in special_ids

    print("  OPT-1 Token ID lookup...", end=" ", flush=True)
    old_t, new_t = _interleaved(opt1_old, opt1_new, n_warmup, n_iter)
    results["OPT-1 Token ID lookup"] = {"old": old_t, "new": new_t}
    _print_paired(old_t, new_t)

    # --- OPT-3: Avoid retokenization ---
    test_text = "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino on September 12."
    dummy_map = list(range(15))

    def opt3_old():
        return len(model.processor._tokenize_text(test_text))

    def opt3_new():
        return len(dummy_map)

    print("  OPT-3 Avoid retokenization...", end=" ", flush=True)
    old_t, new_t = _interleaved(opt3_old, opt3_new, n_warmup, n_iter)
    results["OPT-3 Avoid retokenization"] = {"old": old_t, "new": new_t}
    _print_paired(old_t, new_t)

    # --- OPT-4: Deepcopy ---
    schema_dict = {
        "json_structures": [{"person": {"name": "", "age": "", "job": ""}}],
        "entities": {"company": "", "location": ""},
        "relations": [],
        "classifications": [],
    }
    record = {
        "text": "Apple CEO Tim Cook announced iPhone 15." * 3,
        "schema": schema_dict,
    }

    def opt4_old():
        return copy.deepcopy(record)

    def opt4_new():
        return {"text": record["text"], "schema": copy.deepcopy(record["schema"])}

    print("  OPT-4 Deepcopy...", end=" ", flush=True)
    old_t, new_t = _interleaved(opt4_old, opt4_new, n_warmup, n_iter)
    results["OPT-4 Deepcopy"] = {"old": old_t, "new": new_t}
    _print_paired(old_t, new_t)

    # --- OPT-6: Token cache ---
    special_tokens = [
        "[SEP_STRUCT]",
        "[SEP_TEXT]",
        "[P]",
        "[C]",
        "[E]",
        "[R]",
        "[L]",
        "[EXAMPLE]",
        "[OUTPUT]",
        "[DESCRIPTION]",
        "(",
        ")",
        ",",
        "|",
    ]
    cache = {tok: tokenizer.tokenize(tok) for tok in special_tokens}
    test_tokens = special_tokens * 10

    def opt6_old():
        for tok in test_tokens:
            tokenizer.tokenize(tok)

    def opt6_new():
        for tok in test_tokens:
            if tok in cache:
                _ = cache[tok]
            else:
                tokenizer.tokenize(tok)

    print("  OPT-6 Token cache...", end=" ", flush=True)
    old_t, new_t = _interleaved(opt6_old, opt6_new, n_warmup, n_iter)
    results["OPT-6 Token cache"] = {"old": old_t, "new": new_t}
    _print_paired(old_t, new_t)

    # --- OPT-12: Skip DataLoader ---
    collator = ExtractorCollator(model.processor, is_training=False)
    text_norm = "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino on September 12, 2023."
    schema_e = model.create_schema().entities(
        ["company", "person", "product", "location", "date"]
    )
    sd = schema_e.build()
    for c in sd.get("classifications", []):
        c.setdefault("true_label", ["N/A"])
    small_dataset = [(text_norm, sd)]

    def opt12_old():
        loader = DataLoader(
            small_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
        return list(loader)

    def opt12_new():
        return [collator(small_dataset)]

    print("  OPT-12 Skip DataLoader...", end=" ", flush=True)
    old_t, new_t = _interleaved(opt12_old, opt12_new, n_warmup, n_iter)
    results["OPT-12 Skip DataLoader"] = {"old": old_t, "new": new_t}
    _print_paired(old_t, new_t)

    return results


def _interleaved(old_fn, new_fn, n_warmup, n_iter):
    """Run old/new interleaved to eliminate ordering effects. Returns paired lists."""
    # Warmup both
    for _ in range(n_warmup):
        old_fn()
        new_fn()
    sync()

    old_times = []
    new_times = []
    for _ in range(n_iter):
        # Randomize order each iteration to eliminate systematic bias
        if random.random() < 0.5:
            sync()
            t0 = time.perf_counter()
            old_fn()
            sync()
            old_times.append((time.perf_counter() - t0) * 1000)
            sync()
            t0 = time.perf_counter()
            new_fn()
            sync()
            new_times.append((time.perf_counter() - t0) * 1000)
        else:
            sync()
            t0 = time.perf_counter()
            new_fn()
            sync()
            new_times.append((time.perf_counter() - t0) * 1000)
            sync()
            t0 = time.perf_counter()
            old_fn()
            sync()
            old_times.append((time.perf_counter() - t0) * 1000)

    return old_times, new_times


def _print_paired(old_t, new_t):
    m_old, m_new = statistics.mean(old_t), statistics.mean(new_t)
    t_stat, p_val, mean_diff, hw = paired_test(old_t, new_t)
    speedup = m_old / m_new if m_new > 0 else float("inf")
    print(
        f"{m_old:.4f} -> {m_new:.4f} ms  ({speedup:.1f}x)  "
        f"diff={mean_diff:.4f}±{hw:.4f}ms  p={fmt_p(p_val)}"
    )


# ─── Compare mode ────────────────────────────────────────────────


def compare(baseline_path, optimized_path):
    """Compare two end-to-end result files with Welch's t-test."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(optimized_path) as f:
        optimized = json.load(f)

    print(
        f"\nBaseline:  {baseline_path}  (device={baseline['device']}, n={baseline.get('n', '?')})"
    )
    print(
        f"Optimized: {optimized_path}  (device={optimized['device']}, n={optimized.get('n', '?')})"
    )

    print(
        f"\n{'Scenario':<25} {'Baseline':>18} {'Optimized':>18} {'Diff':>14} {'Speedup':>8} {'p-value':>10}"
    )
    print("=" * 100)

    for name in baseline["e2e"]:
        b = baseline["e2e"][name]
        o = optimized["e2e"][name]

        m_b, ci_b = statistics.mean(b), ci95(b)
        m_o, ci_o = statistics.mean(o), ci95(o)
        diff = m_b - m_o
        diff_ci = math.sqrt(ci_b**2 + ci_o**2)  # approximate CI of difference
        speedup = m_b / m_o if m_o > 0 else float("inf")
        t_stat, p_val = welch_test(b, o)

        sig = "*" if p_val < 0.05 else " "
        if p_val < 0.01:
            sig = "**"
        if p_val < 0.001:
            sig = "***"

        print(
            f"{name:<25} {m_b:>7.2f}±{ci_b:>5.2f}ms  {m_o:>7.2f}±{ci_o:>5.2f}ms  "
            f"{diff:>+6.2f}±{diff_ci:>4.2f}ms  {speedup:>7.3f}x  {fmt_p(p_val):>9}{sig}"
        )

    # Micro-benchmarks (if present in optimized)
    if "micro" in optimized:
        print(
            f"\n{'Component':<30} {'Old':>16} {'New':>16} {'Diff (paired)':>18} {'Speedup':>8} {'p-value':>10}"
        )
        print("=" * 105)

        for name, data in optimized["micro"].items():
            old_t = data["old"]
            new_t = data["new"]
            m_old, ci_old = statistics.mean(old_t), ci95(old_t)
            m_new, ci_new = statistics.mean(new_t), ci95(new_t)
            t_stat, p_val, mean_diff, hw = paired_test(old_t, new_t)
            speedup = m_old / m_new if m_new > 0 else float("inf")

            sig = "*" if p_val < 0.05 else " "
            if p_val < 0.01:
                sig = "**"
            if p_val < 0.001:
                sig = "***"

            print(
                f"{name:<30} {m_old:>6.4f}±{ci_old:>6.4f}ms  {m_new:>6.4f}±{ci_new:>6.4f}ms  "
                f"{mean_diff:>+7.4f}±{hw:>6.4f}ms  {speedup:>7.1f}x  {fmt_p(p_val):>9}{sig}"
            )


# ─── Main ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Tag for this run (baseline or optimized)")
    parser.add_argument("--n", type=int, default=300, help="Iterations per scenario")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "OPTIMIZED"),
        help="Compare two result files",
    )
    args = parser.parse_args()

    if args.compare:
        compare(
            f"bench_stats_{args.compare[0]}.json", f"bench_stats_{args.compare[1]}.json"
        )
        return

    if not args.tag:
        parser.error("--tag is required (or use --compare)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Iterations: {args.n}, Warmup: {args.warmup}\n")

    output = {"tag": args.tag, "device": device, "n": args.n}

    # End-to-end
    print("END-TO-END BENCHMARKS")
    print("-" * 60)
    e2e = run_e2e(args.n, args.warmup)
    output["e2e"] = e2e

    # Micro-benchmarks (only meaningful for optimized run since we inline both versions)
    print("\nCOMPONENT MICRO-BENCHMARKS (interleaved old/new)")
    print("-" * 60)
    micro = run_micro(args.n, args.warmup)
    output["micro"] = {k: v for k, v in micro.items()}

    out_path = f"bench_stats_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"\nRaw timings saved to {out_path}")


if __name__ == "__main__":
    main()
