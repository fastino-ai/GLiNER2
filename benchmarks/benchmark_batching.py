"""
Benchmark: Batched vs Loop-based Post-Encoder Operations

Measures the impact of batched span representation and fast embedding
extraction on inference latency.

Two measurement modes:
  1. End-to-end: encoder + post-encoder (shows real-world improvement)
  2. Post-encoder only: isolates the optimized code paths

Design:
  - A/B interleaved within the same process (not sequential runs)
  - Model loaded once, alternating baseline and optimized each iteration
  - Baseline = loop-based paths (_extract_embeddings_loop + per-sample compute_span_rep)
  - Optimized = batched paths (_extract_embeddings_fast + compute_span_rep_batched)
  - CUDA synchronize before all timing points on GPU

Test matrix:
  - Batch sizes: 1, 8, 16, 32
  - Input lengths: ~20 words (short), ~80 words (medium), ~200 words (long)
  - Device: CPU, GPU (if available)

Protocol (per CLAUDE.md):
  - 10 warmup iterations (discarded)
  - 30 measured iterations per condition
  - Reports mean, median, stdev per condition
  - Welch's t-test for significance (p < 0.05)
  - Interleaved A/B within same process

Usage:
  python benchmarks/benchmark_batching.py
"""

import time
import statistics
from typing import List, Dict, Tuple

import torch
from scipy import stats

from gliner2 import GLiNER2
from gliner2.processor import PreprocessedBatch
from gliner2.training.trainer import ExtractorCollator


# ---------------------------------------------------------------------------
# Test inputs
# ---------------------------------------------------------------------------

SHORT_TEXTS = [
    "Apple released iPhone 15 in September.",
    "Google unveiled Pixel 8 at a press event.",
    "Microsoft launched Windows 11 in Redmond.",
    "Amazon announced new Echo devices today.",
    "Tesla delivered record vehicles last quarter.",
    "Meta revealed Quest 3 headset specs.",
    "Samsung launched Galaxy S24 phone.",
    "Intel announced new Core Ultra processors.",
    "Nvidia released RTX 5090 graphics card.",
    "Sony showed PlayStation 6 prototype.",
    "Apple hired new AI research director.",
    "Netflix expanded into gaming market.",
    "Twitter rebranded to X platform.",
    "Adobe released Photoshop AI features.",
    "Spotify launched audiobook subscription.",
    "Uber expanded autonomous vehicle fleet.",
    "Airbnb introduced new host features.",
    "Zoom added AI meeting summaries.",
    "Slack released workflow automation tools.",
    "Discord launched activity features.",
    "Reddit went public on NYSE.",
    "Pinterest added shopping features.",
    "LinkedIn launched AI writing tools.",
    "Snapchat released AR glasses prototype.",
    "TikTok expanded e-commerce features.",
    "Oracle acquired cloud startup.",
    "Salesforce integrated AI assistant.",
    "VMware completed Broadcom merger.",
    "Dropbox launched AI-powered search.",
    "Shopify released new commerce tools.",
    "PayPal introduced stablecoin payment.",
    "Square renamed to Block officially.",
]

MEDIUM_TEXTS = [
    (
        "Apple Inc. CEO Tim Cook announced the launch of the iPhone 15 Pro Max "
        "at a special event held at the Steve Jobs Theater in Cupertino, California "
        "on September 12, 2023. The device features a titanium design and starts at $1199."
    ),
    (
        "Google CEO Sundar Pichai unveiled the Pixel 8 smartphone at a press conference "
        "in Mountain View, California. The device features Google's custom Tensor G3 chip "
        "and will be available starting at $699 in multiple colors."
    ),
    (
        "Microsoft CEO Satya Nadella presented Windows 11 24H2 at the Build developer "
        "conference in Seattle. The update includes Copilot integration and improved "
        "performance on ARM-based Surface devices priced from $999."
    ),
    (
        "Amazon's Andy Jassy revealed new Echo Show devices and Ring security cameras "
        "at an event in Arlington, Virginia. The new Echo features a larger display "
        "and improved Alexa capabilities with generative AI support."
    ),
    (
        "Tesla CEO Elon Musk announced record quarterly deliveries of 466,000 vehicles "
        "during the Q3 2023 earnings call. The Model Y remains the best-selling "
        "electric vehicle globally with deliveries across 40 countries."
    ),
    (
        "Meta CEO Mark Zuckerberg demonstrated the Quest 3 mixed reality headset "
        "at the Connect conference in Menlo Park. The device costs $499 and features "
        "improved passthrough cameras and the Snapdragon XR2 Gen 2 processor."
    ),
    (
        "Samsung Electronics President JH Han introduced the Galaxy S24 Ultra smartphone "
        "at the Unpacked event in San Jose. The device features Galaxy AI and a titanium "
        "frame with an improved 200MP camera system starting at $1299."
    ),
    (
        "Intel CEO Pat Gelsinger announced the Core Ultra processor lineup at the "
        "Innovation event in San Jose. The new chips feature an integrated neural "
        "processing unit for AI acceleration and improved battery life in laptops."
    ),
    (
        "Nvidia CEO Jensen Huang revealed the RTX 5090 graphics card at the GTC "
        "conference in San Jose. The GPU features 32GB of GDDR7 memory and delivers "
        "twice the ray tracing performance of the previous generation."
    ),
    (
        "Sony Interactive Entertainment CEO Jim Ryan presented the PlayStation 6 "
        "development roadmap at a Tokyo press event. The next-gen console will feature "
        "custom AMD hardware and backwards compatibility with PS5 games."
    ),
    (
        "Apple's AI research division hired former Google Brain researcher to lead "
        "a new generative AI team in Cupertino. The team will focus on on-device "
        "language models and privacy-preserving machine learning techniques."
    ),
    (
        "Netflix co-CEO Greg Peters announced the expansion of their gaming division "
        "with the acquisition of three independent studios. The streaming platform "
        "now offers over 80 mobile games included with every subscription."
    ),
    (
        "Twitter officially rebranded to X under Elon Musk's direction, unveiling "
        "a new logo and expanded payment features at their San Francisco headquarters. "
        "The platform aims to become an everything app similar to WeChat."
    ),
    (
        "Adobe released major AI-powered features for Photoshop and Illustrator at "
        "their MAX conference in Los Angeles. Generative Fill and Generative Expand "
        "use the Firefly model trained exclusively on licensed content."
    ),
    (
        "Spotify CEO Daniel Ek launched the audiobook subscription tier at a Stockholm "
        "press event. Premium subscribers now get 15 hours of audiobook listening per "
        "month from a catalog of over 200,000 titles."
    ),
    (
        "Uber CEO Dara Khosrowshahi announced the expansion of the autonomous vehicle "
        "partnership with Waymo in Phoenix, Arizona. The ride-hailing giant plans to "
        "deploy driverless vehicles in additional markets by mid-2024."
    ),
] * 2  # 32 texts

LONG_TEXTS = [
    (
        "Apple Inc., the multinational technology company headquartered in Cupertino, "
        "California, held its annual fall product launch event at the Steve Jobs Theater "
        "on September 12, 2023. CEO Tim Cook took the stage to announce several major "
        "product updates. The highlight was the iPhone 15 Pro Max, featuring a titanium "
        "chassis, the new A17 Pro chip manufactured on TSMC's 3nm process, and an "
        "improved 48MP camera system with 5x optical zoom. The device starts at $1199 "
        "and is available in four colors: Natural Titanium, Blue Titanium, White "
        "Titanium, and Black Titanium. Cook also announced the Apple Watch Series 9 "
        "with a new S9 chip enabling on-device Siri processing and a Double Tap gesture "
        "feature. The watch starts at $399 for the aluminum model. Additionally, the "
        "company revealed AirPods Pro 2nd generation with USB-C charging and improved "
        "adaptive audio features. Apple's senior VP of hardware engineering, John Ternus, "
        "presented the technical details of the new devices."
    ),
    (
        "Google CEO Sundar Pichai opened the annual I/O developer conference at the "
        "Shoreline Amphitheatre in Mountain View, California with a series of major "
        "announcements. The company unveiled PaLM 2, its latest large language model "
        "that powers over 25 Google products including Bard, Search, and Workspace. "
        "Pichai demonstrated new AI features in Google Maps, Photos, and Gmail that "
        "leverage generative AI capabilities. The Pixel 8 Pro smartphone was teased "
        "with its Tensor G3 chip designed in collaboration with Samsung's semiconductor "
        "division. Google Cloud CEO Thomas Kurian announced new enterprise AI services "
        "including Duet AI for Google Workspace and new security features powered by "
        "the Sec-PaLM model. The conference also featured Android 14 preview with "
        "improved privacy controls and satellite connectivity support. Google DeepMind "
        "CEO Demis Hassabis presented Gemini, the company's next-generation multimodal "
        "AI model that combines text, image, and code understanding capabilities."
    ),
    (
        "Microsoft CEO Satya Nadella delivered the keynote address at the Build 2023 "
        "developer conference held at the Seattle Convention Center. The presentation "
        "focused heavily on the company's AI strategy, with Nadella announcing the "
        "integration of Copilot across the entire Microsoft 365 suite. The company "
        "demonstrated how GitHub Copilot X uses GPT-4 to assist developers with code "
        "generation, documentation, and pull request reviews. Azure AI services "
        "received major updates including Azure OpenAI Service general availability "
        "with GPT-4 and DALL-E 3 support. Windows 11 received a major update with "
        "Copilot integration in the taskbar and improved snap layouts. Microsoft Teams "
        "added AI-powered meeting recaps, action items, and intelligent speaker "
        "attribution. The Xbox division announced new cloud gaming features and "
        "expanded Game Pass to additional markets. Chief Technology Officer Kevin Scott "
        "outlined the company's responsible AI framework and new tools for detecting "
        "AI-generated content. The Surface team previewed the next generation Surface "
        "Pro powered by custom ARM processors developed with Qualcomm."
    ),
] * 11  # 33 texts

ENTITY_TYPES = ["company", "person", "product", "location", "date"]

N_WARMUP = 10
N_MEASURE = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sync(device):
    """Synchronize GPU if needed."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_batch(model, texts: List[str], entity_types: List[str]) -> PreprocessedBatch:
    """Create a preprocessed batch."""
    collator = ExtractorCollator(model.processor)
    schema_obj = model.create_schema()
    schema_obj.entities(entity_types)
    schema = schema_obj.build()
    return collator([(t, schema) for t in texts])


def pad_texts(texts: List[str], bs: int) -> List[str]:
    """Ensure we have exactly bs texts."""
    if len(texts) >= bs:
        return texts[:bs]
    return (texts * ((bs // len(texts)) + 1))[:bs]


# ---------------------------------------------------------------------------
# Timing functions
# ---------------------------------------------------------------------------

def time_e2e_baseline(model, batch_dev, device):
    """End-to-end baseline: encoder + loop embedding extraction + per-sample span rep."""
    sync(device)
    t0 = time.perf_counter()

    outputs = model.encoder(input_ids=batch_dev.input_ids, attention_mask=batch_dev.attention_mask)
    token_embs = outputs.last_hidden_state

    all_tok, all_sch = model.processor._extract_embeddings_loop(
        token_embs, batch_dev.input_ids, batch_dev
    )

    for i in range(len(batch_dev)):
        if any(t != "classifications" for t in batch_dev.task_types[i]) and all_tok[i].numel() > 0:
            model.compute_span_rep(all_tok[i])

    sync(device)
    return time.perf_counter() - t0


def time_e2e_optimized(model, batch_dev, device):
    """End-to-end optimized: encoder + fast embedding extraction + batched span rep."""
    sync(device)
    t0 = time.perf_counter()

    outputs = model.encoder(input_ids=batch_dev.input_ids, attention_mask=batch_dev.attention_mask)
    token_embs = outputs.last_hidden_state

    all_tok, all_sch = model.processor._extract_embeddings_fast(token_embs, batch_dev)

    span_embs = [
        all_tok[i] for i in range(len(batch_dev))
        if any(t != "classifications" for t in batch_dev.task_types[i]) and all_tok[i].numel() > 0
    ]
    if span_embs:
        model.compute_span_rep_batched(span_embs)

    sync(device)
    return time.perf_counter() - t0


def time_post_baseline(model, token_embs, batch_dev, device):
    """Post-encoder only baseline: loop embedding + per-sample span rep."""
    sync(device)
    t0 = time.perf_counter()

    all_tok, all_sch = model.processor._extract_embeddings_loop(
        token_embs, batch_dev.input_ids, batch_dev
    )

    for i in range(len(batch_dev)):
        if any(t != "classifications" for t in batch_dev.task_types[i]) and all_tok[i].numel() > 0:
            model.compute_span_rep(all_tok[i])

    sync(device)
    return time.perf_counter() - t0


def time_post_optimized(model, token_embs, batch_dev, device):
    """Post-encoder only optimized: fast embedding + batched span rep."""
    sync(device)
    t0 = time.perf_counter()

    all_tok, all_sch = model.processor._extract_embeddings_fast(token_embs, batch_dev)

    span_embs = [
        all_tok[i] for i in range(len(batch_dev))
        if any(t != "classifications" for t in batch_dev.task_types[i]) and all_tok[i].numel() > 0
    ]
    if span_embs:
        model.compute_span_rep_batched(span_embs)

    sync(device)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def run_condition(model, batch, device, n_warmup, n_measure):
    """Run one condition: interleaved A/B for both e2e and post-encoder."""
    batch_dev = batch.to(device)

    # Pre-compute encoder output once for post-encoder measurements
    with torch.no_grad():
        sync(device)
        enc_out = model.encoder(input_ids=batch_dev.input_ids, attention_mask=batch_dev.attention_mask)
        token_embs = enc_out.last_hidden_state
        sync(device)

    e2e_base, e2e_opt = [], []
    post_base, post_opt = [], []

    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            time_e2e_baseline(model, batch_dev, device)
            time_e2e_optimized(model, batch_dev, device)
            time_post_baseline(model, token_embs, batch_dev, device)
            time_post_optimized(model, token_embs, batch_dev, device)

        # Measured — interleaved A/B
        for _ in range(n_measure):
            e2e_base.append(time_e2e_baseline(model, batch_dev, device))
            e2e_opt.append(time_e2e_optimized(model, batch_dev, device))
            post_base.append(time_post_baseline(model, token_embs, batch_dev, device))
            post_opt.append(time_post_optimized(model, token_embs, batch_dev, device))

    return e2e_base, e2e_opt, post_base, post_opt


def compute_stats(baseline, optimized):
    """Compute summary statistics and significance test."""
    b_mean = statistics.mean(baseline)
    b_med = statistics.median(baseline)
    b_std = statistics.stdev(baseline)
    o_mean = statistics.mean(optimized)
    o_med = statistics.median(optimized)
    o_std = statistics.stdev(optimized)

    sp_mean = (b_mean - o_mean) / b_mean * 100 if b_mean > 0 else 0
    sp_med = (b_med - o_med) / b_med * 100 if b_med > 0 else 0

    t_stat, p_val = stats.ttest_ind(baseline, optimized, equal_var=False)
    sig = p_val < 0.05

    return {
        "b_mean": b_mean, "b_med": b_med, "b_std": b_std,
        "o_mean": o_mean, "o_med": o_med, "o_std": o_std,
        "sp_mean": sp_mean, "sp_med": sp_med,
        "t_stat": t_stat, "p_val": p_val, "sig": sig,
    }


def fmt_ms(s):
    return f"{s*1000:.1f}"


def print_stats(label, st):
    sig_mark = "*" if st["sig"] else "(ns)"
    print(f"  {label}")
    print(f"    Baseline:  mean={fmt_ms(st['b_mean'])}ms  median={fmt_ms(st['b_med'])}ms  stdev={fmt_ms(st['b_std'])}ms")
    print(f"    Optimized: mean={fmt_ms(st['o_mean'])}ms  median={fmt_ms(st['o_med'])}ms  stdev={fmt_ms(st['o_std'])}ms")
    print(f"    Speedup:   mean={st['sp_mean']:+.1f}%  median={st['sp_med']:+.1f}%  p={st['p_val']:.4f} {sig_mark}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 74)
    print("  Batching Optimization Benchmark")
    print("  n_warmup=%d  n_measure=%d  interleaved A/B" % (N_WARMUP, N_MEASURE))
    print("=" * 74)

    print("\nLoading model...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    model.eval()

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    text_configs = [
        ("short (~20w)", SHORT_TEXTS),
        ("medium (~80w)", MEDIUM_TEXTS),
        ("long (~200w)", LONG_TEXTS),
    ]
    batch_sizes = [1, 8, 16, 32]

    all_e2e = []
    all_post = []

    for device in devices:
        if device.type == "cuda":
            model = model.to(device)

        print(f"\n{'='*74}")
        print(f"  Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
        print(f"{'='*74}")

        for text_label, all_texts in text_configs:
            print(f"\n  --- {text_label} ---")

            for bs in batch_sizes:
                texts = pad_texts(all_texts, bs)
                batch = make_batch(model, texts, ENTITY_TYPES)
                cond = f"{device}, {text_label}, bs={bs}"

                e2e_b, e2e_o, post_b, post_o = run_condition(
                    model, batch, device, N_WARMUP, N_MEASURE
                )

                e2e_st = compute_stats(e2e_b, e2e_o)
                post_st = compute_stats(post_b, post_o)

                print()
                print_stats(f"[E2E]  {cond}", e2e_st)
                print_stats(f"[POST] {cond}", post_st)

                all_e2e.append({"cond": cond, **e2e_st})
                all_post.append({"cond": cond, **post_st})

        if device.type == "cuda":
            model = model.to("cpu")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*74}")
    print("  SUMMARY")
    print(f"{'='*74}")

    for tag, results in [("End-to-end", all_e2e), ("Post-encoder only", all_post)]:
        sig = [r for r in results if r["sig"]]
        print(f"\n  {tag}:")
        if sig:
            speedups = [r["sp_med"] for r in sig]
            print(f"    Significant: {len(sig)}/{len(results)} conditions")
            print(f"    Median speedup range: {min(speedups):+.1f}% to {max(speedups):+.1f}%")
            print(f"    Overall median: {statistics.median(speedups):+.1f}%")
        else:
            print(f"    No significant improvements ({len(results)} conditions tested)")

        # Regressions
        reg = [r for r in results if r["sp_med"] < -5 and r["sig"]]
        if reg:
            print(f"    WARNING regressions:")
            for r in reg:
                print(f"      {r['cond']}: {r['sp_med']:+.1f}%")

    # Table summary
    print(f"\n{'='*74}")
    print("  DETAILED TABLE (median speedup %)")
    print(f"{'='*74}")
    print(f"  {'Condition':<48} {'E2E':>8} {'Post':>8} {'E2E sig':>8} {'Post sig':>8}")
    print(f"  {'-'*48} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for e, p in zip(all_e2e, all_post):
        e_sig = "*" if e["sig"] else ""
        p_sig = "*" if p["sig"] else ""
        print(f"  {e['cond']:<48} {e['sp_med']:>+7.1f}% {p['sp_med']:>+7.1f}% {e_sig:>8} {p_sig:>8}")


if __name__ == "__main__":
    main()
