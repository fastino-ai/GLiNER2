#!/usr/bin/env python3
"""Measure when DisentangledFlash's packed layout beats the padded one.

Usage:
  python benchmarks/benchmark_packed_threshold.py --dtype fp16 --output packed_fp16.json
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time

import torch

from gliner2 import AutoExtractor

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def lengths_for_padding(batch_size: int, max_length: int, padding: float) -> list[int]:
    """Row lengths for a right-padded batch with the given padding fraction."""
    if batch_size == 1:
        return [max_length]
    rows = batch_size - 1
    real_tokens = round((1.0 - padding) * batch_size * max_length)
    remaining = min(max(real_tokens - max_length, rows), rows * max_length)
    base, extra = divmod(remaining, rows)
    lengths = [base + (1 if index < extra else 0) for index in range(rows)]
    for index in range(0, rows - 1, 2):
        shift = min(lengths[index] - 1, max_length - lengths[index + 1]) // 2
        lengths[index] -= shift
        lengths[index + 1] += shift
    return [max_length, *lengths]


def make_batch(lengths: list[int], max_length: int, vocab_size: int, device) -> tuple:
    input_ids = torch.randint(5, vocab_size, (len(lengths), max_length), device=device)
    positions = torch.arange(max_length, device=device)
    attention_mask = (positions[None, :] < torch.tensor(lengths, device=device)[:, None]).long()
    input_ids = input_ids * attention_mask
    return input_ids, attention_mask


def time_encoder(encoder, input_ids, attention_mask, warmup: int, measure: int) -> float:
    cuda = input_ids.device.type == "cuda"
    with torch.inference_mode():
        for _ in range(warmup):
            encoder(input_ids=input_ids, attention_mask=attention_mask)
        if cuda:
            torch.cuda.synchronize()
        samples = []
        for _ in range(measure):
            start = time.perf_counter()
            encoder(input_ids=input_ids, attention_mask=attention_mask)
            if cuda:
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="fastino/gliner2-base-v1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default=None,
                        help="Default: fp16 on CUDA, fp32 otherwise")
    parser.add_argument("--backend", choices=["auto", "torch", "triton"], default="auto")
    parser.add_argument("--tuning-mode", choices=["auto", "heuristic", "autotune"],
                        default="heuristic",
                        help="Triton launch-config selection (default: heuristic)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--paddings", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=20)
    parser.add_argument("--output", default=None, help="Write all measurements as JSON")
    args = parser.parse_args()

    dtype_name = args.dtype or ("fp16" if args.device.startswith("cuda") else "fp32")
    model = AutoExtractor.from_pretrained(
        args.model, map_location=args.device, attention_backend="standard"
    ).eval()
    if dtype_name != "fp32":
        model = model.to(DTYPES[dtype_name])

    padded = copy.deepcopy(model).eval().enable_disentangled_flash(
        backend=args.backend, packed=False, tuning=args.tuning_mode
    )
    packed = copy.deepcopy(model).eval().enable_disentangled_flash(
        backend=args.backend, packed=True, tuning=args.tuning_mode
    )
    del model
    vocab_size = padded.encoder.config.vocab_size
    max_positions = padded.encoder.config.max_position_embeddings

    print(f"model={args.model} device={args.device} dtype={dtype_name} "
          f"backend={padded._disentangled_flash_backend} tuning={args.tuning_mode}")
    if args.device.startswith("cuda"):
        print(f"gpu={torch.cuda.get_device_name()}")
    print(f"{'B':>4} {'L':>5} {'pad':>6} {'padded ms':>10} {'packed ms':>10} {'speedup':>8}")

    results = []
    break_even = {}
    for batch_size in args.batch_sizes:
        for max_length in args.lengths:
            if max_length > max_positions:
                continue
            for padding in sorted(args.paddings):
                lengths = lengths_for_padding(batch_size, max_length, padding)
                actual_padding = 1.0 - sum(lengths) / (batch_size * max_length)
                input_ids, attention_mask = make_batch(
                    lengths, max_length, vocab_size, args.device
                )
                padded_ms = time_encoder(
                    padded.encoder, input_ids, attention_mask, args.warmup, args.measure
                )
                packed_ms = time_encoder(
                    packed.encoder, input_ids, attention_mask, args.warmup, args.measure
                )
                speedup = padded_ms / packed_ms
                results.append({
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "padding": actual_padding,
                    "padded_ms": padded_ms,
                    "packed_ms": packed_ms,
                    "speedup": speedup,
                })
                print(f"{batch_size:>4} {max_length:>5} {actual_padding:>6.2f} "
                      f"{padded_ms:>10.3f} {packed_ms:>10.3f} {speedup:>7.2f}x")
                key = (batch_size, max_length)
                if speedup > 1.0 and key not in break_even:
                    break_even[key] = actual_padding

    print("\nBreak-even padding (smallest measured padding where packed is faster):")
    for batch_size in args.batch_sizes:
        for max_length in args.lengths:
            value = break_even.get((batch_size, max_length))
            label = f"{value:.2f}" if value is not None else "never in range"
            print(f"  B={batch_size:<4} L={max_length:<5} {label}")

    if break_even:
        suggested = max(break_even.values())
        work = {key: key[0] * key[1] ** 2 for key in break_even}
        suggested_work = min(work.values())
        never = sorted(
            (batch_size, max_length)
            for batch_size in args.batch_sizes
            for max_length in args.lengths
            if (batch_size, max_length) not in break_even
            and batch_size * max_length**2 >= suggested_work
            and max_length <= max_positions
        )
        print(f"\nSuggested packed_min_padding (max break-even): {suggested:.2f}")
        print(f"Suggested packed_min_work (smallest B*L^2 that broke even): {suggested_work}")
        if never:
            print(f"  Note: {never} reach that work but never broke even.")
    else:
        suggested = suggested_work = None
        print("\nPacked was never faster in the measured range; keep packed=False.")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump({
                "model": args.model,
                "device": args.device,
                "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
                "dtype": dtype_name,
                "backend": padded._disentangled_flash_backend,
                "tuning_mode": args.tuning_mode,
                "results": results,
                "suggested_min_padding": suggested,
                "suggested_min_work": suggested_work,
            }, handle, indent=2)


if __name__ == "__main__":
    main()
