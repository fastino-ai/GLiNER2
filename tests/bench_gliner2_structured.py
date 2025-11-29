#!/usr/bin/env python
"""Compare GLiNER2 structured extraction with and without inference packing."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Dict, List, Sequence

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gliner2 import GLiNER2, InferencePackingConfig


DEFAULT_TEXTS: List[str] = [
    "iPhone 15 Pro Max with 256GB storage, A17 Pro chip, priced at $1199. Available in titanium and black colors.",
    "Galaxy S24 priced at $899 with 512GB and Exynos chipset.",
    "Pixel 8 available for $699 with 128GB and Google's Tensor G3.",
    "MacBook Air M2 costs $1199 with 8GB RAM and 256GB SSD.",
    "Dell XPS 13 sells for $999 with 16GB RAM and 512GB SSD.",
]

DEFAULT_STRUCTURES: Dict[str, List[str]] = {
    "product": [
        "name::str::Full product name and model",
        "price::str::Product price with currency",
        "storage::str::Storage capacity like 256GB or 1TB",
        "chip::str::Processor information",
    ]
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GLiNER2 structured extraction with and without inference packing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fastino/gliner2-base-v1",
        help="Model name or local path for GLiNER2.from_pretrained",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (e.g. 'cpu', 'cuda', 'cuda:0')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoder forward passes",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold passed to GLiNER2",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override maximum packed sequence length (defaults to model max position embeddings)",
    )
    parser.add_argument(
        "--streams-per-batch",
        type=int,
        default=1,
        help="Number of packed streams to generate per batch",
    )
    return parser.parse_args()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _run_once(
        extractor: GLiNER2,
        *,
        texts: Sequence[str],
        structures: Dict[str, List[str]],
        batch_size: int,
        threshold: float,
        device: torch.device,
        packing_config: InferencePackingConfig | None,
) -> tuple[List[dict], float]:
    _sync_if_cuda(device)
    start = time.perf_counter()
    preds = extractor.batch_extract_json(
        list(texts),
        structures,
        batch_size=batch_size,
        threshold=threshold,
        packing_config=packing_config,
    )
    _sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    return preds, elapsed


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    texts: List[str] = list(DEFAULT_TEXTS)
    structures: Dict[str, List[str]] = dict(DEFAULT_STRUCTURES)

    extractor = GLiNER2.from_pretrained(args.model, map_location=str(device))
    extractor.to(device)
    extractor.eval()

    max_length = args.max_length
    if max_length is None:
        max_length = getattr(extractor.encoder.config, "max_position_embeddings", None)
    if max_length is None:
        raise ValueError("Unable to infer max sequence length for packing; please pass --max-length")

    sep_token_id = getattr(extractor.processor.tokenizer, "sep_token_id", None)
    if sep_token_id is None:
        sep_token_id = getattr(extractor.processor.tokenizer, "eos_token_id", None)
    if sep_token_id is None:
        sep_token_id = getattr(extractor.processor.tokenizer, "pad_token_id", None)

    packing_cfg = InferencePackingConfig(
        max_length=int(max_length),
        sep_token_id=sep_token_id,
        streams_per_batch=args.streams_per_batch,
    )

    print("Running baseline (no packing)...")
    baseline_preds, baseline_time = _run_once(
        extractor,
        texts=texts,
        structures=structures,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=device,
        packing_config=None,
    )

    print("Running with inference packing...")
    packed_preds, packed_time = _run_once(
        extractor,
        texts=texts,
        structures=structures,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=device,
        packing_config=packing_cfg,
    )

    identical = baseline_preds == packed_preds

    print("Timing summary:")
    print(f"  Without packing: {baseline_time:.3f} s")
    print(f"  With packing   : {packed_time:.3f} s")
    if packed_time > 0:
        print(f"  Speedup        : {baseline_time / packed_time:.2f}x")
    print(f"Predictions identical: {identical}")

    if not identical:
        print("Baseline predictions:", baseline_preds)
        print("Packed predictions   :", packed_preds)


if __name__ == "__main__":
    main()
