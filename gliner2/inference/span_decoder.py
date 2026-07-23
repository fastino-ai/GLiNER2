"""Core helpers for deterministic entity-span decoding."""

from __future__ import annotations

from typing import List, Sequence, Tuple

RawSpan = Tuple[str, float, int, int]


def finalize_spans(
    raw_spans: Sequence[RawSpan],
    *,
    dtype: str = "list",
    gate_open: bool = True,
    suppress: bool = True,
) -> List[RawSpan]:
    """Decode thresholded entity spans using the production contract.

    Scores are ordered deterministically, overlapping spans are greedily
    suppressed, and scalar fields emit only their highest-ranked surviving span.
    """
    if not gate_open:
        return []
    spans = sorted(raw_spans, key=lambda span: (-span[1], span[2], span[3]))
    if suppress:
        kept: List[RawSpan] = []
        for span in spans:
            _, _, start, end = span
            if any(
                not (end <= kept_start or start >= kept_end)
                for _, _, kept_start, kept_end in kept
            ):
                continue
            kept.append(span)
        spans = kept
    return spans if dtype == "list" else spans[:1]
