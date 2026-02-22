"""
Test: max_len parameter in batch_extract / extract_entities.

Shows how max_len truncates long inputs, how it affects span coverage
and wall-clock time.

Run:
    python tests/test_max_len.py
"""

import time
from dataclasses import dataclass
from typing import Optional

from gliner2 import GLiNER2
from gliner2.inference.engine import Schema


MODEL_ID   = "fastino/gliner2-base-v1"
SHORT_TEXT = "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino on September 12."
LONG_TEXT  = SHORT_TEXT * 50   # ~700 tokens, ~4 100 chars
ENTITY_TYPES = ["company", "person", "product", "location"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _flatten(raw) -> list:
    """entities dict → flat list of span dicts, with 'label' injected from the key."""
    if isinstance(raw, dict):
        result = []
        for label, spans in raw.items():
            for e in (spans if isinstance(spans, list) else [spans]):
                if isinstance(e, dict):
                    result.append({**e, "label": label})
        return result
    return raw or []


@dataclass
class Result:
    label:       str
    input_chars: int
    max_len:     Optional[int]
    entities:    list
    elapsed:     float

    @property
    def n(self): return len(self.entities)

    @property
    def max_end(self):
        ends = [e["end"] for e in self.entities if isinstance(e, dict) and "end" in e]
        return max(ends) if ends else 0


def measure(label, model, schema, texts, **kw) -> Result:
    t0 = time.perf_counter()
    out = model.batch_extract(texts, schema, batch_size=len(texts),
                              include_spans=True, **kw)
    return Result(label, len(texts[0]), kw.get("max_len"),
                  _flatten(out[0].get("entities", {})),
                  time.perf_counter() - t0)


# ── formatting ────────────────────────────────────────────────────────────────

LINE_WIDTH = 64  # output box width in characters

def header(title):
    print(f"\n┌{'─' * (LINE_WIDTH - 2)}┐")
    print(f"│  {title:<{LINE_WIDTH - 4}}│")
    print(f"└{'─' * (LINE_WIDTH - 2)}┘")

def row(key, value):
    print(f"  {key:<16}{value}")

def divider():
    print(f"  {'─' * (LINE_WIDTH - 4)}")

def print_result(r: Result, baseline: Optional[Result] = None):
    header(r.label)
    row("input chars",  r.input_chars)
    row("max_len",      f"{r.max_len} tokens" if r.max_len else "none")
    row("max span end", r.max_end)
    row("time",         f"{r.elapsed:.3f}s")
    if baseline and baseline.max_end:
        pct    = r.max_end / baseline.max_end * 100
        d_time = r.elapsed - baseline.elapsed
        divider()
        row("vs baseline",
            f"covered {pct:.0f}% of text  /  time {d_time:+.3f}s")
    if r.entities:
        divider()
        for e in r.entities[:3]:
            if isinstance(e, dict):
                print(f"  [{e.get('start'):4d}:{e.get('end'):4d}]"
                      f"  {e.get('text')!r:22s}  ({e.get('label')})")
        if r.n > 3:
            print(f"  … {r.n - 3} more spans")


# ── demo ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {MODEL_ID} …")
    model  = GLiNER2.from_pretrained(MODEL_ID)
    schema = Schema().entities(ENTITY_TYPES)
    print("Done.\n")

    r_short = measure("short text  |  max_len=384", model, schema, [SHORT_TEXT], max_len=384)
    r_base  = measure("long text   |  no max_len  (baseline)", model, schema, [LONG_TEXT])
    r_384   = measure("long text   |  max_len=384", model, schema, [LONG_TEXT], max_len=384)
    r_50    = measure("long text   |  max_len=50",  model, schema, [LONG_TEXT], max_len=50)

    print_result(r_short)
    print_result(r_base)
    print_result(r_384, baseline=r_base)
    print_result(r_50,  baseline=r_base)

    # mixed batch
    texts = [SHORT_TEXT, LONG_TEXT, SHORT_TEXT]
    t0  = time.perf_counter()
    out = model.batch_extract(texts, schema, batch_size=4,
                              include_spans=True, max_len=384)
    elapsed = time.perf_counter() - t0
    header("mixed batch (short + long + short)  |  max_len=384")
    for i, r in enumerate(out):
        n = len(_flatten(r.get("entities", {})))
        row(f"text[{i}]", f"chars={len(texts[i]):5d}   entities={n}")
    row("time", f"{elapsed:.3f}s")

    print()


if __name__ == "__main__":
    main()
