"""Utilities for long-document extraction.

The helpers in this module are intentionally model-agnostic: they split text
into word-aware windows, remap chunk-local spans back to document offsets, and
merge duplicate predictions produced by overlapping chunks.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


_WORD_PATTERN = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)


@dataclass(frozen=True)
class TextChunk:
    """A chunk of text with offsets into the original document."""

    text: str
    start_char: int
    end_char: int
    start_word: int
    end_word: int


def iter_word_offsets(text: str) -> Iterable[Tuple[str, int, int]]:
    """Yield regex word tokens and character offsets using processor-compatible rules.

    The regex is matched against the original text (it is already case-insensitive)
    so that the reported offsets index the caller's string. Lower-casing before
    matching is unsafe because Unicode case folding can change string length
    (e.g. ``"İ".lower()`` expands to two code points), which would shift offsets.
    """
    for match in _WORD_PATTERN.finditer(text):
        yield match.group(), match.start(), match.end()


def split_text_into_chunks(
    text: str,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> List[TextChunk]:
    """Split text into overlapping word windows.

    Args:
        text: Original document text.
        chunk_size: Maximum number of word tokens per chunk.
        chunk_overlap: Number of word tokens repeated between adjacent chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    tokens = list(iter_word_offsets(text))
    if not tokens:
        return [TextChunk(text=text, start_char=0, end_char=len(text), start_word=0, end_word=0)]

    chunks: List[TextChunk] = []
    step = chunk_size - chunk_overlap
    start_word = 0

    while start_word < len(tokens):
        end_word = min(start_word + chunk_size, len(tokens))
        start_char = tokens[start_word][1]
        end_char = tokens[end_word - 1][2]
        chunks.append(
            TextChunk(
                text=text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
                start_word=start_word,
                end_word=end_word,
            )
        )
        if end_word == len(tokens):
            break
        start_word += step

    return chunks


def remap_result_spans(result: Any, original_text: str, chunk: TextChunk) -> Any:
    """Recursively remap chunk-local span dicts to document-level offsets."""
    if isinstance(result, list):
        return [remap_result_spans(item, original_text, chunk) for item in result]

    if isinstance(result, dict):
        remapped = {key: remap_result_spans(value, original_text, chunk) for key, value in result.items()}
        if _is_span_dict(remapped):
            start = int(remapped["start"]) + chunk.start_char
            end = int(remapped["end"]) + chunk.start_char
            remapped["start"] = start
            remapped["end"] = end
            if 0 <= start <= end <= len(original_text):
                remapped["text"] = original_text[start:end]
        return remapped

    return result


def merge_chunk_results(
    original_text: str,
    chunks: List[TextChunk],
    chunk_results: List[Dict[str, Any]],
    include_confidence: bool = False,
    include_spans: bool = False,
    scalar_entity_labels: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Merge formatted extraction results from one document's chunks.

    ``scalar_entity_labels`` names entity types declared with a non-list dtype;
    those are collapsed to a single best value (mirroring the base API) instead
    of being returned as a list.
    """
    if len(chunks) != len(chunk_results):
        raise ValueError("chunks and chunk_results must have the same length")

    scalar_labels = set(scalar_entity_labels or ())
    remapped_results = [
        remap_result_spans(result, original_text, chunk)
        for chunk, result in zip(chunks, chunk_results)
    ]
    merged = _merge_result_dicts(remapped_results, scalar_labels)
    return _strip_span_metadata(merged, include_confidence, include_spans)


def _merge_result_dicts(
    results: List[Dict[str, Any]],
    scalar_entity_labels: Optional[set] = None,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    keys = []
    seen = set()
    for result in results:
        for key in result:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    for key in keys:
        values = [result.get(key) for result in results if key in result]
        if key == "entities":
            merged[key] = _merge_entity_maps(values, scalar_entity_labels or set())
        elif key == "relation_extraction":
            merged[key] = _merge_relation_maps(values)
        else:
            merged[key] = _merge_values(values)

    return merged


def _merge_entity_maps(values: List[Any], scalar_labels: set) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    labels = []
    seen = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        for label in value:
            if label not in seen:
                seen.add(label)
                labels.append(label)

    for label in labels:
        items: List[Any] = []
        for value in values:
            if isinstance(value, dict) and label in value:
                items.extend(_as_list(value[label]))
        deduped = _dedupe_items(items, remove_overlaps=True)
        if label in scalar_labels:
            # A non-list entity dtype yields a single best value (or None),
            # matching the base engine's scalar contract.
            merged[label] = deduped[0] if deduped else None
        else:
            merged[label] = deduped

    return merged


def _merge_relation_maps(values: List[Any]) -> Dict[str, List[Any]]:
    merged: Dict[str, List[Any]] = {}
    labels = []
    seen = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        for label in value:
            if label not in seen:
                seen.add(label)
                labels.append(label)

    for label in labels:
        items: List[Any] = []
        for value in values:
            if isinstance(value, dict) and label in value:
                items.extend(_as_list(value[label]))
        merged[label] = _dedupe_items(items, remove_overlaps=False)

    return merged


def _merge_values(values: List[Any]) -> Any:
    non_empty = [value for value in values if value not in (None, {}, [])]
    if not non_empty:
        return values[0] if values else None

    if all(_is_classification_dict(value) for value in non_empty):
        return max(non_empty, key=lambda value: value.get("confidence", 0.0))

    if all(isinstance(value, str) for value in non_empty):
        counts = Counter(non_empty)
        return max(non_empty, key=lambda value: (counts[value], -non_empty.index(value)))

    if all(isinstance(value, list) for value in non_empty):
        items: List[Any] = []
        for value in non_empty:
            items.extend(value)
        return _dedupe_items(items, remove_overlaps=False)

    if all(isinstance(value, dict) for value in non_empty):
        return _merge_nested_dicts(non_empty)

    return non_empty[0]


def _merge_nested_dicts(values: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    keys = []
    seen = set()
    for value in values:
        for key in value:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    for key in keys:
        merged[key] = _merge_values([value.get(key) for value in values if key in value])
    return merged


def _dedupe_items(items: List[Any], remove_overlaps: bool) -> List[Any]:
    span_items = [item for item in items if _is_span_dict(item)]
    other_items = [item for item in items if not _is_span_dict(item)]

    deduped: List[Any] = []
    if span_items:
        sorted_spans = sorted(span_items, key=lambda item: item.get("confidence", 0.0), reverse=True)
        selected: List[Dict[str, Any]] = []
        seen_spans = set()
        for item in sorted_spans:
            key = _span_key(item)
            if key in seen_spans:
                continue
            if remove_overlaps and any(_spans_overlap(item, existing) for existing in selected):
                continue
            seen_spans.add(key)
            selected.append(item)
        deduped.extend(sorted(selected, key=lambda item: (item["start"], item["end"], item.get("text", ""))))

    # Non-span items (relations, structure instances, classification dicts) are
    # deduplicated on a confidence-insensitive canonical key so that the same
    # prediction seen in two overlapping chunks with slightly different scores is
    # collapsed. When a duplicate is found, the higher-confidence one is kept.
    seen_other: Dict[str, int] = {}
    other_deduped: List[Any] = []
    for item in other_items:
        key = _canonical_key(item)
        if key not in seen_other:
            seen_other[key] = len(other_deduped)
            other_deduped.append(item)
        else:
            idx = seen_other[key]
            if _representative_confidence(item) > _representative_confidence(other_deduped[idx]):
                other_deduped[idx] = item
    deduped.extend(other_deduped)

    return deduped


_SPAN_RESERVED = frozenset({"text", "confidence", "start", "end"})


def _strip_span_metadata(value: Any, include_confidence: bool, include_spans: bool) -> Any:
    if isinstance(value, list):
        return [_strip_span_metadata(item, include_confidence, include_spans) for item in value]

    if isinstance(value, dict):
        if _is_span_dict(value):
            # Entity spans may carry extra attribute-group payloads (e.g.
            # ``{"text": ..., "sentiment": {...}}``). Those extra keys must be
            # preserved verbatim to match the non-long attribute API, which keeps
            # attribute payloads regardless of the confidence/span flags.
            extras = {k: v for k, v in value.items() if k not in _SPAN_RESERVED}
            if not include_confidence and not include_spans and not extras:
                return value.get("text", "")
            stripped: Dict[str, Any] = {"text": value.get("text", "")}
            if include_confidence and "confidence" in value:
                stripped["confidence"] = value["confidence"]
            if include_spans:
                stripped["start"] = value["start"]
                stripped["end"] = value["end"]
            stripped.update(extras)
            return stripped

        # Classification results, e.g. ``{"label": "positive", "confidence": ...}``.
        if _is_classification_dict(value):
            if include_confidence:
                return {"label": value["label"], "confidence": value["confidence"]}
            return value["label"]

        # Enum/choice structure fields, e.g. ``{"text": "outdoor", "confidence": ...}``
        # (a text/confidence pair with no character offsets).
        if "text" in value and "confidence" in value and "start" not in value and "end" not in value:
            if include_confidence:
                return {"text": value["text"], "confidence": value["confidence"]}
            return value["text"]

        return {
            key: _strip_span_metadata(item, include_confidence, include_spans)
            for key, item in value.items()
        }

    return value


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _is_span_dict(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "text" in value
        and "start" in value
        and "end" in value
        and isinstance(value["start"], int)
        and isinstance(value["end"], int)
    )


def _is_classification_dict(value: Any) -> bool:
    return isinstance(value, dict) and "label" in value and "confidence" in value


def _span_key(item: Dict[str, Any]) -> Tuple[str, int, int]:
    return (str(item.get("text", "")).lower(), int(item["start"]), int(item["end"]))


def _spans_overlap(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return not (left["end"] <= right["start"] or left["start"] >= right["end"])


def _canonical_key(value: Any) -> str:
    if isinstance(value, dict):
        return repr(
            sorted(
                (key, _canonical_key(item))
                for key, item in value.items()
                if key != "confidence"
            )
        )
    if isinstance(value, list):
        return repr([_canonical_key(item) for item in value])
    return repr(value)


def _representative_confidence(value: Any) -> float:
    """Best-effort confidence for a possibly-nested prediction, used to pick the
    survivor when duplicate predictions are merged across overlapping chunks."""
    if isinstance(value, dict):
        if isinstance(value.get("confidence"), (int, float)):
            return float(value["confidence"])
        nested = [_representative_confidence(v) for v in value.values()]
        return max(nested) if nested else 0.0
    if isinstance(value, list):
        nested = [_representative_confidence(v) for v in value]
        return max(nested) if nested else 0.0
    return 0.0
