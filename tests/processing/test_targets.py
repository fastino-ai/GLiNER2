"""Tests for architecture-neutral targets and coordinate conversion (PR 3)."""

from __future__ import annotations

from typing import List, Tuple

import pytest
import torch

from gliner2.processor import WhitespaceTokenSplitter
from gliner2.processing.targets import (
    MentionTarget,
    TargetGraph,
    TargetCapacityError,
    char_span_to_word_boundaries,
    word_boundaries_to_char_span,
    inclusive_tokens_to_boundary_pair,
    build_boundary_targets,
    pad_target_graphs,
    apply_truncation_policy,
    normalize_surface_occurrences,
    SurfaceSpan,
)


def _word_offsets(text: str) -> Tuple[List[int], List[int], List[str]]:
    splitter = WhitespaceTokenSplitter()
    starts, ends, toks = [], [], []
    for tok, s, e in splitter(text, lower=False):
        toks.append(tok)
        starts.append(s)
        ends.append(e)
    return starts, ends, toks


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def test_inclusive_end_converts_to_exclusive_boundary():
    assert inclusive_tokens_to_boundary_pair(2, 4) == (2, 5)


def test_character_offsets_roundtrip():
    text = "Apple acquired Apple Records."
    starts, ends, _ = _word_offsets(text)
    # "Apple Records" occupies chars 15..28.
    tok_s, tok_e = char_span_to_word_boundaries(15, 28, starts, ends)
    cs, ce = word_boundaries_to_char_span(tok_s, tok_e, starts, ends)
    assert (cs, ce) == (15, 28)
    assert text[cs:ce] == "Apple Records"


def test_end_of_document_offset_roundtrip():
    text = "John lives in NYC"
    starts, ends, _ = _word_offsets(text)
    # "NYC" is the final token.
    cs_start = text.index("NYC")
    tok_s, tok_e = char_span_to_word_boundaries(cs_start, len(text), starts, ends)
    cs, ce = word_boundaries_to_char_span(tok_s, tok_e, starts, ends)
    assert text[cs:ce] == "NYC"


def test_unicode_casefold_does_not_shift_offsets():
    # "İ".lower() expands to two code points; offsets must index the original.
    text = "İstanbul is large"
    starts, ends, toks = _word_offsets(text)
    # First token is the original "İstanbul".
    assert text[starts[0]:ends[0]] == "İstanbul"
    tok_s, tok_e = char_span_to_word_boundaries(starts[0], ends[0], starts, ends)
    cs, ce = word_boundaries_to_char_span(tok_s, tok_e, starts, ends)
    assert text[cs:ce] == "İstanbul"


# ---------------------------------------------------------------------------
# Surface spans
# ---------------------------------------------------------------------------

def test_surface_span_requires_text_and_offsets():
    span = SurfaceSpan.from_mapping({"text": "Pushkin", "start": 0, "end": 7})
    assert (span.text, span.start, span.end) == ("Pushkin", 0, 7)

    with pytest.raises(ValueError, match="missing"):
        SurfaceSpan.from_mapping({"text": "Pushkin", "start": 0})
    with pytest.raises(ValueError, match="end > start"):
        SurfaceSpan.from_mapping({"text": "Pushkin", "start": 7, "end": 7})
    with pytest.raises(ValueError, match="non-negative"):
        SurfaceSpan.from_mapping({"text": "Pushkin", "start": -1, "end": 7})
    with pytest.raises(ValueError, match="integers"):
        SurfaceSpan.from_mapping({"text": "Pushkin", "start": "0", "end": 7})


# ---------------------------------------------------------------------------
# Occurrence policy
# ---------------------------------------------------------------------------

def test_repeated_surface_text_requires_explicit_policy():
    matches = [(0, 1), (2, 3)]
    with pytest.raises(ValueError):
        normalize_surface_occurrences(matches, occurrence_policy="error_on_ambiguous")
    assert normalize_surface_occurrences(matches, occurrence_policy="all") == matches
    assert normalize_surface_occurrences(matches, occurrence_policy="first") == [(0, 1)]


# ---------------------------------------------------------------------------
# Dense targets
# ---------------------------------------------------------------------------

def test_build_boundary_targets_shapes_and_values():
    graph = TargetGraph(mentions=(
        MentionTarget(query_id=0, start=0, end=2),
        MentionTarget(query_id=0, start=1, end=3),  # nested, shares nothing
        MentionTarget(query_id=1, start=4, end=5),
    ))
    start, end, inside = build_boundary_targets(graph, query_count=2, text_length=5)
    assert start.shape == (2, 6)
    assert end.shape == (2, 6)
    assert inside.shape == (2, 5)
    # query 0 starts at 0 and 1, ends at 2 and 3
    assert start[0, 0] == 1 and start[0, 1] == 1
    assert end[0, 2] == 1 and end[0, 3] == 1
    assert inside[0, 0] == 1 and inside[0, 2] == 1
    assert start[1, 4] == 1 and end[1, 5] == 1


# ---------------------------------------------------------------------------
# Padding + capacity + device move
# ---------------------------------------------------------------------------

def test_pad_target_graphs_capacity_and_padding():
    graphs = [
        TargetGraph(mentions=(
            MentionTarget(0, 0, 2), MentionTarget(0, 3, 5), MentionTarget(1, 1, 4),
        )),
        TargetGraph(mentions=(MentionTarget(0, 0, 1),)),
    ]
    padded = pad_target_graphs(graphs, query_counts=[2, 2], text_lengths=[6, 6],
                               max_gold_per_query=4)
    assert padded.mention_pairs.shape == (2, 2, 4, 2)
    assert padded.mention_mask[0, 0].sum() == 2
    assert padded.mention_mask[1, 0].sum() == 1
    # device move is a no-op on CPU but must return tensors.
    moved = padded.to("cpu")
    assert moved.mention_pairs.device.type == "cpu"


def test_too_many_gold_pairs_raises_capacity_error():
    graphs = [TargetGraph(mentions=(
        MentionTarget(0, 0, 1), MentionTarget(0, 1, 2), MentionTarget(0, 2, 3),
    ))]
    with pytest.raises(TargetCapacityError):
        pad_target_graphs(graphs, query_counts=[1], text_lengths=[3],
                          max_gold_per_query=2)


def test_capacity_policy_rejects_unknown_value():
    graphs = [TargetGraph(mentions=(MentionTarget(0, 0, 1),))]
    with pytest.raises(ValueError, match="on_capacity_exceeded"):
        pad_target_graphs(graphs, query_counts=[1], text_lengths=[3],
                          max_gold_per_query=2, on_capacity_exceeded="nope")


def test_capacity_policy_truncate_with_warning(caplog):
    graphs = [TargetGraph(mentions=(
        MentionTarget(0, 0, 1), MentionTarget(0, 1, 2), MentionTarget(0, 2, 3),
    ))]
    with caplog.at_level("WARNING"):
        padded = pad_target_graphs(
            graphs, query_counts=[1], text_lengths=[3],
            max_gold_per_query=2, on_capacity_exceeded="truncate_with_warning",
        )
    # Only max_gold_per_query kept; the rest are dropped with an explicit warning.
    assert int(padded.mention_mask[0, 0].sum()) == 2
    assert "truncated" in caplog.text


def test_capacity_policy_skip_sample(caplog):
    graphs = [
        TargetGraph(mentions=(
            MentionTarget(0, 0, 1), MentionTarget(0, 1, 2), MentionTarget(0, 2, 3),
        )),
        TargetGraph(mentions=(MentionTarget(0, 0, 2),)),
    ]
    with caplog.at_level("WARNING"):
        padded = pad_target_graphs(
            graphs, query_counts=[1, 1], text_lengths=[3, 3],
            max_gold_per_query=2, on_capacity_exceeded="skip_sample",
        )
    # The overflowing sample contributes no gold; the second sample is intact.
    assert int(padded.mention_mask[0].sum()) == 0
    assert int(padded.mention_mask[1].sum()) == 1
    assert "skipped" in caplog.text


def test_duplicate_gold_pairs_are_collapsed():
    graphs = [TargetGraph(mentions=(
        MentionTarget(0, 0, 2), MentionTarget(0, 0, 2),  # duplicate
    ))]
    padded = pad_target_graphs(graphs, query_counts=[1], text_lengths=[3],
                               max_gold_per_query=4)
    assert padded.mention_mask[0, 0].sum() == 1


# ---------------------------------------------------------------------------
# Truncation policy
# ---------------------------------------------------------------------------

def test_crossing_truncation_target_raises_by_default():
    graph = TargetGraph(mentions=(MentionTarget(0, 2, 10),))
    with pytest.raises(ValueError):
        apply_truncation_policy(graph, truncated_length=5, policy="error")


def test_drop_target_policy_is_explicit():
    graph = TargetGraph(mentions=(MentionTarget(0, 0, 3), MentionTarget(0, 2, 10)))
    kept = apply_truncation_policy(graph, truncated_length=5, policy="drop_target")
    assert len(kept.mentions) == 1
    assert kept.mentions[0].end == 3


def test_drop_example_policy_is_explicit():
    graph = TargetGraph(mentions=(MentionTarget(0, 2, 10),))
    assert apply_truncation_policy(graph, truncated_length=5, policy="drop_example") is None
