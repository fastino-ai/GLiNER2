"""Unit tests for gliner2.processing.boundary_preprocessing helpers.

Covers the two additive fixes:
  * ``_dedupe_contained_spans`` -- the containment filter that resolves the
    "Kiox" vs "Kiox 300"/"Kiox 400C"/"Kiox 500" phantom-match bug (independent
    per-listed-value re-search matching a short value's tokens as a sub-span
    of an unrelated, longer value's own match).
  * ``_unalignable_entity_error`` -- the enriched ``ValueError`` message for
    an entity that fails to token-align, including literal listed value(s)
    and a text snippet when available.

Integration coverage (through ``ExtractorCollator``/``build_boundary_batch_metadata``)
lives in ``tests/processing/test_boundary_collation.py``.
"""

from __future__ import annotations

from gliner2.processing.boundary_preprocessing import (
    _dedupe_contained_spans,
    _unalignable_entity_error,
)

# ---------------------------------------------------------------------------
# _dedupe_contained_spans
# ---------------------------------------------------------------------------


def test_kiox_family_phantom_matches_collapse_to_genuine_mentions():
    """Exact reproduction of the reported "Kiox" collision.

    Text: "Kiox 300 is popular. Kiox 400C improves battery life. Kiox 500 is
    the top model. A user asked about Kiox compatibility." word-tokenizes to
    ``[kiox, 300, is, popular, ., kiox, 400c, improves, battery, life, .,
    kiox, 500, is, the, top, model, ., a, user, asked, about, kiox,
    compatibility, .]``. Independently re-searching each of "Kiox 300",
    "Kiox 400C", "Kiox 500", and bare "Kiox" (as ``Processor._build_outputs``
    does for a list-valued entity field) produces 7 raw half-open token-span
    matches for the one "display" query: the 3 correct multi-token matches,
    plus 3 phantom single-token "kiox" matches that are really the leading
    token of those same 3 longer matches, plus 1 genuine standalone "Kiox"
    mention at token 22. See ``test_boundary_collation.py`` for the same
    scenario driven end-to-end through ``ExtractorCollator``.
    """
    raw_matches = [
        (0, 2),  # "Kiox 300" (tokens 0-1)
        (5, 7),  # "Kiox 400C" (tokens 5-6)
        (11, 13),  # "Kiox 500" (tokens 11-12)
        (0, 1),  # phantom: bare "kiox" hitting "Kiox 300"'s lead token
        (5, 6),  # phantom: bare "kiox" hitting "Kiox 400C"'s lead token
        (11, 12),  # phantom: bare "kiox" hitting "Kiox 500"'s lead token
        (22, 23),  # genuine standalone "Kiox" mention
    ]
    kept = _dedupe_contained_spans(raw_matches)
    assert sorted(kept) == [(0, 2), (5, 7), (11, 13), (22, 23)]


def test_chain_of_nested_containment_keeps_only_the_longest():
    """A <- B <- C nesting keeps only the outermost span C."""
    pairs = [(5, 6), (5, 8), (4, 9)]
    assert _dedupe_contained_spans(pairs) == [(4, 9)]


def test_exact_duplicates_collapse_to_one_instance():
    pairs = [(2, 4), (2, 4), (2, 4)]
    assert _dedupe_contained_spans(pairs) == [(2, 4)]


def test_partial_overlap_that_is_not_containment_is_untouched():
    """Crossing spans (neither contains the other) are out of scope."""
    pairs = [(0, 3), (2, 5)]
    assert sorted(_dedupe_contained_spans(pairs)) == [(0, 3), (2, 5)]


def test_empty_and_singleton_inputs():
    assert _dedupe_contained_spans([]) == []
    assert _dedupe_contained_spans([(0, 1)]) == [(0, 1)]


def test_disjoint_spans_are_all_kept():
    pairs = [(0, 1), (5, 6), (10, 12)]
    assert sorted(_dedupe_contained_spans(pairs)) == pairs


# ---------------------------------------------------------------------------
# _unalignable_entity_error
# ---------------------------------------------------------------------------


def test_error_message_includes_batch_relative_caveat():
    err = _unalignable_entity_error(
        field_name="PartCode",
        sample_idx=3,
        original_texts=None,
        original_schemas=None,
    )
    msg = str(err)
    assert "'PartCode' was not found in sample 3" in msg
    assert "batch-relative" in msg
    assert "find_unalignable_entities" in msg


def test_error_message_includes_literal_values_and_snippet_when_available():
    text = "See IMG Bosch-eBike-LEDRemote-ABS-BES3-MY2023.png for wiring."
    err = _unalignable_entity_error(
        field_name="PartCode",
        sample_idx=0,
        original_texts=[text],
        original_schemas=[{"entities": {"PartCode": "ABS"}}],
    )
    msg = str(err)
    assert "Listed value(s) for 'PartCode': 'ABS'" in msg
    assert text in msg


def test_error_message_truncates_long_snippets():
    long_text = "word " * 100
    err = _unalignable_entity_error(
        field_name="Field",
        sample_idx=0,
        original_texts=[long_text],
        original_schemas=None,
    )
    msg = str(err)
    assert "\u2026" in msg
    assert long_text not in msg


def test_error_message_tolerates_missing_or_malformed_schema():
    # sample_idx out of range must not raise while building the message.
    err = _unalignable_entity_error(
        field_name="X", sample_idx=5, original_texts=[], original_schemas=["not-a-mapping"]
    )
    assert "'X' was not found in sample 5" in str(err)

    # A non-mapping schema at a valid index must also be tolerated.
    err2 = _unalignable_entity_error(
        field_name="X", sample_idx=0, original_texts=None, original_schemas=["not-a-mapping"]
    )
    assert "'X' was not found in sample 0" in str(err2)
