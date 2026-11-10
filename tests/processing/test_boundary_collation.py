"""Boundary-specific processor/collator integration tests."""

from __future__ import annotations

import pytest

from gliner2.processing.targets import TargetCapacityError
from gliner2.processor import SamplingConfig, SchemaTransformer
from gliner2.training import ExtractorCollator


def _processor(tokenizer, *, shuffle_entities=False):
    return SchemaTransformer(
        tokenizer=tokenizer,
        sampling_config=SamplingConfig(
            shuffle_entities=shuffle_entities,
            synthetic_entity_label_prob=0.0,
        ),
    )


def test_boundary_training_collation_aligns_reordered_entity_targets(
    tiny_tokenizer, monkeypatch
):
    processor = _processor(tiny_tokenizer, shuffle_entities=True)
    monkeypatch.setattr("random.shuffle", lambda values: values.reverse())
    collator = ExtractorCollator(
        processor, is_training=True, architecture="boundary", max_gold_per_query=4
    )

    batch = collator([
        (
            "John works at Apple.",
            {"entities": {"person": ["John"], "company": ["Apple"]}},
        )
    ])

    layout = batch.query_layouts[0]
    assert [query.role_name for query in layout.queries] == ["company", "person"]
    assert batch.targets is not None
    pairs = {
        layout.query(query_id).role_name: (int(start), int(end))
        for query_id, (start, end) in enumerate(batch.targets.mention_pairs[0, :, 0])
    }
    assert pairs == {"company": (3, 4), "person": (0, 1)}


def test_boundary_inference_collation_has_layout_without_targets(tiny_tokenizer):
    processor = _processor(tiny_tokenizer)
    batch = ExtractorCollator(
        processor, is_training=False, architecture="boundary"
    )([("Apple released iPhone.", {"entities": {"company": "", "product": ""}})])

    assert [q.role_name for q in batch.query_layouts[0].queries] == ["company", "product"]
    assert batch.targets is None


def test_pinned_relation_head_collapses_the_gold_cross_product(tiny_tokenizer):
    processor = SchemaTransformer(
        tokenizer=tiny_tokenizer,
        sampling_config=SamplingConfig(
            remove_relations_prob=0.0, swap_head_tail_prob=0.0
        ),
    )
    collator = ExtractorCollator(
        processor, is_training=True, architecture="boundary", max_gold_per_query=4
    )
    text = "Pushkin street runs past the Pushkin monument."

    def gold_pairs(head):
        batch = collator([
            (text, {"relations": [{"near": {"head": head, "tail": "monument"}}]})
        ])
        pairs, mask = batch.targets.edge_targets[:2]
        return pairs[mask].tolist()

    # a repeated head surface supervises both cross-products
    assert gold_pairs("Pushkin") == [[0, 1, 6, 7], [5, 6, 6, 7]]
    assert gold_pairs({"text": "Pushkin", "start": 29, "end": 36}) == [[5, 6, 6, 7]]


def test_boundary_training_rejects_missing_entity_annotation(tiny_tokenizer):
    processor = _processor(tiny_tokenizer)
    collator = ExtractorCollator(processor, is_training=True, architecture="boundary")

    with pytest.raises(ValueError, match="was not found"):
        collator([("Apple released iPhone.", {"entities": {"company": ["Google"]}})])


def test_missing_entity_error_includes_listed_value_and_text_snippet(tiny_tokenizer):
    """Bug 1: the raised error must be actionable, not just batch-relative.

    "ABS" is a real, exact character substring of the surrounding text, but
    the tokenizer's hyphen-continuation rule merges the whole hyphen-joined
    filename into one token, so a standalone-token search for "abs" finds
    nothing. The old message ("entity 'PartCode' was not found in sample 0")
    gave no way to tell *which* listed value failed or where to look; the
    fixed message must surface both.
    """
    processor = _processor(tiny_tokenizer)
    collator = ExtractorCollator(processor, is_training=True, architecture="boundary")
    text = "See IMG Bosch-eBike-LEDRemote-ABS-BES3-MY2023.png for wiring."

    with pytest.raises(ValueError) as exc_info:
        collator([(text, {"entities": {"PartCode": ["ABS"]}})])

    message = str(exc_info.value)
    assert "was not found in sample 0" in message
    assert "batch-relative" in message
    assert "ABS" in message
    assert text in message
    assert "find_unalignable_entities" in message


class TestKioxFamilyContainmentFilter:
    """Bug 2: independent per-value re-search produces phantom gold spans.

    Text: "Kiox 300 is popular. Kiox 400C improves battery life. Kiox 500 is
    the top model. A user asked about Kiox compatibility." Entity type
    "display" lists ``["Kiox 300", "Kiox 400C", "Kiox 500", "Kiox"]`` -- four
    independently, correctly extracted mentions. Searching each value
    independently also matches the bare "kiox" token at the start of every
    "Kiox 300"/"Kiox 400C"/"Kiox 500" occurrence, producing 3 phantom
    sub-span matches in addition to the 4 genuine mentions (7 raw positions
    for what should be 4 distinct mentions of the same query).
    """

    TEXT = (
        "Kiox 300 is popular. Kiox 400C improves battery life. Kiox 500 is "
        "the top model. A user asked about Kiox compatibility."
    )
    SCHEMA = {"entities": {"display": ["Kiox 300", "Kiox 400C", "Kiox 500", "Kiox"]}}

    @staticmethod
    def _mention_pairs(batch):
        pairs = batch.targets.mention_pairs[0, 0]
        mask = batch.targets.mention_mask[0, 0]
        return {tuple(int(x) for x in p) for p, m in zip(pairs, mask) if m}

    def test_default_filters_phantom_sub_span_matches(self, tiny_tokenizer):
        processor = _processor(tiny_tokenizer)
        collator = ExtractorCollator(
            processor, is_training=True, architecture="boundary", max_gold_per_query=8
        )
        batch = collator([(self.TEXT, self.SCHEMA)])

        assert [q.role_name for q in batch.query_layouts[0].queries] == ["display"]
        # 4 genuine mentions: "Kiox 300", "Kiox 400C", "Kiox 500", bare "Kiox".
        assert self._mention_pairs(batch) == {(0, 2), (5, 7), (11, 13), (22, 23)}

    def test_opt_out_reproduces_the_pre_fix_phantom_blowup(self, tiny_tokenizer):
        processor = _processor(tiny_tokenizer)
        collator = ExtractorCollator(
            processor,
            is_training=True,
            architecture="boundary",
            max_gold_per_query=8,
            dedupe_contained_entity_spans=False,
        )
        batch = collator([(self.TEXT, self.SCHEMA)])

        # 7 raw positions: the 4 genuine mentions plus 3 phantom sub-spans
        # that are really the leading token of a longer, real mention.
        assert self._mention_pairs(batch) == {
            (0, 2), (5, 7), (11, 13), (22, 23),  # genuine
            (0, 1), (5, 6), (11, 12),  # phantom sub-spans of the above
        }

    def test_without_the_fix_phantom_matches_can_exceed_gold_capacity(self, tiny_tokenizer):
        """Reproduces the real-world symptom: capacity errors from phantoms.

        With ``max_gold_per_query=6`` the 4 genuine mentions fit comfortably,
        but the pre-fix behavior's 7 raw positions overflow -- exactly the
        "TargetCapacityError only fires as a lucky accident" failure mode
        described in the bug report. The real fix is the containment filter,
        not raising ``max_gold_per_query``.
        """
        processor = _processor(tiny_tokenizer)

        fixed_collator = ExtractorCollator(
            processor, is_training=True, architecture="boundary", max_gold_per_query=6
        )
        fixed_collator([(self.TEXT, self.SCHEMA)])  # does not raise

        buggy_collator = ExtractorCollator(
            processor,
            is_training=True,
            architecture="boundary",
            max_gold_per_query=6,
            dedupe_contained_entity_spans=False,
        )
        with pytest.raises(TargetCapacityError):
            buggy_collator([(self.TEXT, self.SCHEMA)])

    def test_filter_is_scoped_per_query_and_per_sample(self, tiny_tokenizer):
        """A short match under one query must not be filtered by a longer
        match belonging to a *different* query, even at an identical span."""
        processor = _processor(tiny_tokenizer)
        collator = ExtractorCollator(
            processor, is_training=True, architecture="boundary", max_gold_per_query=8
        )
        # Two distinct entity types both listing "Kiox 300"/"Kiox" so their
        # raw matches occupy the same token positions but different queries.
        batch = collator([
            (
                self.TEXT,
                {
                    "entities": {
                        "display": ["Kiox 300"],
                        "mentioned_brand": ["Kiox"],
                    }
                },
            )
        ])
        layout = batch.query_layouts[0]
        role_to_qid = {q.role_name: q.query_id for q in layout.queries}
        pairs = batch.targets.mention_pairs[0]
        mask = batch.targets.mention_mask[0]

        def _pairs_for(role):
            qid = role_to_qid[role]
            return {tuple(int(x) for x in p) for p, m in zip(pairs[qid], mask[qid]) if m}

        # "display" only ever searched "Kiox 300": one clean match, nothing
        # to filter against within its own query.
        assert _pairs_for("display") == {(0, 2)}
        # "mentioned_brand" only ever searched bare "Kiox": every occurrence
        # (including the ones at the start of "Kiox 300"/"400C"/"500") is a
        # genuine match *for this query*, since it never searched the longer
        # values -- there is nothing in this query's own result set to
        # contain them.
        assert _pairs_for("mentioned_brand") == {(0, 1), (5, 6), (11, 12), (22, 23)}
