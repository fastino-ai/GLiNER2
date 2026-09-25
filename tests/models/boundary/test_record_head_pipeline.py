"""Integrated record head: forward, marginalized loss, and global decode.

These tests exercise ``RecordHead`` + ``compute_group_loss`` + ``decode_group``
on synthetic candidate batches (no encoder), so they are fast and deterministic
while covering the full instance-formation contract for all three modes.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytest
import torch

from gliner2.models.outputs import CandidateTensorBatch
from gliner2.models.boundary.records import RecordHead
from gliner2.models.boundary.record_loss import compute_group_loss
from gliner2.models.boundary.record_decode import decode_group, derive_count
from gliner2.processing.records import FieldCardinality, RecordFieldSpec, RecordSpec
from gliner2.processing.targets import RecordFieldTarget, RecordTarget, TargetCapacityError


def _build_tiny_records_model(candidate_pool="per_query"):
    from gliner2 import ExtractorConfig
    from gliner2.inference.engine import BoundaryExtractor
    from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD
    from tests.fixtures.tiny_encoder import build_tiny_encoder_config
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    tokenizer = build_tiny_tokenizer()
    encoder_config = build_tiny_encoder_config(vocab_size=len(tokenizer))
    head = dict(TINY_BOUNDARY_HEAD)
    head.update(
        enable_records=True,
        record_dim=24,
        record_instance_queries=8,
        candidate_pool=candidate_pool,
    )
    config = ExtractorConfig(
        model_name="tiny-bert-fixture",
        architecture="boundary",
        boundary_head=head,
        token_pooling="first",
    )
    torch.manual_seed(7)
    model = BoundaryExtractor(config, encoder_config=encoder_config, tokenizer=tokenizer)
    model.eval()
    return model


def test_records_model_exposes_head_and_candidate_states():
    model = _build_tiny_records_model()
    assert "record_decoder" in model.task_module_names()
    assert hasattr(model, "record_decoder")

    B, L, Q, H = 2, 12, 2, model.hidden_size
    out = model.boundary_head(
        torch.randn(B, L, H), torch.ones(B, L, dtype=torch.bool),
        torch.randn(B, Q, H), torch.ones(B, Q, dtype=torch.bool),
    )
    cs = out.candidates.candidate_states
    assert cs is not None
    assert cs.shape[:2] == (B, Q) and cs.shape[-1] == H


def test_lora_record_head_alias_resolves_to_record_decoder():
    from gliner2.training.lora import _resolve_targets

    model = _build_tiny_records_model()
    resolved = _resolve_targets(model, ["record_head"])
    assert resolved, "record_head alias resolved to no modules"
    assert all(name.startswith("record_decoder.") for name in resolved)


def test_records_training_loss_flows_through_model():
    from gliner2.processor import SamplingConfig, SchemaTransformer
    from gliner2.training import ExtractorCollator
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    model = _build_tiny_records_model()
    model.train()
    proc = SchemaTransformer(
        tokenizer=build_tiny_tokenizer(),
        sampling_config=SamplingConfig(
            remove_json_structure_prob=0.0, shuffle_json_fields=False,
            remove_json_field_prob=0.0, synthetic_entity_label_prob=0.0,
        ),
    )
    collator = ExtractorCollator(
        proc, is_training=True, architecture="boundary", max_gold_per_query=16
    )
    output_dict = {
        "json_structures": [
            {"purchase": {"buyer": "Alice", "item": "apples"}},
            {"purchase": {"buyer": "Bob", "item": "oranges"}},
        ],
        "record_metadata": {"purchase": {"mode": "natural", "anchor": "buyer"}},
    }
    batch = collator([("Alice bought apples and Bob sold oranges", output_dict)])
    assert isinstance(batch.targets.record_targets, tuple)
    assert len(batch.targets.record_targets) == 9
    if batch.query_layouts[0].extractive_count() == 0:
        pytest.skip("tiny tokenizer produced no extractive queries for the structure")

    out = model.forward(batch)
    assert out.total_loss is not None and torch.isfinite(out.total_loss)
    assert "record_field_loss" in out.losses
    out.total_loss.backward()
    grads = [p.grad for p in model.record_decoder.parameters() if p.grad is not None]
    assert grads, "no gradient reached the record head"


def test_shared_records_use_fully_batched_training_path(monkeypatch):
    from gliner2.processor import SamplingConfig, SchemaTransformer
    from gliner2.training import ExtractorCollator
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    model = _build_tiny_records_model(candidate_pool="shared")
    model.train()
    monkeypatch.setattr(
        model.record_decoder,
        "forward_group_dense",
        lambda *args, **kwargs: pytest.fail("per-group shared path was used"),
    )
    processor = SchemaTransformer(
        tokenizer=build_tiny_tokenizer(),
        sampling_config=SamplingConfig(
            remove_json_structure_prob=0.0,
            shuffle_json_fields=False,
            remove_json_field_prob=0.0,
            synthetic_entity_label_prob=0.0,
        ),
    )
    batch = ExtractorCollator(
        processor,
        is_training=True,
        architecture="boundary",
        max_gold_per_query=16,
    )([(
        "Alice bought apples",
        {
            "json_structures": [
                {"purchase": {"buyer": "Alice", "item": "apples"}}
            ],
            "record_metadata": {
                "purchase": {"mode": "natural", "anchor": "buyer"}
            },
        },
    )])
    output = model(batch)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.record_decoder.parameters()
    )


def test_engine_decode_records_emits_public_structure_shape():
    from types import SimpleNamespace

    model = _build_tiny_records_model()
    hidden = model.hidden_size
    tokens = ["Alice", "bought", "apples", "then", "Bob", "left", "now"]
    text = " ".join(tokens)
    start_map, end_map, pos = [], [], 0
    for tok in tokens:
        start_map.append(pos)
        end_map.append(pos + len(tok))
        pos += len(tok) + 1

    # anchors "Alice"(0,1) and "Bob"(4,5); one item candidate "apples"(2,3).
    cands = make_candidates([[(0, 1), (4, 5)], [(2, 3)]], hidden, high_logit_field=0)
    spec = RecordSpec(
        task_index=0, task_name="purchase", task_type="json_structures", mode="natural",
        fields=(
            RecordFieldSpec(0, "buyer", 0, FieldCardinality.REQUIRED_ONE, is_anchor=True),
            RecordFieldSpec(1, "item", 1, FieldCardinality.OPTIONAL_ONE),
        ),
        anchor_query_id=0,
    )
    batch = SimpleNamespace(record_specs=({0: spec},))
    core = {"query_states": torch.randn(1, 2, hidden)}

    out = model._decode_records(
        batch, 0, core, cands, offset=0, start_map=start_map, end_map=end_map,
        text=text, text_len=len(tokens), include_confidence=False, include_spans=False,
    )
    assert "purchase" in out
    instances = out["purchase"]
    # Both confident anchors become records; anchor field binds its own surface.
    assert len(instances) == 2
    buyers = {inst["buyer"] for inst in instances}
    assert buyers == {"Alice", "Bob"}

    scored = model._decode_records(
        batch, 0, core, cands, offset=0, start_map=start_map, end_map=end_map,
        text=text, text_len=len(tokens), include_confidence=True, include_spans=False,
    )
    assert all("confidence" not in inst for inst in instances)
    assert all(0.0 <= inst["confidence"] <= 1.0 for inst in scored["purchase"])


def test_engine_choice_field_uses_per_record_assignment_not_global_fallback(
    monkeypatch,
):
    """A ``choices=`` field inside a repeated record must not collapse every
    instance to the same document-level answer.

    ``decode_group()`` already assigns each record instance its own field
    candidate (see the hand-crafted ``assign_logits`` below: instance 0 is
    forced onto the "positive" enum token, instance 1 onto "negative"). But
    those enum tokens live in the schema prefix (positions before ``offset``),
    which ``_format_field``'s document-range bounds check used to drop
    unconditionally -- so every record's ``formatted`` list came back empty
    and every record fell through to ``_decode_choice_field()``'s
    document/field-level fallback instead, which scores the enum choices once
    per field (not once per record) and is therefore identical for every
    instance. We pin that fallback to a fixed, obviously-wrong answer
    ("negative" for everyone) via ``score_explicit_spans`` below: if the two
    records still come back identical, the per-record signal was discarded.
    """
    from types import SimpleNamespace

    model = _build_tiny_records_model()
    hidden = model.hidden_size

    # Schema prefix (positions 0-1): the literal enum tokens for `sentiment`.
    # Document (positions 2-8, offset=2): "Alice bought apples then Bob left now".
    tokens = ["Alice", "bought", "apples", "then", "Bob", "left", "now"]
    text = " ".join(tokens)
    start_map, end_map, pos = [], [], 0
    for tok in tokens:
        start_map.append(pos)
        end_map.append(pos + len(tok))
        pos += len(tok) + 1
    offset = 2

    # field 0 (anchor, "name"): "Alice" abs (2,3), "Bob" abs (6,7).
    # field 1 ("sentiment", choices): "positive" abs (0,1), "negative" abs (1,2).
    cands = make_candidates(
        [[(offset + 0, offset + 1), (offset + 4, offset + 5)], [(0, 1), (1, 2)]],
        hidden,
        high_logit_field=0,
    )
    spec = RecordSpec(
        task_index=0, task_name="dish", task_type="json_structures", mode="natural",
        fields=(
            RecordFieldSpec(0, "name", 0, FieldCardinality.REQUIRED_ONE, is_anchor=True),
            RecordFieldSpec(1, "sentiment", 1, FieldCardinality.REQUIRED_ONE),
        ),
        anchor_query_id=0,
    )
    batch = SimpleNamespace(
        record_specs=({0: spec},),
        text_tokens=[["positive", "negative"] + tokens],
    )
    total_tokens = offset + len(tokens)
    core = {
        "query_states": torch.randn(1, 2, hidden),
        "query_mask": torch.ones(1, 2, dtype=torch.bool),
        "text_states": torch.randn(1, total_tokens, hidden),
        "text_mask": torch.ones(1, total_tokens, dtype=torch.bool),
    }

    # Force instance 0 (Alice) onto "positive" and instance 1 (Bob) onto
    # "negative" in decode_group's own per-instance assignment.
    original_forward_group = model.record_decoder.forward_group

    def patched_forward_group(spec_, query_states_i, candidates_, sample_index_):
        group = original_forward_group(spec_, query_states_i, candidates_, sample_index_)
        with torch.no_grad():
            group.assign_logits[1] = torch.tensor([
                [-10.0, 10.0, -10.0],   # instance 0 -> candidate 0 ("positive")
                [-10.0, -10.0, 10.0],   # instance 1 -> candidate 1 ("negative")
            ])
        return group

    monkeypatch.setattr(model.record_decoder, "forward_group", patched_forward_group)

    # The document/field-level fallback _decode_choice_field() falls back to
    # if per-record data is discarded: pin it to a fixed wrong answer so a
    # regression (both records reading "negative") is unambiguous.
    monkeypatch.setattr(
        model.boundary_head,
        "score_explicit_spans",
        lambda *args, **kwargs: torch.tensor([[[-5.0, 5.0]]]),
    )

    out = model._decode_records(
        batch, 0, core, cands, offset=offset, start_map=start_map, end_map=end_map,
        text=text, text_len=len(tokens), include_confidence=True, include_spans=False,
        metadata={"field_metadata": {"dish.sentiment": {"choices": ["positive", "negative"]}}},
    )

    assert "dish" in out
    by_name = {inst["name"]["text"]: inst["sentiment"]["text"] for inst in out["dish"]}
    assert by_name == {"Alice": "positive", "Bob": "negative"}


def test_engine_does_not_fall_back_to_legacy_for_empty_record_decode(
    monkeypatch,
):
    model = _build_tiny_records_model()
    seen = {}

    def capture_records(*args, **kwargs):
        seen["record_threshold"] = kwargs.get("threshold")
        return {}

    monkeypatch.setattr(model, "_decode_records", capture_records)

    def capture_legacy(
        batch,
        sample_index,
        core,
        specs,
        grouped_candidates,
        metadata,
        skip_names,
        *args,
    ):
        seen["skip_names"] = skip_names
        return {}

    monkeypatch.setattr(model, "_decode_legacy_structures", capture_legacy)
    schema = model.create_schema()
    (
        schema.structure("purchase", mode="natural", anchor="buyer")
        .field("buyer", dtype="str")
        .field("item", dtype="str")
    )

    model.extract("Alice bought apples", schema, threshold=0.17)

    assert seen["skip_names"] == {"purchase"}
    assert seen["record_threshold"] == 0.17


def test_boundary_json_convenience_schema_enables_record_mode():
    model = _build_tiny_records_model()

    schema = model._json_schema({
        "transaction": [
            "merchant::str",
            "amount::str",
            "tags::list",
        ],
    })
    built = schema.build()

    assert built["record_metadata"]["transaction"] == {
        "mode": "natural",
        "anchor": "merchant",
        "fields": {
            "merchant": {
                "cardinality": "required_one",
                "exclusive": True,
            },
            "amount": {
                "cardinality": "required_one",
                "exclusive": True,
            },
            "tags": {
                "cardinality": "zero_or_more",
                "exclusive": True,
            },
        },
    }


def test_choice_decoder_honors_record_local_preference(monkeypatch):
    from types import SimpleNamespace

    model = _build_tiny_records_model()
    hidden = model.hidden_size
    monkeypatch.setattr(
        model.boundary_head,
        "score_explicit_spans",
        lambda *args, **kwargs: torch.tensor([[[0.0, 2.0]]]),
    )
    batch = SimpleNamespace(text_tokens=[["food", "transport"]])
    core = {
        "text_states": torch.randn(1, 2, hidden),
        "text_mask": torch.ones(1, 2, dtype=torch.bool),
        "query_states": torch.randn(1, 1, hidden),
        "query_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    value = model._decode_choice_field(
        batch,
        sample_index=0,
        core=core,
        query_id=0,
        choices=["food", "transport"],
        dtype="str",
        configured_threshold=None,
        default_threshold=0.5,
        prefix_length=2,
        include_confidence=True,
        preferred_choices=[("transport", 10, 19)],
        include_spans=True,
    )

    assert value["text"] == "transport"
    assert value["start"] == 10
    assert value["end"] == 19
    assert value["confidence"] > 0.5


def test_literal_choices_bind_to_preceding_record_anchor():
    from gliner2.inference.engine import BoundaryExtractor

    text = (
        "Amazon charged $12 for books. "
        "Amazon charged $8 for music."
    )
    has_mentions, assigned = (
        BoundaryExtractor._record_local_choice_mentions(
            text,
            ["books", "music", "food"],
            [(0, 6), (30, 36)],
        )
    )

    assert has_mentions is True
    assert assigned == {
        0: [("books", 23, 28)],
        1: [("music", 52, 57)],
    }


def test_literal_choices_stay_with_preceding_anchor_across_clauses():
    from gliner2.inference.engine import BoundaryExtractor

    text = (
        "Hilton was booked. The room includes breakfast and parking. "
        "Grand Hotel was booked. Amenities include breakfast, wifi, and spa."
    )
    second_anchor = text.index("Grand Hotel")
    has_mentions, assigned = (
        BoundaryExtractor._record_local_choice_mentions(
            text,
            ["breakfast", "parking", "wifi", "spa"],
            [(0, len("Hilton")), (second_anchor, second_anchor + 11)],
        )
    )

    assert has_mentions is True
    assert [choice for choice, _, _ in assigned[0]] == [
        "breakfast",
        "parking",
    ]
    assert [choice for choice, _, _ in assigned[1]] == [
        "breakfast",
        "wifi",
        "spa",
    ]


def make_candidates(
    fields: List[List[Tuple[int, int]]],
    hidden: int,
    *,
    high_logit_field: int = None,
) -> CandidateTensorBatch:
    """Build a single-sample candidate batch: ``fields[q]`` = spans for query q."""
    q = len(fields)
    c = max(len(f) for f in fields)
    indices = torch.zeros(1, q, c, 2, dtype=torch.long)
    valid = torch.zeros(1, q, c, dtype=torch.bool)
    pair = torch.zeros(1, q, c)
    states = torch.randn(1, q, c, hidden)
    for qi, spans in enumerate(fields):
        for ci, (s, e) in enumerate(spans):
            indices[0, qi, ci, 0] = s
            indices[0, qi, ci, 1] = e
            valid[0, qi, ci] = True
            if high_logit_field is not None and qi == high_logit_field:
                pair[0, qi, ci] = 6.0
    return CandidateTensorBatch(
        indices=indices,
        proposal_logits=pair.clone(),
        pair_logits=pair,
        valid_mask=valid,
        query_mask=torch.ones(1, q, dtype=torch.bool),
        candidate_states=states,
    )


def test_record_head_masks_out_of_range_field_query_ids():
    hidden = 24
    candidates = make_candidates([[(0, 1), (2, 3)]], hidden)
    spec = RecordSpec(
        task_index=0,
        task_name="event",
        task_type="json_structures",
        mode="natural",
        fields=(
            RecordFieldSpec(
                7, "anchor", 0, FieldCardinality.REQUIRED_ONE, is_anchor=True
            ),
        ),
        anchor_query_id=7,
    )
    head = RecordHead(hidden, record_dim=hidden, instance_queries=4)
    group = head.forward_group_dense(
        spec,
        torch.randn(1, hidden),
        candidates,
        sample_index=0,
    )
    assert not group.instance_mask.any()
    assert not group.field_membership.any()
    assert torch.isfinite(group.assign_logits).all()


def test_natural_mode_overfits_two_records_and_derives_count():
    torch.manual_seed(0)
    hidden = 24
    # query 0 = anchor (buyer): 2 anchor spans -> 2 instances.
    # query 1 = item (scalar): 3 candidates.
    cands = make_candidates(
        [[(0, 1), (3, 4)], [(1, 2), (4, 5), (6, 7)]],
        hidden,
        high_logit_field=0,  # anchors are confidently detected
    )
    spec = RecordSpec(
        task_index=0, task_name="purchase", task_type="json_structures",
        mode="natural",
        fields=(
            RecordFieldSpec(0, "buyer", 0, FieldCardinality.REQUIRED_ONE, is_anchor=True),
            RecordFieldSpec(1, "item", 1, FieldCardinality.OPTIONAL_ONE),
        ),
        anchor_query_id=0,
    )
    query_states = torch.randn(2, hidden)
    records = [
        RecordTarget("0:0", 0, (
            RecordFieldTarget(0, (((0, 1),),)),
            RecordFieldTarget(1, (((1, 2),),)),
        ), anchor_query_id=0),
        RecordTarget("0:1", 0, (
            RecordFieldTarget(0, (((3, 4),),)),
            RecordFieldTarget(1, (((4, 5),),)),
        ), anchor_query_id=0),
    ]

    head = RecordHead(hidden, record_dim=24, instance_queries=8)
    opt = torch.optim.Adam(head.parameters(), lr=5e-3)
    for _ in range(400):
        opt.zero_grad()
        group = head.forward_group(spec, query_states, cands, 0)
        losses = compute_group_loss(group, records)
        (losses["object_loss"] + losses["field_loss"]).backward()
        opt.step()

    with torch.no_grad():
        group = head.forward_group(spec, query_states, cands, 0)
    decoded = decode_group(group, anchor_threshold=0.5, field_threshold=0.2)
    assert derive_count(decoded) == 2
    by_anchor = {rec.anchor_span: rec.fields[1] for rec in decoded}
    assert by_anchor[(0, 1)] == [(1, 2)]
    assert by_anchor[(3, 4)] == [(4, 5)]


def test_latent_mode_recovers_grouped_records():
    torch.manual_seed(1)
    hidden = 24
    # two fields, 3 candidates each; two gold records grouping specific cands.
    cands = make_candidates([[(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4), (5, 6)]], hidden)
    spec = RecordSpec(
        task_index=0, task_name="deal", task_type="json_structures", mode="latent",
        fields=(
            RecordFieldSpec(0, "a", 0, FieldCardinality.OPTIONAL_ONE),
            RecordFieldSpec(1, "b", 1, FieldCardinality.OPTIONAL_ONE),
        ),
    )
    query_states = torch.randn(2, hidden)
    records = [
        RecordTarget("0:0", 0, (
            RecordFieldTarget(0, (((0, 1),),)),
            RecordFieldTarget(1, (((1, 2),),)),
        )),
        RecordTarget("0:1", 0, (
            RecordFieldTarget(0, (((2, 3),),)),
            RecordFieldTarget(1, (((3, 4),),)),
        )),
    ]
    head = RecordHead(hidden, record_dim=24, instance_queries=8)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    first = None
    for step in range(500):
        opt.zero_grad()
        group = head.forward_group(spec, query_states, cands, 0)
        losses = compute_group_loss(group, records)
        total = losses["object_loss"] + losses["field_loss"]
        if first is None:
            first = float(total.detach())
        last = float(total.detach())
        total.backward()
        opt.step()
    assert last < first  # learning happened

    with torch.no_grad():
        group = head.forward_group(spec, query_states, cands, 0)
    decoded = decode_group(group, anchor_threshold=0.5, field_threshold=0.3)
    got = {
        (tuple(rec.fields.get(0, [])), tuple(rec.fields.get(1, [])))
        for rec in decoded
    }
    assert ((( 0, 1),), ((1, 2),)) in got
    assert (((2, 3),), ((3, 4),)) in got


def test_score_candidates_returns_auxiliary_logits_when_requested():
    from gliner2.processor import SamplingConfig, SchemaTransformer
    from gliner2.training import ExtractorCollator
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    model = _build_tiny_records_model()
    proc = SchemaTransformer(
        tokenizer=build_tiny_tokenizer(),
        sampling_config=SamplingConfig(
            remove_json_structure_prob=0.0, shuffle_json_fields=False,
            remove_json_field_prob=0.0, synthetic_entity_label_prob=0.0,
        ),
    )
    collator = ExtractorCollator(
        proc, is_training=True, architecture="boundary", max_gold_per_query=16
    )
    output_dict = {
        "json_structures": [{"purchase": {"buyer": "Alice", "item": "apples"}}],
        "record_metadata": {"purchase": {"mode": "natural", "anchor": "buyer"}},
    }
    batch = collator([("Alice bought apples", output_dict)])

    plain = model.score_candidates(batch)
    result = model.score_candidates(batch, return_auxiliary_logits=True)

    # Default contract unchanged: a bare CandidateTensorBatch.
    assert not isinstance(plain, tuple)
    # Requested contract: (candidates, aux) with the backing boundary marginals.
    assert isinstance(result, tuple) and len(result) == 2
    candidates, aux = result
    assert torch.equal(plain.indices, candidates.indices)
    assert set(aux) == {"start_logits", "end_logits", "inside_logits"}
    for key in ("start_logits", "end_logits"):
        assert aux[key] is not None
        assert aux[key].dim() == 3 and aux[key].shape[0] == len(batch)


def test_matched_record_loss_keeps_gradients_despite_nograd_cost():
    """The Hungarian cost is built under ``no_grad`` but matched object/field
    losses must still backpropagate into the head parameters."""
    torch.manual_seed(11)
    hidden = 24
    cands = make_candidates([[(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4), (5, 6)]], hidden)
    spec = RecordSpec(
        task_index=0, task_name="deal", task_type="json_structures", mode="latent",
        fields=(
            RecordFieldSpec(0, "a", 0, FieldCardinality.OPTIONAL_ONE),
            RecordFieldSpec(1, "b", 1, FieldCardinality.OPTIONAL_ONE),
        ),
    )
    query_states = torch.randn(2, hidden)
    records = [
        RecordTarget("0:0", 0, (
            RecordFieldTarget(0, (((0, 1),),)),
            RecordFieldTarget(1, (((1, 2),),)),
        )),
        RecordTarget("0:1", 0, (
            RecordFieldTarget(0, (((2, 3),),)),
            RecordFieldTarget(1, (((3, 4),),)),
        )),
    ]
    head = RecordHead(hidden, record_dim=24, instance_queries=8)
    group = head.forward_group(spec, query_states, cands, 0)
    losses = compute_group_loss(group, records)
    total = losses["object_loss"] + losses["field_loss"]
    assert total.requires_grad
    total.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads, "matched record loss must produce gradients on head parameters"
    assert all(torch.isfinite(g).all() for g in grads)


def test_anchorless_capacity_error_when_gold_exceeds_queries():
    torch.manual_seed(2)
    hidden = 16
    cands = make_candidates([[(0, 1), (2, 3), (4, 5)]], hidden)
    spec = RecordSpec(
        task_index=0, task_name="ev", task_type="json_structures", mode="anchorless",
        fields=(RecordFieldSpec(0, "a", 0, FieldCardinality.OPTIONAL_ONE),),
    )
    query_states = torch.randn(1, hidden)
    records = [
        RecordTarget(f"0:{i}", 0, (RecordFieldTarget(0, (((2 * i, 2 * i + 1),),)),))
        for i in range(3)
    ]
    head = RecordHead(hidden, record_dim=16, instance_queries=2)  # < 3 gold
    group = head.forward_group(spec, query_states, cands, 0)
    with pytest.raises(TargetCapacityError):
        compute_group_loss(group, records)


def test_decode_respects_exclusive_candidate():
    torch.manual_seed(3)
    hidden = 16
    # single scalar exclusive field shared between two natural anchors; the
    # winning (higher object) instance should claim the candidate.
    cands = make_candidates([[(0, 1), (3, 4)], [(5, 6)]], hidden, high_logit_field=0)
    spec = RecordSpec(
        task_index=0, task_name="p", task_type="json_structures", mode="natural",
        fields=(
            RecordFieldSpec(0, "anchor", 0, FieldCardinality.REQUIRED_ONE, is_anchor=True),
            RecordFieldSpec(1, "shared", 1, FieldCardinality.OPTIONAL_ONE, exclusive=True),
        ),
        anchor_query_id=0,
    )
    query_states = torch.randn(2, hidden)
    head = RecordHead(hidden, record_dim=16, instance_queries=8)
    # Force both instances to strongly prefer the single shared candidate.
    with torch.no_grad():
        group = head.forward_group(spec, query_states, cands, 0)
        group.assign_logits[1][:] = torch.tensor([[-5.0, 5.0], [-5.0, 5.0]])
    decoded = decode_group(group, anchor_threshold=0.5, field_threshold=0.2)
    holders = [rec for rec in decoded if rec.fields.get(1)]
    assert len(holders) == 1  # exclusivity honored across instances


def test_decode_jointly_assigns_exclusive_scalar_fields():
    torch.manual_seed(4)
    hidden = 16
    cands = make_candidates(
        [[(0, 1), (3, 4)], [(1, 2), (4, 5)]],
        hidden,
        high_logit_field=0,
    )
    spec = RecordSpec(
        task_index=0,
        task_name="purchase",
        task_type="json_structures",
        mode="natural",
        fields=(
            RecordFieldSpec(
                0,
                "buyer",
                0,
                FieldCardinality.REQUIRED_ONE,
                is_anchor=True,
            ),
            RecordFieldSpec(
                1,
                "item",
                1,
                FieldCardinality.REQUIRED_ONE,
                exclusive=True,
            ),
        ),
        anchor_query_id=0,
    )
    head = RecordHead(hidden, record_dim=16, instance_queries=8)
    with torch.no_grad():
        group = head.forward_group(spec, torch.randn(2, hidden), cands, 0)
        # Greedy decoding gives anchor 0 candidate 0 and forces anchor 1 onto
        # candidate 1. The globally optimal one-to-one assignment is reversed.
        group.assign_logits[1][:] = torch.tensor([
            [-10.0, 5.0, 4.0],
            [-10.0, 4.9, -5.0],
        ])

    decoded = decode_group(group, anchor_threshold=0.5, field_threshold=0.2)
    by_anchor = {record.anchor_span: record.fields[1] for record in decoded}
    assert by_anchor[(0, 1)] == [(4, 5)]
    assert by_anchor[(3, 4)] == [(1, 2)]


def test_decode_required_scalar_does_not_select_absent():
    torch.manual_seed(5)
    hidden = 16
    cands = make_candidates(
        [[(0, 1)], [(1, 2)]],
        hidden,
        high_logit_field=0,
    )
    spec = RecordSpec(
        task_index=0,
        task_name="purchase",
        task_type="json_structures",
        mode="natural",
        fields=(
            RecordFieldSpec(
                0,
                "buyer",
                0,
                FieldCardinality.REQUIRED_ONE,
                is_anchor=True,
            ),
            RecordFieldSpec(
                1,
                "item",
                1,
                FieldCardinality.REQUIRED_ONE,
            ),
        ),
        anchor_query_id=0,
    )
    head = RecordHead(hidden, record_dim=16, instance_queries=8)
    with torch.no_grad():
        group = head.forward_group(spec, torch.randn(2, hidden), cands, 0)
        group.assign_logits[1][:] = torch.tensor([[5.0, 4.0]])

    decoded = decode_group(group, anchor_threshold=0.5, field_threshold=0.2)
    assert decoded[0].fields[1] == [(1, 2)]
    assert decoded[0].field_scores[1][0] > 0
