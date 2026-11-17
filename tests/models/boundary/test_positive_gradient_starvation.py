"""Tests for positive-gradient starvation and unresolved-field supervision fixes."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from gliner2.configuration import BoundaryHeadSettings, validate_boundary_head
from gliner2.inference.schema import Schema
from gliner2.models.boundary.losses import candidate_pair_loss
from gliner2.models.boundary.records import (
    DenseRecordBatchOutput,
    _instance_field_loss,
    _list_field_bce,
    compute_dense_batch_loss,
)
from gliner2.processing.records import FieldCardinality, RecordFieldSpec
from gliner2.processing.targets import RecordFieldTarget, RecordTarget
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def test_list_field_bce_splits_positive_and_negative_means():
    row_small = torch.zeros(5, requires_grad=True)   # 4 candidates
    row_wide = torch.zeros(65, requires_grad=True)   # 64 candidates
    _list_field_bce(row_small, [1]).backward()
    _list_field_bce(row_wide, [1]).backward()
    torch.testing.assert_close(row_small.grad[1], torch.tensor(-0.5))
    torch.testing.assert_close(row_wide.grad[1], torch.tensor(-0.5))


def test_instance_field_loss_withholds_unresolved_gold_and_supervises_true_empty():
    fspec = RecordFieldSpec(0, "Attacker", 0, FieldCardinality.OPTIONAL_ONE)

    # Non-empty gold span (5, 6) absent from candidate pool -> withheld (0 loss, 0 ABSENT push)
    row_miss = torch.zeros(4, requires_grad=True)
    grp_miss = SimpleNamespace(
        object_logits=torch.zeros(1),
        field_specs=(fspec,),
        assign_logits=[row_miss.unsqueeze(0)],
    )
    non_empty = RecordTarget(
        instance_id=0,
        task_index=0,
        fields=(RecordFieldTarget(query_id=0, values=([(5, 6)],)),),
    )
    loss_miss = _instance_field_loss(grp_miss, 0, non_empty, [{(1, 2): 0, (3, 4): 1}])
    assert loss_miss.item() == 0.0

    # Genuinely empty field -> supervises ABSENT (col 0)
    row_empty = torch.zeros(4, requires_grad=True)
    grp_empty = SimpleNamespace(
        object_logits=torch.zeros(1),
        field_specs=(fspec,),
        assign_logits=[row_empty.unsqueeze(0)],
    )
    truly_empty = RecordTarget(instance_id=0, task_index=0, fields=())
    loss_empty = _instance_field_loss(grp_empty, 0, truly_empty, [{(1, 2): 0, (3, 4): 1}])
    loss_empty.backward()
    assert row_empty.grad[0].item() < -0.7


def test_dense_batch_loss_splits_list_bce_and_withholds_unresolved_fields():
    # 1 batch, 1 natural group, 1 instance, 2 fields (anchor + scalar optional), 4 pool candidates
    assign = torch.zeros(1, 1, 1, 2, 5, requires_grad=True)
    dense = DenseRecordBatchOutput(
        object_logits=torch.zeros(1, 1, 1),
        assign_logits=assign,
        instance_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        group_mask=torch.ones(1, 1, dtype=torch.bool),
        field_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        scalar_fields=torch.tensor([[[True, True]]]),
        modes=torch.zeros(1, 1, dtype=torch.long),
        anchor_fields=torch.zeros(1, 1, dtype=torch.long),
        field_membership=torch.ones(1, 1, 2, 4, dtype=torch.bool),
        pool_spans=torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4]]]),
    )
    # Gold: anchor resolved at cand 0; field 1 had gold (has_gold=True) that missed the pool (gold_indicator=False)
    gold_indicator = torch.zeros(1, 1, 1, 2, 4, dtype=torch.bool)
    gold_indicator[0, 0, 0, 0, 0] = True
    has_gold = torch.tensor([[[[True, True]]]])
    record_mask = torch.ones(1, 1, 1, dtype=torch.bool)

    losses = compute_dense_batch_loss(dense, gold_indicator, record_mask, has_gold=has_gold)
    losses["field_loss"].backward()
    # Unresolved field 1 must receive zero gradient on ABSENT (col 0)
    assert assign.grad[0, 0, 0, 1, 0].item() == 0.0


def test_candidate_pair_loss_split_positive_negative_and_setting_default():
    cfg = BoundaryHeadSettings(**validate_boundary_head({}))
    assert cfg.pair_split_reduction is True

    logits = torch.zeros(1, 20, 10, requires_grad=True)
    labels = torch.zeros(1, 20, 10)
    labels[0, 0, 0] = 1.0
    valid = torch.ones(1, 20, 10, dtype=torch.bool)
    loss = candidate_pair_loss(logits, labels, valid, split_positive_negative=True)
    loss.backward()
    torch.testing.assert_close(logits.grad[0, 0, 0], torch.tensor(-0.5))


def test_boundary_model_record_forward_backward_and_assignment_confidence():
    model = build_tiny_boundary_model()
    sch = (
        Schema()
        .structure("incident", mode="natural", anchor="trigger")
        .field("trigger")
        .field("actor", cardinality="zero_or_more")
        .build()
    )
    train_schema = dict(sch)
    train_schema["json_structures"] = [{"incident": {"trigger": "hit", "actor": ["Alice"]}}]
    batch = model.processor.collate_fn_train(
        [("Alice hit Bob yesterday.", train_schema)], architecture="boundary"
    )
    out = model(batch)
    assert torch.isfinite(out.total_loss)
    out.total_loss.backward()
