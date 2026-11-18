"""PR-29--32 dense structured-head contracts."""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F

from gliner2.configuration import BoundaryHeadSettings, migrate_config_dict
from gliner2.models.base import QueryLayout
from gliner2.models.boundary.records import (
    DenseRecordBatchOutput,
    DenseRecordGroupOutput,
    build_dense_record_cost,
    compute_dense_batch_loss,
    compute_dense_group_loss,
)
from gliner2.models.boundary.relations import (
    RelationProposalSettings,
    RelationTypeSpec,
    TypedRelationPairGenerator,
)
from gliner2.models.outputs import CandidateTensorBatch
from gliner2.processing.records import (
    FieldCardinality,
    RecordFieldSpec,
    RecordSpec,
)
from gliner2.processing.targets import (
    RecordFieldTarget,
    RecordTarget,
    TargetCapacityError,
)


def _relation_candidates(batch: int, queries: int, candidates: int):
    starts = torch.arange(candidates).view(1, 1, candidates)
    indices = torch.stack((starts, starts + 1), -1).expand(batch, queries, -1, -1).clone()
    logits = torch.randn(batch, queries, candidates)
    valid = torch.rand(batch, queries, candidates) > 0.2
    return CandidateTensorBatch(
        indices=indices,
        proposal_logits=logits,
        pair_logits=logits,
        valid_mask=valid,
        query_mask=torch.ones(batch, queries, dtype=torch.bool),
    )


def _reference_pairs(candidates, specs, settings):
    result = set()
    for b, sample_specs in enumerate(specs):
        for r, spec in enumerate(sample_specs):
            heads, tails = [], []
            for qids, output in ((spec.head_query_ids, heads), (spec.tail_query_ids, tails)):
                for qid in qids:
                    for c in range(candidates.indices.shape[2]):
                        prob = float(torch.sigmoid(candidates.pair_logits[b, qid, c]))
                        if candidates.valid_mask[b, qid, c] and prob >= settings.argument_threshold:
                            span = candidates.indices[b, qid, c]
                            output.append((prob, int(span[0]), int(span[1]), qid))
                output.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
            heads = heads[: settings.heads_per_relation]
            tails = tails[: settings.tails_per_relation]
            pairs = [
                (hp * tp, hs, he, ts, te)
                for hp, hs, he, _ in heads
                for tp, ts, te, _ in tails
                if spec.allow_self or (hs, he) != (ts, te)
            ]
            pairs.sort(key=lambda x: -x[0])
            result.update((b, r, *pair[1:]) for pair in pairs[: settings.pair_cap])
    return result


def test_tensorized_relation_pair_set_matches_reference_randomized():
    random.seed(9)
    torch.manual_seed(9)
    for _ in range(25):
        candidates = _relation_candidates(3, 4, 7)
        specs = [
            [
                RelationTypeSpec("r0", (0, 1), (2, 3), allow_self=False),
                RelationTypeSpec("r1", (2,), (0, 3), allow_self=True),
            ]
            for _ in range(3)
        ]
        settings = RelationProposalSettings(
            heads_per_relation=4, tails_per_relation=5, pair_cap=11,
            argument_threshold=0.25,
        )
        pairs = TypedRelationPairGenerator(settings).generate_batched(
            candidates, [QueryLayout(queries=())] * 3, specs, compact=False
        )
        mask = pairs.pair_mask
        got = set(zip(
            pairs.batch_index[mask].tolist(), pairs.relation_index[mask].tolist(),
            pairs.head_start[mask].tolist(), pairs.head_end[mask].tolist(),
            pairs.tail_start[mask].tolist(), pairs.tail_end[mask].tolist(),
        ))
        assert got == _reference_pairs(candidates, specs, settings)


def test_dense_record_cost_matches_scalar_reference():
    torch.manual_seed(4)
    instances, fields, candidates, gold_count = 5, 2, 7, 3
    spec = RecordSpec(
        task_index=0, task_name="x", task_type="json_structures", mode="latent",
        fields=(
            RecordFieldSpec(0, "scalar", 0, FieldCardinality.OPTIONAL_ONE),
            RecordFieldSpec(1, "list", 1, FieldCardinality.ZERO_OR_MORE),
        ),
    )
    group = DenseRecordGroupOutput(
        spec=spec,
        object_logits=torch.randn(instances),
        assign_logits=torch.randn(instances, fields, 1 + candidates),
        instance_mask=torch.ones(instances, dtype=torch.bool),
        field_membership=torch.ones(fields, candidates, dtype=torch.bool),
        pool_spans=torch.stack((torch.arange(candidates), torch.arange(1, candidates + 1)), -1),
        field_specs=spec.fields,
        field_query_ids=torch.tensor([0, 1]),
        instance_pool_index=torch.arange(instances),
    )
    gold = torch.rand(gold_count, fields, candidates) > 0.75
    cost = build_dense_record_cost(group, gold)
    reference = torch.empty_like(cost)
    for i in range(instances):
        for n in range(gold_count):
            score = F.logsigmoid(group.object_logits[i])
            scalar_cols = gold[n, 0].nonzero().flatten() + 1
            if not len(scalar_cols):
                scalar_cols = torch.tensor([0])
            score += torch.logsumexp(F.log_softmax(group.assign_logits[i, 0], -1)[scalar_cols], 0)
            score -= F.binary_cross_entropy_with_logits(
                group.assign_logits[i, 1, 1:], gold[n, 1].float(), reduction="sum"
            )
            reference[i, n] = -score
    torch.testing.assert_close(cost, reference)


def test_dense_natural_loss_skips_missing_anchor_instead_of_using_row_zero():
    spec = RecordSpec(
        task_index=0,
        task_name="event",
        task_type="json_structures",
        mode="natural",
        fields=(
            RecordFieldSpec(
                0, "anchor", 0, FieldCardinality.REQUIRED_ONE, is_anchor=True
            ),
        ),
        anchor_query_id=0,
    )
    group = DenseRecordGroupOutput(
        spec=spec,
        object_logits=torch.zeros(2),
        assign_logits=torch.zeros(2, 1, 3),
        instance_mask=torch.ones(2, dtype=torch.bool),
        field_membership=torch.ones(1, 2, dtype=torch.bool),
        pool_spans=torch.tensor([[0, 1], [2, 3]]),
        field_specs=spec.fields,
        field_query_ids=torch.tensor([0]),
        instance_pool_index=torch.tensor([0, 1]),
    )
    missing = RecordTarget(
        "event:0",
        0,
        (RecordFieldTarget(0, (((5, 6),),)),),
        anchor_query_id=0,
    )
    losses = compute_dense_group_loss(group, [missing])
    assert losses["field_loss"].item() == 0.0
    assert losses["field_count"] == 0


def test_dense_record_capacity_counts_only_real_instances():
    spec = RecordSpec(
        task_index=0,
        task_name="event",
        task_type="json_structures",
        mode="latent",
        fields=(RecordFieldSpec(0, "value", 0, FieldCardinality.OPTIONAL_ONE),),
    )
    group = DenseRecordGroupOutput(
        spec=spec,
        object_logits=torch.zeros(4),
        assign_logits=torch.zeros(4, 1, 5),
        instance_mask=torch.tensor([True, False, False, False]),
        field_membership=torch.ones(1, 4, dtype=torch.bool),
        pool_spans=torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]]),
        field_specs=spec.fields,
        field_query_ids=torch.tensor([0]),
        instance_pool_index=torch.arange(4),
    )
    records = [
        RecordTarget(str(index), 0, (), anchor_query_id=None)
        for index in range(2)
    ]
    with pytest.raises(TargetCapacityError, match="1 valid instance"):
        compute_dense_group_loss(group, records)


def _list_group(assign_logits: torch.Tensor, real: int) -> DenseRecordGroupOutput:
    spec = RecordSpec(
        task_index=0, task_name="x", task_type="json_structures", mode="latent",
        fields=(RecordFieldSpec(0, "items", 0, FieldCardinality.ZERO_OR_MORE),),
    )
    width = assign_logits.shape[-1] - 1
    membership = torch.zeros(1, width, dtype=torch.bool)
    membership[:, :real] = True
    return DenseRecordGroupOutput(
        spec=spec,
        object_logits=torch.zeros(1),
        assign_logits=assign_logits,
        instance_mask=torch.ones(1, dtype=torch.bool),
        field_membership=membership,
        pool_spans=torch.stack((torch.arange(width), torch.arange(1, width + 1)), -1),
        field_specs=spec.fields,
        field_query_ids=torch.tensor([0]),
        instance_pool_index=torch.tensor([0]),
    )


def test_dense_list_loss_ignores_pool_padding():
    """Issue #180: padding columns must not dilute the list-field term."""
    torch.manual_seed(7)
    real, padded = 3, 16
    logits = torch.randn(1, 1, 1 + padded)
    gold = torch.zeros(1, 1, padded, dtype=torch.bool)
    gold[0, 0, 1] = True
    record = [RecordTarget("0", 0, (), anchor_query_id=None)]

    loss_padded = compute_dense_group_loss(_list_group(logits, real), record, gold)
    loss_real = compute_dense_group_loss(
        _list_group(logits[..., : 1 + real], real), record, gold[..., :real]
    )
    expected = F.binary_cross_entropy_with_logits(
        logits[0, 0, 1 : 1 + real], gold[0, 0, :real].float()
    )
    torch.testing.assert_close(loss_padded["field_loss"], expected)
    torch.testing.assert_close(loss_real["field_loss"], expected)


def test_dense_batch_list_loss_ignores_pool_padding():
    torch.manual_seed(8)
    real, padded = 3, 16
    logits = torch.randn(1, 1, 1, 1, 1 + padded)
    membership = torch.zeros(1, 1, 1, padded, dtype=torch.bool)
    membership[..., :real] = True
    gold = torch.zeros(1, 1, 1, 1, padded, dtype=torch.bool)
    gold[..., 1] = True
    output = DenseRecordBatchOutput(
        object_logits=torch.zeros(1, 1, 1),
        assign_logits=logits,
        instance_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        field_membership=membership,
        pool_spans=torch.stack((torch.arange(padded), torch.arange(1, padded + 1)), -1)[None],
        field_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        scalar_fields=torch.zeros(1, 1, 1, dtype=torch.bool),
        modes=torch.ones(1, 1, dtype=torch.long),
        anchor_fields=torch.zeros(1, 1, dtype=torch.long),
        group_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    losses = compute_dense_batch_loss(output, gold, torch.ones(1, 1, 1, dtype=torch.bool))
    expected = F.binary_cross_entropy_with_logits(
        logits[0, 0, 0, 0, 1 : 1 + real], gold[0, 0, 0, 0, :real].float()
    )
    torch.testing.assert_close(losses["field_loss"], expected)


def test_new_and_legacy_structured_config_defaults():
    assert BoundaryHeadSettings().enable_records
    assert BoundaryHeadSettings().enable_relations
    migrated = migrate_config_dict({
        "architecture": "boundary", "config_version": 2, "boundary_head": {}
    })
    assert migrated["boundary_head"]["enable_records"] is False
    assert migrated["boundary_head"]["enable_relations"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_relation_training_pair_generation_has_no_host_transfer_cuda(monkeypatch):
    candidates = _relation_candidates(2, 3, 8)
    candidates = CandidateTensorBatch(
        **{
            name: getattr(candidates, name).cuda()
            for name in ("indices", "proposal_logits", "pair_logits", "valid_mask", "query_mask")
        }
    )
    original_cpu = torch.Tensor.cpu

    def forbidden_cpu(self, *args, **kwargs):
        raise AssertionError("relation proposal training path transferred to CPU")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_cpu)
    try:
        TypedRelationPairGenerator().generate_batched(
            candidates,
            [QueryLayout(queries=())] * 2,
            [[RelationTypeSpec("r", (0,), (1,))] for _ in range(2)],
            compact=False,
        )
    finally:
        monkeypatch.setattr(torch.Tensor, "cpu", original_cpu)
