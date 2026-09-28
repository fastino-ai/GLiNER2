"""CPU contracts for the flag-gated shared document candidate pool."""

from __future__ import annotations

import torch
import pytest

from gliner2.configuration import BoundaryHeadSettings, validate_boundary_head
import gliner2.models.boundary.model as boundary_model_module
from gliner2.models.boundary.losses import (
    build_candidate_labels,
    candidate_pair_loss,
    reranker_listwise_loss,
    select_hard_negative_candidates,
)
from gliner2.models.boundary.model import BoundaryHead
from gliner2.models.boundary.pool import (
    DocumentCandidatePool,
    OverlapBiasedCandidateAttention,
    classify_overlap_buckets,
)
from gliner2.processing.targets import MentionTarget, TargetGraph, pad_target_graphs


def _reference_bucket(a, b):
    s1, e1 = a
    s2, e2 = b
    if a == b:
        return 0
    if s1 == s2:
        return 4
    if e1 == e2:
        return 5
    if e1 <= s2:
        return 6
    if e2 <= s1:
        return 7
    if s1 > s2 and e1 < e2:
        return 1
    if s1 < s2 and e1 > e2:
        return 2
    return 3


def test_overlap_buckets_exhaustive_against_reference():
    spans = [(s, e) for s in range(5) for e in range(s + 1, 6)]
    indices = torch.tensor(spans).unsqueeze(0)
    actual = classify_overlap_buckets(indices)[0]
    expected = torch.tensor([
        [_reference_bucket(a, b) for b in spans] for a in spans
    ])
    assert torch.equal(actual, expected)
    assert set(actual.reshape(-1).tolist()) == set(range(8))


def test_overlap_attention_is_permutation_equivariant():
    torch.manual_seed(4)
    layer = OverlapBiasedCandidateAttention(12, 3, 0.0).eval()
    indices = torch.tensor([[[0, 2], [1, 4], [4, 5], [0, 4], [2, 4]]])
    states = torch.randn(1, 5, 12)
    mask = torch.tensor([[True, True, True, True, False]])
    permutation = torch.tensor([2, 0, 4, 1, 3])
    expected = layer(states, indices, mask)[:, permutation]
    actual = layer(
        states[:, permutation], indices[:, permutation], mask[:, permutation]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_pool_retains_query_quota_and_injected_gold():
    pool = DocumentCandidatePool(
        4, pool_boundary_top_k=6, pool_size=5, min_pool_per_query=2
    )
    with torch.no_grad():
        pool.start_projection.weight.zero_()
        pool.start_projection.bias.zero_()
        pool.end_projection.weight.zero_()
        pool.end_projection.bias.zero_()
    states = torch.zeros(1, 7, 4)
    boundary_mask = torch.ones(1, 7, dtype=torch.bool)
    query_mask = torch.ones(1, 2, dtype=torch.bool)
    start = torch.full((1, 2, 7), -10.0)
    end = torch.full((1, 2, 7), -10.0)
    start[0, 0, :2] = torch.tensor([10.0, 9.0])
    end[0, 0, 1:3] = torch.tensor([10.0, 9.0])
    start[0, 1, 4:6] = torch.tensor([10.0, 9.0])
    end[0, 1, 5:7] = torch.tensor([9.0, 10.0])
    gold_pairs = torch.tensor([[[[2, 5]], [[0, 0]]]])
    gold_mask = torch.tensor([[[True], [False]]])
    result = pool(
        states,
        boundary_mask,
        query_mask,
        start,
        end,
        gold_pairs=gold_pairs,
        gold_mask=gold_mask,
        gold_injection_prob=1.0,
    )
    retained = set(map(tuple, result.indices[0, result.mask[0]].tolist()))
    assert (0, 1) in retained
    assert (4, 6) in retained
    assert (2, 5) in retained
    assert result.gold_mask[..., 0].any()


def test_shared_oracle_recall_uses_retained_preinjection_pool():
    pool = DocumentCandidatePool(
        2, pool_boundary_top_k=4, pool_size=1, min_pool_per_query=0
    )
    with torch.no_grad():
        pool.start_projection.weight.zero_()
        pool.start_projection.bias.zero_()
        pool.end_projection.weight.zero_()
        pool.end_projection.bias.zero_()
    result = pool(
        torch.zeros(1, 4, 2),
        torch.ones(1, 4, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([[[10.0, 0.0, 1.0, -5.0]]]),
        torch.tensor([[[-5.0, 10.0, 0.0, 1.0]]]),
        gold_pairs=torch.tensor([[[[2, 3]]]]),
        gold_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        gold_injection_prob=0.0,
        return_stats=True,
    )
    assert result.indices[result.mask].tolist() == [[0, 1]]
    assert result.stats.gold_hit_without_injection.item() == 0


def test_pool_warns_when_oversubscribed(caplog):
    """More distinct candidate spans than ``pool_size`` -> a warning fires.

    This is silent by design in the return value (no exception, no field);
    the warning is the only signal that low-priority candidates were
    dropped after each query's ``min_pool_per_query`` quota was filled,
    which is how record/entity fields end up null or bound to the wrong
    instance without the caller ever seeing an error.
    """
    torch.manual_seed(0)
    pool = DocumentCandidatePool(
        4, pool_boundary_top_k=6, pool_size=2, min_pool_per_query=0
    )
    states = torch.randn(1, 8, 4)
    boundary_mask = torch.ones(1, 8, dtype=torch.bool)
    query_mask = torch.ones(1, 3, dtype=torch.bool)
    start = torch.randn(1, 3, 8)
    end = torch.randn(1, 3, 8)

    import logging

    with caplog.at_level(logging.WARNING, logger="gliner2.models.boundary.pool"):
        pool(states, boundary_mask, query_mask, start, end)

    assert any(
        "DocumentCandidatePool" in record.message and "pool_size" in record.message
        for record in caplog.records
    )


def test_pool_does_not_warn_when_capacity_is_sufficient(caplog):
    torch.manual_seed(0)
    pool = DocumentCandidatePool(
        4, pool_boundary_top_k=6, pool_size=64, min_pool_per_query=2
    )
    states = torch.randn(1, 8, 4)
    boundary_mask = torch.ones(1, 8, dtype=torch.bool)
    query_mask = torch.ones(1, 3, dtype=torch.bool)
    start = torch.randn(1, 3, 8)
    end = torch.randn(1, 3, 8)

    import logging

    with caplog.at_level(logging.WARNING, logger="gliner2.models.boundary.pool"):
        pool(states, boundary_mask, query_mask, start, end)

    assert not any("DocumentCandidatePool" in record.message for record in caplog.records)


def test_single_query_union_is_unchanged_by_padded_queries():
    torch.manual_seed(8)
    pool = DocumentCandidatePool(
        4, pool_boundary_top_k=5, pool_size=8, min_pool_per_query=2
    ).eval()
    states = torch.randn(1, 6, 4)
    boundary_mask = torch.ones(1, 6, dtype=torch.bool)
    start = torch.randn(1, 1, 6)
    end = torch.randn(1, 1, 6)
    single = pool(
        states, boundary_mask, torch.ones(1, 1, dtype=torch.bool), start, end
    )
    padded = pool(
        states,
        boundary_mask,
        torch.tensor([[True, False]]),
        torch.cat((start, torch.full_like(start, 1000.0)), 1),
        torch.cat((end, torch.full_like(end, 1000.0)), 1),
    )
    assert torch.equal(single.indices, padded.indices)
    assert torch.equal(single.mask, padded.mask)
    torch.testing.assert_close(
        single.proposal_logits, padded.proposal_logits, rtol=0, atol=0
    )


def test_axis_generic_pooled_losses_match_per_query_layout():
    indices = torch.tensor([[[0, 1], [2, 4], [1, 3]]])
    valid = torch.tensor([[True, True, True]])
    gold = torch.tensor([[[[0, 1]], [[2, 4]]]])
    gold_mask = torch.ones(1, 2, 1, dtype=torch.bool)
    pooled_labels = build_candidate_labels(
        indices, valid, gold, gold_mask, query_axis=2, candidate_axis=1
    )
    expanded_indices = indices.unsqueeze(1).expand(1, 2, 3, 2)
    expanded_valid = valid.unsqueeze(1).expand(1, 2, 3)
    per_query_labels = build_candidate_labels(
        expanded_indices, expanded_valid, gold, gold_mask
    )
    assert torch.equal(pooled_labels.transpose(1, 2), per_query_labels)

    pooled_logits = torch.tensor([[[3.0, -1.0], [0.0, 4.0], [2.0, 1.0]]])
    pooled_valid = valid.unsqueeze(-1).expand_as(pooled_logits)
    hard = select_hard_negative_candidates(
        pooled_logits,
        pooled_labels,
        pooled_valid,
        negatives_per_positive=1,
        minimum_negatives=1,
        query_axis=2,
        candidate_axis=1,
    )
    loss = candidate_pair_loss(
        pooled_logits,
        pooled_labels,
        pooled_valid,
        hard,
        query_mask=torch.ones(1, 2, dtype=torch.bool),
        query_axis=2,
        candidate_axis=1,
    )
    listwise = reranker_listwise_loss(
        pooled_logits,
        pooled_labels,
        pooled_valid,
        torch.ones(1, 2, dtype=torch.bool),
        query_axis=2,
        candidate_axis=1,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(listwise)


def _settings(**changes):
    values = dict(
        boundary_dim=8,
        pair_dim=8,
        start_top_k=6,
        end_top_k=6,
        ends_per_start=3,
        starts_per_end=3,
        candidate_budget=12,
        training_candidate_budget=16,
        max_gold_per_query=4,
        end_block_size=8,
        multihead_pair_compat_heads=2,
        dropout=0.0,
        negative_query_ratio=0.0,
        enable_abstention=False,
        enable_count_head=False,
        soft_iou_aux_weight=0.0,
    )
    values.update(changes)
    return BoundaryHeadSettings(**values)


def test_shared_pool_end_to_end_training_loss_and_public_adapter(monkeypatch):
    torch.manual_seed(12)
    settings = _settings(
        candidate_pool="shared",
        pool_boundary_top_k=8,
        pool_size=16,
        min_pool_per_query=2,
        candidate_attention_layers=1,
        candidate_attention_heads=2,
        query_attention_layers=1,
        enable_span_content=True,
        content_dim=4,
    )
    head = BoundaryHead(16, settings, query_dim=16).train()
    tokens = torch.randn(2, 10, 16)
    text_mask = torch.ones(2, 10, dtype=torch.bool)
    queries = torch.randn(2, 2, 16)
    query_mask = torch.ones(2, 2, dtype=torch.bool)
    targets = pad_target_graphs(
        [
            TargetGraph((MentionTarget(0, 0, 2), MentionTarget(1, 4, 9))),
            TargetGraph((MentionTarget(0, 2, 10),)),
        ],
        [2, 2],
        [10, 10],
        4,
    )
    original = boundary_model_module.build_candidate_labels
    seen_dimensions = []

    def capture_dimensions(candidate_indices, *args, **kwargs):
        seen_dimensions.append(candidate_indices.dim())
        return original(candidate_indices, *args, **kwargs)

    monkeypatch.setattr(
        boundary_model_module, "build_candidate_labels", capture_dimensions
    )
    output = head(
        tokens, text_mask, queries, query_mask, targets,
        collect_diagnostics=True,
    )
    assert output.candidates.indices.shape == (2, 2, 16, 2)
    assert output.candidates.pair_logits.shape == (2, 2, 16)
    assert torch.isfinite(output.total_loss)
    assert "per_query_proposal_gold_hit" in output.metrics
    assert seen_dimensions == [3], "shared losses must consume [B,C,2] indices"
    output.total_loss.backward()
    assert head.shared_pool_scorer.film.weight.grad is not None
    assert torch.isfinite(head.shared_pool_scorer.film.weight.grad).all()


def test_per_query_path_ignores_shared_only_ablation_flags_exactly():
    tokens = torch.randn(1, 7, 16)
    text_mask = torch.ones(1, 7, dtype=torch.bool)
    queries = torch.randn(1, 1, 16)
    query_mask = torch.ones(1, 1, dtype=torch.bool)
    torch.manual_seed(91)
    first = BoundaryHead(
        16,
        _settings(
            candidate_pool="per_query",
            candidate_attention_layers=0,
            query_attention_layers=0,
        ),
        query_dim=16,
    ).eval()
    torch.manual_seed(91)
    second = BoundaryHead(
        16,
        _settings(
            candidate_pool="per_query",
            candidate_attention_layers=2,
            query_attention_layers=1,
        ),
        query_dim=16,
    ).eval()
    with torch.no_grad():
        a = first(tokens, text_mask, queries, query_mask)
        b = second(tokens, text_mask, queries, query_mask)
    assert torch.equal(a.candidates.indices, b.candidates.indices)
    assert torch.equal(a.candidates.pair_logits, b.candidates.pair_logits)


def test_shared_pool_config_defaults_and_validation():
    defaults = validate_boundary_head({})
    assert defaults["candidate_pool"] == "per_query"
    for bad in ("global", "", "shared_pool"):
        try:
            validate_boundary_head({"candidate_pool": bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid candidate_pool={bad!r}")
    with pytest.raises(ValueError, match="candidate or query attention"):
        validate_boundary_head({
            "pair_dim": 10,
            "multihead_pair_compat_heads": 2,
            "candidate_attention_layers": 0,
            "candidate_attention_heads": 4,
            "query_attention_layers": 1,
        })
