"""Sparse pair scorer: shapes, invalid masking, and finite gradients."""

from __future__ import annotations

import pytest
import torch

from gliner2.configuration import BoundaryHeadSettings
from gliner2.models.boundary.heads import BoundaryQueryHead
from gliner2.models.boundary.model import BoundaryHead
from gliner2.models.boundary.proposal import ProposalSettings, SparseBoundaryProposer
from gliner2.models.boundary.scoring import (
    SparseBoundaryPairScorer,
    continuous_length_features,
    gather_boundary_states,
)


def test_gather_boundary_states_shape():
    b, n, d, q, c = 2, 7, 8, 3, 4
    states = torch.randn(b, n, d)
    idx = torch.randint(0, n, (b, q, c))
    out = gather_boundary_states(states, idx)
    assert out.shape == (b, q, c, d)
    # spot check one gather
    assert torch.equal(out[0, 0, 0], states[0, idx[0, 0, 0]])


def test_continuous_length_features_no_max_lookup():
    starts = torch.tensor([[[0, 1]]])
    ends = torch.tensor([[[3, 2]]])
    tl = torch.tensor([8])
    feats = continuous_length_features(starts, ends, tl)
    assert feats.shape == (1, 1, 2, 3)
    assert torch.isfinite(feats).all()


def _pipeline(l=8, b=1, q=2, d=8, seed=0):
    torch.manual_seed(seed)
    settings = ProposalSettings(
        start_top_k=4, end_top_k=4, ends_per_start=3, starts_per_end=3,
        candidate_budget=8, training_candidate_budget=12, max_gold_per_query=6,
        end_block_size=4, bidirectional=True,
    )
    boundary_states = torch.randn(b, l + 1, d, requires_grad=True)
    query_states = torch.randn(b, q, d, requires_grad=True)
    text_states = torch.randn(b, l, d, requires_grad=True)
    boundary_mask = torch.ones(b, l + 1, dtype=torch.bool)
    text_mask = torch.ones(b, l, dtype=torch.bool)
    query_mask = torch.ones(b, q, dtype=torch.bool)

    head = BoundaryQueryHead(hidden_size=d, boundary_dim=d, query_dim=d)
    marg = head(boundary_states, boundary_mask, text_states, text_mask, query_states, query_mask)
    proposer = SparseBoundaryProposer(boundary_dim=d, query_dim=d, settings=settings).eval()
    proposals = proposer(
        boundary_states, boundary_mask, query_states, query_mask,
        marg.start_logits, marg.end_logits,
    )
    scorer = SparseBoundaryPairScorer(boundary_dim=d, query_dim=d, pair_dim=d)
    text_lengths = text_mask.sum(dim=1).long()
    logits = scorer(
        boundary_states, query_states, proposals,
        marg.start_logits, marg.end_logits, marg.inside_prefix, text_lengths,
    )
    return logits, proposals, boundary_states, query_states


def test_scorer_shape_and_invalid_masked():
    logits, proposals, _, _ = _pipeline()
    b, q, c = proposals.valid_mask.shape
    assert logits.shape == (b, q, c)
    min_val = torch.finfo(logits.dtype).min
    invalid = ~proposals.valid_mask
    if invalid.any():
        assert (logits[invalid] == min_val).all()
    assert torch.isfinite(logits[proposals.valid_mask]).all()


def test_scorer_finite_gradients():
    logits, proposals, boundary_states, query_states = _pipeline(seed=2)
    valid = proposals.valid_mask
    loss = torch.where(valid, logits, torch.zeros_like(logits)).sum()
    loss.backward()
    assert boundary_states.grad is not None
    assert torch.isfinite(boundary_states.grad).all()
    assert torch.isfinite(query_states.grad).all()


@pytest.mark.parametrize("candidate_pool", ["per_query", "shared"])
def test_explicit_span_scoring_matches_selected_candidate_logits(candidate_pool):
    torch.manual_seed(19)
    settings = BoundaryHeadSettings(
        boundary_dim=8,
        pair_dim=8,
        start_top_k=4,
        end_top_k=4,
        ends_per_start=3,
        starts_per_end=3,
        candidate_budget=8,
        training_candidate_budget=12,
        max_gold_per_query=6,
        end_block_size=4,
        dropout=0.0,
        candidate_pool=candidate_pool,
    )
    head = BoundaryHead(8, settings, query_dim=8).eval()
    token_states = torch.randn(2, 7, 8)
    text_mask = torch.tensor([
        [True] * 7,
        [True] * 5 + [False] * 2,
    ])
    query_states = torch.randn(2, 3, 8)
    query_mask = torch.tensor([
        [True, True, True],
        [True, True, False],
    ])

    output = head(token_states, text_mask, query_states, query_mask)
    explicit = head.score_explicit_spans(
        token_states,
        text_mask,
        query_states,
        query_mask,
        output.candidates.indices,
        output.candidates.valid_mask,
    )

    valid = output.candidates.valid_mask & query_mask.unsqueeze(-1)
    assert torch.allclose(
        explicit[valid], output.candidates.pair_logits[valid], atol=1e-6
    )
