"""Masked boundary/pair/inside losses.

Start and end objectives are multi-label BCE (nested spans may share a
boundary — never a softmax over positions). All losses are masking-aware and
empty-query safe: denominators use ``clamp_min(1)`` so a query with no positive
span still contributes finite negative supervision without NaNs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from gliner2.models.boundary.constants import MASK_LOGIT


def _to_query_candidate(
    tensor: torch.Tensor, query_axis: int, candidate_axis: int
) -> torch.Tensor:
    """Canonicalize candidate scores to ``[B,Q,C]`` without copying."""
    return torch.movedim(tensor, (query_axis, candidate_axis), (1, 2))


def _from_query_candidate(
    tensor: torch.Tensor, query_axis: int, candidate_axis: int
) -> torch.Tensor:
    return torch.movedim(tensor, (1, 2), (query_axis, candidate_axis))


def _safe_bce(logits: torch.Tensor, targets: torch.Tensor, keep: torch.BoolTensor) -> torch.Tensor:
    """Elementwise BCE-with-logits with extreme masked logits neutralized."""
    safe_logits = torch.where(keep, logits, torch.zeros_like(logits))
    safe_targets = torch.where(keep, targets, torch.zeros_like(targets))
    return F.binary_cross_entropy_with_logits(safe_logits, safe_targets, reduction="none")


def _reduce(
    elementwise: torch.Tensor,
    keep: torch.BoolTensor,
    query_mask: Optional[torch.BoolTensor],
    mode: str,
) -> torch.Tensor:
    """Reduce a masked ``[B, Q, N]`` loss globally, per query, or by sum."""
    keep_f = keep.to(elementwise.dtype)
    if mode == "sum":
        if query_mask is not None and keep_f.dim() >= 2:
            keep_f = keep_f * query_mask.unsqueeze(-1).to(keep_f.dtype)
        return (elementwise * keep_f).sum()
    if mode == "global":
        return (elementwise * keep_f).sum() / keep_f.sum().clamp_min(1)
    if mode != "per_query":
        raise ValueError(f"unknown reduction mode {mode!r}")
    numerator = (elementwise * keep_f).sum(-1)
    denominator = keep_f.sum(-1).clamp_min(1)
    per_query = numerator / denominator
    active = keep.any(-1)
    if query_mask is not None:
        active = active & query_mask
    active_f = active.to(per_query.dtype)
    return (per_query * active_f).sum() / active_f.sum().clamp_min(1)


def balanced_multilabel_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.BoolTensor,
    *,
    negative_weight: float = 1.0,
    query_mask: Optional[torch.BoolTensor] = None,
    reduction: str = "global",
) -> torch.Tensor:
    """Mean multi-label BCE over valid positions.

    Negative (non-boundary) positions are scaled by ``negative_weight``. The
    default of ``1.0`` applies **no** down-weighting; pass a value in ``(0, 1)``
    (wired from ``BoundaryHeadSettings.boundary_negative_weight``) to counter the
    negative-dominance of sparse boundary targets.
    """
    bce = _safe_bce(logits, targets, valid_mask)
    weight = torch.where(targets > 0.5, torch.ones_like(targets), torch.full_like(targets, negative_weight))
    return _reduce(bce * weight, valid_mask, query_mask, reduction)


def asymmetric_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.BoolTensor,
    *,
    gamma_positive: float = 0.0,
    gamma_negative: float = 2.0,
    clip: float = 0.05,
    negative_weight: float = 1.0,
    query_mask: Optional[torch.BoolTensor] = None,
    reduction: str = "global",
) -> torch.Tensor:
    """Asymmetric focal loss for imbalanced multi-label boundary targets."""
    keep = valid_mask
    safe_logits = torch.where(keep, logits, torch.zeros_like(logits))
    p = torch.sigmoid(safe_logits)
    if clip > 0:
        p_neg = (1 - p + clip).clamp(max=1.0)
    else:
        p_neg = 1 - p
    los_pos = targets * torch.log(p.clamp_min(1e-8)) * ((1 - p) ** gamma_positive)
    los_neg = (
        (1 - targets)
        * torch.log(p_neg.clamp_min(1e-8))
        * (p ** gamma_negative)
        * negative_weight
    )
    return _reduce(-(los_pos + los_neg), valid_mask, query_mask, reduction)


def build_candidate_labels(
    candidate_indices: torch.LongTensor,   # [B,Q,C,2] or pooled [B,C,2]
    candidate_mask: torch.BoolTensor,      # [B,Q,C] or pooled [B,C]
    gold_pairs: torch.LongTensor,          # [B, Q, G, 2]
    gold_mask: torch.BoolTensor,           # [B, Q, G]
    *,
    return_iou: bool = False,
    query_axis: int = 1,
    candidate_axis: int = 2,
):
    """Build labels in the caller's score-axis order.

    For a shared pool pass ``candidate_indices=[B,C,2]``,
    ``candidate_mask=[B,C]``, ``candidate_axis=1`` and ``query_axis=2``; the
    result is ``[B,C,Q]``.
    """
    pooled = candidate_indices.dim() == 3
    if pooled:
        cand = candidate_indices.unsqueeze(2).unsqueeze(3)  # [B,C,1,1,2]
        gold = gold_pairs.unsqueeze(1)                       # [B,1,Q,G,2]
        same = (cand == gold).all(dim=-1) & gold_mask.unsqueeze(1)
        labels = same.any(-1).to(torch.float) * candidate_mask.unsqueeze(-1)
        candidate_weight = candidate_mask.unsqueeze(-1).to(labels.dtype)
    else:
        canonical_indices = torch.movedim(
            candidate_indices, (query_axis, candidate_axis), (1, 2)
        )
        canonical_mask = _to_query_candidate(
            candidate_mask, query_axis, candidate_axis
        )
        cand = canonical_indices.unsqueeze(3)       # [B,Q,C,1,2]
        gold = gold_pairs.unsqueeze(2)              # [B,Q,1,G,2]
        same = (cand == gold).all(dim=-1) & gold_mask.unsqueeze(2)
        canonical_labels = same.any(dim=-1).to(torch.float)
        canonical_labels = canonical_labels * canonical_mask.to(canonical_labels.dtype)
        labels = _from_query_candidate(
            canonical_labels, query_axis, candidate_axis
        )
        candidate_weight = canonical_mask.to(canonical_labels.dtype)
    if not return_iou:
        return labels

    candidate_start, candidate_end = cand[..., 0], cand[..., 1]
    gold_start, gold_end = gold[..., 0], gold[..., 1]
    intersection = (
        torch.minimum(candidate_end, gold_end)
        - torch.maximum(candidate_start, gold_start)
    ).clamp_min(0)
    union = (
        candidate_end - candidate_start
        + gold_end - gold_start
        - intersection
    )
    iou = intersection.to(torch.float) / union.clamp_min(1).to(torch.float)
    iou = iou * (
        gold_mask.unsqueeze(1) if pooled else gold_mask.unsqueeze(2)
    ).to(iou.dtype)
    canonical_soft = iou.amax(-1) * candidate_weight
    soft_targets = (
        canonical_soft
        if pooled
        else _from_query_candidate(
            canonical_soft, query_axis, candidate_axis
        )
    )
    return labels, soft_targets


def select_hard_negative_candidates(
    pair_logits: torch.Tensor,        # [B, Q, C]
    labels: torch.Tensor,             # [B, Q, C]
    valid_mask: torch.BoolTensor,     # [B, Q, C]
    *,
    negatives_per_positive: int,
    minimum_negatives: int,
    keep_all_when_no_positive: bool = False,
    query_axis: int = 1,
    candidate_axis: int = 2,
) -> torch.BoolTensor:
    """Select the highest-scoring negatives per query (positives always kept)."""
    original_query_axis, original_candidate_axis = query_axis, candidate_axis
    pair_logits = _to_query_candidate(pair_logits, query_axis, candidate_axis)
    labels = _to_query_candidate(labels, query_axis, candidate_axis)
    valid_mask = _to_query_candidate(valid_mask, query_axis, candidate_axis)
    positive = (labels > 0.5) & valid_mask
    negative = (labels <= 0.5) & valid_mask
    n_positive = positive.sum(-1, keepdim=True)
    floor = torch.finfo(pair_logits.dtype).min
    negative_scores = pair_logits.masked_fill(~negative, floor)
    order = torch.argsort(negative_scores, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)
    n_keep = (negatives_per_positive * n_positive).clamp_min(minimum_negatives)
    if keep_all_when_no_positive:
        n_keep = torch.where(
            n_positive == 0,
            torch.full_like(n_keep, pair_logits.shape[-1]),
            n_keep,
        )
    selected = positive | (negative & (rank < n_keep))
    return _from_query_candidate(
        selected, original_query_axis, original_candidate_axis
    )


def candidate_pair_loss(
    logits: torch.Tensor,             # [B, Q, C]
    labels: torch.Tensor,             # [B, Q, C]
    valid_mask: torch.BoolTensor,     # [B, Q, C]
    hard_negative_mask: Optional[torch.BoolTensor] = None,
    *,
    query_mask: Optional[torch.BoolTensor] = None,
    reduction: str = "global",
    query_axis: int = 1,
    candidate_axis: int = 2,
    split_positive_negative: bool = False,
) -> torch.Tensor:
    """BCE over candidates; optionally restrict negatives to a hard subset.

    When ``split_positive_negative=True``, positive candidates and selected
    negative candidates are averaged over their own counts and summed so
    positive gradients are not diluted by large negative/absent-query pools.
    """
    logits = _to_query_candidate(logits, query_axis, candidate_axis)
    labels = _to_query_candidate(labels, query_axis, candidate_axis)
    valid_mask = _to_query_candidate(valid_mask, query_axis, candidate_axis)
    if hard_negative_mask is not None:
        hard_negative_mask = _to_query_candidate(
            hard_negative_mask, query_axis, candidate_axis
        )
    if hard_negative_mask is not None:
        effective = valid_mask & ((labels > 0.5) | hard_negative_mask)
    else:
        effective = valid_mask
    bce = _safe_bce(logits, labels, effective)
    if split_positive_negative:
        pos_mask = effective & (labels > 0.5)
        neg_mask = effective & (labels <= 0.5)
        return (
            _reduce(bce, pos_mask, query_mask, reduction)
            + _reduce(bce, neg_mask, query_mask, reduction)
        )
    return _reduce(bce, effective, query_mask, reduction)


def inside_consistency_loss(
    inside_logits: torch.Tensor,      # [B, Q, L]
    inside_targets: torch.Tensor,     # [B, Q, L]
    text_mask: torch.BoolTensor,      # [B, L]
    query_mask: torch.BoolTensor,     # [B, Q]
    *,
    negative_weight: float = 1.0,
    reduction: str = "global",
) -> torch.Tensor:
    keep = text_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
    return balanced_multilabel_bce(
        inside_logits,
        inside_targets,
        keep,
        negative_weight=negative_weight,
        query_mask=query_mask,
        reduction=reduction,
    )


def proposal_listwise_loss(
    proposal_logits: torch.Tensor,
    gold_mask: torch.BoolTensor,
    valid_mask: torch.BoolTensor,
    query_mask: torch.BoolTensor,
    *,
    query_axis: int = 1,
    candidate_axis: int = 2,
) -> torch.Tensor:
    """Rank injected gold candidates above other valid proposals."""
    proposal_logits = _to_query_candidate(
        proposal_logits, query_axis, candidate_axis
    )
    gold_mask = _to_query_candidate(gold_mask, query_axis, candidate_axis)
    valid_mask = _to_query_candidate(valid_mask, query_axis, candidate_axis)
    floor = MASK_LOGIT
    logits = proposal_logits.masked_fill(~valid_mask, floor)
    all_lse = torch.logsumexp(logits, dim=-1)
    gold_lse = torch.logsumexp(logits.masked_fill(~gold_mask, floor), dim=-1)
    has_gold = gold_mask.any(-1) & query_mask
    loss = torch.where(
        has_gold, all_lse - gold_lse, torch.zeros_like(all_lse)
    )
    return loss.sum() / has_gold.to(loss.dtype).sum().clamp_min(1)


def reranker_listwise_loss(
    pair_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.BoolTensor,
    query_mask: torch.BoolTensor,
    *,
    query_axis: int = 1,
    candidate_axis: int = 2,
) -> torch.Tensor:
    """Listwise gold-mass loss over reranked candidates, empty-query safe."""
    gold_mask = (labels > 0.5) & valid_mask
    return proposal_listwise_loss(
        pair_logits,
        gold_mask,
        valid_mask,
        query_mask,
        query_axis=query_axis,
        candidate_axis=candidate_axis,
    )


def marginal_pair_consistency_loss(
    pair_logits: torch.Tensor,
    indices: torch.LongTensor,
    valid_mask: torch.BoolTensor,
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    boundary_keep: torch.BoolTensor,
) -> torch.Tensor:
    """Match boundary marginals to candidate-level noisy-OR probabilities."""
    probabilities = torch.sigmoid(pair_logits) * valid_mask.to(pair_logits.dtype)
    # The clamp epsilon must be representable in the tensor's own dtype. In half precision
    # `1.0 - 1e-6` rounds to exactly 1.0 (bf16 eps is 7.8e-3, fp16 eps is 9.8e-4), so the
    # clamp becomes a no-op, a saturated probability reaches 1.0, and log1p(-1.0) is -inf.
    # The forward still looks healthy -- the -inf is summed and `1 - exp(-inf)` is 1 -- but
    # d/dp log1p(-p) = -1/(1-p) is infinite, and a masked entry's zero grad_output turns
    # that into 0 * inf = NaN.
    # max() leaves float32 bit-identical, since 1e-6 already exceeds its 1.2e-7 eps.
    eps = max(1e-6, torch.finfo(probabilities.dtype).eps)
    log_survival = torch.log1p(-probabilities.clamp(max=1.0 - eps))
    b, q, n = start_logits.shape

    def accumulate(index: torch.LongTensor):
        safe_index = index.clamp(0, n - 1)
        total = torch.zeros(
            b, q, n, dtype=log_survival.dtype, device=log_survival.device
        )
        total.scatter_add_(2, safe_index, log_survival)
        count = torch.zeros_like(total)
        count.scatter_add_(2, safe_index, valid_mask.to(total.dtype))
        return 1.0 - torch.exp(total), count > 0

    predicted_start, reached_start = accumulate(indices[..., 0])
    predicted_end, reached_end = accumulate(indices[..., 1])
    result = pair_logits.new_zeros(())
    for predicted, marginal, reached in (
        (predicted_start, start_logits, reached_start),
        (predicted_end, end_logits, reached_end),
    ):
        keep = reached & boundary_keep
        target = torch.sigmoid(torch.where(keep, marginal, torch.zeros_like(marginal)))
        squared = (predicted - target) ** 2
        result = result + (squared * keep).sum() / keep.sum().clamp_min(1)
    return result * 0.5


def abstention_loss(
    null_logits: torch.Tensor,
    mention_mask: torch.BoolTensor,
    query_mask: torch.BoolTensor,
) -> torch.Tensor:
    """Train a per-query gate whose target is one for an absent query."""
    target = (~mention_mask.any(-1)).to(null_logits.dtype)
    elementwise = F.binary_cross_entropy_with_logits(
        null_logits, target, reduction="none"
    )
    keep = query_mask.to(elementwise.dtype)
    return (elementwise * keep).sum() / keep.sum().clamp_min(1)


def count_log_rate_loss(
    count_log_rate: torch.Tensor,
    mention_mask: torch.BoolTensor,
    query_mask: torch.BoolTensor,
) -> torch.Tensor:
    """Poisson NLL for a count head whose output is the logarithm of its rate."""
    target = mention_mask.sum(-1).to(count_log_rate.dtype)
    elementwise = F.poisson_nll_loss(
        count_log_rate,
        target,
        log_input=True,
        full=False,
        reduction="none",
    )
    keep = query_mask.to(elementwise.dtype)
    return (elementwise * keep).sum() / keep.sum().clamp_min(1)
