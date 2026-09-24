"""Boundary architecture model and trainable head bundle.

``BoundaryHead`` is the architecture-neutral trainable core: it turns token
states + query states into boundary marginals, sparse candidate proposals,
reranked pair logits, and (given targets) the weighted component losses. It is
deliberately decoupled from the encoder/tokenizer so it can be unit-overfit on
synthetic states (see ``tests/models/boundary/test_overfit_head.py``).

``BoundaryExtractorModel`` wraps a shared transformer encoder + classification
head around ``BoundaryHead`` and provides architecture-stamped serialization.
Half-open ``[start, end)`` coordinates throughout; there is no width axis.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig

if TYPE_CHECKING:
    from peft import PeftModel

from gliner2.configuration import BoundaryHeadSettings, ExtractorConfig
from gliner2.layers import create_mlp
from gliner2.models.base import (
    BaseExtractorModel,
    EncodedBatch,
    load_extractor_tokenizer,
)
from gliner2.models.loading import (
    apply_post_load_options,
    checkpoint_file,
    load_checkpoint_state_dict,
    reconcile_encoder_embeddings,
    split_load_kwargs,
)
from gliner2.models.boundary.encoding import BoundaryEncoder
from gliner2.models.boundary.constants import MASK_LOGIT
from gliner2.models.boundary.heads import BoundaryMarginals, BoundaryQueryHead
from gliner2.models.boundary.losses import (
    asymmetric_focal_loss,
    balanced_multilabel_bce,
    build_candidate_labels,
    candidate_pair_loss,
    inside_consistency_loss,
    proposal_listwise_loss,
    reranker_listwise_loss,
    marginal_pair_consistency_loss,
    abstention_loss,
    count_log_rate_loss,
    select_hard_negative_candidates,
)
from gliner2.models.boundary.proposal import (
    BoundaryProposals,
    ProposalSettings,
    SparseBoundaryProposer,
)
from gliner2.models.boundary.scoring import SparseBoundaryPairScorer, gather_boundary_states
from gliner2.models.boundary.pool import (
    DocumentCandidatePool,
    PooledCandidates,
    SharedPoolScorer,
)
from gliner2.models.boundary.targets_device import dense_targets_from_pairs
from gliner2.models.boundary.validation import (
    safe_relation_indices,
    validate_candidate_indices,
    validate_masked_spans,
)
from gliner2.models.boundary.relations import (
    RelationProposalSettings,
    RelationTypeSpec,
    SparseRelationScorer,
    TypedRelationPairGenerator,
)
from gliner2.models.base import QueryLayout, QuerySpec
from gliner2.models.outputs import CandidateTensorBatch, ExtractorOutput
from gliner2.processing.targets import (
    MentionTarget,
    PaddedTargetBatch,
    TargetCapacityError,
    TargetGraph,
    pad_target_graphs,
)
from gliner2.utils.device_errors import is_fatal_device_error


DEFAULT_LOSS_WEIGHTS = {"start": 1.0, "end": 1.0, "pair": 1.0, "inside": 0.5}
logger = logging.getLogger(__name__)


def _finite_loss_term(value: torch.Tensor) -> torch.Tensor:
    """Discard non-finite scalar loss contributions without poisoning training."""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _pad_states(
    sequences: List[torch.Tensor],
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """Pad ``[N_i, H]`` sequences and return their validity mask."""
    batch = len(sequences)
    if batch == 0 or all(sequence.shape[0] == 0 for sequence in sequences):
        return (
            torch.zeros(batch, 0, hidden_size, device=device, dtype=dtype),
            torch.zeros(batch, 0, device=device, dtype=torch.bool),
        )
    padded = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor(
        [sequence.shape[0] for sequence in sequences],
        dtype=torch.long,
        device=device,
    )
    mask = torch.arange(padded.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask


def proposal_settings_from_head(settings: BoundaryHeadSettings) -> ProposalSettings:
    """Map validated ``BoundaryHeadSettings`` to runtime ``ProposalSettings``."""
    return ProposalSettings(
        start_top_k=settings.start_top_k,
        end_top_k=settings.end_top_k,
        ends_per_start=settings.ends_per_start,
        starts_per_end=settings.starts_per_end,
        candidate_budget=settings.candidate_budget,
        training_candidate_budget=settings.training_candidate_budget,
        max_gold_per_query=settings.max_gold_per_query,
        end_block_size=settings.end_block_size,
        bidirectional=settings.bidirectional_proposals,
        export_mode=settings.export_mode,
        vectorized_pair_elements=settings.vectorized_pair_elements,
        enable_rotary_endpoints=settings.enable_rotary_endpoints,
        rotary_base=settings.rotary_base,
        boundary_top_k_alpha=settings.boundary_top_k_alpha,
        boundary_top_k_max=settings.boundary_top_k_max,
        boundary_top_k_bucket=settings.boundary_top_k_bucket,
    )


class BoundaryHead(nn.Module):
    """Composable boundary head: encoding, marginals, proposal, scoring, losses."""

    def __init__(
        self,
        hidden_size: int,
        settings: BoundaryHeadSettings,
        query_dim: Optional[int] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        hard_negatives_per_positive: Optional[int] = None,
        minimum_hard_negatives: Optional[int] = None,
        build_candidate_states: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.settings = settings
        self.query_dim = query_dim if query_dim is not None else hidden_size
        self.loss_weights = dict(loss_weights or DEFAULT_LOSS_WEIGHTS)
        self.hard_negatives_per_positive = (
            settings.hard_negatives_per_positive
            if hard_negatives_per_positive is None else hard_negatives_per_positive
        )
        self.minimum_hard_negatives = (
            settings.minimum_hard_negatives
            if minimum_hard_negatives is None else minimum_hard_negatives
        )
        self.collect_diagnostics = False
        self._gold_injection_prob = 1.0
        self._consistency_scale = 1.0
        self._soft_iou_scale = 1.0

        d = settings.boundary_dim
        # Candidate contextual states for the record head (endpoint-derived);
        # only built when records are enabled so legacy state dicts are unchanged.
        self.candidate_encoder = (
            nn.Linear(2 * d, hidden_size) if build_candidate_states else None
        )
        self.boundary_encoder = BoundaryEncoder(
            hidden_size,
            d,
            settings.dropout,
            settings.boundary_refinement_layers,
            settings.boundary_ffn_multiplier,
            settings.boundary_attention_layers,
            settings.boundary_attention_heads,
            settings.boundary_attention_window,
        )
        self.boundary_query_head = BoundaryQueryHead(
            hidden_size, d, self.query_dim, settings.dropout
        )
        self.boundary_proposer = SparseBoundaryProposer(
            d, self.query_dim, proposal_settings_from_head(settings)
        )
        self.pair_scorer = SparseBoundaryPairScorer(
            d, self.query_dim, settings.pair_dim,
            use_inside_evidence=settings.use_inside_evidence,
            dropout=settings.dropout,
            enable_span_content=settings.enable_span_content,
            content_dim=settings.content_dim,
            content_soft_max_pool=settings.content_soft_max_pool,
            enable_rotary_endpoints=settings.enable_rotary_endpoints,
            rotary_base=settings.rotary_base,
            query_conditioned_inside_weight=settings.query_conditioned_inside_weight,
            endpoint_difference_features=settings.endpoint_difference_features,
            reranker_endpoint_compat=settings.reranker_endpoint_compat,
            multihead_pair_compat_heads=settings.multihead_pair_compat_heads,
            content_hidden_size=hidden_size,
        )
        self.use_inside_evidence = settings.use_inside_evidence
        auxiliary_rng_state = torch.random.get_rng_state()
        self.null_projection = (
            nn.Linear(self.query_dim, 1) if settings.enable_abstention else None
        )
        if self.null_projection is not None:
            nn.init.zeros_(self.null_projection.weight)
            nn.init.zeros_(self.null_projection.bias)
        self.count_head = (
            nn.Linear(self.query_dim, 1) if settings.enable_count_head else None
        )
        if self.count_head is not None:
            nn.init.zeros_(self.count_head.weight)
            nn.init.zeros_(self.count_head.bias)
        torch.random.set_rng_state(auxiliary_rng_state)
        # Shared-pool parameters are present for checkpoint transparency even
        # while the default per-query path is selected. Preserve the caller's
        # RNG stream so flag-off forward numerics remain exactly unchanged.
        shared_rng_state = torch.random.get_rng_state()
        self.shared_pool_builder = DocumentCandidatePool(
            d,
            pool_boundary_top_k=settings.pool_boundary_top_k,
            pool_size=settings.pool_size,
            min_pool_per_query=settings.min_pool_per_query,
        )
        self.shared_pool_scorer = SharedPoolScorer(
            d,
            self.query_dim,
            settings.pair_dim,
            dropout=settings.dropout,
            candidate_attention_layers=settings.candidate_attention_layers,
            candidate_attention_heads=settings.candidate_attention_heads,
            query_attention_layers=settings.query_attention_layers,
            enable_span_content=settings.enable_span_content,
            content_dim=settings.content_dim,
            content_soft_max_pool=settings.content_soft_max_pool,
            text_hidden_size=hidden_size,
        )
        torch.random.set_rng_state(shared_rng_state)

    def set_gold_injection_prob(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"gold_injection_prob must be in [0, 1], got {value}"
            )
        self._gold_injection_prob = float(value)

    def set_consistency_scale(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"consistency scale must be in [0, 1], got {value}")
        self._consistency_scale = float(value)

    def set_soft_iou_scale(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"soft IoU scale must be in [0, 1], got {value}")
        self._soft_iou_scale = float(value)

    def score_explicit_spans(
        self,
        token_states: torch.Tensor,
        text_mask: torch.BoolTensor,
        query_states: torch.Tensor,
        query_mask: torch.BoolTensor,
        indices: torch.LongTensor,
        valid_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Score caller-provided half-open spans for every query.

        ``indices`` is ``[B, Q, C, 2]``.  Unlike normal inference, this path
        bypasses sparse proposal selection while retaining the exact proposal
        compatibility prior and pair-reranker computation used by ordinary
        candidates.  It is intended for span-conditioned features such as
        entity attributes and constrained structure fields.
        """
        if indices.dim() != 4 or indices.shape[-1] != 2:
            raise ValueError(
                f"indices must be [B, Q, C, 2], got {tuple(indices.shape)}"
            )
        b, q, _, _ = indices.shape
        if (
            b != token_states.shape[0]
            or b != query_states.shape[0]
            or q != query_states.shape[1]
        ):
            raise ValueError("explicit span batch/query dimensions do not match states")

        text_lengths = text_mask.sum(dim=1).long()
        starts = indices[..., 0]
        ends = indices[..., 1]
        legal = (
            (starts >= 0)
            & (ends > starts)
            & (ends <= text_lengths.view(b, 1, 1))
            & query_mask.unsqueeze(-1)
        )
        if valid_mask is not None:
            if valid_mask.shape != indices.shape[:-1]:
                raise ValueError(
                    "valid_mask must match indices [B, Q, C], got "
                    f"{tuple(valid_mask.shape)} and {tuple(indices.shape)}"
                )
            legal = legal & valid_mask

        encoding = self.boundary_encoder(token_states, text_mask)
        marginals = self.boundary_query_head(
            encoding.states,
            encoding.mask,
            token_states,
            text_mask,
            query_states,
            query_mask,
        )
        compatibility = self.boundary_proposer.score_explicit_pairs(
            encoding.states, query_states, indices, legal
        )
        proposals = BoundaryProposals(
            indices=indices,
            logits=None,
            valid_mask=legal,
            compat_logits=compatibility,
        )
        inside_prefix = (
            marginals.inside_prefix if self.use_inside_evidence else None
        )
        return self.pair_scorer(
            encoding.states,
            query_states,
            proposals,
            marginals.start_logits,
            marginals.end_logits,
            inside_prefix,
            text_lengths,
            token_states,
            text_mask,
            inside_prefix_mean=marginals.inside_prefix_mean,
        )

    def forward(
        self,
        token_states: torch.Tensor,        # [B, L, H]
        text_mask: torch.BoolTensor,       # [B, L]
        query_states: torch.Tensor,        # [B, Q, Hq]
        query_mask: torch.BoolTensor,      # [B, Q]
        targets: Optional[PaddedTargetBatch] = None,
        *,
        return_candidates: bool = True,
        gold_injection_prob: Optional[float] = None,
        collect_diagnostics: Optional[bool] = None,
    ) -> ExtractorOutput:
        b, l, _ = token_states.shape
        text_lengths = text_mask.sum(dim=1).long()

        encoding = self.boundary_encoder(token_states, text_mask)
        marginals = self.boundary_query_head(
            encoding.states, encoding.mask,
            token_states, text_mask,
            query_states, query_mask,
        )

        gold_pairs = None
        gold_mask = None
        if targets is not None:
            gold_pairs = targets.mention_pairs
            gold_mask = targets.mention_mask

        diagnostics = (
            self.collect_diagnostics
            if collect_diagnostics is None else collect_diagnostics
        )
        injection = (
            self._gold_injection_prob
            if gold_injection_prob is None else gold_injection_prob
        )
        inside_prefix = marginals.inside_prefix if self.use_inside_evidence else None
        pooled: Optional[PooledCandidates] = None
        pooled_candidate_states = None
        comparison_stats = None
        comparison_indices = None
        comparison_valid = None
        if self.settings.candidate_pool == "shared":
            pooled = self.shared_pool_builder(
                encoding.states,
                encoding.mask,
                query_mask,
                marginals.start_logits,
                marginals.end_logits,
                gold_pairs=gold_pairs,
                gold_mask=gold_mask,
                gold_injection_prob=injection if self.training else 0.0,
                return_stats=diagnostics,
            )
            pooled_logits, _ = self.shared_pool_scorer(
                encoding.states,
                query_states,
                query_mask,
                pooled,
                marginals.start_logits,
                marginals.end_logits,
                inside_prefix,
                text_lengths,
                token_states,
                text_mask,
                inside_prefix_mean=marginals.inside_prefix_mean,
            )
            q = query_states.shape[1]
            c = pooled.indices.shape[1]
            proposals = BoundaryProposals(
                indices=pooled.indices.unsqueeze(1).expand(b, q, c, 2),
                logits=(
                    pooled.proposal_logits.unsqueeze(1).expand(b, q, c)
                    if pooled.proposal_logits is not None else None
                ),
                valid_mask=pooled.mask.unsqueeze(1).expand(b, q, c),
                gold_mask=(
                    pooled.gold_mask.transpose(1, 2)
                    if pooled.gold_mask is not None else None
                ),
                stats=pooled.stats,
                compat_logits=(
                    pooled.compat_logits.unsqueeze(1).expand(b, q, c)
                    if pooled.compat_logits is not None else None
                ),
            )
            pair_logits = pooled_logits.transpose(1, 2)
            if self.candidate_encoder is not None:
                from gliner2.models.boundary.indexing import gather_rows
                ps = gather_rows(encoding.states, pooled.indices[..., 0])
                pe = gather_rows(encoding.states, pooled.indices[..., 1])
                pooled_candidate_states = self.candidate_encoder(
                    torch.cat((ps, pe), -1)
                ).masked_fill(~pooled.mask.unsqueeze(-1), 0.0)
            # Dry-run A/B: the historical proposer is evaluated alongside the
            # shared pool without invoking its reranker.
            if diagnostics:
                comparison_proposals = self.boundary_proposer(
                    encoding.states,
                    encoding.mask,
                    query_states,
                    query_mask,
                    marginals.start_logits,
                    marginals.end_logits,
                    gold_pairs=gold_pairs,
                    gold_mask=gold_mask,
                    return_stats=True,
                    return_proposal_logits=False,
                    gold_injection_prob=0.0,
                )
                comparison_stats = comparison_proposals.stats
                comparison_indices = comparison_proposals.indices
                comparison_valid = comparison_proposals.valid_mask
        else:
            scorer_start_states, scorer_end_states = self.pair_scorer.project_endpoints(
                encoding.states
            )
            proposals = self.boundary_proposer(
                encoding.states, encoding.mask,
                query_states, query_mask,
                marginals.start_logits, marginals.end_logits,
                gold_pairs=gold_pairs, gold_mask=gold_mask,
                return_stats=diagnostics,
                return_proposal_logits=self.training,
                gold_injection_prob=injection,
                scorer_start_states=scorer_start_states,
                scorer_end_states=scorer_end_states,
            )
            pair_logits = self.pair_scorer(
                encoding.states, query_states, proposals,
                marginals.start_logits, marginals.end_logits,
                inside_prefix,
                text_lengths,
                token_states,
                text_mask,
                inside_prefix_mean=marginals.inside_prefix_mean,
            )
            if diagnostics:
                comparison_pool = self.shared_pool_builder(
                    encoding.states,
                    encoding.mask,
                    query_mask,
                    marginals.start_logits,
                    marginals.end_logits,
                    gold_pairs=gold_pairs,
                    gold_mask=gold_mask,
                    gold_injection_prob=0.0,
                    return_stats=True,
                )
                comparison_stats = comparison_pool.stats
                comparison_indices = comparison_pool.indices.unsqueeze(1).expand(
                    b, query_states.shape[1], comparison_pool.indices.shape[1], 2
                )
                comparison_valid = comparison_pool.mask.unsqueeze(1).expand(
                    b, query_states.shape[1], comparison_pool.mask.shape[1]
                )
        self._last_proposal_stats = proposals.stats
        null_logits = (
            self.null_projection(query_states).squeeze(-1)
            if self.null_projection is not None else None
        )
        count_log_rates = (
            self.count_head(query_states).squeeze(-1)
            if self.count_head is not None else None
        )

        candidates = None
        if return_candidates:
            candidate_states = pooled_candidate_states
            if self.candidate_encoder is not None and pooled is None:
                starts = proposals.indices[..., 0]
                ends = proposals.indices[..., 1]
                g_start = gather_boundary_states(encoding.states, starts)  # [B,Q,C,d]
                g_end = gather_boundary_states(encoding.states, ends)
                candidate_states = self.candidate_encoder(
                    torch.cat([g_start, g_end], dim=-1)
                )
                candidate_states = candidate_states.masked_fill(
                    ~proposals.valid_mask.unsqueeze(-1), 0.0
                )
            candidates = (
                pooled.to_candidate_batch(
                    pooled_logits, query_mask, candidate_states
                )
                if pooled is not None
                else CandidateTensorBatch(
                    indices=proposals.indices,
                    proposal_logits=proposals.logits,
                    pair_logits=pair_logits,
                    valid_mask=proposals.valid_mask,
                    query_mask=query_mask,
                    candidate_states=candidate_states,
                )
            )

        losses: Optional[Dict[str, torch.Tensor]] = None
        total_loss = None
        if targets is not None:
            losses = self._compute_losses(
                marginals,
                proposals,
                pair_logits,
                targets,
                query_mask,
                encoding.mask,
                text_mask,
                null_logits,
                count_log_rates,
                pooled=pooled,
                pooled_pair_logits=(
                    pooled_logits if pooled is not None else None
                ),
            )
            total_loss = losses["total_loss"]

        metrics = None
        if proposals.stats is not None:
            stats = proposals.stats
            metrics = {
                "proposal_gold_hit": stats.gold_hit_without_injection,
                "proposal_gold_total": stats.gold_total,
                "start_hit": stats.start_hit,
                "end_hit": stats.end_hit,
                "boundary_total": stats.boundary_total,
                "unique_candidates": stats.unique_candidates,
                "valid_queries": query_mask.sum(),
            }
            if targets is not None:
                same = (
                    proposals.indices.unsqueeze(2)
                    == targets.mention_pairs.unsqueeze(3)
                ).all(-1)
                gold_hit = (
                    same & proposals.valid_mask.unsqueeze(2)
                ).any(-1) & targets.mention_mask
                lengths = (
                    targets.mention_pairs[..., 1]
                    - targets.mention_pairs[..., 0]
                )
                for label, in_bucket in (
                    ("1", lengths == 1),
                    ("2", lengths == 2),
                    ("3_4", (lengths >= 3) & (lengths <= 4)),
                    ("5_8", (lengths >= 5) & (lengths <= 8)),
                    ("9_plus", lengths >= 9),
                ):
                    bucket_gold = in_bucket & targets.mention_mask
                    metrics[f"length_{label}_hit"] = (
                        gold_hit & bucket_gold
                    ).sum()
                    metrics[f"length_{label}_total"] = bucket_gold.sum()
                absent = query_mask & ~targets.mention_mask.any(-1)
                predicted = (
                    (pair_logits > 0) & proposals.valid_mask
                ).any(-1)
                metrics["absent_query_false_positive"] = (
                    predicted & absent
                ).sum()
                metrics["absent_query_total"] = absent.sum()
            metrics = {key: value for key, value in metrics.items() if value is not None}
            if comparison_stats is not None:
                prefix = (
                    "per_query" if self.settings.candidate_pool == "shared"
                    else "shared"
                )
                for key, value in (
                    ("proposal_gold_hit", comparison_stats.gold_hit_without_injection),
                    ("proposal_gold_total", comparison_stats.gold_total),
                    ("start_hit", comparison_stats.start_hit),
                    ("end_hit", comparison_stats.end_hit),
                    ("boundary_total", comparison_stats.boundary_total),
                    ("unique_candidates", comparison_stats.unique_candidates),
                ):
                    if value is not None:
                        metrics[f"{prefix}_{key}"] = value
                if (
                    targets is not None
                    and comparison_indices is not None
                    and comparison_valid is not None
                ):
                    comparison_same = (
                        comparison_indices.unsqueeze(2)
                        == targets.mention_pairs.unsqueeze(3)
                    ).all(-1)
                    comparison_hit = (
                        comparison_same & comparison_valid.unsqueeze(2)
                    ).any(-1) & targets.mention_mask
                    lengths = (
                        targets.mention_pairs[..., 1]
                        - targets.mention_pairs[..., 0]
                    )
                    for label, in_bucket in (
                        ("1", lengths == 1),
                        ("2", lengths == 2),
                        ("3_4", (lengths >= 3) & (lengths <= 4)),
                        ("5_8", (lengths >= 5) & (lengths <= 8)),
                        ("9_plus", lengths >= 9),
                    ):
                        bucket_gold = in_bucket & targets.mention_mask
                        metrics[f"{prefix}_length_{label}_hit"] = (
                            comparison_hit & bucket_gold
                        ).sum()
                        metrics[f"{prefix}_length_{label}_total"] = (
                            bucket_gold.sum()
                        )

        return ExtractorOutput(
            loss=total_loss,
            total_loss=total_loss,
            losses=losses,
            candidates=candidates,
            start_logits=marginals.start_logits,
            end_logits=marginals.end_logits,
            inside_logits=marginals.inside_logits,
            metrics=metrics,
            null_logits=null_logits,
            count_log_rates=count_log_rates,
            batch_size=b,
        )

    def _compute_losses(
        self,
        marginals: BoundaryMarginals,
        proposals: BoundaryProposals,
        pair_logits: torch.Tensor,
        targets: PaddedTargetBatch,
        query_mask: torch.BoolTensor,
        boundary_mask: torch.BoolTensor,
        text_mask: torch.BoolTensor,
        null_logits: Optional[torch.Tensor],
        count_log_rates: Optional[torch.Tensor],
        *,
        pooled: Optional[PooledCandidates] = None,
        pooled_pair_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        boundary_keep = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)  # [B,Q,L+1]
        n_boundary = marginals.start_logits.shape[2]
        start_targets = targets.start_targets
        end_targets = targets.end_targets
        inside_targets = targets.inside_targets
        if start_targets is None or end_targets is None or inside_targets is None:
            safe_pairs = targets.mention_pairs.clamp(0, n_boundary - 1)
            start_targets, end_targets, inside_targets = dense_targets_from_pairs(
                safe_pairs, targets.mention_mask, text_mask.shape[-1]
            )
        if start_targets.shape[-1] != n_boundary:
            start_targets = start_targets[..., :n_boundary] if start_targets.shape[-1] > n_boundary else torch.nn.functional.pad(start_targets, (0, n_boundary - start_targets.shape[-1]))
            end_targets = end_targets[..., :n_boundary] if end_targets.shape[-1] > n_boundary else torch.nn.functional.pad(end_targets, (0, n_boundary - end_targets.shape[-1]))
        if inside_targets.shape[-1] != marginals.inside_logits.shape[-1]:
            tgt_n = marginals.inside_logits.shape[-1]
            inside_targets = inside_targets[..., :tgt_n] if inside_targets.shape[-1] > tgt_n else torch.nn.functional.pad(inside_targets, (0, tgt_n - inside_targets.shape[-1]))
        start_targets = start_targets.to(marginals.start_logits.dtype)
        end_targets = end_targets.to(marginals.end_logits.dtype)
        inside_targets = inside_targets.to(marginals.inside_logits.dtype)

        neg_weight = getattr(self.settings, "boundary_negative_weight", 1.0)
        use_focal = getattr(self.settings, "boundary_marginal_loss", "bce") == "asymmetric_focal"
        reduction = getattr(self.settings, "loss_reduction", "global")

        def _marginal_loss(logits, tgt, keep):
            if use_focal:
                return asymmetric_focal_loss(
                    logits,
                    tgt,
                    keep,
                    gamma_positive=self.settings.boundary_focal_gamma_positive,
                    gamma_negative=self.settings.boundary_focal_gamma_negative,
                    clip=self.settings.boundary_focal_clip,
                    negative_weight=neg_weight,
                    query_mask=query_mask,
                    reduction=reduction,
                )
            return balanced_multilabel_bce(
                logits,
                tgt,
                keep,
                negative_weight=neg_weight,
                query_mask=query_mask,
                reduction=reduction,
            )

        start_loss = _marginal_loss(
            marginals.start_logits, start_targets, boundary_keep
        )
        end_loss = _marginal_loss(
            marginals.end_logits, end_targets, boundary_keep
        )

        use_soft_iou = self.settings.soft_iou_aux_weight > 0
        pooled_loss = pooled is not None
        if pooled_loss:
            if pooled_pair_logits is None:
                raise ValueError("shared-pool loss requires candidate-major logits")
            loss_logits = pooled_pair_logits
            loss_valid = (
                pooled.mask.unsqueeze(-1)
                & query_mask.unsqueeze(1)
            )
            label_indices = pooled.indices
            label_mask = pooled.mask
            query_axis, candidate_axis = 2, 1
        else:
            loss_logits = pair_logits
            loss_valid = proposals.valid_mask
            label_indices = proposals.indices
            label_mask = proposals.valid_mask
            query_axis, candidate_axis = 1, 2
        label_output = build_candidate_labels(
            label_indices,
            label_mask,
            targets.mention_pairs,
            targets.mention_mask,
            return_iou=use_soft_iou,
            query_axis=query_axis,
            candidate_axis=candidate_axis,
        )
        if use_soft_iou:
            labels, soft_iou_targets = label_output
        else:
            labels = label_output
            soft_iou_targets = None
        hard = select_hard_negative_candidates(
            loss_logits.detach(), labels, loss_valid,
            negatives_per_positive=self.hard_negatives_per_positive,
            minimum_negatives=self.minimum_hard_negatives,
            keep_all_when_no_positive=self.settings.hard_negative_keep_all_when_absent,
            query_axis=query_axis,
            candidate_axis=candidate_axis,
        )
        pair_query_mask = query_mask
        if self.training and self.settings.negative_query_ratio > 0:
            positive_queries = labels.any(candidate_axis) & query_mask
            absent_queries = query_mask & ~positive_queries
            priorities = torch.rand(
                absent_queries.shape, device=absent_queries.device
            ).masked_fill(~absent_queries, -1.0)
            flat_priorities = priorities.reshape(-1)
            order = flat_priorities.argsort(descending=True)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(order.numel(), device=order.device)
            n_keep = (
                positive_queries.sum() * self.settings.negative_query_ratio
            ).ceil().clamp(
                min=1, max=self.settings.max_negative_queries_per_batch
            )
            selected = (rank.view_as(absent_queries) < n_keep) & absent_queries
            pair_query_mask = positive_queries | selected
        pair_loss = candidate_pair_loss(
            loss_logits,
            labels,
            loss_valid,
            hard_negative_mask=hard,
            query_mask=pair_query_mask,
            reduction=reduction,
            query_axis=query_axis,
            candidate_axis=candidate_axis,
        )
        soft_iou_loss = loss_logits.new_zeros(())
        if soft_iou_targets is not None and self._soft_iou_scale > 0:
            effective = loss_valid & ((labels > 0.5) | hard)
            soft_iou_loss = candidate_pair_loss(
                loss_logits,
                soft_iou_targets,
                effective,
                query_mask=pair_query_mask,
                reduction=reduction,
                query_axis=query_axis,
                candidate_axis=candidate_axis,
            )
        rerank_loss = loss_logits.new_zeros(())
        if self.settings.rerank_listwise_weight > 0:
            rerank_loss = reranker_listwise_loss(
                loss_logits,
                labels,
                loss_valid,
                pair_query_mask,
                query_axis=query_axis,
                candidate_axis=candidate_axis,
            )

        inside_loss = inside_consistency_loss(
            marginals.inside_logits, inside_targets, text_mask, query_mask,
            negative_weight=neg_weight,
            reduction=reduction,
        )

        proposal_loss = loss_logits.new_zeros(())
        proposal_logits = proposals.logits
        proposal_gold = proposals.gold_mask
        proposal_valid = proposals.valid_mask
        proposal_query_axis, proposal_candidate_axis = 1, 2
        if pooled_loss:
            proposal_logits = (
                pooled.proposal_logits.unsqueeze(-1).expand_as(loss_logits)
                if pooled.proposal_logits is not None else None
            )
            proposal_gold = pooled.gold_mask
            proposal_valid = loss_valid
            proposal_query_axis, proposal_candidate_axis = 2, 1
        if (
            proposal_logits is not None
            and proposal_gold is not None
            and self.settings.proposal_loss_weight > 0
        ):
            proposal_loss = proposal_listwise_loss(
                proposal_logits,
                proposal_gold,
                proposal_valid,
                query_mask,
                query_axis=proposal_query_axis,
                candidate_axis=proposal_candidate_axis,
            )
        consistency_loss = loss_logits.new_zeros(())
        if self.settings.consistency_loss_weight > 0:
            consistency_loss = marginal_pair_consistency_loss(
                pair_logits,
                proposals.indices,
                proposals.valid_mask,
                marginals.start_logits,
                marginals.end_logits,
                boundary_keep,
            )
        null_loss = pair_logits.new_zeros(())
        if null_logits is not None and self.settings.abstention_loss_weight > 0:
            null_loss = abstention_loss(
                null_logits, targets.mention_mask, query_mask
            )
        count_loss = pair_logits.new_zeros(())
        if count_log_rates is not None and self.settings.count_loss_weight > 0:
            count_loss = count_log_rate_loss(
                count_log_rates, targets.mention_mask, query_mask
            )

        (
            start_loss,
            end_loss,
            pair_loss,
            soft_iou_loss,
            rerank_loss,
            inside_loss,
            proposal_loss,
            consistency_loss,
            null_loss,
            count_loss,
        ) = (
            _finite_loss_term(value)
            for value in (
                start_loss,
                end_loss,
                pair_loss,
                soft_iou_loss,
                rerank_loss,
                inside_loss,
                proposal_loss,
                consistency_loss,
                null_loss,
                count_loss,
            )
        )
        w = self.loss_weights
        total = (
            w.get("start", 1.0) * start_loss
            + w.get("end", 1.0) * end_loss
            + w.get("pair", 1.0) * pair_loss
            + w.get("inside", 0.5) * inside_loss
            + (
                self.settings.soft_iou_aux_weight
                * self._soft_iou_scale
                * soft_iou_loss
            )
            + self.settings.rerank_listwise_weight * rerank_loss
            + self.settings.proposal_loss_weight * proposal_loss
            + (
                self.settings.consistency_loss_weight
                * self._consistency_scale
                * consistency_loss
            )
            + self.settings.abstention_loss_weight * null_loss
            + self.settings.count_loss_weight * count_loss
        )
        return {
            "total_loss": total,
            "start_loss": start_loss,
            "end_loss": end_loss,
            "pair_loss": pair_loss,
            "soft_iou_loss": soft_iou_loss,
            "rerank_listwise_loss": rerank_loss,
            "inside_loss": inside_loss,
            "proposal_loss": proposal_loss,
            "consistency_loss": consistency_loss,
            "abstention_loss": null_loss,
            "count_loss": count_loss,
        }


def _group_scored_candidates(
    candidates: CandidateTensorBatch,
    *,
    threshold: Union[float, torch.Tensor] = 0.5,
    probabilities: Optional[torch.Tensor] = None,
    count_log_rates: Optional[torch.Tensor] = None,
    adaptive_threshold: bool = False,
) -> List[List[List[Tuple[float, int, int]]]]:
    """Threshold candidates and optionally add top scores up to a predicted count."""
    b, q, c, _ = candidates.indices.shape
    probs = (
        torch.sigmoid(candidates.pair_logits)
        if probabilities is None else probabilities
    )
    eligible = candidates.valid_mask & candidates.query_mask.unsqueeze(-1)
    threshold_value = threshold
    if isinstance(threshold, torch.Tensor):
        if threshold.shape != probs.shape[:2]:
            raise ValueError(
                "tensor threshold must be [B, Q], got "
                f"{tuple(threshold.shape)} for probabilities {tuple(probs.shape)}"
            )
        threshold_value = threshold.to(
            device=probs.device, dtype=probs.dtype
        ).unsqueeze(-1)
    keep = eligible & (probs >= threshold_value)
    if adaptive_threshold:
        if count_log_rates is None:
            raise ValueError(
                "adaptive threshold decoding requires count_log_rates"
            )
        predicted_count = torch.exp(count_log_rates).round().long().clamp(
            min=0, max=c
        )
        ranked_scores = probs.masked_fill(~eligible, MASK_LOGIT)
        order = torch.argsort(
            ranked_scores, dim=-1, descending=True, stable=True
        )
        rank = torch.argsort(order, dim=-1)
        # Union with threshold hits: count guidance fills only when fewer than
        # the predicted number survived, and never removes threshold hits.
        keep = keep | (
            eligible & (rank < predicted_count.unsqueeze(-1))
        )
    bi, qi, ci = keep.nonzero(as_tuple=True)
    out: List[List[List[Tuple[float, int, int]]]] = [
        [[] for _ in range(q)] for _ in range(b)
    ]
    if bi.numel() == 0:
        return out
    rows = torch.stack(
        (
            bi,
            qi,
            candidates.indices[bi, qi, ci, 0],
            candidates.indices[bi, qi, ci, 1],
        ),
        dim=-1,
    ).cpu().numpy()
    scores = probs[bi, qi, ci].float().cpu().numpy()
    segment_ids = rows[:, 0] * q + rows[:, 1]
    change_points = (
        (segment_ids[1:] != segment_ids[:-1]).nonzero()[0] + 1
    ).tolist()
    starts = [0, *change_points]
    ends = [*change_points, len(rows)]
    for lo, hi in zip(starts, ends):
        sample, query = int(rows[lo, 0]), int(rows[lo, 1])
        out[sample][query] = [
            (float(score), int(start), int(end))
            for score, start, end in zip(
                scores[lo:hi], rows[lo:hi, 2], rows[lo:hi, 3]
            )
        ]
    return out


def decode_candidates(
    candidates: CandidateTensorBatch,
    *,
    threshold: float = 0.5,
) -> List[List[List[Tuple[int, int]]]]:
    """Threshold pair logits into score-ordered per-query half-open spans."""
    grouped = _group_scored_candidates(candidates, threshold=threshold)
    return [
        [
            [
                (start, end)
                for _, start, end in sorted(
                    scored, key=lambda item: (-item[0], item[1], item[2])
                )
            ]
            for scored in sample
        ]
        for sample in grouped
    ]


def _extractive_field_names(schema_tokens: List[str]) -> List[str]:
    """Child field/label names in a schema group (the token after each marker)."""
    names: List[str] = []
    for j in range(len(schema_tokens) - 1):
        if schema_tokens[j] in ("[E]", "[C]", "[R]"):
            names.append(schema_tokens[j + 1])
    return names


def _iter_inclusive_spans(value: Any):
    """Yield inclusive ``(start, end)`` token spans from a structure-label field.

    Field values are lists of ``(start, end)`` tuples (all surface occurrences),
    possibly nested; ``(-1, -1)`` and ``None`` mark "not found" and are skipped.
    """
    if value is None:
        return
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(x, int) for x in value)
    ):
        if value != (-1, -1):
            yield value
        return
    if isinstance(value, (list, tuple)):
        for sub in value:
            yield from _iter_inclusive_spans(sub)


def _schema_group_name(schema_tokens: List[str]) -> str:
    """Recover the human-readable task/schema name from its prompt token."""
    if len(schema_tokens) > 2:
        return schema_tokens[2].split(" [DESCRIPTION] ")[0]
    return schema_tokens[0] if schema_tokens else ""


class BoundaryExtractorModel(BaseExtractorModel):
    """Boundary architecture: shared encoder + classification head + boundary head."""

    config_class = ExtractorConfig
    architecture = "boundary"

    def task_module_names(self) -> Tuple[str, ...]:
        base = ("classifier", "boundary_head")
        extras = ()
        if getattr(self, "enable_records", False):
            extras += ("record_decoder",)
        if getattr(self, "enable_relations", False):
            extras += ("relation_scorer",)
        return base + extras

    def _head_touch(self, device) -> torch.Tensor:
        """Exactly-zero term touching every enabled optional head."""
        total = torch.zeros((), device=device)
        for name in ("record_decoder", "relation_scorer", "relation_pair_generator"):
            module = getattr(self, name, None)
            if module is not None:
                for parameter in getattr(module, "parameters", lambda: ())():
                    total = total + parameter.sum() * 0.0
        return total

    def _zero_loss(self, device) -> torch.Tensor:
        """Differentiable zero used when a micro-batch has no supervision."""
        total = self._head_touch(device)
        for parameter in self.parameters():
            if parameter.requires_grad:
                total = total + parameter.sum() * 0.0
        return total

    def __init__(
        self,
        config: ExtractorConfig,
        encoder_config=None,
        tokenizer=None,
        use_flashdeberta: Optional[bool] = None,
        word_splitter=None,
        load_encoder: bool = True,
    ):
        """Build the boundary model.

        Args:
            config: Extractor config with ``architecture="boundary"``.
            encoder_config: Encoder config; required when ``load_encoder`` is False.
            tokenizer: Optional tokenizer; defaults to ``config.model_name``'s.
            use_flashdeberta: Force the FlashDeBERTa backend on or off.
            word_splitter: Optional word splitter name or callable.
            load_encoder: When False, build heads only and set ``encoder`` to
                ``None``; inference then requires ``hidden_states``.

        Raises:
            ValueError: On a non-boundary config, or ``load_encoder=False``
                without ``encoder_config``.
        """
        super().__init__(config)
        if config.architecture != "boundary":
            raise ValueError(
                f"BoundaryExtractorModel requires architecture='boundary', "
                f"got {config.architecture!r}"
            )
        self.config = config

        from gliner2.processor import SchemaTransformer
        if tokenizer is not None:
            self.processor = SchemaTransformer(
                tokenizer=tokenizer,
                token_pooling=config.token_pooling,
                word_splitter=word_splitter,
            )
        else:
            self.processor = SchemaTransformer(
                config.model_name,
                token_pooling=config.token_pooling,
                word_splitter=word_splitter,
            )

        if load_encoder:
            self.encoder = self._load_encoder(
                config.model_name,
                encoder_config,
                getattr(config, "attn_implementation", "sdpa"),
                use_flashdeberta=use_flashdeberta,
            )
            self.encoder.resize_token_embeddings(len(self.processor.tokenizer))
            self.hidden_size = self.encoder.config.hidden_size
        else:
            if encoder_config is None:
                raise ValueError("load_encoder=False requires encoder_config")
            self.encoder = None
            self.hidden_size = encoder_config.hidden_size

        self.classifier = create_mlp(
            input_dim=self.hidden_size,
            intermediate_dims=[self.hidden_size * 2],
            output_dim=1,
            dropout=config.boundary_head.get("dropout", 0.1),
            activation="relu",
            add_layer_norm=False,
        )

        settings = BoundaryHeadSettings(**config.boundary_head)
        self.boundary_settings = settings
        self.enable_records = settings.enable_records
        self.enable_relations = settings.enable_relations
        self.boundary_head = BoundaryHead(
            self.hidden_size, settings, query_dim=self.hidden_size,
            build_candidate_states=settings.enable_records,
        )
        if self.enable_records:
            from gliner2.models.boundary.records import RecordHead
            self.record_decoder = RecordHead(
                self.hidden_size,
                settings.record_dim,
                settings.record_instance_queries,
            )
        if self.enable_relations:
            self.relation_pair_generator = TypedRelationPairGenerator(
                RelationProposalSettings(
                    heads_per_relation=settings.relation_heads_per_type,
                    tails_per_relation=settings.relation_tails_per_type,
                    pair_cap=settings.relation_pair_cap,
                    argument_threshold=settings.relation_argument_proposal_threshold,
                )
            )
            self.relation_scorer = SparseRelationScorer(
                self.hidden_size,
                dropout=settings.dropout,
                relation_query_dim=(
                    2 * self.hidden_size
                    if settings.directional_relation_states
                    else self.hidden_size
                ),
                use_biaffine_content=settings.relation_biaffine_content,
            )

        self._lora_layers = {}
        self._adapter_config = None

    def quantize(self, method: str = "fp16") -> "BoundaryExtractorModel":
        """Cast the complete boundary model for lower-precision inference.

        Args:
            method: ``"fp16"`` (span-parity default) or ``"bf16"``.
        """
        normalized = str(method).strip().lower()
        aliases = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        if normalized not in aliases:
            raise ValueError(
                "quantize method must be 'fp16' or 'bf16', "
                f"got {method!r}"
            )
        self.to(dtype=aliases[normalized])
        logger.info("Converted boundary model to %s", normalized)
        return self

    def compile(self, dynamic: bool = True) -> "BoundaryExtractorModel":
        """Compile the backbone and tensor-heavy boundary regions in place."""
        if not hasattr(torch, "compile"):
            raise RuntimeError("BoundaryExtractorModel.compile requires torch.compile")
        if self.encoder is not None:
            self.encoder = torch.compile(self.encoder, dynamic=dynamic)
        self.boundary_head.boundary_encoder = torch.compile(
            self.boundary_head.boundary_encoder, dynamic=dynamic
        )
        self.boundary_head.boundary_query_head = torch.compile(
            self.boundary_head.boundary_query_head, dynamic=dynamic
        )
        self.boundary_head.boundary_proposer = torch.compile(
            self.boundary_head.boundary_proposer, dynamic=dynamic
        )
        self.boundary_head.pair_scorer = torch.compile(
            self.boundary_head.pair_scorer, dynamic=dynamic
        )
        if self.boundary_settings.candidate_pool == "shared":
            self.boundary_head.shared_pool_builder = torch.compile(
                self.boundary_head.shared_pool_builder, dynamic=dynamic
            )
            self.boundary_head.shared_pool_scorer = torch.compile(
                self.boundary_head.shared_pool_scorer, dynamic=dynamic
            )
        return self

    def apply_lora(
        self,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        targets: Optional[List[str]] = None,
        use_dora: bool = False,
    ) -> "PeftModel":
        """Apply PEFT LoRA adapters and return the wrapped boundary model."""
        from peft import LoraConfig as PeftLoraConfig, get_peft_model
        from gliner2.training.lora import _cast_lora_dtype, _resolve_targets

        base_id = (
            getattr(self, "name_or_path", "")
            or getattr(self.config, "_name_or_path", "")
            or None
        )
        peft_config = PeftLoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=_resolve_targets(self, targets or ["encoder"]),
            bias="none",
            use_dora=use_dora,
            base_model_name_or_path=base_id,
        )
        peft_model = get_peft_model(self, peft_config)
        _cast_lora_dtype(peft_model)
        return peft_model

    # =========================================================================
    # Encoding
    # =========================================================================

    def _encode_core(self, batch, hidden_states=None) -> Dict[str, Any]:
        """Encode a ``PreprocessedBatch`` into padded states + query enumeration.

        Every extractive schema child (``[E]``/``[C]``/``[R]`` marker) becomes one
        boundary query; its query embedding is the marker's contextual embedding
        (``embs[1:]``), aligned 1:1 with the gold field order in
        ``structure_labels``. Classification schemas are enumerated separately and
        scored by the shared classifier. No fixed cross-sample query layout is
        required, so training-time task shuffling is handled naturally.

        Args:
            batch: The collated batch.
            hidden_states: Optional per-row encoder output; see ``encode_tokens``.

        Returns:
            Padded text/query states, masks, and per-sample query specs.
        """
        device = next(self.parameters()).device
        batch = batch.to(device)
        token_embeddings = self.encode_tokens(batch, hidden_states)
        fast_routing = (
            getattr(self.processor, "token_pooling", None) == "first"
            and getattr(batch, "text_word_indices", None) is not None
            and getattr(batch, "text_word_mask", None) is not None
            and getattr(batch, "query_marker_indices", None) is not None
            and getattr(batch, "query_marker_mask", None) is not None
            and len(getattr(batch, "query_layouts", ())) == len(batch)
        )
        if fast_routing:
            h = token_embeddings.shape[-1]

            def gather_routed(indices, mask):
                safe_idx = indices.clamp(0, token_embeddings.shape[1] - 1)
                states = token_embeddings.gather(
                    1, safe_idx.unsqueeze(-1).expand(-1, -1, h)
                )
                return states * mask.unsqueeze(-1).to(states.dtype)

            text_states = gather_routed(
                batch.text_word_indices, batch.text_word_mask
            )
            query_states = gather_routed(
                batch.query_marker_indices, batch.query_marker_mask
            )
            text_mask = batch.text_word_mask
            query_mask = batch.query_marker_mask
            cls_states = gather_routed(
                batch.cls_marker_indices, batch.cls_marker_mask
            )

            ext_specs: List[List[Dict[str, Any]]] = []
            cls_specs: List[List[Dict[str, Any]]] = []
            rel_specs: List[List[Dict[str, Any]]] = []
            word_offsets: List[int] = []
            for i in range(len(batch)):
                layout = batch.query_layouts[i]
                specs_i = [
                    {
                        "group_index": spec.task_index,
                        "field_index": spec.role_index,
                        "task_type": spec.task_type,
                        "task_name": spec.task_name,
                        "field_name": spec.role_name,
                    }
                    for spec in layout.queries
                ]
                ext_specs.append(specs_i)
                text_len_i = (
                    len(batch.start_mappings[i])
                    if batch.start_mappings else int(batch.text_word_counts[i])
                )
                word_offsets.append(max(int(batch.text_word_counts[i]) - text_len_i, 0))

                cls_i = []
                cls_offset = 0
                for group_index in range(batch.schema_counts[i]):
                    if batch.task_types[i][group_index] != "classifications":
                        continue
                    choice_count = max(
                        len(batch.schema_special_indices[i][group_index]) - 1, 0
                    )
                    if choice_count:
                        schema_tokens = batch.schema_tokens_list[i][group_index]
                        cls_i.append({
                            "group_index": group_index,
                            "task_name": _schema_group_name(schema_tokens),
                            "schema_tokens": schema_tokens,
                            "choice_states": cls_states[
                                i, cls_offset:cls_offset + choice_count
                            ],
                            # Legacy decode helper slices ``embs[1:]``; the
                            # group marker itself is not scored.
                            "group_embs": torch.cat((
                                cls_states.new_zeros((1, h)),
                                cls_states[
                                    i, cls_offset:cls_offset + choice_count
                                ],
                            )),
                        })
                    cls_offset += choice_count
                cls_specs.append(cls_i)

                rel_i = []
                groups = {}
                for query_id, spec in enumerate(specs_i):
                    groups.setdefault(spec["group_index"], []).append(query_id)
                for group_index, role_ids_list in groups.items():
                    if batch.task_types[i][group_index] != "relations":
                        continue
                    role_ids = tuple(role_ids_list)
                    if len(role_ids) < 2:
                        continue
                    head_id, tail_id = role_ids[:2]
                    max_q = query_states.shape[1] - 1
                    head_id = min(head_id, max_q)
                    tail_id = min(tail_id, max_q)
                    role_states = query_states[i, [head_id, tail_id]]
                    relation_state = (
                        torch.cat((role_states[0], role_states[1]), dim=-1)
                        if self.boundary_settings.directional_relation_states
                        else role_states.mean(dim=0)
                    )
                    rel_i.append({
                        "group_index": group_index,
                        "relation_type": specs_i[head_id]["task_name"],
                        "spec": RelationTypeSpec(
                            specs_i[head_id]["task_name"],
                            head_query_ids=(head_id,),
                            tail_query_ids=(tail_id,),
                        ),
                        "query_state": relation_state,
                    })
                rel_specs.append(rel_i)

            return {
                "text_states": text_states,
                "text_mask": text_mask,
                "text_lengths": text_mask.sum(-1).long(),
                "query_states": query_states,
                "query_mask": query_mask,
                "ext_specs": ext_specs,
                "cls_specs": cls_specs,
                "rel_specs": rel_specs,
                "word_offsets": word_offsets,
            }

        all_token_embs, all_schema_embs = self.processor.extract_embeddings_from_batch(
            token_embeddings, batch.input_ids, batch
        )

        n = len(batch)
        h = self.hidden_size
        text_lengths = [t.shape[0] for t in all_token_embs]
        text_states, text_mask = _pad_states(
            all_token_embs, h, device, token_embeddings.dtype
        )

        ext_specs: List[List[Dict[str, Any]]] = []
        ext_embs: List[List[torch.Tensor]] = []
        cls_specs: List[List[Dict[str, Any]]] = []
        rel_specs: List[List[Dict[str, Any]]] = []
        word_offsets: List[int] = []

        for i in range(n):
            specs_i: List[Dict[str, Any]] = []
            embs_i: List[torch.Tensor] = []
            cls_i: List[Dict[str, Any]] = []
            rel_i: List[Dict[str, Any]] = []
            text_len_i = len(batch.start_mappings[i]) if batch.start_mappings else text_lengths[i]
            word_offsets.append(max(text_lengths[i] - text_len_i, 0))

            for g in range(batch.schema_counts[i]):
                task_type = batch.task_types[i][g]
                schema_tokens = batch.schema_tokens_list[i][g]
                group_embs = all_schema_embs[i][g]
                if not group_embs:
                    continue
                field_names = _extractive_field_names(schema_tokens)
                name = _schema_group_name(schema_tokens)
                if task_type == "classifications":
                    if len(group_embs) > 1:
                        cls_i.append({
                            "group_index": g,
                            "task_name": name,
                            "schema_tokens": schema_tokens,
                            "group_embs": torch.stack(group_embs),
                        })
                    continue
                first_query_id = len(specs_i)
                for fidx, fname in enumerate(field_names):
                    if 1 + fidx >= len(group_embs):
                        break
                    specs_i.append({
                        "group_index": g,
                        "field_index": fidx,
                        "task_type": task_type,
                        "task_name": name,
                        "field_name": fname,
                    })
                    embs_i.append(group_embs[1 + fidx])
                if task_type == "relations" and len(specs_i) > first_query_id:
                    role_ids = tuple(range(first_query_id, len(specs_i)))
                    if len(role_ids) >= 2:
                        head_id, tail_id = role_ids[:2]
                        role_states = torch.stack(group_embs[1:3])
                        relation_state = (
                            torch.cat((role_states[0], role_states[1]), dim=-1)
                            if self.boundary_settings.directional_relation_states
                            else role_states.mean(dim=0)
                        )
                        rel_i.append({
                            "group_index": g,
                            "relation_type": name,
                            "spec": RelationTypeSpec(
                                name, head_query_ids=(head_id,), tail_query_ids=(tail_id,)
                            ),
                            "query_state": relation_state,
                        })

            ext_specs.append(specs_i)
            ext_embs.append(embs_i)
            cls_specs.append(cls_i)
            rel_specs.append(rel_i)

        query_sequences = [
            torch.stack(embs_i) if embs_i else token_embeddings.new_zeros((0, h))
            for embs_i in ext_embs
        ]
        query_states, query_mask = _pad_states(
            query_sequences, h, device, token_embeddings.dtype
        )

        return {
            "text_states": text_states,
            "text_mask": text_mask,
            "text_lengths": torch.tensor(text_lengths, dtype=torch.long, device=device),
            "query_states": query_states,
            "query_mask": query_mask,
            "ext_specs": ext_specs,
            "cls_specs": cls_specs,
            "rel_specs": rel_specs,
            "word_offsets": word_offsets,
        }

    def encode(self, batch) -> EncodedBatch:
        """Encode a ``PreprocessedBatch`` into a dense ``EncodedBatch``."""
        core = self._encode_core(batch)
        layouts: List[QueryLayout] = []
        for specs in core["ext_specs"]:
            queries = tuple(
                QuerySpec(
                    query_id=j,
                    task_index=spec["group_index"],
                    task_type=spec["task_type"],
                    task_name=spec["task_name"],
                    role_index=spec["field_index"],
                    role_name=spec["field_name"],
                    field_path=(spec["task_name"], spec["field_name"]),
                    extractive=True,
                )
                for j, spec in enumerate(specs)
            )
            layouts.append(QueryLayout(queries=queries))
        return EncodedBatch(
            text_states=core["text_states"],
            text_mask=core["text_mask"],
            text_lengths=core["text_lengths"],
            query_states=core["query_states"],
            query_mask=core["query_mask"],
            query_layouts=tuple(layouts),
        )

    def _targets_from_structure(self, batch, core: Dict[str, Any]) -> Optional[PaddedTargetBatch]:
        """Build a :class:`PaddedTargetBatch` from ``structure_labels`` + queries."""
        structure_labels = getattr(batch, "structure_labels", None)
        if not structure_labels:
            return None
        graphs: List[TargetGraph] = []
        query_counts: List[int] = []
        text_lengths: List[int] = []
        for i in range(len(batch)):
            specs = core["ext_specs"][i]
            length = int(core["text_lengths"][i])
            mentions: List[MentionTarget] = []
            if i < len(structure_labels):
                sample_labels = structure_labels[i]
                for qid, spec in enumerate(specs):
                    group_index = spec["group_index"]
                    if group_index >= len(sample_labels):
                        continue
                    structure = sample_labels[group_index]
                    if not structure or (isinstance(structure, (list, tuple)) and structure[0] == 0):
                        continue
                    if isinstance(structure, (int, float)):
                        continue
                    fidx = spec["field_index"]
                    instances = structure[1] if len(structure) > 1 else []
                    for inst in instances:
                        if fidx >= len(inst):
                            continue
                        for (s, e_inc) in _iter_inclusive_spans(inst[fidx]):
                            if 0 <= s <= e_inc < length:
                                mentions.append(MentionTarget(qid, s, e_inc + 1))
            graphs.append(TargetGraph(mentions=tuple(mentions)))
            query_counts.append(len(specs))
            text_lengths.append(length)
        if not any(query_counts):
            return None
        return pad_target_graphs(
            graphs, query_counts, text_lengths,
            self.boundary_head.settings.max_gold_per_query,
            build_dense=False,
        )

    def _classification_loss(self, batch, core: Dict[str, Any]) -> torch.Tensor:
        """Binary cross-entropy over classification choices (shared classifier).

        Normalized by the number of supervised labels (so it is a mean, on the
        same scale as the boundary marginal losses) and scaled by the configured
        ``classification_loss_weight``. A label/logit shape disagreement is a
        real preprocessing bug and raises rather than silently dropping the
        group's supervision.
        """
        device = core["text_states"].device
        total = torch.zeros((), device=device)
        structure_labels = getattr(batch, "structure_labels", None)
        if not structure_labels:
            return total
        label_count = 0
        for i in range(len(batch)):
            if i >= len(structure_labels):
                continue
            for cls in core["cls_specs"][i]:
                group_index = cls["group_index"]
                if group_index >= len(structure_labels[i]):
                    continue
                labels_raw = structure_labels[i][group_index]
                if not labels_raw or (isinstance(labels_raw, (int, float)) and labels_raw == 0):
                    continue
                if isinstance(labels_raw, (list, tuple)) and len(labels_raw) >= 2 and isinstance(labels_raw[0], int) and isinstance(labels_raw[1], list):
                    continue
                choice_states = (
                    cls["choice_states"]
                    if "choice_states" in cls else cls["group_embs"][1:]
                )
                logits = self.classifier(choice_states).squeeze(-1)
                labels = torch.tensor(labels_raw, dtype=logits.dtype, device=device)
                if labels.shape != logits.shape:
                    continue
                total = total + F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="sum"
                )
                label_count += labels.numel()
        if label_count == 0:
            return total
        weight = getattr(self.boundary_settings, "classification_loss_weight", 1.0)
        return weight * total / label_count

    @staticmethod
    def _single_sample_candidates(
        candidates: CandidateTensorBatch, sample_index: int
    ) -> CandidateTensorBatch:
        """Keep one sample while preserving the padded candidate contract."""
        return CandidateTensorBatch(
            indices=candidates.indices[sample_index:sample_index + 1],
            proposal_logits=(
                candidates.proposal_logits[sample_index:sample_index + 1]
                if candidates.proposal_logits is not None else None
            ),
            pair_logits=candidates.pair_logits[sample_index:sample_index + 1],
            valid_mask=candidates.valid_mask[sample_index:sample_index + 1],
            query_mask=candidates.query_mask[sample_index:sample_index + 1],
            candidate_states=(
                candidates.candidate_states[sample_index:sample_index + 1]
                if candidates.candidate_states is not None else None
            ),
        )

    def _relation_loss(self, batch, core, candidates, targets=None) -> Optional[torch.Tensor]:
        """Binary relation-pair loss over sparse, gold-inclusive proposals."""
        if candidates is None:
            return None
        relation_schemas = [
            [entry["spec"] for entry in rel_specs]
            for rel_specs in core["rel_specs"]
        ]
        if not any(relation_schemas):
            return None
        edge_targets = getattr(targets, "edge_targets", None)
        routing = (
            edge_targets[2:6]
            if isinstance(edge_targets, tuple) and len(edge_targets) >= 6
            else None
        )
        pairs = self.relation_pair_generator.generate_batched(
            candidates,
            [QueryLayout(queries=())] * candidates.indices.shape[0],
            relation_schemas,
            compact=False,
            routing=routing,
        )
        relation_dim = (
            2 * self.hidden_size
            if self.boundary_settings.directional_relation_states
            else self.hidden_size
        )
        relation_states = torch.nn.utils.rnn.pad_sequence(
            [
                torch.stack([entry["query_state"] for entry in rel_specs])
                if rel_specs else core["text_states"].new_zeros((0, relation_dim))
                for rel_specs in core["rel_specs"]
            ],
            batch_first=True,
        )
        logits = self.relation_scorer(
            core["text_states"], relation_states, candidates, pairs
        )
        if not isinstance(edge_targets, tuple):
            return logits.sum() * 0.0
        gold_pairs, gold_mask = edge_targets[:2]
        if pairs.relation_index.numel() == 0:
            return logits.sum() * 0.0
        max_rel_dim = gold_pairs.shape[1]
        if max_rel_dim == 0 or gold_pairs.shape[0] == 0:
            return logits.sum() * 0.0
        safe_rel_idx, valid_rel = safe_relation_indices(
            pairs.relation_index, max_rel_dim
        )
        safe_batch_idx = pairs.batch_index.clamp(
            min=0, max=gold_pairs.shape[0] - 1
        )
        valid_batch = (
            (pairs.batch_index >= 0)
            & (pairs.batch_index < gold_pairs.shape[0])
        )
        pair_coords = torch.stack(
            (pairs.head_start, pairs.head_end, pairs.tail_start, pairs.tail_end),
            dim=-1,
        )
        selected_gold = gold_pairs[safe_batch_idx, safe_rel_idx]
        selected_mask = gold_mask[safe_batch_idx, safe_rel_idx]
        selected_mask = selected_mask & (valid_rel & valid_batch).unsqueeze(-1)
        labels = (
            (pair_coords.unsqueeze(1) == selected_gold).all(-1)
            & selected_mask
        ).any(-1).to(logits.dtype)
        if labels.shape != logits.shape:
            labels = labels.view_as(logits) if labels.numel() == logits.numel() else labels[:logits.shape[0]]
        pair_mask = (
            pairs.pair_mask
            if pairs.pair_mask is not None
            else torch.ones_like(labels, dtype=torch.bool)
        )
        pair_mask = pair_mask & valid_rel & valid_batch
        if pair_mask.shape != logits.shape:
            pair_mask = pair_mask[:logits.shape[0]] if pair_mask.numel() >= logits.numel() else torch.ones_like(logits, dtype=torch.bool)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = (loss * pair_mask.to(loss.dtype)).sum() / pair_mask.sum().clamp_min(1)
        return self.boundary_settings.relation_loss_weight * loss

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(
        self,
        batch,
        *,
        return_candidates: bool = True,
        return_individual_losses: bool = False,
        gold_injection_prob: Optional[float] = None,
        collect_diagnostics: Optional[bool] = None,
    ) -> ExtractorOutput:
        core = self._encode_core(batch)
        targets = getattr(batch, "targets", None)
        if targets is None and self.training:
            targets = self._targets_from_structure(batch, core)
        if targets is not None:
            target_lengths = core["text_lengths"].to(
                targets.mention_pairs.device
            )
            validate_masked_spans(
                targets.mention_pairs,
                targets.mention_mask,
                target_lengths,
                name="mention_pairs",
            )
            # Move to the model device regardless of origin: collator targets
            # arrive on CPU, and the fallback path (_targets_from_structure ->
            # pad_target_graphs) also allocates on CPU. Either must meet the
            # accelerator logits, so normalize here.
            targets = targets.to(core["text_states"].device)

        if core["query_states"].shape[1] > 0:
            output = self.boundary_head(
                core["text_states"], core["text_mask"],
                core["query_states"], core["query_mask"],
                targets, return_candidates=return_candidates,
                gold_injection_prob=gold_injection_prob,
                collect_diagnostics=collect_diagnostics,
            )
            if output.candidates is not None:
                validate_candidate_indices(
                    output.candidates.indices,
                    output.candidates.valid_mask,
                    core["text_lengths"],
                )
        else:
            # Classification-only batch: no extractive queries to score.
            output = ExtractorOutput(
                candidates=None,
                total_loss=None,
                loss=None,
                losses={},
                batch_size=len(batch),
            )

        cls_loss = _finite_loss_term(self._classification_loss(batch, core))

        record_loss = None
        if (
            self.enable_records
            and self.training
            and targets is not None
            and getattr(targets, "records", None) is not None
            and output.candidates is not None
            and output.candidates.candidate_states is not None
        ):
            try:
                record_loss = self._record_loss(batch, core, output.candidates, targets)
            except RuntimeError as exc:
                if is_fatal_device_error(exc):
                    raise
                logger.warning(
                    "Dropped record auxiliary loss after RuntimeError: %s", exc
                )
                record_loss = None
            except (IndexError, ValueError) as exc:
                logger.warning(
                    "Dropped record auxiliary loss after %s: %s",
                    type(exc).__name__,
                    exc,
                )
                record_loss = None
        relation_loss = None
        if self.enable_relations and self.training and output.candidates is not None:
            try:
                relation_loss = self._relation_loss(
                    batch, core, output.candidates, targets
                )
            except RuntimeError as exc:
                if is_fatal_device_error(exc):
                    raise
                logger.warning(
                    "Dropped relation auxiliary loss after RuntimeError: %s", exc
                )
                relation_loss = None
            except (IndexError, ValueError) as exc:
                logger.warning(
                    "Dropped relation auxiliary loss after %s: %s",
                    type(exc).__name__,
                    exc,
                )
                relation_loss = None
        if record_loss is not None:
            record_loss = {
                name: _finite_loss_term(value)
                for name, value in record_loss.items()
            }
        if relation_loss is not None:
            relation_loss = _finite_loss_term(relation_loss)

        # Explicit supervision flag. ``bool(tensor)`` on a loss is wrong in
        # principle (a device sync that also conflates "zero loss" with "no
        # loss"), so decide from the presence of classification queries rather
        # than the value of ``cls_loss``.
        has_cls_supervision = any(bool(specs) for specs in core["cls_specs"])
        has_supervision = (
            targets is not None
            or has_cls_supervision
            or record_loss is not None
            or relation_loss is not None
        )
        if has_supervision:
            span_total = (
                _finite_loss_term(output.total_loss)
                if output.total_loss is not None
                else torch.zeros((), device=cls_loss.device)
            )
            combined = span_total + cls_loss + self._head_touch(cls_loss.device)
            if record_loss is not None:
                combined = combined + record_loss["total"]
            if relation_loss is not None:
                combined = combined + relation_loss
            output.total_loss = combined
            output.loss = combined
            if output.losses is not None:
                output.losses["classification_loss"] = cls_loss
                if record_loss is not None:
                    output.losses["record_object_loss"] = record_loss["object"]
                    output.losses["record_field_loss"] = record_loss["field"]
                if relation_loss is not None:
                    output.losses["relation_loss"] = relation_loss
        return output

    def _record_loss(self, batch, core, candidates, targets) -> Dict[str, torch.Tensor]:
        """Aggregate record object + field-assignment losses across the batch."""
        from gliner2.models.boundary.records import (
            compute_dense_batch_loss,
            compute_dense_group_loss,
            compute_group_loss,
        )

        device = core["query_states"].device
        obj_total = torch.zeros((), device=device)
        field_total = torch.zeros((), device=device)
        object_count = 0
        field_count = 0
        record_specs = getattr(batch, "record_specs", ())
        per_sample_records = targets.records  # List[List[RecordTarget]]
        packed_records = getattr(targets, "record_targets", None)
        weight = self.boundary_settings.record_loss_weight
        if (
            self.boundary_settings.candidate_pool == "shared"
            and isinstance(packed_records, tuple)
            and len(packed_records) >= 9
        ):
            packed_spans, packed_mask, record_mask = packed_records[:3]
            dense = self.record_decoder.forward_groups_dense(
                core["query_states"], candidates, packed_records[3:9]
            )
            gold_indicator = (
                (
                    packed_spans.unsqueeze(-2)
                    == dense.pool_spans[:, None, None, None, None, :, :]
                ).all(-1)
                & packed_mask.unsqueeze(-1)
            ).any(-2)
            gold_indicator &= dense.field_membership.unsqueeze(2)
            try:
                losses = compute_dense_batch_loss(
                    dense, gold_indicator, record_mask
                )
            except TargetCapacityError:
                return {
                    "total": obj_total,
                    "object": obj_total,
                    "field": field_total,
                }
            object_mean = losses["object_loss"]
            field_mean = losses["field_loss"]
            return {
                "total": weight * (object_mean + field_mean),
                "object": object_mean,
                "field": field_mean,
            }

        for i in range(len(batch)):
            if i >= len(record_specs):
                continue
            specs = record_specs[i]
            if not specs:
                continue
            sample_records = (
                per_sample_records[i] if i < len(per_sample_records) else []
            )
            query_states_i = core["query_states"][i]
            for group_index, (task_index, spec) in enumerate(specs.items()):
                recs = [r for r in sample_records if r.task_index == task_index]
                try:
                    if self.boundary_settings.candidate_pool == "shared":
                        group = self.record_decoder.forward_group_dense(
                            spec, query_states_i, candidates, i
                        )
                        gold_indicator = None
                        if isinstance(packed_records, tuple):
                            packed_spans, packed_mask, _ = packed_records[:3]
                            spans = packed_spans[
                                i, group_index, :len(recs), :len(spec.fields)
                            ]
                            span_mask = packed_mask[
                                i, group_index, :len(recs), :len(spec.fields)
                            ]
                            gold_indicator = (
                                (
                                    spans.unsqueeze(-2)
                                    == group.pool_spans[None, None, None, :, :]
                                ).all(-1)
                                & span_mask.unsqueeze(-1)
                            ).any(-2)
                            gold_indicator = (
                                gold_indicator
                                & group.field_membership.unsqueeze(0)
                            )
                        losses = compute_dense_group_loss(
                            group, recs, gold_indicator=gold_indicator
                        )
                    else:
                        group = self.record_decoder.forward_group(
                            spec, query_states_i, candidates, i
                        )
                        losses = compute_group_loss(group, recs)
                except (TargetCapacityError, ValueError, IndexError):
                    continue
                group_object_count = int(losses.get("object_count", 1))
                group_field_count = int(losses.get("field_count", 1))
                obj_total = (
                    obj_total + losses["object_loss"] * group_object_count
                )
                field_total = (
                    field_total + losses["field_loss"] * group_field_count
                )
                object_count += group_object_count
                field_count += group_field_count

        obj = obj_total / max(object_count, 1)
        field = field_total / max(field_count, 1)
        return {"object": obj, "field": field, "total": weight * (obj + field)}

    def score_candidates(
        self, batch, *, return_auxiliary_logits: bool = False
    ) -> Union[CandidateTensorBatch, Tuple[CandidateTensorBatch, Dict[str, torch.Tensor]]]:
        """Score sparse boundary candidates for ``batch``.

        By default returns the :class:`CandidateTensorBatch`. When
        ``return_auxiliary_logits=True`` returns ``(candidates, aux)`` where
        ``aux`` carries the boundary marginals backing those candidates:
        ``start_logits``/``end_logits`` (``[B, Q, L+1]``) and ``inside_logits``
        (``[B, Q, L]``, ``None`` when inside evidence is disabled).
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                output = self.forward(batch, return_candidates=True)
        finally:
            self.train(was_training)
        if not return_auxiliary_logits:
            return output.candidates
        aux = {
            "start_logits": output.start_logits,
            "end_logits": output.end_logits,
            "inside_logits": output.inside_logits,
        }
        return output.candidates, aux

    def score_explicit_spans(
        self,
        encoded: EncodedBatch,
        indices: torch.LongTensor,
        valid_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Score explicit spans in an already encoded batch.

        This architecture-aware public primitive bypasses proposal top-k while
        preserving the checkpoint's proposal compatibility and pair reranker.
        """
        return self.boundary_head.score_explicit_spans(
            encoded.text_states,
            encoded.text_mask,
            encoded.query_states,
            encoded.query_mask,
            indices,
            valid_mask,
        )

    # =========================================================================
    # Serialization
    # =========================================================================

    def push_to_hub(self, repo_id: str, private: bool = True):
        """Save and upload this boundary checkpoint to the Hugging Face Hub."""
        from huggingface_hub import HfApi

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            api = HfApi()
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
            return api.upload_folder(repo_id=repo_id, folder_path=tmp_dir)

    def save_pretrained(self, save_directory: str, **kwargs):
        from safetensors.torch import save_file

        if self.encoder is None:
            raise ValueError("cannot save a boundary model built with load_encoder=False")
        os.makedirs(save_directory, exist_ok=True)
        self.config.architecture = "boundary"
        self.config.architectures = [type(self).__name__]
        self.config.save_pretrained(save_directory)

        encoder_config_path = os.path.join(save_directory, "encoder_config")
        os.makedirs(encoder_config_path, exist_ok=True)
        self.encoder.config.save_pretrained(encoder_config_path)

        save_file(self.state_dict(), os.path.join(save_directory, "model.safetensors"))
        self.processor.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, **kwargs):
        config = kwargs.pop("config", None)
        model_options, hub_kwargs = split_load_kwargs(
            kwargs, context=f"{cls.__name__}.from_pretrained"
        )
        quantize = model_options.pop("quantize", False)
        compile_model = model_options.pop("compile", False)
        map_location = model_options.pop("map_location", None)
        use_flashdeberta = model_options.pop("use_flashdeberta", None)
        word_splitter = model_options.pop("word_splitter", None)

        if config is None:
            config = cls.config_class.from_pretrained(
                checkpoint_file(repo_or_dir, "config.json", hub_kwargs)
            )
        encoder_config = AutoConfig.from_pretrained(
            checkpoint_file(
                repo_or_dir, "encoder_config/config.json", hub_kwargs
            )
        )
        tokenizer_source = repo_or_dir
        if os.path.isdir(str(repo_or_dir)) and hub_kwargs.get("subfolder"):
            tokenizer_source = os.path.join(
                str(repo_or_dir), str(hub_kwargs["subfolder"])
            )
        tokenizer = load_extractor_tokenizer(tokenizer_source)
        model = cls(
            config,
            encoder_config=encoder_config,
            tokenizer=tokenizer,
            use_flashdeberta=use_flashdeberta,
            word_splitter=word_splitter,
        )

        state_dict = load_checkpoint_state_dict(repo_or_dir, hub_kwargs)
        reconcile_encoder_embeddings(model, state_dict)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint at {repo_or_dir!r} does not match this model's "
                "boundary-head configuration. Parameter-changing flags such as "
                "enable_span_content, enable_rotary_endpoints, "
                "boundary_attention_layers, query_conditioned_inside_weight, "
                "endpoint_difference_features, multihead_pair_compat_heads, "
                "candidate_pool, pool_boundary_top_k, pool_size, "
                "candidate_attention_layers, candidate_attention_heads, "
                "query_attention_layers, "
                "enable_abstention, enable_count_head, enable_records, and "
                "enable_relations must match the training configuration. "
                f"Original error:\n{exc}"
            ) from exc

        model.config._name_or_path = repo_or_dir
        model.name_or_path = repo_or_dir
        return apply_post_load_options(
            model,
            map_location=map_location,
            quantize=quantize,
            compile_model=compile_model,
            compile_dynamic=True,
        )


__all__ = [
    "BoundaryHead",
    "BoundaryExtractorModel",
    "decode_candidates",
    "proposal_settings_from_head",
]
