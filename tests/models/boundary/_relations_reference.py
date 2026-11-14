"""FROZEN REFERENCE: TypedRelationPairGenerator.generate_batched as it was before the shared-order select rewrite.

The per-relation [B, R, Q*K] sort it does is the behaviour the rewritten ``select`` must reproduce exactly;
tests/models/boundary/test_relation_select_equivalence.py compares every output slot against it. Do not edit.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from gliner2.models.boundary.relations import (  # noqa: F401
    CandidateTensorBatch, QueryLayout, RelationPairBatch, RelationTypeSpec, TypedRelationPairGenerator, _query_type,
)


class ReferenceGenerator(TypedRelationPairGenerator):
    def generate_batched(
        self,
        candidates: CandidateTensorBatch,
        query_layouts: Sequence[QueryLayout],
        relation_schemas: Sequence[Sequence[RelationTypeSpec]],
        *,
        compact: bool = False,
        routing: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> RelationPairBatch:
        """Generate all ``[B,R,pair_cap]`` proposals with tensor operations.

        ``compact=False`` is the synchronization-free training representation:
        invalid padded proposals remain present and are identified by
        :attr:`RelationPairBatch.pair_mask`. ``compact=True`` is the
        backward-compatible decode representation and materializes Python keys.
        """
        s = self.settings
        device = candidates.indices.device
        bsz, queries, cand_count = candidates.valid_mask.shape
        rel_count = (
            routing[0].shape[1]
            if routing is not None
            else max((len(x) for x in relation_schemas), default=0)
        )
        if rel_count == 0:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return RelationPairBatch(
                empty, empty, empty, empty, empty, empty,
                candidates.pair_logits.new_zeros(0),
                candidates.pair_logits.new_zeros(0),
                pair_mask=torch.zeros(0, dtype=torch.bool, device=device),
            )

        if routing is not None:
            head_member, tail_member, relation_valid, allow_self = routing
            head_member = head_member.to(device=device)
            tail_member = tail_member.to(device=device)
            relation_valid = relation_valid.to(device=device)
            allow_self = allow_self.to(device=device)
        else:
            # Decode-only fallback. Training supplies collator-built routing and
            # never performs device scalar writes in this loop.
            head_member = torch.zeros(
                bsz, rel_count, queries, dtype=torch.bool, device=device
            )
            tail_member = torch.zeros_like(head_member)
            relation_valid = torch.zeros(
                bsz, rel_count, dtype=torch.bool, device=device
            )
            allow_self = torch.zeros_like(relation_valid)
            for batch_index, schemas in enumerate(relation_schemas):
                for relation_index, spec in enumerate(schemas):
                    relation_valid[batch_index, relation_index] = True
                    allow_self[batch_index, relation_index] = spec.allow_self
                    valid_h = [q for q in spec.head_query_ids if 0 <= q < queries]
                    valid_t = [q for q in spec.tail_query_ids if 0 <= q < queries]
                    if valid_h:
                        head_member[batch_index, relation_index, valid_h] = True
                    if valid_t:
                        tail_member[batch_index, relation_index, valid_t] = True

        probs = torch.sigmoid(candidates.pair_logits)
        base_valid = candidates.valid_mask & candidates.query_mask.unsqueeze(-1)
        flat_prob = probs.reshape(bsz, 1, queries * cand_count).expand(-1, rel_count, -1)
        flat_valid = base_valid.reshape(bsz, 1, -1).expand(-1, rel_count, -1)
        head_valid = flat_valid & head_member.unsqueeze(-1).expand(
            -1, -1, -1, cand_count
        ).reshape(bsz, rel_count, -1)
        tail_valid = flat_valid & tail_member.unsqueeze(-1).expand(
            -1, -1, -1, cand_count
        ).reshape(bsz, rel_count, -1)
        threshold = flat_prob >= s.argument_threshold
        head_valid &= threshold
        tail_valid &= threshold
        floor = torch.finfo(flat_prob.dtype).min
        flat_spans = candidates.indices.reshape(bsz, queries * cand_count, 2)

        def select(valid: torch.BoolTensor, requested: int):
            take = min(requested, queries * cand_count)
            secondary = torch.arange(
                queries * cand_count, device=device
            ).view(1, 1, -1).expand(bsz, rel_count, -1)
            end_key = flat_spans[..., 1].unsqueeze(1).expand(
                -1, rel_count, -1
            )
            end_order = torch.argsort(
                end_key.gather(-1, secondary), dim=-1, stable=True
            )
            secondary = secondary.gather(-1, end_order)
            start_key = flat_spans[..., 0].unsqueeze(1).expand(
                -1, rel_count, -1
            )
            start_order = torch.argsort(
                start_key.gather(-1, secondary), dim=-1, stable=True
            )
            secondary = secondary.gather(-1, start_order)
            ordered_score = flat_prob.gather(-1, secondary)
            ordered_valid = valid.gather(-1, secondary)
            rank_in_secondary = torch.argsort(
                ordered_score.masked_fill(~ordered_valid, floor),
                dim=-1, descending=True, stable=True,
            )[..., :take]
            ranked = secondary.gather(-1, rank_in_secondary)
            selected_valid = valid.gather(-1, ranked)
            selected_prob = flat_prob.gather(-1, ranked)
            if take < requested:
                pad = requested - take
                ranked = F.pad(ranked, (0, pad))
                selected_valid = F.pad(selected_valid, (0, pad), value=False)
                selected_prob = F.pad(selected_prob, (0, pad))
            qslot = torch.div(ranked, cand_count, rounding_mode="floor")
            cslot = ranked - qslot * cand_count
            qslot = qslot.clamp(0, queries - 1)
            cslot = cslot.clamp(0, cand_count - 1)
            batch = torch.arange(bsz, device=device)[:, None, None]
            spans = candidates.indices[batch, qslot, cslot]
            return selected_prob, qslot, spans, selected_valid

        hp, hq, hspan, hvalid = select(head_valid, s.heads_per_relation)
        tp, tq, tspan, tvalid = select(tail_valid, s.tails_per_relation)
        pair_score = hp.unsqueeze(-1) * tp.unsqueeze(-2)
        pair_valid = hvalid.unsqueeze(-1) & tvalid.unsqueeze(-2)
        same_span = (hspan.unsqueeze(-2) == tspan.unsqueeze(-3)).all(-1)
        pair_valid &= allow_self[..., None, None] | ~same_span
        pair_valid &= relation_valid[..., None, None]
        flat_pair_score = pair_score.flatten(2)
        flat_pair_valid = pair_valid.flatten(2)
        take = min(s.pair_cap, flat_pair_score.shape[-1])
        keep = torch.argsort(
            flat_pair_score.masked_fill(~flat_pair_valid, floor),
            dim=-1, descending=True, stable=True,
        )[..., :take]
        kept_valid = flat_pair_valid.gather(-1, keep)
        if take < s.pair_cap:
            keep = F.pad(keep, (0, s.pair_cap - take))
            kept_valid = F.pad(kept_valid, (0, s.pair_cap - take), value=False)
        hi = torch.div(keep, s.tails_per_relation, rounding_mode="floor")
        ti = keep - hi * s.tails_per_relation
        hi = hi.clamp(0, s.heads_per_relation - 1)
        ti = ti.clamp(0, s.tails_per_relation - 1)

        def gather_selected(values: torch.Tensor, index: torch.LongTensor):
            return values.gather(
                2, index.clamp(0, values.shape[2] - 1).unsqueeze(-1).expand(*index.shape, values.shape[-1])
            )

        hs = gather_selected(hspan, hi)
        ts = gather_selected(tspan, ti)
        hp_out = hp.gather(2, hi.clamp(0, hp.shape[2] - 1))
        tp_out = tp.gather(2, ti.clamp(0, tp.shape[2] - 1))
        hq_out = hq.gather(2, hi.clamp(0, hq.shape[2] - 1))
        tq_out = tq.gather(2, ti.clamp(0, tq.shape[2] - 1))
        bi = torch.arange(bsz, device=device)[:, None, None].expand_as(keep)
        ri = torch.arange(rel_count, device=device)[None, :, None].expand_as(keep)

        flat_mask = kept_valid.reshape(-1)
        tensors = [
            bi.reshape(-1), ri.reshape(-1), hs[..., 0].reshape(-1),
            hs[..., 1].reshape(-1), ts[..., 0].reshape(-1),
            ts[..., 1].reshape(-1), hp_out.reshape(-1), tp_out.reshape(-1),
            hq_out.reshape(-1), tq_out.reshape(-1),
        ]
        if compact:
            tensors = [value[flat_mask] for value in tensors]
            flat_mask = torch.ones_like(tensors[0], dtype=torch.bool)
        out = RelationPairBatch(
            batch_index=tensors[0], relation_index=tensors[1],
            head_start=tensors[2], head_end=tensors[3],
            tail_start=tensors[4], tail_end=tensors[5],
            head_prob=tensors[6], tail_prob=tensors[7],
            pair_mask=flat_mask,
        )
        if compact:
            # Decode-only metadata: Python conversion is intentionally absent
            # from the training representation.
            for index in range(len(out)):
                batch_index = int(out.batch_index[index])
                relation_index = int(out.relation_index[index])
                spec = relation_schemas[batch_index][relation_index]
                layout = (
                    query_layouts[batch_index]
                    if batch_index < len(query_layouts) else None
                )
                out.relation_types.append(spec.relation_type)
                # Endpoint type names are only presentation metadata. For the
                # common single-type schema use the declared query directly.
                hquery = int(tensors[8][index])
                tquery = int(tensors[9][index])
                out.head_keys.append((
                    _query_type(layout, hquery) if layout is not None else str(hquery),
                    int(out.head_start[index]), int(out.head_end[index]),
                ))
                out.tail_keys.append((
                    _query_type(layout, tquery) if layout is not None else str(tquery),
                    int(out.tail_start[index]), int(out.tail_end[index]),
                ))
        return out
