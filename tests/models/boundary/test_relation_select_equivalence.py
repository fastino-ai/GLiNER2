"""The rewritten relation ``select`` (one candidate order + blocked topk) reproduces the original per-relation sort.

Every output slot of ``generate_batched`` is compared against the frozen pre-rewrite implementation
(``_relations_reference.py``), in both representations, on random cases built to hit the edges: tied scores,
invalid candidates and masked queries, relations with fewer eligible candidates than ``heads_per_relation``
(the invalid padding), more requested than candidates exist, batch > 1, and a budget small enough to force
one relation per block.
"""
import sys
from pathlib import Path

import pytest
import torch

from gliner2.models.boundary.relations import (
    CandidateTensorBatch, RelationProposalSettings, RelationTypeSpec, TypedRelationPairGenerator,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _relations_reference import ReferenceGenerator  # noqa: E402

FIELDS = ("batch_index", "relation_index", "head_start", "head_end", "tail_start", "tail_end",
          "head_prob", "tail_prob", "pair_mask")


def _case(seed: int, bsz: int, queries: int, cands: int, rels: int):
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, 12, (bsz, queries, cands), generator=g)
    ends = starts + torch.randint(1, 4, (bsz, queries, cands), generator=g)
    logits = torch.randint(-3, 4, (bsz, queries, cands), generator=g).float() / 2   # few values: many ties
    candidates = CandidateTensorBatch(
        indices=torch.stack([starts, ends], -1), proposal_logits=None, pair_logits=logits,
        valid_mask=torch.rand(bsz, queries, cands, generator=g) > 0.3,
        query_mask=torch.rand(bsz, queries, generator=g) > 0.2,
    )
    schemas = []
    for _ in range(bsz):
        specs = []
        for r in range(rels):
            k = 1 + r % 2   # one or two member queries: often fewer eligible than requested
            heads = tuple(torch.randperm(queries, generator=g)[:k].tolist())
            tails = tuple(torch.randperm(queries, generator=g)[:k].tolist())
            specs.append(RelationTypeSpec(f"rel{r}", heads, tails, allow_self=bool(r % 3 == 0)))
        schemas.append(specs)
    return candidates, schemas


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_select_matches_the_original_per_relation_sort(seed, compact):
    settings = RelationProposalSettings(heads_per_relation=5, tails_per_relation=4, pair_cap=7)
    candidates, schemas = _case(seed, bsz=2, queries=4, cands=6, rels=5)
    new = TypedRelationPairGenerator(settings)
    new.select_budget = 1   # one relation per block
    a = ReferenceGenerator(settings).generate_batched(candidates, [], schemas, compact=compact)
    b = new.generate_batched(candidates, [], schemas, compact=compact)
    for f in FIELDS:
        assert torch.equal(getattr(a, f), getattr(b, f)), f
    assert a.relation_types == b.relation_types


def test_more_requested_than_candidates_pads_like_the_original():
    settings = RelationProposalSettings(heads_per_relation=40, tails_per_relation=40, pair_cap=50)
    candidates, schemas = _case(7, bsz=1, queries=3, cands=4, rels=3)   # 12 candidates < 40 requested
    a = ReferenceGenerator(settings).generate_batched(candidates, [], schemas, compact=False)
    b = TypedRelationPairGenerator(settings).generate_batched(candidates, [], schemas, compact=False)
    for f in FIELDS:
        assert torch.equal(getattr(a, f), getattr(b, f)), f
