"""Sparse typed relation extraction: typed/capped pairing + scorer overfit."""

from __future__ import annotations

import torch

from gliner2.models.base import QueryLayout, QuerySpec
from gliner2.models.outputs import CandidateTensorBatch
from gliner2.models.boundary.relations import (
    RelationProposalSettings,
    RelationTypeSpec,
    SparseRelationScorer,
    TypedRelationPairGenerator,
)


def _layout(role_names):
    return QueryLayout(
        queries=tuple(
            QuerySpec(query_id=i, task_index=0, task_type="entities",
                      task_name="entities", role_index=i, role_name=name,
                      field_path=(name,), extractive=True)
            for i, name in enumerate(role_names)
        )
    )


def _candidates(spans_per_query, length):
    """Build a CandidateTensorBatch (B=1) from ``spans_per_query`` = list per query."""
    q = len(spans_per_query)
    c = max((len(s) for s in spans_per_query), default=1)
    indices = torch.zeros(1, q, c, 2, dtype=torch.long)
    valid = torch.zeros(1, q, c, dtype=torch.bool)
    pair_logits = torch.full((1, q, c), -10.0)
    for qi, spans in enumerate(spans_per_query):
        for ci, (s, e) in enumerate(spans):
            indices[0, qi, ci, 0] = s
            indices[0, qi, ci, 1] = e
            valid[0, qi, ci] = True
            pair_logits[0, qi, ci] = 5.0
    return CandidateTensorBatch(
        indices=indices,
        proposal_logits=torch.zeros(1, q, c),
        pair_logits=pair_logits,
        valid_mask=valid,
        query_mask=torch.ones(1, q, dtype=torch.bool),
    )


def test_typed_pair_generation_respects_endpoint_types():
    layout = _layout(["person", "org"])
    cands = _candidates([[(0, 1), (4, 5)], [(2, 3), (5, 6)]], length=7)
    spec = RelationTypeSpec("works_for", head_query_ids=(0,), tail_query_ids=(1,))
    pairs = TypedRelationPairGenerator().generate(cands, [layout], [spec])

    assert len(pairs) == 4  # 2 heads x 2 tails
    assert all(k[0] == "person" for k in pairs.head_keys)
    assert all(k[0] == "org" for k in pairs.tail_keys)
    assert all(rt == "works_for" for rt in pairs.relation_types)


def test_pairing_is_capped_and_independent_of_sequence_length():
    layout = _layout(["person", "org"])
    # Same small candidate counts regardless of a long sequence -> no N^2 blowup.
    cands = _candidates([[(0, 1), (4, 5), (8, 9)], [(2, 3), (6, 7)]], length=500)
    spec = RelationTypeSpec("works_for", (0,), (1,))
    settings = RelationProposalSettings(heads_per_relation=2, tails_per_relation=2, pair_cap=3)
    pairs = TypedRelationPairGenerator(settings).generate(cands, [layout], [spec])
    # top-2 heads x top-2 tails = 4, capped to pair_cap=3.
    assert len(pairs) == 3


def test_no_self_pairs_by_default():
    layout = _layout(["thing", "thing2"])
    cands = _candidates([[(0, 1)], [(0, 1)]], length=5)
    spec = RelationTypeSpec("same", (0,), (1,))
    pairs = TypedRelationPairGenerator().generate(cands, [layout], [spec])
    assert len(pairs) == 0  # identical span filtered as a self-pair


def test_sparse_relation_scorer_overfits_gold_pairs():
    torch.manual_seed(0)
    hidden = 16
    length = 6
    boundary_states = torch.randn(1, length, hidden)
    rel_query = torch.randn(1, 1, hidden)

    layout = _layout(["person", "org"])
    cands = _candidates([[(0, 1), (4, 5)], [(2, 3), (5, 6)]], length=length)
    spec = RelationTypeSpec("works_for", (0,), (1,))
    pairs = TypedRelationPairGenerator().generate(cands, [layout], [spec])

    # Gold: (person 0,1) -> (org 2,3)
    gold = torch.tensor([
        1.0 if (int(hs), int(he), int(ts), int(te)) == (0, 1, 2, 3) else 0.0
        for hs, he, ts, te in zip(pairs.head_start, pairs.head_end, pairs.tail_start, pairs.tail_end)
    ])

    scorer = SparseRelationScorer(hidden)
    opt = torch.optim.Adam(scorer.parameters(), lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        logits = scorer(boundary_states, rel_query, cands, pairs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, gold)
        loss.backward()
        opt.step()

    probs = torch.sigmoid(scorer(boundary_states, rel_query, cands, pairs).detach())
    pred = (probs > 0.5).float()
    assert torch.equal(pred, gold), (probs, gold)


def test_boundary_model_wires_sparse_relation_loss():
    from gliner2 import ExtractorConfig
    from gliner2.inference.engine import BoundaryExtractor
    from gliner2.processor import SamplingConfig, SchemaTransformer
    from gliner2.training import ExtractorCollator
    from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD
    from tests.fixtures.tiny_encoder import build_tiny_encoder_config
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    tokenizer = build_tiny_tokenizer()
    head = dict(TINY_BOUNDARY_HEAD)
    head.update(enable_relations=True)
    model = BoundaryExtractor(
        ExtractorConfig(
            model_name="tiny-bert-fixture",
            architecture="boundary",
            boundary_head=head,
            token_pooling="first",
        ),
        encoder_config=build_tiny_encoder_config(vocab_size=len(tokenizer)),
        tokenizer=tokenizer,
    )
    processor = SchemaTransformer(
        tokenizer=tokenizer,
        sampling_config=SamplingConfig(
            remove_relations_prob=0.0,
            swap_head_tail_prob=0.0,
            synthetic_entity_label_prob=0.0,
        ),
    )
    collator = ExtractorCollator(processor, is_training=True, architecture="boundary")
    batch = collator([(
        "Alice works for Acme",
        {"relations": [{"works_for": {"head": "Alice", "tail": "Acme"}}]},
    )])
    assert isinstance(batch.targets.edge_targets, tuple)
    assert len(batch.targets.edge_targets) == 6
    model.train()
    output = model(batch)
    assert "relation_scorer" in model.task_module_names()
    assert output.losses["relation_loss"].requires_grad

    # Exercise public-boundary relation decoding with non-self synthetic
    # endpoint proposals; an untrained proposer is not expected to supply them.
    with torch.no_grad():
        model.relation_scorer.mlp[-1].bias.fill_(10.0)
    candidates = CandidateTensorBatch(
        indices=torch.tensor([[[[0, 1]], [[3, 4]]]]),
        proposal_logits=torch.zeros(1, 2, 1),
        pair_logits=torch.tensor([[[5.0], [2.0]]]),
        valid_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        query_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    decode_args = (
        0,
        {
            "rel_specs": [[{
                "spec": RelationTypeSpec(
                    "works_for: a person is employed by an organization",
                    (0,),
                    (1,),
                ),
                "query_state": torch.zeros(model.hidden_size),
            }]],
            "text_states": torch.zeros(1, 4, model.hidden_size),
        },
        candidates,
        {
            "relation_metadata": {},
            "relation_descriptions": {
                "works_for": "a person is employed by an organization",
            },
        },
    )
    decode_kwargs = dict(
        threshold=0.5,
        offset=0,
        start_map=[0, 6, 12, 16],
        end_map=[5, 11, 15, 20],
        text="Alice works for Acme",
        text_len=4,
        include_spans=True,
    )
    decoded = model._decode_relations(
        *decode_args, include_confidence=False, **decode_kwargs
    )
    assert decoded["works_for"] == [{
        "head": {"text": "Alice", "start": 0, "end": 5},
        "tail": {"text": "Acme", "start": 16, "end": 20},
    }]

    scored = model._decode_relations(
        *decode_args, include_confidence=True, **decode_kwargs
    )["works_for"][0]
    head_prob, tail_prob = torch.sigmoid(candidates.pair_logits).flatten().tolist()
    assert scored["head"]["confidence"] == head_prob
    assert scored["tail"]["confidence"] == tail_prob
    assert 0.0 <= scored["confidence"] <= 1.0


def test_relation_decoder_collapses_partial_and_repeated_mentions():
    from gliner2.inference.engine import BoundaryExtractor

    edges = [
        {
            "score": 0.99,
            "head": ("Anika Rao", 0, 9, 0.97),
            "tail": ("Helio", 20, 25, 0.62),
        },
        {
            "score": 0.98,
            "head": ("Anika Rao", 0, 9, 0.97),
            "tail": ("Helio Robotics", 20, 34, 0.88),
        },
        {
            "score": 0.995,
            "head": ("Anika Rao", 0, 9, 0.97),
            "tail": ("Helio Robotics", 80, 94, 0.71),
        },
    ]

    decoded = BoundaryExtractor._deduplicate_relation_edges(edges)

    assert len(decoded) == 1
    assert decoded[0]["head"] == ("Anika Rao", 0, 9, 0.97)
    # The containing mention is canonical, and the closest occurrence wins
    # over a marginally higher-scoring distant cross-product.
    assert decoded[0]["tail"] == ("Helio Robotics", 20, 34, 0.88)
    assert decoded[0]["score"] == 0.99


def test_relation_description_survives_schema_roundtrip():
    from gliner2.inference.engine import BoundaryExtractor
    from gliner2.inference.schema import Schema

    schema = Schema()
    schema.relations({
        "acquired": {
            "description": "completed purchase of a company",
            "threshold": 0.75,
        },
    })

    built = schema.build()
    assert built["relation_descriptions"] == {
        "acquired": "completed purchase of a company",
    }
    serialized = schema.to_dict()
    assert serialized["relations"]["acquired"] == {
        "description": "completed purchase of a company",
        "threshold": 0.75,
    }
    assert Schema.from_dict(serialized).build()["relation_descriptions"] == (
        built["relation_descriptions"]
    )
    _, metadata = BoundaryExtractor._build_schema_dicts_and_metadata(
        object(),
        [schema],
    )
    assert metadata[0]["relation_descriptions"] == built[
        "relation_descriptions"
    ]
