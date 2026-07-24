import math

import pytest

from gliner2.joint_ie.candidates import (
    CandidateBuilder,
    CandidateSource,
    RelationHypothesis,
    center_logit,
    center_logits,
)


def test_centered_logits_use_probability_threshold():
    assert center_logit(0.0, 0.5) == 0.0
    assert center_logit(math.log(3.0), 0.75) == pytest.approx(0.0)
    assert center_logits([[0.0, 1.0]], 0.5) == [[0.0, 1.0]]


def test_build_keeps_overlaps_and_typed_duplicate_spans():
    # Shape [types=2, L=2, W=2]. Invalid end positions are ignored.
    logits = [
        [[2.0, 1.5], [-5.0, -5.0]],
        [[1.0, -5.0], [-5.0, -5.0]],
    ]
    problem = CandidateBuilder().build(logits, ["person", "org"])
    keys = {node.key for node in problem.nodes}
    assert ("person", 0, 1) in keys
    assert ("person", 0, 2) in keys  # no early overlap NMS
    assert ("org", 0, 1) in keys     # same span remains typed


def test_relation_rescues_typed_endpoints_and_builds_capped_pairs():
    entities = [
        [[-4.0], [-4.0], [-4.0]],  # person
        [[-4.0], [-4.0], [-4.0]],  # org
    ]
    roles = [[
        [[3.0], [1.0], [-3.0]],  # head
        [[-3.0], [1.0], [4.0]],  # tail
    ]]
    hypothesis = RelationHypothesis(
        "works_for", roles, head_types=["person"], tail_types=["org"]
    )
    problem = CandidateBuilder(relation_role_cap=2, relation_pair_cap=2).build(
        entities, ["person", "org"], [hypothesis]
    )
    assert len(problem.edges) == 2
    assert all(problem.node_by_id[e.head].entity_type == "person" for e in problem.edges)
    assert all(problem.node_by_id[e.tail].entity_type == "org" for e in problem.edges)
    assert all(node.source == CandidateSource.RELATION_RESCUE for node in problem.nodes)


def test_duplicate_edges_collapse_deterministically_before_cap():
    entities = [[[2.0], [2.0]]]
    roles = [[[[3.0], [-2.0]], [[-2.0], [3.0]]]]
    hypotheses = [
        RelationHypothesis("next", roles, ["item"], ["item"], hypothesis_id="h"),
        RelationHypothesis("next", roles, ["item"], ["item"], hypothesis_id="h"),
    ]
    problem = CandidateBuilder(relation_role_cap=1, relation_pair_cap=4,
                               max_edges_per_type=1).build(
        entities, ["item"], hypotheses
    )
    assert len(problem.edges) == 1
    assert problem.edges[0].head != problem.edges[0].tail


def test_rejects_wrong_lattice_shapes():
    with pytest.raises(ValueError, match=r"\[types, L, W\]"):
        CandidateBuilder().build([[1.0]], ["thing"])


def test_candidate_and_decision_thresholds_are_separate_and_inf_is_skipped():
    logits = [[[math.log(0.2 / 0.8)], [float("-inf")]]]
    problem = CandidateBuilder(candidate_threshold=.05).build(
        logits, ["person"], entity_thresholds={"person": .5},
        entity_candidate_thresholds={"person": .1})
    assert len(problem.nodes) == 1
    assert problem.nodes[0].probability == pytest.approx(.2)
    assert problem.nodes[0].utility < 0


def test_relation_candidate_threshold_is_separate_from_utility_threshold():
    entities = [[[0.0], [0.0]]]
    role_logit = math.log(.2 / .8)
    roles = [[[[role_logit], [float("-inf")]], [[float("-inf")], [role_logit]]]]
    hypothesis = RelationHypothesis("next", roles, ["item"], ["item"],
                                    threshold=.5, candidate_threshold=.1)
    problem = CandidateBuilder().build(entities, ["item"], [hypothesis],
        entity_candidate_thresholds={"item": .9})
    assert len(problem.edges) == 1
    assert problem.edges[0].score < 0
    assert all(math.isfinite(node.score) for node in problem.nodes)
