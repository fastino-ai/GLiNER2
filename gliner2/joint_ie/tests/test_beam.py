import pytest

from gliner2.joint_ie.candidates import EdgeCandidate, JointProblem, NodeCandidate
from gliner2.joint_ie.optimizers import BeamOptimizer, GreedyOptimizer


def node(name, score):
    return NodeCandidate(name, 0, 1, score, candidate_id=name)


def test_endpoint_nodes_are_selected_atomically_and_counted_once():
    a, b = node("a", 2.0), node("b", 3.0)
    edges = (
        EdgeCandidate("r", "a", "b", 4.0, slot=0, hypothesis="h"),
        EdgeCandidate("s", "a", "b", 5.0, slot=0, hypothesis="other"),
    )
    result = GreedyOptimizer().optimize(JointProblem((a, b), edges))
    assert set(result.node_ids) == {"a", "b"}
    assert len(result.edges) == 2
    assert result.score == pytest.approx(14.0)  # 2 + 3 + 4 + 5


def test_constraints_are_duck_typed_and_penalties_apply():
    class Constraint:
        def allow_node(self, candidate):
            return candidate.entity_type != "blocked"

        def allow_edge(self, candidate, nodes):
            return candidate.relation_type != "forbidden"

        def penalty_edge(self, candidate):
            return 2.5

    a, b, blocked = node("a", 1.0), node("b", 1.0), node("blocked", 9.0)
    edges = (
        EdgeCandidate("ok", "a", "b", 3.0),
        EdgeCandidate("forbidden", "a", "b", 100.0),
    )
    result = BeamOptimizer().optimize(JointProblem((a, b, blocked), edges, (Constraint(),)))
    assert {e.relation_type for e in result.edges} == {"ok"}
    assert "blocked" not in result.node_ids
    assert result.score == pytest.approx(2.5)


def test_beam_tracks_slots_and_mutually_exclusive_count_alternatives():
    a, b, c = node("a", 0.0), node("b", 0.0), node("c", 0.0)
    edges = (
        EdgeCandidate("r", "a", "b", 5.0, slot=0, hypothesis="count", count_alternative=1),
        EdgeCandidate("r", "a", "c", 4.0, slot=1, hypothesis="count", count_alternative=1),
        EdgeCandidate("r", "b", "c", 8.0, slot=0, hypothesis="count", count_alternative=2),
        EdgeCandidate("r", "c", "a", 8.0, slot=0, hypothesis="count", count_alternative=2),
    )
    result = BeamOptimizer(beam_width=16).optimize(JointProblem((a, b, c), edges))
    # Alternative 1 may consume both distinct slots (9); alternative 2 has one
    # colliding slot and can consume only one edge (8).
    assert result.score == pytest.approx(9.0)
    assert {e.count_alternative for e in result.edges} == {1}
    assert {e.slot for e in result.edges} == {0, 1}


def test_beam_is_never_worse_than_greedy_and_adds_positive_free_nodes():
    a, b, c, free = node("a", -2.0), node("b", 0.0), node("c", 0.0), node("free", 1.5)
    edges = (
        EdgeCandidate("r", "a", "b", 7.0, slot=0, hypothesis="h"),
        EdgeCandidate("r", "a", "c", 6.0, slot=0, hypothesis="h"),
    )
    problem = JointProblem((a, b, c, free), edges)
    greedy = GreedyOptimizer().optimize(problem)
    beam = BeamOptimizer(beam_width=1).optimize(problem)
    assert beam.score >= greedy.score
    assert "free" in beam.node_ids
