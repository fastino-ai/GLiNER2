"""Deterministic greedy joint optimizer."""

from __future__ import annotations

from typing import Hashable, List, Set

from .base import BaseOptimizer, JointSolution
from ..candidates import EdgeCandidate, JointProblem


class GreedyOptimizer(BaseOptimizer):
    """Select profitable edges atomically, then independent positive nodes."""

    @staticmethod
    def _rank(edge: EdgeCandidate):
        return (-edge.score, edge.relation_type, str(edge.hypothesis), str(edge.slot),
                str(edge.head), str(edge.tail))

    def optimize(self, problem: JointProblem) -> JointSolution:
        node_by_id = problem.node_by_id
        selected: Set[Hashable] = set()
        edges: List[EdgeCandidate] = []
        used = set()
        score = 0.0

        # Re-rank by true initial atomic utility so a strong endpoint can make a
        # relation preferable without ever double-counting that endpoint later.
        ranked = sorted(problem.edges, key=lambda edge: (
            -(edge.score + node_by_id[edge.head].score +
              (0.0 if edge.tail == edge.head else node_by_id[edge.tail].score)),
            self._rank(edge),
        ))
        for edge in ranked:
            if self.edge_conflicts(edge, used):
                continue
            new_ids = [value for value in (edge.head, edge.tail) if value not in selected]
            proposed_nodes = [node_by_id[value] for value in new_ids]
            current_nodes = [node_by_id[value] for value in selected]
            if not all(self.allow_node(problem, node, current_nodes + proposed_nodes, edges)
                       for node in proposed_nodes):
                continue
            if not self.allow_edge(problem, edge, current_nodes + proposed_nodes, edges):
                continue
            gain = edge.score + sum(node.score for node in proposed_nodes)
            gain -= self.edge_penalty(problem, edge, current_nodes + proposed_nodes, edges)
            if gain < 0.0:
                continue
            selected.update(new_ids)
            edges.append(edge)
            used.update(self.edge_usage(edge))
            score += gain

        # Nodes need not participate in a relation.
        for node in sorted(problem.nodes, key=lambda n: (-n.score, n.entity_type, n.start, n.end)):
            if node.candidate_id in selected or node.score <= 0.0:
                continue
            current_nodes = [node_by_id[value] for value in selected]
            if self.allow_node(problem, node, current_nodes + [node], edges):
                selected.add(node.candidate_id)
                score += node.score
        return self.solution(problem, selected, edges, score)
