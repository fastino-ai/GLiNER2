"""Candidate lattice construction for joint entity and relation decoding.

Spans use half-open ``[start, end)`` offsets.  Candidate generation deliberately
performs no overlap suppression: global constraints belong in the optimizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple


class CandidateSource(str, Enum):
    """How a node entered the joint lattice."""

    ENTITY = "entity"
    RELATION_RESCUE = "relation_rescue"
    PROVIDED = "provided"


def probability_to_logit(probability: float) -> float:
    """Return the log-odds, accepting exact zero and one."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    if probability == 0.0:
        return -math.inf
    if probability == 1.0:
        return math.inf
    return math.log(probability) - math.log1p(-probability)


def center_logit(logit: float, threshold: float = 0.5) -> float:
    """Center a raw logit so positive utility means passing ``threshold``."""
    return float(logit) - probability_to_logit(threshold)


def center_logits(logits: Any, threshold: float = 0.5) -> Any:
    """Center nested Python, NumPy, or torch logits without requiring either."""
    offset = probability_to_logit(threshold)
    try:
        return logits - offset
    except (TypeError, AttributeError):
        if isinstance(logits, (list, tuple)):
            values = [center_logits(value, threshold) for value in logits]
            return type(logits)(values)
        return float(logits) - offset


def _number(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)

def sigmoid(value: float) -> float:
    value = float(value)
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _shape(value: Any) -> Tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(v) for v in shape)
    result: List[int] = []
    current = value
    while isinstance(current, (list, tuple)):
        result.append(len(current))
        if not current:
            break
        current = current[0]
    return tuple(result)


@dataclass(frozen=True)
class NodeCandidate:
    """A typed entity-span candidate."""

    entity_type: str
    start: int
    end: int
    score: float
    probability: Optional[float] = None
    source: CandidateSource = CandidateSource.ENTITY
    candidate_id: Optional[Hashable] = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("node spans must be non-empty and non-negative")
        if self.probability is None:
            object.__setattr__(self, "probability", sigmoid(self.score))
        if not isinstance(self.source, CandidateSource):
            object.__setattr__(self, "source", CandidateSource(self.source))
        if self.candidate_id is None:
            object.__setattr__(self, "candidate_id", self.key)

    @property
    def key(self) -> Tuple[str, int, int]:
        return (self.entity_type, self.start, self.end)

    @property
    def utility(self) -> float:
        return self.score


@dataclass(frozen=True)
class EdgeCandidate:
    """A typed directed relation between two node candidates."""

    relation_type: str
    head: Hashable
    tail: Hashable
    score: float
    head_probability: Optional[float] = None
    tail_probability: Optional[float] = None
    head_entity_probability: Optional[float] = None
    tail_entity_probability: Optional[float] = None
    count_probability: Optional[float] = None
    derived: bool = False
    slot: Optional[Hashable] = None
    hypothesis: Optional[Hashable] = None
    count_alternative: Optional[Hashable] = None
    candidate_id: Optional[Hashable] = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.candidate_id is None:
            object.__setattr__(self, "candidate_id", self.key)

    @property
    def key(self) -> Tuple[Any, ...]:
        return (self.relation_type, self.head, self.tail, self.slot, self.count_alternative)

    @property
    def utility(self) -> float:
        return self.score

    @property
    def exclusion_keys(self) -> Tuple[Hashable, ...]:
        """Exact resources consumed by this edge.

        Slot use is scoped to a count alternative. Count-choice compatibility is
        checked separately, allowing several slots from the same alternative.
        """
        if self.slot is None:
            return ()
        return (("slot", self.hypothesis, self.count_alternative, self.slot),)

    @property
    def count_choice(self) -> Optional[Tuple[Hashable, Hashable]]:
        if self.count_alternative is None or self.hypothesis is None:
            return None
        return (self.hypothesis, self.count_alternative)


@dataclass(frozen=True)
class RelationHypothesis:
    """One relation lattice with shape ``[count_slots, 2, L, W]``."""

    relation_type: str
    role_logits: Any
    head_types: Sequence[str]
    tail_types: Sequence[str]
    threshold: Optional[float] = None
    candidate_threshold: Optional[float] = None
    count_probability: float = 1.0
    count_utility: float = 0.0
    count_alternative: Optional[Hashable] = None
    hypothesis_id: Optional[Hashable] = None


@dataclass(frozen=True)
class JointProblem:
    nodes: Tuple[NodeCandidate, ...]
    edges: Tuple[EdgeCandidate, ...]
    constraints: Tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        ids = [node.candidate_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node candidate ids must be unique")
        known = set(ids)
        for edge in self.edges:
            if edge.head not in known or edge.tail not in known:
                raise ValueError("edge endpoints must refer to nodes in the problem")

    @property
    def node_by_id(self) -> Dict[Hashable, NodeCandidate]:
        return {node.candidate_id: node for node in self.nodes}


class CandidateBuilder:
    """Build a bounded joint problem from entity and relation lattices.

    Caps are applied after deterministic duplicate collapse. Role candidates are
    not thresholded before pairing: the best ``relation_role_cap`` entries can
    rescue typed endpoints missing from the entity threshold lattice.
    """

    def __init__(
        self,
        *,
        candidate_threshold: float = 0.05,
        relation_role_threshold: float = 0.05,
        top_k_entities: int = 32,
        top_k_roles: int = 12,
        count_top_k: int = 2,
        entity_weight: float = 1.0,
        role_weight: float = 1.0,
        count_weight: float = 1.0,
        entity_threshold: Optional[float] = None,
        max_nodes_per_type: Optional[int] = None,
        relation_role_cap: Optional[int] = None,
        relation_pair_cap: int = 128,
        max_edges_per_type: int = 256,
        rescue_per_role: Optional[int] = None,
    ) -> None:
        max_nodes_per_type = top_k_entities if max_nodes_per_type is None else max_nodes_per_type
        relation_role_cap = top_k_roles if relation_role_cap is None else relation_role_cap
        entity_threshold = candidate_threshold if entity_threshold is None else entity_threshold
        for name, value in (("max_nodes_per_type", max_nodes_per_type),
                            ("relation_role_cap", relation_role_cap),
                            ("relation_pair_cap", relation_pair_cap),
                            ("max_edges_per_type", max_edges_per_type)):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.entity_threshold = entity_threshold
        self.relation_role_threshold = relation_role_threshold
        self.count_top_k = count_top_k
        self.entity_weight, self.role_weight, self.count_weight = entity_weight, role_weight, count_weight
        self.max_nodes_per_type = max_nodes_per_type
        self.relation_role_cap = relation_role_cap
        self.relation_pair_cap = relation_pair_cap
        self.max_edges_per_type = max_edges_per_type
        self.rescue_per_role = relation_role_cap if rescue_per_role is None else rescue_per_role

    @staticmethod
    def _span_entries(lattice: Any) -> List[Tuple[float, int, int]]:
        shape = _shape(lattice)
        if len(shape) != 2:
            raise ValueError(f"span lattice must have shape [L, W], got {shape}")
        length, widths = shape
        entries = []
        for start in range(length):
            for width in range(widths):
                end = start + width + 1
                if end <= length:
                    entries.append((_number(lattice[start][width]), start, end))
        return entries

    @staticmethod
    def _node_rank(node: NodeCandidate) -> Tuple[Any, ...]:
        source_rank = 0 if node.source == CandidateSource.ENTITY else 1
        return (-node.score, node.start, node.end, node.entity_type, source_rank)

    @staticmethod
    def _edge_rank(edge: EdgeCandidate) -> Tuple[Any, ...]:
        return (-edge.score, str(edge.hypothesis), str(edge.slot), str(edge.count_alternative),
                str(edge.head), str(edge.tail), edge.relation_type)

    def build(
        self,
        entity_logits: Any,
        entity_types: Sequence[str],
        relation_hypotheses: Sequence[RelationHypothesis | Mapping[str, Any]] = (),
        *,
        entity_thresholds: Optional[Mapping[str, Optional[float]]] = None,
        entity_candidate_thresholds: Optional[Mapping[str, Optional[float]]] = None,
        entity_max_candidates: Optional[Mapping[str, int]] = None,
        constraints: Iterable[Any] = (),
    ) -> JointProblem:
        shape = _shape(entity_logits)
        if len(shape) != 3 or shape[0] != len(entity_types):
            raise ValueError(
                f"entity logits must have shape [types, L, W], got {shape} "
                f"for {len(entity_types)} types"
            )
        thresholds = dict(entity_thresholds or {})
        candidate_thresholds = dict(entity_candidate_thresholds or {})
        maxima = dict(entity_max_candidates or {})
        node_map: Dict[Tuple[str, int, int], NodeCandidate] = {}
        raw_entity_scores: Dict[Tuple[str, int, int], float] = {}

        for type_index, entity_type in enumerate(entity_types):
            threshold = thresholds.get(entity_type) or 0.5
            candidate_threshold = candidate_thresholds.get(entity_type)
            if candidate_threshold is None:
                candidate_threshold = self.entity_threshold
            candidates: List[NodeCandidate] = []
            for raw, start, end in self._span_entries(entity_logits[type_index]):
                if not math.isfinite(raw):
                    continue
                score = center_logit(raw, threshold) * self.entity_weight
                probability = sigmoid(raw)
                raw_entity_scores[(entity_type, start, end)] = (score, probability)
                if probability >= candidate_threshold:
                    candidates.append(NodeCandidate(entity_type, start, end, score, probability))
            for node in sorted(candidates, key=self._node_rank)[:maxima.get(entity_type, self.max_nodes_per_type)]:
                node_map[node.key] = node

        edge_groups: Dict[str, List[EdgeCandidate]] = {}
        for index, value in enumerate(relation_hypotheses):
            hypothesis = value if isinstance(value, RelationHypothesis) else RelationHypothesis(**value)
            role_shape = _shape(hypothesis.role_logits)
            if len(role_shape) != 4 or role_shape[1] != 2:
                raise ValueError(
                    f"relation role logits must have shape [count_slots, 2, L, W], got {role_shape}"
                )
            hypothesis_id = hypothesis.hypothesis_id
            if hypothesis_id is None:
                hypothesis_id = (hypothesis.relation_type, index)
            final_threshold = hypothesis.threshold if hypothesis.threshold is not None else 0.5
            candidate_threshold = hypothesis.candidate_threshold
            if candidate_threshold is None:
                candidate_threshold = self.relation_role_threshold
            relation_offset = probability_to_logit(final_threshold)
            for count_slot in range(role_shape[0]):
                role_options: List[List[Tuple[float, int, int]]] = []
                for role in range(2):
                    entries = [(raw, start, end) for raw, start, end in self._span_entries(hypothesis.role_logits[count_slot][role])
                               if math.isfinite(raw) and sigmoid(raw) >= candidate_threshold]
                    entries.sort(key=lambda item: (-item[0], item[1], item[2]))
                    role_options.append(entries[:self.relation_role_cap])

                typed_roles: List[List[Tuple[float, NodeCandidate]]] = [[], []]
                for role, types in enumerate((hypothesis.head_types, hypothesis.tail_types)):
                    rescued = 0
                    for raw_role_score, start, end in role_options[role]:
                        for entity_type in types:
                            key = (entity_type, start, end)
                            node = node_map.get(key)
                            if node is None:
                                if rescued >= self.rescue_per_role:
                                    continue
                                node_score, node_probability = raw_entity_scores.get(key, (float("-inf"), 0.0))
                                node = NodeCandidate(
                                    entity_type, start, end, node_score, node_probability,
                                    source=CandidateSource.RELATION_RESCUE,
                                )
                                node_map[key] = node
                                rescued += 1
                            typed_roles[role].append(((raw_role_score - relation_offset) * self.role_weight, node, sigmoid(raw_role_score)))

                pairs: List[Tuple[float, NodeCandidate, NodeCandidate]] = []
                for (head_score, head, head_prob), (tail_score, tail, tail_prob) in product(*typed_roles):
                    pairs.append((head_score + tail_score + hypothesis.count_utility * self.count_weight, head, tail, head_prob, tail_prob))
                pairs.sort(key=lambda item: (-item[0], str(item[1].candidate_id), str(item[2].candidate_id)))
                for score, head, tail, head_prob, tail_prob in pairs[:self.relation_pair_cap]:
                    edge_groups.setdefault(hypothesis.relation_type, []).append(EdgeCandidate(
                        hypothesis.relation_type,
                        head.candidate_id,
                        tail.candidate_id,
                        score,
                        head_probability=head_prob, tail_probability=tail_prob,
                        head_entity_probability=head.probability, tail_entity_probability=tail.probability,
                        count_probability=hypothesis.count_probability,
                        slot=count_slot,
                        hypothesis=hypothesis_id,
                        count_alternative=hypothesis.count_alternative,
                    ))

        # Collapse duplicates before caps. Equal-score ties are resolved by rank.
        edges: List[EdgeCandidate] = []
        for relation_type in sorted(edge_groups):
            unique: Dict[Tuple[Any, ...], EdgeCandidate] = {}
            for edge in sorted(edge_groups[relation_type], key=self._edge_rank):
                previous = unique.get(edge.key)
                if previous is None or edge.score > previous.score:
                    unique[edge.key] = edge
            edges.extend(sorted(unique.values(), key=self._edge_rank)[:self.max_edges_per_type])

        # Relation rescue can exceed entity caps only up to a deterministic per-role
        # bound already imposed above; collapse all typed duplicate spans globally.
        nodes = tuple(sorted(node_map.values(), key=lambda n: (n.entity_type, n.start, n.end, -n.score)))
        return JointProblem(nodes, tuple(sorted(edges, key=self._edge_rank)), tuple(constraints))


class ProblemBuilder(CandidateBuilder):
    """Backward-compatible descriptive alias for :class:`CandidateBuilder`."""
