"""Stable Joint IE result objects and optimizer-solution conversion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .lattice import SpanRef, sigmoid


@dataclass(frozen=True)
class JointEntity:
    id: str
    type: str
    text: str
    start: int
    end: int
    confidence: Optional[float] = None
    sentence_id: Optional[int] = None
    rescued: bool = False

    @property
    def label(self) -> str:
        return self.type

    @property
    def span(self) -> Tuple[int, int]:
        return (self.start, self.end)

    def to_dict(self, include_confidence: bool = True,
                include_spans: bool = True) -> Dict[str, Any]:
        value: Dict[str, Any] = {"id": self.id, "type": self.type, "text": self.text}
        if include_spans:
            value.update(start=self.start, end=self.end)
            if self.sentence_id is not None:
                value["sentence_id"] = self.sentence_id
        if include_confidence and self.confidence is not None:
            value["confidence"] = self.confidence
        if self.rescued:
            value["rescued"] = True
        return value


@dataclass(frozen=True)
class JointRelation:
    type: str
    head: str
    tail: str
    confidence: Optional[float] = None
    derived: bool = False

    @property
    def label(self) -> str:
        return self.type

    def to_dict(self, include_confidence: bool = True) -> Dict[str, Any]:
        value: Dict[str, Any] = {"type": self.type, "head": self.head, "tail": self.tail}
        if include_confidence and self.confidence is not None:
            value["confidence"] = self.confidence
        if self.derived:
            value["derived"] = True
        return value


@dataclass
class JointResult:
    text: str
    entities: List[JointEntity] = field(default_factory=list)
    relations: List[JointRelation] = field(default_factory=list)
    default_include_confidence: bool = field(default=True, repr=False, compare=False)
    default_include_spans: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.entities = list(self.entities)
        self.relations = list(self.relations)
        ids = [entity.id for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("entity IDs must be unique")
        known = set(ids)
        if any(rel.head not in known or rel.tail not in known for rel in self.relations):
            raise ValueError("relation endpoints must reference entities in this result")

    def entity(self, entity_id: str) -> JointEntity:
        for value in self.entities:
            if value.id == entity_id:
                return value
        raise KeyError(entity_id)

    get_entity = entity

    def entities_by_type(self, entity_type: str) -> List[JointEntity]:
        return [entity for entity in self.entities if entity.type == entity_type]

    def relations_by_type(self, relation_type: str) -> List[JointRelation]:
        return [relation for relation in self.relations if relation.type == relation_type]

    def outgoing(self, entity: Any, relation_type: Optional[str] = None) -> List[JointRelation]:
        entity_id = entity.id if isinstance(entity, JointEntity) else str(entity)
        return [relation for relation in self.relations
                if relation.head == entity_id and
                (relation_type is None or relation.type == relation_type)]

    def incoming(self, entity: Any, relation_type: Optional[str] = None) -> List[JointRelation]:
        entity_id = entity.id if isinstance(entity, JointEntity) else str(entity)
        return [relation for relation in self.relations
                if relation.tail == entity_id and
                (relation_type is None or relation.type == relation_type)]

    def neighbors(self, entity: Any, relation_type: Optional[str] = None) -> List[JointEntity]:
        entity_id = entity.id if isinstance(entity, JointEntity) else str(entity)
        ids: List[str] = []
        for relation in self.relations:
            if relation_type is not None and relation.type != relation_type:
                continue
            if relation.head == entity_id:
                ids.append(relation.tail)
            elif relation.tail == entity_id:
                ids.append(relation.head)
        seen = set()
        return [self.entity(value) for value in ids if not (value in seen or seen.add(value))]

    def relations_of(self, entity: Any, relation_type: Optional[str] = None) -> List[JointRelation]:
        entity_id = entity.id if isinstance(entity, JointEntity) else str(entity)
        return [relation for relation in self.relations if
                (relation.head == entity_id or relation.tail == entity_id) and
                (relation_type is None or relation.type == relation_type)]

    def to_dict(self, include_confidence: Optional[bool] = None,
                include_spans: Optional[bool] = None, include_text: bool = False) -> Dict[str, Any]:
        if include_confidence is None:
            include_confidence = self.default_include_confidence
        if include_spans is None:
            include_spans = self.default_include_spans
        value = {
            "entities": [entity.to_dict(include_confidence, include_spans)
                         for entity in self.entities],
            "relations": [relation.to_dict(include_confidence)
                          for relation in self.relations],
        }
        if include_text:
            value = {"text": self.text, **value}
        return value

    def to_networkx(self) -> Any:
        """Return a directed multigraph; networkx remains an optional dependency."""
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("to_networkx() requires the optional 'networkx' package") from exc
        graph = nx.MultiDiGraph()
        for entity in self.entities:
            attrs = entity.to_dict()
            attrs.pop("id")
            graph.add_node(entity.id, **attrs)
        for relation in self.relations:
            attrs = relation.to_dict()
            attrs.pop("head")
            attrs.pop("tail")
            graph.add_edge(relation.head, relation.tail, **attrs)
        return graph


class ResultBuilder:
    """Build a stable result from a duck-typed optimizer solution and problem.

    Supported selections are records in ``solution.entities``/``relations`` or
    indices in ``selected_entities``/``selected_relations`` into corresponding
    problem candidate collections. Records may be mappings or objects.
    """

    def __init__(self, include_confidence: bool = True,
                 include_spans: bool = True, include_count: bool = True,
                 config: Any = None):
        self.include_confidence = bool(_get(config, "include_confidence", default=include_confidence))
        self.include_spans = bool(_get(config, "include_spans", default=include_spans))
        self.include_count = include_count

    def build(self, solution: Any, problem: Any = None, text: Optional[str] = None,
              include_confidence: Optional[bool] = None,
              include_spans: Optional[bool] = None, candidates: Any = None,
              candidate_set: Any = None, lattice: Any = None, **_: Any) -> JointResult:
        confidence_flag = self.include_confidence if include_confidence is None else include_confidence
        spans_flag = self.include_spans if include_spans is None else include_spans
        problem = problem or candidates or candidate_set
        if problem is None:
            raise TypeError("ResultBuilder requires a candidate problem")
        document = text if text is not None else str(_get(lattice, "text", default=_get(problem, "text", default="")))
        entity_records = self._selected(solution, problem, "entities")
        normalized = [self._entity_parts(item, problem, document, lattice) for item in entity_records]
        normalized.sort(key=lambda item: (item[1].char_start, item[1].char_end, item[0], item[1].text))

        entities: List[JointEntity] = []
        record_to_id: Dict[Any, str] = {}
        span_label_to_id: Dict[Tuple[str, int, int], str] = {}
        for index, (label, span, scores, rescued, source) in enumerate(normalized):
            entity_id = f"e{index + 1}"
            confidence = _geometric_mean(scores) if confidence_flag else None
            entity = JointEntity(entity_id, label, span.text, span.char_start, span.char_end, confidence,
                                 span.sentence_id, rescued)
            entities.append(entity)
            record_to_id[_identity(source)] = entity_id
            candidate_id = _get(source, "candidate_id", "id", default=None)
            if candidate_id is not None:
                record_to_id[_identity(candidate_id)] = entity_id
            span_label_to_id[(label, span.char_start, span.char_end)] = entity_id

        relation_records = self._selected(solution, problem, "relations")
        relations: List[JointRelation] = []
        for item in relation_records:
            label = str(_get(item, "type", "label", "relation", "relation_type"))
            head = self._endpoint_id(_get(item, "head", "source", "head_entity"), record_to_id,
                                     span_label_to_id, normalized)
            tail = self._endpoint_id(_get(item, "tail", "target", "tail_entity"), record_to_id,
                                     span_label_to_id, normalized)
            scores = _scores(item, self.include_count)
            derived = bool(_get(item, "derived", "is_derived", default=False))
            relations.append(JointRelation(label, head, tail,
                _geometric_mean(scores) if confidence_flag else None, derived))
        relations.sort(key=lambda rel: (rel.type, rel.head, rel.tail))
        return JointResult(document, entities, relations, confidence_flag, spans_flag)

    __call__ = build

    @staticmethod
    def _selected(solution: Any, problem: Any, kind: str) -> List[Any]:
        aliases = (kind, "nodes") if kind == "entities" else (kind, "edges")
        direct = _get(solution, *aliases, default=None)
        if direct is not None:
            return list(direct)
        selected = _get(solution, f"selected_{kind}", default=[])
        problem_aliases = (f"{kind}_candidates", f"candidate_{kind}", kind,
                           "nodes" if kind == "entities" else "edges")
        candidates = list(_get(problem, *problem_aliases, default=[]))
        result = []
        for value in selected:
            if isinstance(value, int):
                result.append(candidates[value])
            elif isinstance(value, bool):
                continue
            else:
                result.append(value)
        return result

    @staticmethod
    def _entity_parts(item: Any, problem: Any, text: str,
                      lattice: Any = None) -> Tuple[str, SpanRef, List[float], bool, Any]:
        label = str(_get(item, "type", "label", "entity_type"))
        span = _get(item, "span", "span_ref", default=None)
        if isinstance(span, int):
            span = list(_get(problem, "spans", "span_candidates"))[span]
        if span is None:
            span = item
        if not isinstance(span, SpanRef):
            token_start = int(_get(span, "token_start", "start_token", "start", default=0))
            token_end = int(_get(span, "token_end", "end_token", "end", default=token_start))
            starts = _get(lattice, "start_mappings", default=None)
            ends = _get(lattice, "end_mappings", default=None)
            char_start_value = _get(span, "char_start", default=None)
            char_end_value = _get(span, "char_end", default=None)
            char_start = int(starts[token_start] if char_start_value is None and starts is not None else
                             token_start if char_start_value is None else char_start_value)
            last_token = max(token_start, token_end - 1)
            char_end = int(ends[last_token] if char_end_value is None and ends is not None else
                           token_end if char_end_value is None else char_end_value)
            value = _get(span, "text", default=text[char_start:char_end])
            sentence_id = _get(span, "sentence_id", "sentence", default=None)
            span = SpanRef(token_start, token_end, char_start, char_end, str(value), sentence_id)
        source = _get(item, "source", default="")
        rescued = bool(_get(item, "rescued", "is_rescued", default=False)) or "rescue" in str(source).lower()
        return label, span, _scores(item, True), rescued, item

    @staticmethod
    def _endpoint_id(endpoint: Any, records: Mapping[Any, str],
                     spans: Mapping[Tuple[str, int, int], str],
                     normalized: Sequence[Tuple[Any, ...]]) -> str:
        if isinstance(endpoint, str) and endpoint.startswith("e"):
            return endpoint
        identity = _identity(endpoint)
        if identity in records:
            return records[identity]
        if isinstance(endpoint, int) and 0 <= endpoint < len(normalized):
            return records[_identity(normalized[endpoint][4])]
        label = str(_get(endpoint, "type", "label", "entity_type", default=""))
        span = _get(endpoint, "span", "span_ref", default=endpoint)
        start = int(_get(span, "char_start", "start", default=-1))
        end = int(_get(span, "char_end", "end", default=-1))
        try:
            return spans[(label, start, end)]
        except KeyError as exc:
            raise ValueError("relation endpoint does not identify a selected entity") from exc


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _identity(value: Any) -> Any:
    try:
        hash(value)
        return ("value", value)
    except TypeError:
        return ("object", id(value))


def _scores(value: Any, include_count: bool) -> List[float]:
    scores: List[float] = []
    explicit = _get(value, "probabilities", "confidences", "scores", default=None)
    if explicit is not None and not isinstance(explicit, (str, bytes, Mapping)):
        scores.extend(float(item) for item in explicit)
    else:
        for name in ("entity_confidence", "head_confidence", "tail_confidence", "confidence",
                     "head_probability", "tail_probability", "head_entity_probability",
                     "tail_entity_probability", "probability"):
            score = _get(value, name, default=None)
            if score is not None:
                scores.append(float(score))
    if include_count:
        count = _get(value, "count_confidence", "count_probability", default=None)
        if count is not None:
            scores.append(float(count))
        else:
            count_logit = _get(value, "count_logit", default=None)
            if count_logit is not None:
                scores.append(sigmoid(count_logit))
    return scores


def _geometric_mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("confidence components must be probabilities in [0, 1]")
    if any(value == 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))
