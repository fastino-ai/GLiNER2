"""Post-decoding constraints for joint entity and relation extraction.

The implementation deliberately uses structural (duck-typed) accessors so it
can operate on dictionaries, dataclasses, or decoder result objects without
importing the schema or compiler modules.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _label(value: Any) -> Any:
    return _get(value, "label", "type", "name", "entity_type", "relation", "relation_type")


def _endpoint(relation: Any, side: str) -> Any:
    return _get(relation, side, f"{side}_entity")


def _identity(value: Any) -> Any:
    """Return a stable entity identity, preferring offsets over surface text."""
    if value is None:
        return None
    identifier = _get(value, "id", "entity_id", "candidate_id", "index")
    if identifier is not None:
        return identifier
    start, end = _get(value, "start"), _get(value, "end")
    if start is not None and end is not None:
        return (start, end, _label(value))
    text = _get(value, "text", "value")
    if text is not None:
        return (text, _label(value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _relation_key(value: Any) -> tuple[Any, Any, Any]:
    return (_label(value), _identity(_endpoint(value, "head")), _identity(_endpoint(value, "tail")))


def _matches(relation: Any, relation_type: Optional[str]) -> bool:
    return relation_type is None or _label(relation) == relation_type


class Constraint(ABC):
    """Base API for incremental, decoder-independent constraints.

    ``allows`` checks a candidate against already accepted output. ``apply``
    performs deterministic greedy filtering. Constraints that describe a
    required companion edge override ``validate`` for whole-result checks.
    """

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return True

    def allow_node(self, candidate: Any, nodes: Sequence[Any] = (), edges: Sequence[Any] = ()) -> bool:
        return True

    def allow_edge(self, candidate: Any, nodes: Sequence[Any] = (), edges: Sequence[Any] = ()) -> bool:
        return self.allows(candidate, edges, nodes)

    def penalty_edge(self, candidate: Any, nodes: Sequence[Any] = (), edges: Sequence[Any] = ()) -> float:
        return 0.0

    def __call__(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return self.allows(candidate, relations, entities)

    def apply(self, relations: Iterable[Any], entities: Sequence[Any] = ()) -> list[Any]:
        accepted: list[Any] = []
        for candidate in relations:
            if self.allows(candidate, accepted, entities):
                accepted.append(candidate)
        return accepted

    def validate(self, relations: Sequence[Any], entities: Sequence[Any] = ()) -> bool:
        accepted: list[Any] = []
        for candidate in relations:
            if not self.allows(candidate, accepted, entities):
                return False
            accepted.append(candidate)
        return True

    def to_dict(self) -> dict[str, Any]:
        data = {"type": type(self).__name__}
        data.update(self.__dict__)
        return data


@dataclass(frozen=True)
class TypedEndpoints(Constraint):
    relation: Optional[str] = None
    head_types: tuple[str, ...] = ()
    tail_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_types", tuple(self.head_types))
        object.__setattr__(self, "tail_types", tuple(self.tail_types))

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        head, tail = _label(_endpoint(candidate, "head")), _label(_endpoint(candidate, "tail"))
        return (not self.head_types or head in self.head_types) and (not self.tail_types or tail in self.tail_types)


@dataclass(frozen=True)
class NoSelfLoops(Constraint):
    relation: Optional[str] = None

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return not _matches(candidate, self.relation) or _identity(_endpoint(candidate, "head")) != _identity(_endpoint(candidate, "tail"))


@dataclass(frozen=True)
class UniqueRelationPair(Constraint):
    relation: Optional[str] = None
    directed: bool = True

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        head, tail = _identity(_endpoint(candidate, "head")), _identity(_endpoint(candidate, "tail"))
        for existing in relations:
            if _label(existing) != _label(candidate):
                continue
            old_head, old_tail = _identity(_endpoint(existing, "head")), _identity(_endpoint(existing, "tail"))
            if (head, tail) == (old_head, old_tail) or (not self.directed and (head, tail) == (old_tail, old_head)):
                return False
        return True


@dataclass(frozen=True)
class UniqueRelationSlot(Constraint):
    relation: Optional[str] = None
    slot: str = "head"

    def __post_init__(self) -> None:
        if self.slot not in {"head", "tail", "slot"}:
            raise ValueError("slot must be 'head', 'tail', or 'slot'")

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        value = _identity(_endpoint(candidate, self.slot))
        return all(_label(old) != _label(candidate) or _identity(_endpoint(old, self.slot)) != value for old in relations)


@dataclass(frozen=True)
class EntityOverlapPolicy(Constraint):
    policy: str = "disallow"

    def __post_init__(self) -> None:
        if self.policy not in {"allow", "disallow", "nested"}:
            raise ValueError("policy must be 'allow', 'disallow', or 'nested'")

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return True

    def allow_node(self, candidate: Any, nodes: Sequence[Any] = (), edges: Sequence[Any] = ()) -> bool:
        if self.policy == "allow":
            return True
        start, end = _get(candidate, "start"), _get(candidate, "end")
        if start is None or end is None:
            return True
        for old in nodes:
            if old is candidate:
                continue
            old_start, old_end = _get(old, "start"), _get(old, "end")
            if old_start is None or old_end is None or end <= old_start or old_end <= start:
                continue
            nested = ((start >= old_start and end <= old_end) or
                      (old_start >= start and old_end <= end))
            if self.policy == "disallow" or not nested:
                return False
        return True

    def apply_entities(self, entities: Iterable[Any]) -> list[Any]:
        if self.policy == "allow":
            return list(entities)
        accepted: list[Any] = []
        for candidate in entities:
            start, end = _get(candidate, "start"), _get(candidate, "end")
            if start is None or end is None:
                accepted.append(candidate)
                continue
            valid = True
            for old in accepted:
                old_start, old_end = _get(old, "start"), _get(old, "end")
                if old_start is None or old_end is None or end <= old_start or old_end <= start:
                    continue
                nested = (start >= old_start and end <= old_end) or (old_start >= start and old_end <= end)
                if self.policy == "disallow" or not nested:
                    valid = False
                    break
            if valid:
                accepted.append(candidate)
        return accepted


@dataclass(frozen=True)
class MaxRelationsPerHead(Constraint):
    limit: int
    relation: Optional[str] = None

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("limit must be non-negative")

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        key = _identity(_endpoint(candidate, "head"))
        return sum(_matches(old, self.relation) and _identity(_endpoint(old, "head")) == key for old in relations) < self.limit


@dataclass(frozen=True)
class MaxRelationsPerTail(Constraint):
    limit: int
    relation: Optional[str] = None

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("limit must be non-negative")

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        key = _identity(_endpoint(candidate, "tail"))
        return sum(_matches(old, self.relation) and _identity(_endpoint(old, "tail")) == key for old in relations) < self.limit


@dataclass(frozen=True)
class SymmetricRelation(Constraint):
    relation: str

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return True

    def validate(self, relations: Sequence[Any], entities: Sequence[Any] = ()) -> bool:
        keys = {_relation_key(item) for item in relations}
        return all(_label(item) != self.relation or (self.relation, _identity(_endpoint(item, "tail")), _identity(_endpoint(item, "head"))) in keys for item in relations)


@dataclass(frozen=True)
class InverseRelation(Constraint):
    relation: str
    inverse: str

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        return True

    def validate(self, relations: Sequence[Any], entities: Sequence[Any] = ()) -> bool:
        keys = {_relation_key(item) for item in relations}
        for item in relations:
            label = _label(item)
            if label == self.relation and (self.inverse, _identity(_endpoint(item, "tail")), _identity(_endpoint(item, "head"))) not in keys:
                return False
            if label == self.inverse and (self.relation, _identity(_endpoint(item, "tail")), _identity(_endpoint(item, "head"))) not in keys:
                return False
        return True


@dataclass(frozen=True)
class AcyclicRelation(Constraint):
    relation: str

    def allows(self, candidate: Any, relations: Sequence[Any] = (), entities: Sequence[Any] = ()) -> bool:
        if not _matches(candidate, self.relation):
            return True
        head, tail = _identity(_endpoint(candidate, "head")), _identity(_endpoint(candidate, "tail"))
        if head == tail:
            return False
        graph: dict[Any, list[Any]] = defaultdict(list)
        for old in relations:
            if _matches(old, self.relation):
                graph[_identity(_endpoint(old, "head"))].append(_identity(_endpoint(old, "tail")))
        stack, seen = [tail], set()
        while stack:
            node = stack.pop()
            if node == head:
                return False
            if node not in seen:
                seen.add(node)
                stack.extend(graph[node])
        return True


_CONSTRAINT_TYPES = {cls.__name__: cls for cls in (
    TypedEndpoints, NoSelfLoops, UniqueRelationPair, UniqueRelationSlot,
    EntityOverlapPolicy, MaxRelationsPerHead, MaxRelationsPerTail,
    SymmetricRelation, InverseRelation, AcyclicRelation,
)}

def constraint_from_dict(data: Mapping[str, Any]) -> Constraint:
    values = dict(data)
    kind = values.pop("type", None)
    try:
        cls = _CONSTRAINT_TYPES[kind]
    except KeyError as exc:
        raise ValueError(f"unknown constraint type {kind!r}") from exc
    return cls(**values)
