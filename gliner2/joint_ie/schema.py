"""Declarative schema for joint entity and relation extraction."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional
from .constraints import AcyclicRelation, Constraint, MaxRelationsPerHead, MaxRelationsPerTail, NoSelfLoops

def _name(value: str, kind: str) -> str:
    if not isinstance(value, str) or not value.strip(): raise ValueError(f"{kind} name must be a non-empty string")
    return value

def _types(value: str | Iterable[str], side: str) -> tuple[str, ...]:
    values=(value,) if isinstance(value,str) else tuple(value)
    if not values: raise ValueError(f"relation {side} must contain at least one entity type")
    if any(not isinstance(x,str) or not x.strip() for x in values): raise ValueError(f"relation {side} contains an invalid entity type")
    if len(set(values)) != len(values): raise ValueError(f"relation {side} entity types must be unique")
    return values

def _prob(value: Optional[float], field: str) -> None:
    if value is not None and not 0 <= value <= 1: raise ValueError(f"{field} must be in [0, 1]")

@dataclass(frozen=True)
class EntitySpec:
    name: str
    description: Optional[str]=None
    threshold: Optional[float]=None
    candidate_threshold: Optional[float]=None
    max_candidates: Optional[int]=None
    allow_nested: Optional[bool]=None
    def __post_init__(self):
        _name(self.name,"entity"); _prob(self.threshold,"threshold"); _prob(self.candidate_threshold,"candidate_threshold")
        if self.max_candidates is not None and self.max_candidates <= 0: raise ValueError("max_candidates must be positive")

@dataclass(frozen=True)
class RelationSpec:
    name: str
    head: tuple[str,...]
    tail: tuple[str,...]
    description: Optional[str]=None
    threshold: Optional[float]=None
    candidate_threshold: Optional[float]=None
    directed: bool=True
    symmetric: bool=False
    inverse: Optional[str]=None
    allow_self: bool=False
    max_per_head: Optional[int]=None
    max_per_tail: Optional[int]=None
    def __post_init__(self):
        _name(self.name,"relation"); object.__setattr__(self,"head",_types(self.head,"head")); object.__setattr__(self,"tail",_types(self.tail,"tail"))
        _prob(self.threshold,"threshold"); _prob(self.candidate_threshold,"candidate_threshold")
        if self.inverse is not None: _name(self.inverse,"inverse relation")
        if self.symmetric and self.inverse: raise ValueError("a relation cannot be both symmetric and inverse")
        if self.symmetric and set(self.head) != set(self.tail):
            raise ValueError("symmetric relations require compatible head and tail types")
        if self.symmetric and self.directed: object.__setattr__(self,"directed",False)
        for n in ("max_per_head","max_per_tail"):
            v=getattr(self,n)
            if v is not None and v < 0: raise ValueError(f"{n} must be non-negative")

class JointSchema:
    def __init__(self): self._entities={}; self._relations={}; self._constraints=[]
    @property
    def entity_specs(self): return tuple(self._entities.values())
    @property
    def relation_specs(self): return tuple(self._relations.values())
    @property
    def constraints(self): return tuple(self._constraints)
    def entity(self,name,description=None,*,threshold=None,candidate_threshold=None,max_candidates=None,allow_nested=None):
        if name in self._entities: raise ValueError(f"entity {name!r} is already defined")
        self._entities[name]=EntitySpec(name,description,threshold,candidate_threshold,max_candidates,allow_nested); return self
    def entities(self, entities):
        if isinstance(entities,str): return self.entity(entities)
        items=entities.items() if isinstance(entities,Mapping) else ((x,None) for x in entities)
        for name,value in items:
            if isinstance(value,Mapping): self.entity(name,**dict(value))
            else: self.entity(name,value)
        return self
    def relation(self,name,head,tail,description=None,*,threshold=None,candidate_threshold=None,directed=True,symmetric=False,inverse=None,allow_self=False,max_per_head=None,max_per_tail=None,**aliases):
        if name in self._relations: raise ValueError(f"relation {name!r} is already defined")
        inverse_of=aliases.pop("inverse_of",None); allow_self_loops=aliases.pop("allow_self_loops",None); no_self=aliases.pop("no_self_loops",None)
        unique_head=aliases.pop("unique_head",False); unique_tail=aliases.pop("unique_tail",False); aliases.pop("unique_pair",None); acyclic=aliases.pop("acyclic",False)
        if aliases: raise TypeError(f"unknown relation options: {sorted(aliases)}")
        if inverse is not None and inverse_of is not None and inverse != inverse_of: raise ValueError("inverse and inverse_of disagree")
        inverse=inverse or inverse_of
        if allow_self_loops is not None: allow_self=allow_self_loops
        if no_self is not None: allow_self=not no_self
        if unique_head and max_per_head is None: max_per_head=1
        if unique_tail and max_per_tail is None: max_per_tail=1
        spec=RelationSpec(name,_types(head,"head"),_types(tail,"tail"),description,threshold,candidate_threshold,directed,symmetric,inverse,allow_self,max_per_head,max_per_tail)
        unknown=(set(spec.head)|set(spec.tail))-set(self._entities)
        if unknown: raise ValueError(f"relation {name!r} references unknown entity types: {sorted(unknown)}")
        self._relations[name]=spec
        if acyclic: self.acyclic(name)
        return self
    def constraint(self,constraint):
        if not isinstance(constraint,Constraint): raise TypeError("constraint must implement Constraint")
        self._constraints.append(constraint); return self
    def _validate_relation_name(self,name):
        if name is not None and name not in self._relations: raise ValueError(f"unknown relation {name!r}")
    def no_self_loops(self,relation=None): self._validate_relation_name(relation); return self.constraint(NoSelfLoops(relation))
    def acyclic(self,relation): self._validate_relation_name(relation); return self.constraint(AcyclicRelation(relation))
    def at_most(self,relation=None,*,per_head=None,per_tail=None,per=None,limit=None):
        # New form: at_most("works_for", per_head=1); old form: at_most(1, "works_for", per="head")
        if isinstance(relation,int):
            old_limit=relation; relation=limit if isinstance(limit,str) else None; limit=old_limit
        self._validate_relation_name(relation)
        if limit is not None:
            if per == "tail": per_tail=limit
            else: per_head=limit
        if per_head is None and per_tail is None: raise ValueError("provide per_head and/or per_tail")
        if per_head is not None: self.constraint(MaxRelationsPerHead(per_head,relation))
        if per_tail is not None: self.constraint(MaxRelationsPerTail(per_tail,relation))
        return self
    def to_dict(self):
        return {"entities":{s.name:{k:v for k,v in asdict(s).items() if k!="name" and v is not None} for s in self.entity_specs},"relations":{s.name:{k:v for k,v in asdict(s).items() if k!="name" and v is not None} for s in self.relation_specs},"constraints":[c.to_dict() for c in self.constraints]}
    def to_json(self,**kwargs): return json.dumps(self.to_dict(),**kwargs)
    @classmethod
    def from_dict(cls,data):
        from .constraints import constraint_from_dict
        s=cls()
        entities=data.get("entities",{})
        if isinstance(entities,Mapping):
            for name,v in entities.items(): s.entity(name,**dict(v)) if isinstance(v,Mapping) else s.entity(name,v)
        else:
            for v in entities: s.entity(v) if isinstance(v,str) else s.entity(**v)
        relations=data.get("relations",{})
        if isinstance(relations,Mapping):
            for name,v in relations.items(): s.relation(name,**dict(v))
        else:
            for v in relations: s.relation(**v)
        for v in data.get("constraints",()): s.constraint(constraint_from_dict(v))
        return s
    @classmethod
    def from_json(cls,value): return cls.from_dict(json.loads(value))
