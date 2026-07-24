"""Compile declarative schemas into processor and decoder contracts."""
from dataclasses import dataclass
from typing import Any
from .constraints import (AcyclicRelation, Constraint, EntityOverlapPolicy, InverseRelation,
 MaxRelationsPerHead, MaxRelationsPerTail, NoSelfLoops, SymmetricRelation, TypedEndpoints,
 UniqueRelationPair, UniqueRelationSlot)
from .schema import EntitySpec, JointSchema, RelationSpec

@dataclass(frozen=True)
class CompiledJointSchema:
    model_schema: dict[str,Any]
    entity_specs: dict[str,EntitySpec]
    relation_specs: dict[str,RelationSpec]
    constraints: tuple[Constraint,...]
    entity_order: tuple[str,...]
    relation_order: tuple[str,...]
    def build(self): return self.model_schema

def _add(values, constraint):
    if constraint not in values: values.append(constraint)

def compile_schema(schema):
    if not isinstance(schema,JointSchema): raise TypeError("schema must be a JointSchema")
    entities={s.name:s for s in schema.entity_specs}; relations={s.name:s for s in schema.relation_specs}
    eo=tuple(entities); ro=tuple(relations)
    model={"json_structures":[],"classifications":[],"entities":{n:"" for n in eo},
      "relations":[{n:{"head":"","tail":""}} for n in ro],"json_descriptions":{},
      "entity_descriptions":{n:s.description for n,s in entities.items() if s.description is not None}}
    constraints=list(schema.constraints)
    if entities:
        policy="allow" if all(s.allow_nested for s in entities.values()) else ("nested" if any(s.allow_nested for s in entities.values()) else "disallow")
        _add(constraints,EntityOverlapPolicy(policy))
    for s in relations.values():
        _add(constraints,TypedEndpoints(s.name,s.head,s.tail))
        if not s.allow_self: _add(constraints,NoSelfLoops(s.name))
        _add(constraints,UniqueRelationPair(s.name,directed=s.directed))
        _add(constraints,UniqueRelationSlot(s.name,"slot"))
        if s.max_per_head is not None: _add(constraints,MaxRelationsPerHead(s.max_per_head,s.name))
        if s.max_per_tail is not None: _add(constraints,MaxRelationsPerTail(s.max_per_tail,s.name))
        if s.symmetric: _add(constraints,SymmetricRelation(s.name))
        if s.inverse:
            if s.inverse not in relations: raise ValueError(f"relation {s.name!r} has unknown inverse {s.inverse!r}")
            other=relations[s.inverse]
            if set(s.head)!=set(other.tail) or set(s.tail)!=set(other.head): raise ValueError(f"inverse endpoint types for {s.name!r} and {s.inverse!r} are incompatible")
            _add(constraints,InverseRelation(s.name,s.inverse))
    return CompiledJointSchema(model,entities,relations,tuple(constraints),eo,ro)
class JointSchemaCompiler:
    def compile(self,schema): return compile_schema(schema)
compile_joint_schema=compile_schema
