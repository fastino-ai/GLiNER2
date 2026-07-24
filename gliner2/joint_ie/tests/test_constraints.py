from gliner2.joint_ie import (
    AcyclicRelation, EntityOverlapPolicy, InverseRelation, MaxRelationsPerHead,
    NoSelfLoops, SymmetricRelation, TypedEndpoints, UniqueRelationPair,
    UniqueRelationSlot,
)


def entity(identifier, label, start=0, end=1):
    return {"id": identifier, "type": label, "start": start, "end": end}


def relation(label, head, tail):
    return {"type": label, "head": head, "tail": tail}


def test_local_relation_constraints_use_duck_typing():
    alice, acme, bob = entity("a", "person"), entity("c", "company"), entity("b", "person")
    edge = relation("works", alice, acme)
    assert TypedEndpoints("works", ("person",), ("company",))(edge)
    assert not TypedEndpoints("works", ("company",), ("person",))(edge)
    assert not NoSelfLoops("works")(relation("works", alice, alice))
    assert not UniqueRelationPair("works")(edge, [edge])
    assert not UniqueRelationSlot("works", "head")(relation("works", alice, bob), [edge])
    assert not MaxRelationsPerHead(1, "works")(relation("works", alice, bob), [edge])


def test_graph_constraints():
    a, b, c = (entity(x, "node") for x in "abc")
    ab, bc, ca = relation("edge", a, b), relation("edge", b, c), relation("edge", c, a)
    assert AcyclicRelation("edge")(bc, [ab])
    assert not AcyclicRelation("edge")(ca, [ab, bc])
    symmetric = [ab, relation("edge", b, a)]
    assert SymmetricRelation("edge").validate(symmetric)
    assert InverseRelation("parent", "child").validate([
        relation("parent", a, b), relation("child", b, a)])


def test_entity_overlap_policies():
    outer = {"start": 0, "end": 10}
    nested = {"start": 2, "end": 5}
    crossing = {"start": 4, "end": 12}
    assert EntityOverlapPolicy("disallow").apply_entities([outer, nested]) == [outer]
    assert EntityOverlapPolicy("nested").apply_entities([outer, nested, crossing]) == [outer, nested]
