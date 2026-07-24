import dataclasses
import json
import pytest

from gliner2.joint_ie import JointSchema, compile_schema


def test_frozen_specs_order_and_roundtrip():
    schema = (JointSchema().entity("person", "A human").entities(["company", "city"])
              .relation("works_for", "person", "company", unique_pair=True,
                        no_self_loops=True)
              .relation("based_in", "company", "city"))
    assert [x.name for x in schema.entity_specs] == ["person", "company", "city"]
    assert [x.name for x in schema.relation_specs] == ["works_for", "based_in"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        schema.entity_specs[0].name = "other"
    assert JointSchema.from_json(schema.to_json()).to_dict() == schema.to_dict()
    assert json.loads(schema.to_json())["relations"]["works_for"]["head"] == ["person"]


def test_relation_endpoint_validation():
    schema = JointSchema().entities(["person", "company"])
    with pytest.raises(ValueError, match="unknown entity"):
        schema.relation("works_for", "human", "company")


def test_compiler_emits_exact_processor_shape_and_flags():
    schema = (JointSchema().entities({"person": "Human", "company": None})
              .relation("works_for", "person", "company", unique_pair=True,
                        no_self_loops=True, acyclic=True))
    compiled = compile_schema(schema)
    assert compiled.model_schema == {
        "json_structures": [], "classifications": [],
        "entities": {"person": "", "company": ""},
        "relations": [{"works_for": {"head": "", "tail": ""}}],
        "json_descriptions": {}, "entity_descriptions": {"person": "Human"},
    }
    assert compiled.build() is compiled.model_schema
    names = [type(x).__name__ for x in compiled.constraints]
    assert "TypedEndpoints" in names
    assert "NoSelfLoops" in names
    assert "UniqueRelationPair" in names
    assert "AcyclicRelation" in names
