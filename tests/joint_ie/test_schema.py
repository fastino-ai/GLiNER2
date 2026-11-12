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
        "relation_descriptions": {},
    }
    assert compiled.build() is compiled.model_schema
    names = [type(x).__name__ for x in compiled.constraints]
    assert "TypedEndpoints" in names
    assert "NoSelfLoops" in names
    assert "UniqueRelationPair" in names
    assert "AcyclicRelation" in names


def test_compiler_preserves_relation_descriptions():
    schema = (JointSchema().entities({"person": "Human", "company": None})
              .relation("works_for", "person", "company", "Employment relationship")
              .relation("founded", "person", "company", "")
              .relation("owns", "person", "company"))

    compiled = compile_schema(JointSchema.from_json(schema.to_json()))

    assert compiled.build()["relation_descriptions"] == {
        "works_for": "Employment relationship", "founded": "",
    }
    assert compiled.build()["entity_descriptions"] == {"person": "Human"}


@pytest.mark.parametrize("description, expected", [
    ("Employment relationship", "works_for: Employment relationship"),
    ("", "works_for"),
    (None, "works_for"),
])
def test_compiled_relation_description_reaches_processor(tiny_tokenizer, description, expected):
    from gliner2.processor import SchemaTransformer

    schema = (JointSchema().entities(["person", "company"])
              .relation("works_for", "person", "company", description)
              .relation("owns", "person", "company"))
    processor = SchemaTransformer(tokenizer=tiny_tokenizer)
    processor.change_mode(False)

    record = processor.transform_and_format(
        "John works at Apple.", compile_schema(schema).build()
    )

    prompts = [tokens[2] for tokens, task in zip(record.schema_tokens_list, record.task_types)
               if task == "relations"]
    assert prompts == [expected, "owns"]
