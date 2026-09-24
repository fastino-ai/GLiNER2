"""Schema.from_dict covers every option the builder takes, and nothing else."""

from __future__ import annotations

import copy

import pytest
import torch
from pydantic import ValidationError

from gliner2.inference.runtime import ExtractorRuntimeMixin
from gliner2.inference.schema import AttributeGroup, Schema

FULL_SCHEMA = {
    "entities": {
        "person": "A human being",
        "company": {"description": "An organisation", "threshold": 0.7},
        "city": {"dtype": "str"},
        "product": {},
    },
    "entity_attributes": {
        "sentiment": {
            "labels": ["positive", "negative"],
            "threshold": 0.3,
            "applies_to": ["person", "company"],
        },
        "role": {"labels": ["founder", "employee"], "multi_label": True, "qualify_labels": True},
    },
    "structures": {
        "purchase": {
            "fields": [
                {"name": "buyer", "dtype": "str", "cardinality": "required_one"},
                {"name": "item", "threshold": 0.4, "exclusive": True},
                {"name": "channel", "dtype": "str", "choices": ["web", "store"]},
            ],
            "mode": "natural",
            "anchor": "buyer",
            "occurrence_policy": "first",
        }
    },
    "classifications": [
        {
            "task": "topics",
            "labels": {"tech": "Technology news", "finance": "Money and markets"},
            "multi_label": True,
            "cls_threshold": 0.2,
            "top_k": 1,
            "class_act": "sigmoid",
        },
        {
            "task": "sentiment",
            "labels": ["positive", "negative"],
            "prompt": "overall tone",
            "examples": [["I love it", "positive"]],
        },
    ],
    "relations": {
        "works_for": {"description": "Employment", "threshold": 0.6},
        "founded": {},
    },
}


def _builder_schema() -> Schema:
    schema = Schema()
    schema.entities(
        {
            "person": "A human being",
            "company": {"description": "An organisation", "threshold": 0.7},
            "city": {"dtype": "str"},
            "product": {},
        }
    )
    schema.entity_attributes(
        {
            "sentiment": AttributeGroup(
                ["positive", "negative"], threshold=0.3, applies_to=["person", "company"]
            ),
            "role": AttributeGroup(["founder", "employee"], multi_label=True, qualify_labels=True),
        }
    )
    (
        schema.structure("purchase", mode="natural", anchor="buyer", occurrence_policy="first")
        .field("buyer", dtype="str", cardinality="required_one")
        .field("item", threshold=0.4, exclusive=True)
        .field("channel", dtype="str", choices=["web", "store"])
    )
    schema.classification(
        "topics",
        {"tech": "Technology news", "finance": "Money and markets"},
        multi_label=True,
        cls_threshold=0.2,
        top_k=1,
        class_act="sigmoid",
    )
    schema.classification(
        "sentiment",
        ["positive", "negative"],
        prompt="overall tone",
        examples=[("I love it", "positive")],
    )
    schema.relations({"works_for": {"description": "Employment", "threshold": 0.6}, "founded": {}})
    return schema


def test_round_trip_is_exact():
    assert Schema.from_dict(FULL_SCHEMA).to_dict() == FULL_SCHEMA


def test_from_dict_builds_what_the_builder_builds():
    from_dict = Schema.from_dict(FULL_SCHEMA)
    builder = _builder_schema()
    assert from_dict.build() == builder.build()
    assert from_dict._entity_metadata == builder._entity_metadata
    assert from_dict._field_metadata == builder._field_metadata
    assert from_dict._relation_metadata == builder._relation_metadata
    assert from_dict._entity_attribute_groups == builder._entity_attribute_groups


def test_builder_to_dict_matches_the_dict_form():
    assert _builder_schema().to_dict() == FULL_SCHEMA


def test_threshold_metadata_reaches_the_schema():
    schema = Schema.from_dict(FULL_SCHEMA)
    built = schema.build()
    topics = built["classifications"][0]
    assert topics["cls_threshold"] == 0.2
    assert topics["top_k"] == 1
    assert topics["label_descriptions"] == {
        "tech": "Technology news",
        "finance": "Money and markets",
    }
    assert schema._entity_metadata["company"]["threshold"] == 0.7
    assert schema._field_metadata["purchase.item"]["threshold"] == 0.4
    assert schema._relation_metadata["works_for"]["threshold"] == 0.6


def test_to_dict_lists_only_declared_entities():
    schema = Schema().entities(["person"])
    schema.entity_attributes({"mood": AttributeGroup(["happy", "sad"])})
    assert schema.build()["entities"].keys() >= {"person", "happy", "sad"}
    assert schema.to_dict() == {
        "entities": ["person"],
        "entity_attributes": {"mood": {"labels": ["happy", "sad"]}},
    }


def test_to_dict_keeps_undescribed_entities_beside_described_ones():
    data = {"entities": {"person": "A human", "city": {}}}
    assert Schema.from_dict(data).to_dict() == data


def test_plain_forms_round_trip_unchanged():
    data = {
        "entities": ["a", "b"],
        "classifications": [{"task": "t", "labels": ["x", "y"]}],
        "relations": ["r"],
    }
    assert Schema.from_dict(data).to_dict() == data


def _with(path: tuple, key: str, value: object) -> dict:
    data = copy.deepcopy(FULL_SCHEMA)
    target = data
    for step in path:
        target = target[step]
    target[key] = value
    return data


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("entities", "company"),
        ("entity_attributes", "sentiment"),
        ("structures", "purchase"),
        ("structures", "purchase", "fields", 0),
        ("classifications", 0),
        ("relations", "works_for"),
    ],
)
def test_unknown_key_raises_at_every_level(path):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Schema.from_dict(_with(path, "unknown_option", True))


@pytest.mark.parametrize(
    ("path", "key", "value"),
    [
        (("classifications", 0), "cls_threshold", 1.5),
        (("classifications", 0), "top_k", 0),
        (("classifications", 0), "class_act", "relu"),
        (("entities", "company"), "threshold", -0.1),
        (("entities", "company"), "dtype", "dict"),
        (("structures", "purchase", "fields", 1), "threshold", 2),
        (("relations", "works_for"), "threshold", 1.1),
        (("entity_attributes", "sentiment"), "threshold", 3),
        (("entity_attributes", "sentiment"), "labels", []),
    ],
)
def test_out_of_range_values_raise(path, key, value):
    with pytest.raises(ValidationError):
        Schema.from_dict(_with(path, key, value))


def test_entity_attributes_require_entities():
    with pytest.raises(ValidationError, match="entity_attributes requires entities"):
        Schema.from_dict({"relations": ["r"], "entity_attributes": {"mood": {"labels": ["happy"]}}})


def test_label_descriptions_need_two_labels():
    with pytest.raises(ValidationError, match="at least 2"):
        Schema.from_dict({"classifications": [{"task": "t", "labels": {"x": "only"}}]})


@pytest.mark.parametrize(("cls_threshold", "top_k"), [(-0.1, None), (0.5, 0)])
def test_builder_rejects_bad_classification_options(cls_threshold, top_k):
    with pytest.raises(ValueError):
        Schema().classification("t", ["x", "y"], cls_threshold=cls_threshold, top_k=top_k)


def test_builder_omits_top_k_when_unset():
    built = Schema().classification("t", ["x", "y"]).build()
    assert "top_k" not in built["classifications"][0]


class _FixedHead(torch.nn.Module):
    def __init__(self, logits: list[float]):
        super().__init__()
        self.logits = torch.tensor(logits).unsqueeze(-1)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        return self.logits


class _Decoder(ExtractorRuntimeMixin):
    def __init__(self, logits: list[float]):
        self.classifier = _FixedHead(logits)


def _decode(schema: Schema, logits: list[float]):
    results: dict = {}
    _Decoder(logits)._extract_classification_result(
        results,
        "topics",
        schema.build(),
        torch.zeros(len(logits) + 1, 4),
        ["(", "[P]", "topics", "("],
    )
    return results["topics"]


def test_top_k_caps_multi_label_by_probability():
    schema = Schema().classification(
        "topics", ["a", "b", "c"], multi_label=True, cls_threshold=0.1, top_k=2
    )
    decoded = _decode(schema, [0.0, 3.0, 1.0])
    assert [label for label, _ in decoded] == ["b", "c"]


def test_multi_label_order_is_unchanged_without_top_k():
    schema = Schema().classification("topics", ["a", "b", "c"], multi_label=True, cls_threshold=0.1)
    decoded = _decode(schema, [0.0, 3.0, 1.0])
    assert [label for label, _ in decoded] == ["a", "b", "c"]


def test_cls_threshold_from_dict_filters_multi_label():
    schema = Schema.from_dict(
        {
            "classifications": [
                {
                    "task": "topics",
                    "labels": ["a", "b", "c"],
                    "multi_label": True,
                    "cls_threshold": 0.9,
                }
            ]
        }
    )
    decoded = _decode(schema, [0.0, 3.0, 1.0])
    assert [label for label, _ in decoded] == ["b"]
