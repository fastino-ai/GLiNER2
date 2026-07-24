from types import SimpleNamespace

import pytest

from gliner2.joint_ie.long_text import extract_long_text
from gliner2.joint_ie.lattice import SpanRef
from gliner2.joint_ie.result import JointEntity, JointRelation, JointResult, ResultBuilder


def test_result_navigation_and_stable_serialization():
    result = JointResult("Ada works at Acme", [
        JointEntity("e0", "person", "Ada", 0, 3, 0.9),
        JointEntity("e1", "org", "Acme", 13, 17, 0.8, rescued=True),
    ], [JointRelation("works_for", "e0", "e1", 0.7)])
    assert result.entity("e0").text == "Ada"
    assert result.entities_by_type("org") == [result.entities[1]]
    assert result.outgoing("e0") == result.relations
    assert result.incoming(result.entities[1]) == result.relations
    assert result.neighbors("e0") == [result.entities[1]]
    assert list(result.to_dict()) == ["entities", "relations"]
    assert result.to_dict(include_confidence=False, include_spans=False)["entities"][0] == {
        "id": "e0", "type": "person", "text": "Ada"
    }


def test_result_builder_sorts_ids_and_geometric_confidence():
    person = {"type": "person", "span": SpanRef(0, 1, 0, 3, "Ada"),
              "probabilities": [0.81, 1.0]}
    org = {"type": "org", "span": SpanRef(3, 4, 13, 17, "Acme"),
           "confidence": 0.64, "rescued": True}
    relation = {"type": "works_for", "head": person, "tail": org,
                "probabilities": [0.81, 0.64], "count_probability": 1.0}
    built = ResultBuilder().build(
        SimpleNamespace(entities=[org, person], relations=[relation]),
        SimpleNamespace(text="Ada works at Acme"),
    )
    assert [(entity.id, entity.type) for entity in built.entities] == [("e1", "person"), ("e2", "org")]
    assert built.entities[0].confidence == pytest.approx(0.9)
    assert built.relations[0].head == "e1"
    assert built.relations[0].tail == "e2"
    assert built.relations[0].confidence == pytest.approx((0.81 * 0.64) ** (1 / 3))


def test_long_text_remaps_dedupes_and_keeps_relations_chunk_local():
    class Engine:
        def extract_joint(self, text, include_confidence=True, include_spans=True):
            entities = []
            for word, kind in (("Bob", "person"), ("Acme", "org")):
                start = 0
                while True:
                    start = text.find(word, start)
                    if start < 0:
                        break
                    entities.append(JointEntity(f"x{len(entities)}", kind, word,
                                                start, start + len(word), 0.8))
                    start += len(word)
            relations = []
            people = [entity for entity in entities if entity.type == "person"]
            orgs = [entity for entity in entities if entity.type == "org"]
            for person, org in zip(people, orgs):
                relations.append(JointRelation("works_for", person.id, org.id, 0.7))
            return JointResult(text, entities, relations)

    text = "Bob works at Acme and Bob works at Acme"
    result = extract_long_text(Engine(), text, chunk_size=6, chunk_overlap=3)
    assert [(entity.id, entity.start, entity.end) for entity in result.entities] == [
        ("e1", 0, 3), ("e2", 13, 17), ("e3", 22, 25), ("e4", 35, 39)
    ]
    assert all(relation.head in {"e1", "e3"} for relation in result.relations)
