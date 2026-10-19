"""A ``choices=`` field puts its literals in a schema prefix; records must not be anchored there."""

from types import SimpleNamespace

from gliner2.models.boundary.engine import BoundaryExtractor
from tests.models.boundary.test_record_head_pipeline import _build_tiny_records_model

TEXTS = [
    "Alice bought apples in Paris on Monday",
    "Bob sold oranges to Carol in Rome",
    "The camera is excellent but the battery is weak",
]


def _spec(choice_anchor: bool):
    fields = (SimpleNamespace(query_id=0, name="org"), SimpleNamespace(query_id=1, name="tone"))
    metadata = {"field_metadata": {"report.tone": {"choices": ["positive", "negative"]}}}
    if choice_anchor:
        metadata["field_metadata"]["report.org"] = {"choices": ["a", "b"]}
    return SimpleNamespace(task_name="report", anchor_query_id=0, fields=fields), metadata


def test_prefix_and_straddling_anchors_are_dropped():
    records = [SimpleNamespace(anchor_span=span) for span in [(8, 10), (8, 15), (11, 14), None]]
    spec, metadata = _spec(choice_anchor=False)
    kept = BoundaryExtractor._drop_prefix_anchored(records, spec, metadata, offset=11)
    assert [r.anchor_span for r in kept] == [(11, 14), None]


def test_a_choice_anchor_may_sit_in_the_prefix():
    records = [SimpleNamespace(anchor_span=(8, 10)), SimpleNamespace(anchor_span=(11, 14))]
    spec, metadata = _spec(choice_anchor=True)
    assert BoundaryExtractor._drop_prefix_anchored(records, spec, metadata, offset=11) == records


def test_choice_structure_decodes_without_prefix_anchored_records():
    model = _build_tiny_records_model()
    schema = (
        model.create_schema()
        .structure("report", mode="natural", anchor="buyer")
        .field("buyer", dtype="str")
        .field("tone", dtype="str", choices=["positive", "negative", "neutral"])
    )
    results = model.batch_extract(TEXTS, schema, batch_size=1, threshold=0.0, include_spans=True)

    reports = [inst for r in results for inst in r.get("report", [])]
    assert reports
    assert all(inst["buyer"] is not None for inst in reports)


def test_a_choice_anchor_decodes_without_raising():
    model = _build_tiny_records_model()
    schema = (
        model.create_schema()
        .structure("report", mode="natural", anchor="tone")
        .field("tone", dtype="str", choices=["positive", "negative", "neutral"])
        .field("buyer", dtype="str")
    )
    results = model.batch_extract(TEXTS, schema, batch_size=1, threshold=0.0, include_spans=True)
    assert len(results) == len(TEXTS)
