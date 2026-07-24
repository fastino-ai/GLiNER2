"""JSON schema (from_dict/to_dict) support for per-field thresholds and regex
validators, and per-task classification thresholds.

These exercise only the torch-free schema layer (``gliner2.inference.schema`` /
``schema_model``), so they need no model download.
"""

import pytest

from gliner2.inference.schema import RegexValidator, Schema


def test_from_dict_wires_field_threshold_and_validators():
    schema = Schema.from_dict(
        {
            "structures": {
                "job": {
                    "fields": [
                        {"name": "title", "dtype": "str", "threshold": 0.7},
                        {
                            "name": "email",
                            "dtype": "str",
                            "validators": [
                                {"pattern": r"[^@]+@[^@]+", "mode": "partial", "ignore_case": False}
                            ],
                        },
                    ]
                }
            }
        }
    )

    assert schema._field_metadata["job.title"]["threshold"] == 0.7

    validators = schema._field_metadata["job.email"]["validators"]
    assert len(validators) == 1
    v = validators[0]
    assert isinstance(v, RegexValidator)
    assert v.mode == "partial"
    assert v.exclude is False
    assert v.flags == 0  # ignore_case=False
    assert v.validate("a@b.com") is True
    assert v.validate("nope") is False


def test_from_dict_wires_cls_threshold():
    schema = Schema.from_dict(
        {"classifications": [{"task": "remote", "labels": ["yes", "no"], "cls_threshold": 0.8}]}
    )
    cfg = schema.schema["classifications"][0]
    assert cfg["cls_threshold"] == 0.8


def test_to_dict_round_trips_thresholds_and_validators():
    data = {
        "structures": {
            "job": {
                "fields": [
                    {"name": "title", "dtype": "str", "threshold": 0.7},
                    {
                        "name": "email",
                        "validators": [
                            {"pattern": r"[^@]+@[^@]+", "mode": "partial", "exclude": False, "ignore_case": False}
                        ],
                    },
                ]
            }
        },
        "classifications": [{"task": "remote", "labels": ["yes", "no"], "cls_threshold": 0.8}],
    }
    out = Schema.from_dict(data).to_dict()

    fields = {f["name"]: f for f in out["structures"]["job"]["fields"]}
    assert fields["title"]["threshold"] == 0.7
    assert fields["email"]["validators"] == [
        {"pattern": r"[^@]+@[^@]+", "mode": "partial", "exclude": False, "ignore_case": False}
    ]
    assert out["classifications"][0]["cls_threshold"] == 0.8

    # Idempotent: re-parsing the emitted dict yields the same dict.
    assert Schema.from_dict(out).to_dict() == out


def test_defaults_omitted_from_to_dict():
    out = Schema.from_dict(
        {
            "structures": {"x": {"fields": [{"name": "f"}]}},
            "classifications": [{"task": "t", "labels": ["a", "b"]}],
        }
    ).to_dict()
    field = out["structures"]["x"]["fields"][0]
    assert "threshold" not in field
    assert "validators" not in field
    assert "cls_threshold" not in out["classifications"][0]


@pytest.mark.parametrize(
    "bad",
    [
        {"structures": {"x": {"fields": [{"name": "f", "threshold": 1.5}]}}},
        {"structures": {"x": {"fields": [{"name": "f", "validators": [{"pattern": "("}]}]}}},
        {"classifications": [{"task": "t", "labels": ["a", "b"], "cls_threshold": 2}]},
    ],
)
def test_invalid_inputs_rejected(bad):
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        Schema.from_dict(bad)
