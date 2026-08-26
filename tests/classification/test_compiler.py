"""Phase 3 gate: a compiled schema either round-trips through the real
processor with exact label alignment, or fails loudly at compile. No third
outcome.
"""

from __future__ import annotations

import pytest

from gliner2.classification import constraints as C
from gliner2.classification.compiler import (
    CompiledClassificationSchema,
    _assert_model_schema,
    compile_schema,
)
from gliner2.classification.constraints import DictAssignment, Iff
from gliner2.classification.errors import SchemaError
from gliner2.classification.schema import ClassificationSchema


# ---- real-processor fixture -------------------------------------------


@pytest.fixture(scope="module")
def processor():
    """A real SchemaTransformer. Skips if the tokenizer cannot be loaded
    offline; when present, this is the adversarial contract check."""
    try:
        from gliner2.processor import SchemaTransformer

        return SchemaTransformer(model_name="fastino/gliner2-base-v1")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"real tokenizer unavailable: {exc!r}")


def _l_names(tokens):
    return [tokens[i + 1] for i in range(len(tokens) - 1) if tokens[i] == "[L]"]


def _basic():
    return (
        ClassificationSchema()
        .single("intent", ["read", "write", "delete"])
        .multi("effects", ["read_only", "create", "modify", "delete"], min_labels=1)
        .ordinal("risk", ["safe", "low", "elevated", "dangerous"])
    )


# ---- T-P1 : the adversarial one ----------------------------------------


def test_compiled_schema_survives_real_processor(processor):
    schema = _basic()
    compiled = compile_schema(schema)
    batch = processor.collate_fn_inference([("delete the file", compiled.build())])

    # (a) every task is a classification task
    assert batch.task_types[0] == ["classifications"] * len(schema.task_order)

    # (b) the fallback record was NOT taken
    flat = [tok for task_tokens in batch.schema_tokens_list[0] for tok in task_tokens]
    assert "dummy" not in flat

    # (c) the encoded [L] names equal label_names, in order, per task
    for task_tokens, spec in zip(batch.schema_tokens_list[0], schema.task_specs):
        assert _l_names(task_tokens) == list(spec.label_names)


# ---- T-P2 : inference does not read true_label; training does ----------

_NO_TRUE_LABEL = {
    "json_structures": [],
    "entities": {},
    "relations": [],
    "json_descriptions": {},
    "entity_descriptions": {},
    "classifications": [
        {
            "task": "intent",
            "labels": ["a", "b"],
            "multi_label": False,
            "cls_threshold": 0.5,
            "class_act": "auto",
        }
    ],
}


def test_classification_inference_without_true_label_collates(processor):
    batch = processor.collate_fn_inference(
        [("hello", _NO_TRUE_LABEL)], error_policy="raise"
    )
    assert batch.task_types[0] == ["classifications"]
    assert batch.structure_labels[0] == [[0, 0]]
    flat = [tok for task_tokens in batch.schema_tokens_list[0] for tok in task_tokens]
    assert "dummy" not in flat


def test_missing_true_label_is_not_required_by_assert():
    _assert_model_schema(_NO_TRUE_LABEL)


def test_missing_true_label_raises_when_targets_are_requested(processor):
    with pytest.raises(SchemaError, match="true_label"):
        processor.collate_fn_inference(
            [("hello", _NO_TRUE_LABEL)],
            error_policy="raise",
            build_targets=True,
        )
    with pytest.raises(SchemaError, match="true_label"):
        processor.collate_fn_train([("hello", _NO_TRUE_LABEL)], error_policy="raise")


# ---- T-P4 : instruction -> prompt, resolves back -----------------------


def test_instruction_reaches_prompt_and_resolves(processor):
    schema = (
        ClassificationSchema()
        .single("sentiment", ["pos", "neg"], instruction="Judge the tone.")
        .single("sent", ["a", "b"])
    )  # prefix without boundary -> allowed
    compiled = compile_schema(schema)
    batch = processor.collate_fn_inference([("hello", compiled.build())])

    prompt_strs = [task_tokens[2] for task_tokens in batch.schema_tokens_list[0]]
    assert "sentiment: Judge the tone." in prompt_strs

    from gliner2.inference.engine import GLiNER2

    classifications = compiled.build()["classifications"]
    for prompt_str in prompt_strs:
        resolved = GLiNER2._resolve_classification_config(prompt_str, classifications)
        assert resolved is not None
    # prefix shadowing: "sent" prompt must not resolve to "sentiment"
    resolved = GLiNER2._resolve_classification_config(
        "sentiment: Judge the tone.", classifications
    )
    assert resolved["task"] == "sentiment"


# ---- T-P5 : prefix-colliding task names --------------------------------


def test_prefix_colliding_task_names_rejected():
    schema = (
        ClassificationSchema()
        .single("risk", ["a", "b"])
        .single("risk level", ["c", "d"])
    )
    with pytest.raises(SchemaError, match="prefix"):
        compile_schema(schema)


def test_bare_prefix_without_boundary_is_allowed():
    schema = (
        ClassificationSchema()
        .single("sent", ["a", "b"])
        .single("sentiment", ["c", "d"])
    )
    compile_schema(schema)  # must not raise


# ---- T-P6 : descriptions and examples appear in the encoded prompt -----


def test_descriptions_and_examples_in_prompt(processor):
    schema = ClassificationSchema().single(
        "intent",
        {"read": "a read operation", "write": "a write operation"},
        examples=[("cat file", "read"), ("rm file", "write")],
    )
    compiled = compile_schema(schema)
    batch = processor.collate_fn_inference([("hello", compiled.build())])
    prompt_str = batch.schema_tokens_list[0][0][2]
    assert "[DESCRIPTION]" in prompt_str
    assert "[EXAMPLE]" in prompt_str


# ---- T-C6 : default lowering -------------------------------------------


def test_default_lowering_semantics():
    schema = ClassificationSchema().multi(
        "tox", ["violence", "hate", "benign"], threshold=0.4, default="benign"
    )
    compiled = compile_schema(schema)
    lowered = [c for c in compiled.constraints if isinstance(c, Iff)]
    assert lowered, "default should lower into an Iff"
    rule = lowered[0]

    def decide(sel):
        return DictAssignment(compiled, {"tox": set(sel)}, decided=["tox"])

    assert rule.evaluate(decide(["benign"])) is True  # default alone: legal
    assert rule.evaluate(decide(["benign", "hate"])) is False  # default + real: illegal
    assert rule.evaluate(decide(["hate"])) is True  # real alone: legal
    # the v2 regression: the default is reachable at all
    assert rule.evaluate(decide(["benign"])) is not False


# ---- T-P7 : fingerprints -----------------------------------------------


def test_fingerprint_stable_and_constraint_sensitive():
    schema = _basic()
    assert compile_schema(schema).fingerprint == compile_schema(schema).fingerprint

    with_constraint = _basic().constrain(
        C.implies(("intent", "delete"), ("effects", "delete"))
    )
    assert (
        compile_schema(with_constraint).fingerprint
        != compile_schema(schema).fingerprint
    )


# ---- T-P8 : idempotent compile -----------------------------------------


def test_compile_is_idempotent():
    compiled = compile_schema(_basic())
    assert compile_schema(compiled) is compiled
    assert isinstance(compiled, CompiledClassificationSchema)


# ---- T-P9 : reserved token that somehow reached emission ---------------


def test_assert_rejects_reserved_token_in_emission():
    model = {
        "json_structures": [],
        "entities": {},
        "relations": [],
        "json_descriptions": {},
        "entity_descriptions": {},
        "classifications": [
            {
                "task": "intent",
                "labels": ["a", "b[L]bad"],
                "multi_label": False,
                "cls_threshold": 0.5,
                "class_act": "auto",
            }
        ],
    }
    with pytest.raises(SchemaError, match="reserved"):
        _assert_model_schema(model)


# ---- static feasibility ------------------------------------------------


def test_static_feasibility_catches_contradiction():
    schema = (
        ClassificationSchema()
        .single("x", ["a", "b"])
        .constrain(C.all_of(("x", "a"), ("x", "b")))
    )
    with pytest.raises(SchemaError):
        compile_schema(schema)


def test_task_lookup_raises_schema_error():
    compiled = compile_schema(_basic())
    with pytest.raises(SchemaError):
        compiled.task("nonexistent")
