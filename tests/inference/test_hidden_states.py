"""Precomputed encoder hidden states reproduce every batch inference API."""

from __future__ import annotations

import pytest
import torch

from gliner2 import ExtractorConfig, Schema
from gliner2.classification import ClassificationConfig, ClassificationSchema, Classifier
from gliner2.inference.engine import BoundaryExtractor
from gliner2.joint_ie import JointIE, JointIEConfig
from gliner2.joint_ie.schema import JointSchema
from gliner2.training.data import InputExample
from gliner2.training.trainer import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD
from tests.fixtures.tiny_encoder import build_tiny_encoder_config
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model
from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

TEXTS = ["Alice works at Acme in Paris", "Bob joined Globex", "Carol is happy"]


def _boundary_model(load_encoder: bool = True) -> BoundaryExtractor:
    tokenizer = build_tiny_tokenizer()
    head = dict(TINY_BOUNDARY_HEAD)
    head.update(
        enable_relations=True,
        relation_heads_per_type=8,
        relation_tails_per_type=8,
        relation_pair_cap=16,
        relation_argument_proposal_threshold=0.0,
    )
    torch.manual_seed(3)
    return BoundaryExtractor(
        ExtractorConfig(
            model_name="tiny-bert-fixture",
            architecture="boundary",
            boundary_head=head,
            token_pooling="first",
        ),
        encoder_config=build_tiny_encoder_config(vocab_size=len(tokenizer)),
        tokenizer=tokenizer,
        load_encoder=load_encoder,
    ).eval()


def _encoderless_copy(model: BoundaryExtractor) -> BoundaryExtractor:
    heads = _boundary_model(load_encoder=False)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
    heads.load_state_dict(state)
    return heads.eval()


def _capture_rows(model, run):
    """Run ``run()`` and return its output plus the encoder's per-row states."""
    rows = []

    def hook(module, args, kwargs, output):
        mask = kwargs["attention_mask"]
        states = output.last_hidden_state
        rows.extend(states[i, : int(mask[i].sum())].clone() for i in range(mask.shape[0]))

    handle = model.encoder.register_forward_hook(hook, with_kwargs=True)
    try:
        result = run()
    finally:
        handle.remove()
    return result, rows


def _assert_encoder_unused(model, run):
    calls = []
    handle = model.encoder.register_forward_hook(lambda *_: calls.append(1))
    try:
        result = run()
    finally:
        handle.remove()
    assert calls == []
    return result


def _schema() -> Schema:
    return (
        Schema()
        .entities(["person", "company", "location"])
        .classification("sentiment", ["positive", "negative"])
        .relations(["works_for"])
    )


@pytest.mark.parametrize("build", [_boundary_model, build_tiny_span_model])
def test_batch_extract_matches_with_hidden_states(build):
    model = build()
    schema = _schema()

    def run(**kwargs):
        return model.batch_extract(
            TEXTS, schema, threshold=0.0, include_confidence=True, include_spans=True, **kwargs
        )

    expected, rows = _capture_rows(model, run)
    assert len(rows) == len(TEXTS)
    assert any(result["entities"] for result in expected)
    assert _assert_encoder_unused(model, lambda: run(hidden_states=rows)) == expected


def test_encoderless_boundary_model_matches_full_model():
    model = _boundary_model()
    heads = _encoderless_copy(model)
    schema = _schema()

    def run(target, **kwargs):
        return target.batch_extract(TEXTS, schema, threshold=0.0, include_confidence=True, **kwargs)

    expected, rows = _capture_rows(model, lambda: run(model))
    assert heads.encoder is None
    assert set(heads.state_dict()) == {
        k for k in model.state_dict() if not k.startswith("encoder.")
    }
    assert run(heads, hidden_states=rows) == expected


def test_classifier_matches_with_hidden_states_and_per_text_schemas():
    model = _boundary_model()
    schemas = [
        ClassificationSchema().single("intent", ["read", "delete"]),
        ClassificationSchema().multi("effects", ["read_only", "delete"], min_labels=1),
        ClassificationSchema().single("intent", ["read", "delete"]),
    ]
    config = ClassificationConfig(decoder="exact")

    def run(target, **kwargs):
        results = Classifier(target).batch_classify(TEXTS, schemas, config=config, **kwargs)
        return [result.to_dict() for result in results]

    expected, rows = _capture_rows(model, lambda: run(model))
    assert [str(r) for r in expected] != [str(expected[0])] * len(TEXTS)
    assert "effects" in str(expected[1]) and "intent" in str(expected[0])
    assert _assert_encoder_unused(model, lambda: run(model, hidden_states=rows)) == expected
    assert run(_encoderless_copy(model), hidden_states=rows) == expected


def test_joint_ie_matches_with_hidden_states():
    model = _boundary_model()
    schema = (
        JointSchema()
        .entity("person", threshold=0.1, candidate_threshold=0.0)
        .entity("org", threshold=0.1, candidate_threshold=0.0)
        .relation("works_for", "person", "org", threshold=0.1)
    )
    config = JointIEConfig(candidate_threshold=0.0, optimizer="greedy")

    def run(target, **kwargs):
        engine = JointIE(target)
        return [r.to_dict() for r in engine.batch_extract(TEXTS, schema, config=config, **kwargs)]

    expected, rows = _capture_rows(model, lambda: run(model))
    assert any(result["entities"] for result in expected)
    assert _assert_encoder_unused(model, lambda: run(model, hidden_states=rows)) == expected
    assert run(_encoderless_copy(model), hidden_states=rows) == expected


def test_mismatched_hidden_states_raise():
    model = _boundary_model()
    schema = _schema()
    _, rows = _capture_rows(model, lambda: model.batch_extract(TEXTS, schema))

    with pytest.raises(ValueError, match="hidden_states count"):
        model.batch_extract(TEXTS, schema, hidden_states=rows[:-1])
    with pytest.raises(ValueError, match=r"hidden_states\[1\] has shape"):
        model.batch_extract(TEXTS, schema, hidden_states=[rows[0], rows[1][:-1], rows[2]])
    with pytest.raises(ValueError, match=r"hidden_states\[0\] has shape"):
        model.batch_extract(TEXTS, schema, hidden_states=[r[:, :-1] for r in rows])


def test_encoderless_model_requires_hidden_states():
    heads = _boundary_model(load_encoder=False)
    with pytest.raises(ValueError, match="built without an encoder"):
        heads.batch_extract(TEXTS, _schema())
    with pytest.raises(ValueError, match="cannot save"):
        heads.save_pretrained("unused")


def test_encoderless_model_requires_encoder_config():
    config = ExtractorConfig(
        model_name="tiny-bert-fixture",
        architecture="boundary",
        boundary_head=dict(TINY_BOUNDARY_HEAD),
        token_pooling="first",
    )
    with pytest.raises(ValueError, match="requires encoder_config"):
        BoundaryExtractor(config, tokenizer=build_tiny_tokenizer(), load_encoder=False)


def test_training_loss_runs_through_the_encoder_seam(monkeypatch):
    model = _boundary_model().train()
    examples = [
        InputExample(text="Alice works at Acme", entities={"person": ["Alice"], "org": ["Acme"]}),
        InputExample(text="Bob is here", entities={"person": ["Bob"]}),
    ]
    batch = ExtractorCollator(model.processor, is_training=True, architecture="boundary")(
        [(e.text, e.to_dict()["output"]) for e in examples]
    )
    torch.manual_seed(0)
    expected = model(batch).loss

    seen = []
    original = type(model).encode_tokens

    def spy(self, batch, hidden_states=None):
        seen.append(hidden_states)
        return original(self, batch, hidden_states)

    monkeypatch.setattr(type(model), "encode_tokens", spy)
    torch.manual_seed(0)
    assert torch.equal(model(batch).loss, expected)
    assert seen == [None]
