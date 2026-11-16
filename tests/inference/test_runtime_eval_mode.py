"""``batch_extract`` puts the model in eval mode without redundant module walks."""

from __future__ import annotations

from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model

TEXT = "alice works at acme in paris"
LABELS = ["person", "company"]


def _count_train_calls(model, monkeypatch):
    calls = []
    original = model.train

    def train(mode: bool = True):
        calls.append(mode)
        return original(mode)

    monkeypatch.setattr(model, "train", train)
    return calls


def test_batch_extract_skips_eval_when_already_in_eval_mode(monkeypatch):
    model = build_tiny_span_model()
    assert not model.training
    calls = _count_train_calls(model, monkeypatch)

    model.extract_entities(TEXT, LABELS)

    assert calls == []
    assert not model.training


def test_batch_extract_switches_training_model_to_eval(monkeypatch):
    model = build_tiny_span_model()
    model.train()
    calls = _count_train_calls(model, monkeypatch)

    model.extract_entities(TEXT, LABELS)

    assert calls == [False]
    assert not any(module.training for module in model.modules())


def test_batch_extract_output_is_unchanged_by_prior_mode():
    in_eval = build_tiny_span_model().extract_entities(TEXT, LABELS, include_confidence=True)
    model = build_tiny_span_model()
    model.train()
    assert model.extract_entities(TEXT, LABELS, include_confidence=True) == in_eval
