"""Tests for the collator error policy and additive batch fields (PR 3)."""

from __future__ import annotations

import pytest

from gliner2.processor import SchemaTransformer


def _processor(tiny_tokenizer):
    return SchemaTransformer(tokenizer=tiny_tokenizer)


# A classification item missing ``labels`` is malformed at inference and training.
_BAD = ("hello world", {"classifications": [{"task": "t"}]})
_GOOD = ("hello world", {"entities": {"company": "companies"}})


def test_training_collator_raises_on_invalid_annotation(tiny_tokenizer):
    proc = _processor(tiny_tokenizer)
    with pytest.raises(Exception):
        proc.collate_fn_train([_BAD], error_policy="raise")


def test_skip_policy_drops_bad_record(tiny_tokenizer):
    proc = _processor(tiny_tokenizer)
    batch = proc.collate_fn_inference([_BAD], error_policy="skip")
    assert len(batch) == 0


def test_fallback_policy_is_opt_in(tiny_tokenizer):
    proc = _processor(tiny_tokenizer)
    batch = proc.collate_fn_inference([_BAD], error_policy="fallback")
    assert len(batch) == 1


def test_additive_fields_default_empty_and_survive_device_move(tiny_tokenizer):
    proc = _processor(tiny_tokenizer)
    batch = proc.collate_fn_inference([_GOOD])
    assert batch.query_layouts == ()
    assert batch.targets is None
    assert batch.model_texts == ()
    moved = batch.to("cpu")
    assert moved.input_ids.device.type == "cpu"
    assert moved.targets is None


def test_unknown_error_policy_raises(tiny_tokenizer):
    proc = _processor(tiny_tokenizer)
    with pytest.raises(ValueError):
        proc.collate_fn_inference([_GOOD], error_policy="bogus")
