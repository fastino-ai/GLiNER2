"""Tests for extractor checkpoint dependency compatibility."""

from unittest.mock import call

import pytest
import torch

from gliner2 import processor
from gliner2.models import base


def test_legacy_extra_special_tokens_are_normalized(monkeypatch):
    tokenizer = object()
    calls = []

    def fake_load(repo_or_dir, **kwargs):
        calls.append(call(repo_or_dir, **kwargs))
        if len(calls) == 1:
            raise AttributeError("'list' object has no attribute 'keys'")
        return tokenizer

    monkeypatch.setattr(base.AutoTokenizer, "from_pretrained", fake_load)

    with pytest.warns(UserWarning, match="legacy list-valued"):
        result = base.load_extractor_tokenizer("checkpoint")

    assert result is tokenizer
    assert calls == [
        call("checkpoint"),
        call("checkpoint", extra_special_tokens={}),
    ]


def test_unrelated_attribute_error_is_not_hidden(monkeypatch):
    def fake_load(repo_or_dir, **kwargs):
        raise AttributeError("unrelated tokenizer failure")

    monkeypatch.setattr(base.AutoTokenizer, "from_pretrained", fake_load)

    with pytest.raises(AttributeError, match="unrelated tokenizer failure"):
        base.load_extractor_tokenizer("checkpoint")


def _raise_tiktoken_error(model_name, **kwargs):
    raise ValueError("`tiktoken` is required to read a `tiktoken` file.")


def test_missing_sentencepiece_dependencies_are_named(monkeypatch):
    monkeypatch.setattr(processor.AutoTokenizer, "from_pretrained", _raise_tiktoken_error)
    monkeypatch.setattr(
        processor, "_module_available", lambda module: module != "google.protobuf"
    )

    with pytest.raises(ImportError, match=r"missing protobuf\..*gliner2\[train\]") as info:
        processor._load_encoder_tokenizer("microsoft/deberta-v3-base")

    assert isinstance(info.value.__cause__, ValueError)


def test_tokenizer_error_is_not_hidden_when_dependencies_exist(monkeypatch):
    monkeypatch.setattr(processor.AutoTokenizer, "from_pretrained", _raise_tiktoken_error)
    monkeypatch.setattr(processor, "_module_available", lambda module: True)

    with pytest.raises(ValueError, match="tiktoken"):
        processor._load_encoder_tokenizer("microsoft/deberta-v3-base")


def test_module_available_handles_missing_parent_package():
    assert processor._module_available("gliner2_missing_parent.child") is False


def test_encoder_is_normalized_to_task_head_dtype(monkeypatch):
    encoder = torch.nn.Linear(2, 2).half()

    monkeypatch.setattr(
        base.AutoModel,
        "from_config",
        lambda config, **kwargs: encoder,
    )

    result = base.BaseExtractorModel._load_encoder(
        "unused",
        encoder_config=object(),
        attn_implementation="eager",
    )

    assert next(result.parameters()).dtype == torch.float32
