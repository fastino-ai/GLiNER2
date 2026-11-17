"""Tests for loading PEFT LoRA adapter directories and subfolder checkpoints via AutoExtractor."""
from __future__ import annotations

import json

import pytest
import torch

from gliner2 import AutoExtractor
from gliner2.training.lora import save_adapter_checkpoint
from tests.fixtures.tiny_boundary_checkpoint import (
    build_tiny_boundary_model,
    save_tiny_boundary_checkpoint,
)


def test_save_and_load_adapter_directory_preserves_weights_and_config(tmp_path):
    base_dir = save_tiny_boundary_checkpoint(tmp_path / "base")
    model = build_tiny_boundary_model().eval()
    model.name_or_path = str(base_dir)
    adapted = model.apply_lora(r=4, alpha=8, targets=["encoder", "boundary_head"]).eval()
    for name, param in adapted.named_parameters():
        if "lora_B" in name:
            param.data.fill_(0.25)
    adapted.get_base_model().config.max_len = 512

    adapter_dir = tmp_path / "adapter"
    save_adapter_checkpoint(adapted, adapter_dir)
    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / "adapter_model.safetensors").is_file()
    assert (adapter_dir / "config.json").is_file()
    assert not (adapter_dir / "model.safetensors").exists()

    loaded = AutoExtractor.from_pretrained(str(adapter_dir), map_location="cpu").eval()
    assert loaded.config.max_len == 512
    for name, param in loaded.named_parameters():
        if "lora_B" in name:
            torch.testing.assert_close(
                param.detach().cpu(),
                torch.full_like(param.detach().cpu(), 0.25),
            )

    expected = adapted.batch_extract(["john works at google"], [{"entities": ["person"]}], threshold=0.0)
    actual = loaded.batch_extract(["john works at google"], [{"entities": ["person"]}], threshold=0.0)
    assert actual == expected

    # A bare PEFT directory without an extractor config.json still loads using the base config
    (adapter_dir / "config.json").unlink()
    loaded_bare = AutoExtractor.from_pretrained(str(adapter_dir), map_location="cpu").eval()
    assert loaded_bare.batch_extract(["john works at google"], [{"entities": ["person"]}], threshold=0.0) == expected


def test_subfolder_checkpoint_forwards_hub_kwargs_to_model_loader(tmp_path, monkeypatch):
    save_tiny_boundary_checkpoint(tmp_path / "repo" / "checkpoint-epoch-1")
    loaded = AutoExtractor.from_pretrained(
        str(tmp_path / "repo"), subfolder="checkpoint-epoch-1", map_location="cpu"
    )
    assert loaded.config.architecture == "boundary"

    # Also verify non-directory Hub path forwards subfolder to model_class.from_pretrained
    seen = {}
    from gliner2.configuration import ExtractorConfig

    cfg = ExtractorConfig(model_name="hf-internal-testing/tiny-random-bert", architecture="boundary")
    monkeypatch.setattr("gliner2.auto._load_config", lambda src, hk: cfg)
    cls = AutoExtractor._resolve_class("boundary")
    monkeypatch.setattr(cls, "from_pretrained", classmethod(lambda c, src, *a, **kw: seen.update(kw) or "ok"))
    assert AutoExtractor.from_pretrained("owner/repo", subfolder="step-100", revision="v1") == "ok"
    assert seen["subfolder"] == "step-100"
    assert seen["revision"] == "v1"


def test_empty_dir_and_missing_base_id_raise_clear_errors(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        AutoExtractor.from_pretrained(str(empty))

    bad_adapter = tmp_path / "bad_adapter"
    bad_adapter.mkdir()
    (bad_adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": ""}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="base_model_name_or_path"):
        AutoExtractor.from_pretrained(str(bad_adapter))
