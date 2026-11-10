"""Unit tests for GLiNER2 LoRA integration via PEFT."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from peft import PeftModel
from unittest.mock import MagicMock, patch

from gliner2.training.lora import _resolve_targets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TinyEncoder(nn.Module):
    """Minimal encoder-like module with attention projection names."""

    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(8, 8, bias=False)
        self.key = nn.Linear(8, 8, bias=False)
        self.value = nn.Linear(8, 8, bias=False)
        self.dense = nn.Linear(8, 8, bias=False)
        self.other = nn.Linear(8, 8, bias=False)


class TinyModel(nn.Module):
    """Minimal GLiNER2-shaped model for LoRA tests."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = TinyEncoder()
        self.classifier = nn.Linear(8, 4, bias=False)
        self.span_rep = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder.query(x) + self.encoder.value(x)
        return self.classifier(enc)


# ---------------------------------------------------------------------------
# _resolve_targets
# ---------------------------------------------------------------------------

def test_resolve_targets_encoder_matches_attention_projections() -> None:
    """'encoder' target should resolve to query/key/value/dense only."""
    model = TinyModel()
    targets = _resolve_targets(model, ["encoder"])
    assert set(targets) == {"encoder.query", "encoder.key", "encoder.value", "encoder.dense"}
    assert "encoder.other" not in targets


def test_resolve_targets_specific_encoder_submodule() -> None:
    """'encoder.query' target should resolve to only that layer."""
    model = TinyModel()
    targets = _resolve_targets(model, ["encoder.query"])
    assert targets == ["encoder.query"]


def test_resolve_targets_task_module_classifier() -> None:
    """'classifier' task module should resolve to the classifier layer."""
    model = TinyModel()
    targets = _resolve_targets(model, ["classifier"])
    assert targets == ["classifier"]


def test_resolve_targets_multiple_groups() -> None:
    """Multiple targets should union their selections."""
    model = TinyModel()
    targets = _resolve_targets(model, ["encoder", "classifier"])
    assert "encoder.query" in targets
    assert "classifier" in targets


def test_resolve_targets_unknown_target_returns_empty() -> None:
    """An unrecognised target name should resolve to nothing."""
    model = TinyModel()
    targets = _resolve_targets(model, ["nonexistent_module"])
    assert targets == []


def test_resolve_targets_returns_sorted_unique_list() -> None:
    """Result should be sorted and contain no duplicates."""
    model = TinyModel()
    targets = _resolve_targets(model, ["encoder", "encoder"])
    assert targets == sorted(set(targets))
    assert len(targets) == len(set(targets))


# ---------------------------------------------------------------------------
# Extractor.apply_lora (tested via a thin stand-in to avoid loading weights)
# ---------------------------------------------------------------------------

def test_apply_lora_returns_peft_model() -> None:
    """apply_lora should wrap the model in a PeftModel."""
    model = TinyModel()
    peft_model = _apply_lora_helper(model, r=4, alpha=8.0, targets=["encoder"])
    assert isinstance(peft_model, PeftModel)


def test_apply_lora_only_lora_params_require_grad() -> None:
    """After apply_lora, only LoRA parameters should have requires_grad."""
    model = TinyModel()
    peft_model = _apply_lora_helper(model, r=4, alpha=8.0, targets=["encoder"])
    trainable_names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in trainable_names)


def test_apply_lora_adapter_save_load_round_trip(tmp_path) -> None:
    """Saved adapter weights should reload exactly onto a fresh model."""
    model = TinyModel()
    peft_model = _apply_lora_helper(model, r=2, alpha=4.0, targets=["encoder"])
    for p in peft_model.parameters():
        if p.requires_grad:
            nn.init.constant_(p, 0.5)

    peft_model.save_pretrained(str(tmp_path))
    fresh = TinyModel()
    loaded = PeftModel.from_pretrained(fresh, str(tmp_path))

    orig_sd = {n: p for n, p in peft_model.named_parameters() if "lora_" in n}
    load_sd = {n: p for n, p in loaded.named_parameters() if "lora_" in n}
    assert orig_sd.keys() == load_sd.keys()
    for key in orig_sd:
        assert torch.allclose(orig_sd[key], load_sd[key])


def test_apply_lora_merge_and_unload_removes_adapter(tmp_path) -> None:
    """merge_and_unload should produce a plain model with no adapter layers."""
    model = TinyModel()
    peft_model = _apply_lora_helper(model, r=2, alpha=4.0, targets=["encoder"])
    merged = peft_model.merge_and_unload()
    assert not isinstance(merged, PeftModel)
    assert isinstance(merged.classifier, nn.Linear)


# ---------------------------------------------------------------------------
# unload_lora_adapter() when `model` itself is not a PeftModel
#
# Extractor.load_adapter() (gliner2/models/span/model.py,
# gliner2/models/boundary/model.py) calls
# gliner2.training.lora.load_lora_adapter(self, adapter_path), which for the
# dual-format ("adapter_config.json" + "adapter_weights.safetensors") save
# path calls get_peft_model(self, ...). get_peft_model() injects LoraLayer
# modules directly into `self`'s own submodule tree in place, but the
# PeftModel wrapper object it returns is discarded -- `self` is never
# reassigned to that wrapper, so `self` stays a plain nn.Module (never an
# isinstance of PeftModel) even though it now contains live LoraLayer
# submodules. Extractor.unload_adapter() then calls
# unload_lora_adapter(self), whose only removal path was
# `if isinstance(model, PeftModel): model.unload()` -- always False here, so
# it silently no-opped: has_adapter/_lora_layers bookkeeping reset to "no
# adapter", but the injected LoraLayer modules kept running in every
# subsequent forward pass with no error or warning.
# ---------------------------------------------------------------------------

def test_unload_lora_adapter_restores_base_layers_when_model_is_not_a_peftmodel() -> None:
    """unload_lora_adapter(model) must unwrap in-place-injected LoraLayers
    even when `model` itself was never turned into a PeftModel instance."""
    from peft.tuners.lora.layer import LoraLayer
    from gliner2.training.lora import unload_lora_adapter

    model = TinyModel()
    original_query = model.encoder.query
    original_value = model.encoder.value

    _apply_lora_helper(model, r=2, alpha=4.0, targets=["encoder"])
    assert not isinstance(model, PeftModel)
    assert isinstance(model.encoder.query, LoraLayer)

    count = unload_lora_adapter(model)

    assert count > 0
    assert not any(isinstance(m, LoraLayer) for m in model.modules())
    # The original frozen base layers -- not new/reconstructed ones -- are
    # what's left in place; LoRA never touches base-layer weights, so
    # identity (not just type) is what proves nothing was mutated.
    assert model.encoder.query is original_query
    assert model.encoder.value is original_value


def test_load_then_unload_lora_adapter_round_trip_restores_original_output(tmp_path) -> None:
    """End-to-end dual-format save/load/unload must return the model to its
    exact pre-adapter behavior -- this is the actual user-facing regression:
    Extractor.load_adapter()/.unload_adapter() on a saved (adapter_only)
    checkpoint, as documented in adapter-switching.md."""
    from gliner2.training.lora import save_lora_adapter, load_lora_adapter, unload_lora_adapter

    torch.manual_seed(0)
    base_state = TinyModel().state_dict()  # frozen base weights, shared below

    model = TinyModel()
    model.load_state_dict(base_state)
    x = torch.randn(2, 8)
    with torch.no_grad():
        baseline = model(x).clone()

    peft_model = _apply_lora_helper(model, r=2, alpha=4.0, targets=["encoder"])
    with torch.no_grad():
        for p in peft_model.parameters():
            if p.requires_grad:
                nn.init.constant_(p, 0.3)  # non-trivial adapter delta
    save_lora_adapter(peft_model, tmp_path)

    fresh = TinyModel()
    fresh.load_state_dict(base_state)  # same base weights as `model`'s baseline
    load_lora_adapter(fresh, tmp_path, auto_unload=True)
    with torch.no_grad():
        adapted = fresh(x)
    assert not torch.allclose(adapted, baseline), "adapter should change output"

    unload_lora_adapter(fresh)
    with torch.no_grad():
        restored = fresh(x)
    assert torch.allclose(restored, baseline), (
        "unload_lora_adapter() must restore the exact pre-adapter output, "
        "not silently leave the injected LoraLayers active"
    )


# ---------------------------------------------------------------------------
# Private helper (avoids loading real Extractor weights in unit tests)
# ---------------------------------------------------------------------------

def _apply_lora_helper(model: nn.Module, r: int = 8, alpha: float = 16.0,
                       dropout: float = 0.0, targets: list[str] | None = None,
                       use_dora: bool = False) -> PeftModel:
    """Mirror Extractor.apply_lora logic for use with TinyModel."""
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=_resolve_targets(model, targets or ["encoder"]),
        bias="none",
        use_dora=use_dora,
    )
    return get_peft_model(model, config)
