#!/usr/bin/env python
"""Capture legacy LoRA behaviour as golden fixtures for backwards-compat tests.

Run ONCE against unmodified upstream/main before the PEFT migration.

Outputs:
    tests/fixtures/compat/legacy_adapter_golden/   (adapter_config.json + adapter_weights.safetensors)
    tests/fixtures/compat/input_batch.pt
    tests/fixtures/compat/legacy_forward_outputs.pt
    tests/fixtures/compat/lora_weights.pt
    tests/fixtures/compat/base_weights.pt
    tests/fixtures/compat/public_surface.json
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "compat"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Tiny model matching the Gliner2Internal test helpers
# ---------------------------------------------------------------------------

class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(8, 8, bias=False)
        self.key = nn.Linear(8, 8, bias=False)
        self.value = nn.Linear(8, 8, bias=False)
        self.dense = nn.Linear(8, 8, bias=False)
        self.other = nn.Linear(8, 8, bias=False)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TinyEncoder()
        self.classifier = nn.Linear(8, 4, bias=False)
        self.span_rep = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder.query(x) + self.encoder.value(x)
        return self.classifier(enc)


# ---------------------------------------------------------------------------
# 2. Seed, build, apply legacy LoRA, capture
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(42)
    model = TinyModel()

    torch.save(
        {n: p.data.clone() for n, p in model.named_parameters()},
        FIXTURE_DIR / "base_weights.pt",
    )

    input_batch = torch.randn(2, 8)
    torch.save(input_batch, FIXTURE_DIR / "input_batch.pt")

    from gliner2.training.lora import (
        LoRAConfig,
        apply_lora_to_model,
        save_lora_adapter,
        LoRALayer,
    )

    cfg = LoRAConfig(enabled=True, r=4, alpha=8.0, dropout=0.0, target_modules=["encoder"])
    model, lora_layers = apply_lora_to_model(model, cfg)

    lora_weight_dict = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALayer):
            lora_weight_dict[f"{name}.lora_A"] = mod.lora_A.data.clone()
            lora_weight_dict[f"{name}.lora_B"] = mod.lora_B.data.clone()
    torch.save(lora_weight_dict, FIXTURE_DIR / "lora_weights.pt")

    with torch.no_grad():
        outputs = model(input_batch)
    torch.save(outputs, FIXTURE_DIR / "legacy_forward_outputs.pt")

    adapter_dir = FIXTURE_DIR / "legacy_adapter_golden"
    save_lora_adapter(model, adapter_dir)
    print(f"Saved legacy adapter to {adapter_dir}")
    print(f"  Files: {sorted(p.name for p in adapter_dir.iterdir())}")

    # ---------------------------------------------------------------------------
    # 3. Public surface snapshot
    # ---------------------------------------------------------------------------
    import gliner2
    from gliner2.model import Extractor
    from gliner2.training.trainer import TrainingConfig

    surface: dict = {"__init__exports": {}, "extractor_methods": {}, "training_config_lora_fields": {}}

    init_symbols = [
        "LoRAConfig", "LoRAAdapterConfig", "LoRALayer",
        "load_lora_adapter", "save_lora_adapter", "unload_lora_adapter",
        "has_lora_adapter", "apply_lora_to_model", "merge_lora_weights",
        "unmerge_lora_weights",
    ]
    for sym_name in init_symbols:
        obj = getattr(gliner2, sym_name, None)
        if obj is None:
            surface["__init__exports"][sym_name] = None
            continue
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            sig = "<no signature>"
        surface["__init__exports"][sym_name] = sig

    for method_name in [
        "load_adapter", "unload_adapter", "merge_lora",
        "save_adapter", "has_adapter", "adapter_config", "save_pretrained",
    ]:
        obj = getattr(Extractor, method_name, None)
        if obj is None:
            surface["extractor_methods"][method_name] = None
            continue
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            sig = "<no signature>"
        surface["extractor_methods"][method_name] = sig

    import dataclasses
    tc = TrainingConfig.__dataclass_fields__
    for field_name in [
        "use_lora", "lora_r", "lora_alpha", "lora_dropout",
        "lora_target_modules", "save_adapter_only",
    ]:
        if field_name in tc:
            f = tc[field_name]
            if f.default is not dataclasses.MISSING:
                default_val = f.default
            elif f.default_factory is not dataclasses.MISSING:
                default_val = f.default_factory()
            else:
                default_val = None
            surface["training_config_lora_fields"][field_name] = {
                "type": str(f.type),
                "default": repr(default_val),
            }

    surface_path = FIXTURE_DIR / "public_surface.json"
    with open(surface_path, "w") as fh:
        json.dump(surface, fh, indent=2, sort_keys=True)
    print(f"Saved public surface to {surface_path}")

    print("\nDone. Fixtures written to:", FIXTURE_DIR)


if __name__ == "__main__":
    main()
