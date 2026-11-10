"""Validate preloaded trainable PEFT adapters before constructing an optimizer."""
from __future__ import annotations

from typing import TYPE_CHECKING

from peft import PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer

from gliner2.training.lora_targets import _resolve_targets

if TYPE_CHECKING:
    from gliner2.training.trainer import TrainingConfig


def validate_loaded_adapter(model: PeftModel, config: TrainingConfig) -> None:
    """Continue the adapter's weights; never silently install a fresh adapter."""
    if not config.use_lora:
        raise ValueError("a preloaded LoRA adapter requires use_lora=True")
    if set(model.peft_config) != {"default"} or model.active_adapters != ["default"]:
        raise ValueError("LoRA continuation requires one active default adapter")
    adapter = model.peft_config["default"]
    expected_targets = set(_resolve_targets(model.get_base_model(), config.lora_target_modules))
    if (adapter.inference_mode or adapter.r != config.lora_r
            or adapter.lora_alpha != config.lora_alpha
            or adapter.lora_dropout != config.lora_dropout
            or adapter.use_dora != config.lora_use_dora
            or set(adapter.target_modules) != expected_targets
            or adapter.bias != "none" or adapter.modules_to_save
            or adapter.rank_pattern or adapter.alpha_pattern or adapter.use_rslora):
        raise ValueError("loaded LoRA adapter does not match the training configuration")
    if any(isinstance(layer, BaseTunerLayer) and (layer.disable_adapters or layer.merged)
           for layer in model.modules()):
        raise ValueError("LoRA continuation requires enabled, unmerged adapter layers")
    if any(parameter.requires_grad and "lora_" not in name
           for name, parameter in model.named_parameters()):
        raise ValueError("LoRA continuation cannot silently train base parameters")
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("load the LoRA adapter with is_trainable=True")
