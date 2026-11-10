"""Validate native trained artifacts before copying into optimizer-owned tensors."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file


def _normalized(config: dict, ignored: set[str]) -> dict:
    return json.loads(json.dumps({key: value for key, value in config.items() if key not in ignored},
                                default=lambda value: sorted(value) if isinstance(value, set) else value.value))


def _read_tensors(directory: Path, stem: str) -> dict[str, torch.Tensor]:
    safe = directory / f"{stem}.safetensors"
    binary = directory / ("pytorch_model.bin" if stem == "model" else f"{stem}.bin")
    if safe.is_file():
        return load_file(str(safe), device="cpu")
    if binary.is_file():
        return torch.load(binary, map_location="cpu", weights_only=True)
    index_path = directory / f"{safe.name}.index.json"
    if not index_path.is_file():
        index_path = directory / f"{binary.name}.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("native artifact has no tensor shard index")
    result = {}
    for filename in sorted(set(weight_map.values())):
        path = directory / filename
        if Path(filename).name != filename or path.is_symlink():
            raise ValueError("native artifact shard must be a local regular file")
        tensors = (load_file(str(path), device="cpu") if filename.endswith(".safetensors")
                   else torch.load(path, map_location="cpu", weights_only=True))
        if set(result) & set(tensors) or any(weight_map.get(key) != filename for key in tensors):
            raise ValueError("native artifact shard index does not match tensors")
        result.update(tensors)
    if set(result) != set(weight_map):
        raise ValueError("native artifact shard index has missing tensors")
    return result


def native_weights(model: torch.nn.Module, directory: Path) -> dict[str, torch.Tensor]:
    """Return a complete validated state mapping; never replace live parameters.

    The service must authorize the artifact and its base-model provenance before
    this call. PEFT files contain adapter tensors only, not proof of base identity.
    """
    current = model.state_dict()
    if (directory / "adapter_config.json").is_file():
        if not isinstance(model, PeftModel):
            raise ValueError("an adapter artifact requires a matching LoRA session")
        adapter = PeftConfig.from_pretrained(str(directory), local_files_only=True)
        ignored = {"inference_mode", "base_model_name_or_path", "revision", "auto_mapping"}
        if _normalized(adapter.to_dict(), ignored) != _normalized(model.peft_config["default"].to_dict(), ignored):
            raise ValueError("native adapter topology does not match session")
        incoming = _read_tensors(directory, "adapter_model")
        expected = get_peft_model_state_dict(model)
        destinations = {key.replace(".default.", "."): key for key in current if "lora_" in key}
    else:
        config = json.loads((directory / "config.json").read_text())
        ignored = {"_name_or_path", "architectures", "transformers_version", "dtype", "torch_dtype"}
        # Rehydrate defaults omitted by Transformers' diff-format config writer.
        restored_config = type(model.config).from_dict(config).to_dict()
        if _normalized(restored_config, ignored) != _normalized(model.config.to_dict(), ignored):
            raise ValueError("native model configuration does not match session")
        incoming = _read_tensors(directory, "model")
        if isinstance(model, PeftModel):
            destinations = {key.removeprefix("base_model.model.").replace(".base_layer.", "."): key
                            for key in current if "lora_" not in key}
        else:
            destinations = {key: key for key in current}
        expected = {key: current[target] for key, target in destinations.items()}
    if set(incoming) != set(expected) or set(destinations) != set(expected):
        raise ValueError("native artifact tensor names do not match session")
    if any(not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
           or value.is_floating_point() != expected[key].is_floating_point()
           or (value.is_floating_point() and not torch.isfinite(value).all()) for key, value in incoming.items()):
        raise ValueError("native artifact tensor shape or values do not match session")
    return current | {destinations[key]: value for key, value in incoming.items()}
