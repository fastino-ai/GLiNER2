"""
Custom LoRA (Low-Rank Adaptation) Implementation for GLiNER2
=============================================================

Parameter-efficient fine-tuning by injecting trainable low-rank matrices
into frozen linear layers of the encoder.

Based on: "LoRA: Low-Rank Adaptation of Large Language Models"
Paper: https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


# =============================================================================
# LoRA Configuration
# =============================================================================


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA parameter-efficient fine-tuning.

    Parameters
    ----------
    enabled : bool
        Whether LoRA is enabled.
    r : int
        Rank of the low-rank decomposition (bottleneck dimension).
        Higher r = more parameters but better approximation.
        Typical values: 4, 8, 16, 32, 64.
    alpha : float
        Scaling factor for LoRA updates. Final scaling is alpha/r.
        Typical values: 8, 16, 32 (often 2*r).
    dropout : float
        Dropout probability applied to LoRA path.
    target_modules : List[str]
        Module names to apply LoRA to. Supported module groups:

        - "encoder" - Applies LoRA to query, key, value, dense layers within encoder
        - "encoder.query" - Only query layers in encoder
        - "encoder.key" - Only key layers in encoder
        - "encoder.value" - Only value layers in encoder
        - "encoder.dense" - Only dense layers in encoder
        - "span_rep" - Applies LoRA to ALL linear layers within span_rep
        - "classifier" - Applies LoRA to ALL linear layers within classifier
        - "count_embed" - Applies LoRA to ALL linear layers within count_embed
        - "count_pred" - Applies LoRA to ALL linear layers within count_pred

        Examples:
        - ["encoder"] - all encoder layers (query, key, value, dense)
        - ["encoder.query", "encoder.key", "encoder.value"] - only attention layers
        - ["encoder.dense"] - only dense (FFN) layers in encoder
        - ["encoder", "span_rep", "classifier"] - encoder + task heads
        - ["classifier"] - classifier fine-tuning only
    """

    enabled: bool = False
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["encoder"])

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {self.r}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be > 0, got {self.alpha}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}")
        if self.enabled and not self.target_modules:
            raise ValueError("target_modules cannot be empty when LoRA is enabled")


@dataclass
class LoRAAdapterConfig:
    """
    Configuration for a saved LoRA adapter.

    This is the config that gets saved with adapter-only checkpoints.
    """

    adapter_type: str = "lora"
    adapter_version: str = "1.0"
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=list)
    created_at: str = ""

    def save(self, path: Union[str, Path]) -> None:
        """Save adapter config to JSON file."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config_path = path / "adapter_config.json"

        # Set created_at if not set
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"

        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=2)

        logger.info(f"Saved adapter config to {config_path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LoRAAdapterConfig":
        """Load adapter config from JSON file or directory."""
        path = Path(path)

        # If path is a directory, look for adapter_config.json
        if path.is_dir():
            config_path = path / "adapter_config.json"
        else:
            config_path = path

        if not config_path.exists():
            raise FileNotFoundError(f"Adapter config not found at {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    @classmethod
    def is_adapter_path(cls, path: Union[str, Path]) -> bool:
        """Check if path contains an adapter."""
        path = Path(path)

        # Check for adapter_config.json
        if path.is_dir():
            return (path / "adapter_config.json").exists()
        else:
            return path.name == "adapter_config.json" and path.exists()


# =============================================================================
# LoRA Layer
# =============================================================================


class LoRALayer(nn.Module):
    """
    LoRA-enhanced Linear layer.

    Computes: output = W*x + (B*A*x) * scaling
    Where:
        - W is the frozen original weight
        - A, B are trainable low-rank matrices
        - scaling = alpha / r

    Parameters
    ----------
    base_layer : nn.Linear
        Original linear layer (will be frozen).
    r : int
        Rank of low-rank decomposition.
    alpha : float
        LoRA scaling factor.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Store frozen base layer
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Get device from base layer to ensure LoRA parameters are on same device
        device = next(base_layer.parameters()).device

        # LoRA low-rank matrices
        # A: (r, in_features) - initialized with small random values
        # B: (out_features, r) - initialized to zero (no change at start)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device))

        # Initialize A with Kaiming uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero-initialized

        # Dropout
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Flag to track if weights are merged
        self.merged = False

    # Expose base layer attributes for compatibility
    @property
    def weight(self):
        """Expose weight from base layer for compatibility."""
        return self.base_layer.weight

    @property
    def bias(self):
        """Expose bias from base layer for compatibility."""
        return self.base_layer.bias

    @property
    def in_features(self):
        """Expose in_features from base layer."""
        return self.base_layer.in_features

    @property
    def out_features(self):
        """Expose out_features from base layer."""
        return self.base_layer.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., in_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., out_features).
        """
        # Base output from frozen weights
        base_output = self.base_layer(x)

        if self.merged:
            # Weights already merged, just use base layer
            return base_output

        # LoRA path: x -> dropout -> A -> B -> scale
        # Equivalent to: (x @ A.T) @ B.T * scaling
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T

        return base_output + lora_output * self.scaling

    def merge_weights(self):
        """Merge LoRA weights (B @ A) into base layer weights."""
        if self.merged:
            # Already merged, silently skip
            return

        with torch.no_grad():
            # Compute LoRA contribution: B @ A * scaling
            lora_weight = (self.lora_B @ self.lora_A) * self.scaling
            # Add to base weight
            self.base_layer.weight.data += lora_weight

        self.merged = True
        logger.debug(f"Merged LoRA weights (r={self.r}) into base layer")

    def unmerge_weights(self):
        """Separate LoRA weights from base layer (reverse of merge)."""
        if not self.merged:
            # Not merged, silently skip
            return

        with torch.no_grad():
            # Subtract LoRA contribution
            lora_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= lora_weight

        self.merged = False
        logger.debug(f"Unmerged LoRA weights (r={self.r}) from base layer")

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}, merged={self.merged}"


# =============================================================================
# LoRA Application Functions
# =============================================================================

# Module-specific patterns for LoRA application
ENCODER_PATTERNS = ["query", "key", "value", "dense"]
ALL_LINEAR_MODULES = ["span_rep", "classifier", "count_embed", "count_pred"]


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
) -> Tuple[nn.Module, Dict[str, LoRALayer]]:
    """
    Apply LoRA to linear layers based on module groups in target_modules.

    Module group behavior:
    - "encoder": Applies LoRA to query, key, value, dense layers within encoder
    - "encoder.query": Only query layers in encoder
    - "encoder.key": Only key layers in encoder
    - "encoder.value": Only value layers in encoder
    - "encoder.dense": Only dense layers in encoder
    - "span_rep", "classifier", "count_embed", "count_pred": Applies LoRA to ALL linear layers

    Parameters
    ----------
    model : nn.Module
        The model to apply LoRA to.
    config : LoRAConfig
        LoRA configuration.

    Returns
    -------
    model : nn.Module
        Modified model with LoRA layers.
    lora_layers : Dict[str, LoRALayer]
        Dictionary mapping layer names to LoRA layers.
    """
    if not config.enabled:
        logger.info("LoRA is disabled, skipping application")
        return model, {}

    lora_layers = {}

    def _should_apply_lora(local_name: str, full_path: str) -> bool:
        """
        Check if LoRA should be applied based on module groups.

        Args:
            local_name: Local module name (e.g., "query", "linear")
            full_path: Full path from model root (e.g., "encoder.layer.0.attention.self.query")

        Returns:
            True if LoRA should be applied to this layer
        """
        for target in config.target_modules:
            if target == "encoder":
                # For encoder, apply only to specific patterns
                if full_path.startswith("encoder."):
                    # Check if local name matches encoder patterns
                    if any(pattern in local_name for pattern in ENCODER_PATTERNS):
                        return True
            elif target.startswith("encoder."):
                # Specific encoder layer (e.g., "encoder.query", "encoder.dense")
                layer_name = target.split(".", 1)[
                    1
                ]  # Extract "query" from "encoder.query"
                if full_path.startswith("encoder.") and layer_name in local_name:
                    return True
            elif target in ALL_LINEAR_MODULES:
                # For these modules, apply to ALL linear layers within
                if full_path.startswith(f"{target}."):
                    return True
        return False

    # Recursively find and replace modules
    def _inject_lora_recursive(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Apply LoRA to matching Linear layers
            if isinstance(child, nn.Linear) and _should_apply_lora(name, full_name):
                # Replace with LoRA layer
                lora_layer = LoRALayer(
                    base_layer=child,
                    r=config.r,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                setattr(module, name, lora_layer)
                lora_layers[full_name] = lora_layer

                logger.debug(
                    f"Applied LoRA to {full_name} "
                    f"(in={child.in_features}, out={child.out_features})"
                )
            else:
                # Recurse into child
                _inject_lora_recursive(child, full_name)

    _inject_lora_recursive(model)

    if not lora_layers:
        logger.warning(
            f"No LoRA layers were applied. Target modules {config.target_modules} "
            f"not found. Check your target_modules configuration."
        )
    else:
        logger.info(f"Applied LoRA to {len(lora_layers)} layers")

    return model, lora_layers


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Extract all LoRA parameters (lora_A and lora_B) from model.

    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.

    Returns
    -------
    List[nn.Parameter]
        List of LoRA parameters.
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Get state dict containing only LoRA parameters.

    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.

    Returns
    -------
    Dict[str, torch.Tensor]
        State dict with LoRA parameters only.
    """
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state[f"{name}.lora_A"] = module.lora_A.data
            lora_state[f"{name}.lora_B"] = module.lora_B.data
    return lora_state


def merge_lora_weights(model: nn.Module) -> int:
    """
    Merge all LoRA weights into base layers and remove LoRA structure.

    After calling this, the model will have standard Linear layers with
    merged weights. LoRA adapters are removed from the model.

    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.

    Returns
    -------
    int
        Number of layers merged and removed.
    """
    count = 0
    already_merged = 0
    for module in model.modules():
        if isinstance(module, LoRALayer):
            if not module.merged:
                module.merge_weights()
                count += 1
            else:
                already_merged += 1

    if count > 0:
        logger.debug(f"Merged LoRA weights in {count} layers")
    if already_merged > 0:
        logger.debug(f"Skipped {already_merged} layers (already merged)")

    # Remove LoRA layers after merging
    if count > 0 or already_merged > 0:
        remove_lora_from_model(model)
        logger.info("Merged and removed LoRA layers from model")

    return count


def unmerge_lora_weights(model: nn.Module) -> int:
    """
    Unmerge all LoRA weights from their base layers.

    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.

    Returns
    -------
    int
        Number of layers unmerged.
    """
    count = 0
    not_merged = 0
    for module in model.modules():
        if isinstance(module, LoRALayer):
            if module.merged:
                module.unmerge_weights()
                count += 1
            else:
                not_merged += 1

    if count > 0:
        logger.debug(f"Unmerged LoRA weights in {count} layers")
    if not_merged > 0:
        logger.debug(f"Skipped {not_merged} layers (not merged)")
    return count


def count_lora_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Count LoRA parameters vs total parameters.

    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.

    Returns
    -------
    lora_params : int
        Number of trainable LoRA parameters.
    total_params : int
        Total number of model parameters.
    percentage : float
        Percentage of trainable parameters.
    """
    lora_params = sum(p.numel() for p in get_lora_parameters(model))
    total_params = sum(p.numel() for p in model.parameters())
    percentage = (lora_params / total_params * 100) if total_params > 0 else 0.0

    return lora_params, total_params, percentage


def print_lora_info(model: nn.Module, config: LoRAConfig):
    """
    Print detailed LoRA configuration and parameter statistics.

    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.
    config : LoRAConfig
        LoRA configuration.
    """
    lora_params, total_params, percentage = count_lora_parameters(model)

    # Count LoRA layers
    num_lora_layers = sum(1 for m in model.modules() if isinstance(m, LoRALayer))

    print("=" * 70)
    print("🔧 LoRA Configuration")
    print("=" * 70)
    print(f"Enabled            : {config.enabled}")
    print(f"Rank (r)           : {config.r}")
    print(f"Alpha              : {config.alpha}")
    print(f"Scaling (α/r)      : {config.alpha / config.r:.4f}")
    print(f"Dropout            : {config.dropout}")
    print(f"Target modules     : {', '.join(config.target_modules)}")
    print(f"LoRA layers        : {num_lora_layers}")
    print("-" * 70)
    print(
        f"Trainable params   : {lora_params:,} / {total_params:,} ({percentage:.2f}%)"
    )
    print(f"Memory savings     : ~{100 - percentage:.1f}% fewer gradients")
    print("=" * 70)


def remove_lora_from_model(model: nn.Module) -> nn.Module:
    """
    Remove LoRA layers and restore original Linear layers.
    Useful for inference with merged weights.

    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.

    Returns
    -------
    nn.Module
        Model with LoRA layers replaced by standard Linear layers.
    """

    def _remove_lora_recursive(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, LoRALayer):
                # Ensure weights are merged
                if not child.merged:
                    child.merge_weights()
                # Replace LoRALayer with its base layer
                setattr(module, name, child.base_layer)
                logger.debug(f"Removed LoRA from {name}, restored base layer")
            else:
                _remove_lora_recursive(child)

    _remove_lora_recursive(model)
    logger.info("Removed all LoRA layers from model")
    return model


# =============================================================================
# Adapter Management Functions
# =============================================================================


def save_lora_adapter(
    model: nn.Module,
    save_path: Union[str, Path],
) -> None:
    """
    Save only LoRA adapter weights and config.

    Args:
        model: Model with LoRA layers (must NOT be merged)
        save_path: Directory to save adapter

    Saves:
        - adapter_config.json
        - adapter_weights.safetensors
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Collect LoRA weights and config
    lora_state = {}
    lora_config = None

    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            if module.merged:
                raise ValueError(
                    "Cannot save adapter with merged weights. "
                    "Call unmerge_lora_weights() first."
                )
            # Save LoRA matrices with full path from model root
            lora_state[f"{name}.lora_A"] = module.lora_A.data
            lora_state[f"{name}.lora_B"] = module.lora_B.data

            # Extract config from first LoRA layer
            if lora_config is None:
                lora_config = {
                    "lora_r": module.r,
                    "lora_alpha": module.alpha,
                    "lora_dropout": (
                        module.lora_dropout.p
                        if hasattr(module.lora_dropout, "p")
                        else 0.0
                    ),
                }

    if not lora_state:
        raise ValueError("No LoRA layers found in model")

    # Save weights
    weights_path = save_path / "adapter_weights.safetensors"
    save_file(lora_state, str(weights_path))
    logger.info(f"Saved {len(lora_state)} LoRA tensors to {weights_path}")

    # Determine target modules from layer names
    # Extract top-level module names (encoder, span_rep, classifier, etc.)
    target_modules = set()
    for key in lora_state.keys():
        # Extract first level module from full path
        # e.g., "encoder.layer.0.attention.self.query.lora_A" -> "encoder"
        # e.g., "span_rep.project_start.0.lora_A" -> "span_rep"
        parts = key.split(".")
        if len(parts) > 0:
            # Get the first level module name
            module_name = parts[0]
            target_modules.add(module_name)

    # Create and save adapter config
    adapter_config = LoRAAdapterConfig(
        adapter_type="lora",
        adapter_version="1.0",
        lora_r=lora_config["lora_r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=sorted(list(target_modules)),
        created_at=datetime.utcnow().isoformat() + "Z",
    )
    adapter_config.save(save_path)

    logger.info(f"Saved LoRA adapter to {save_path}")


def load_lora_adapter(
    model: nn.Module,
    adapter_path: Union[str, Path],
    auto_unload: bool = True,
) -> Dict[str, LoRALayer]:
    """
    Load LoRA adapter onto model.

    Args:
        model: Base model (should not have LoRA applied)
        adapter_path: Path to adapter directory
        auto_unload: If True, unload existing adapter first

    Returns:
        Dict of LoRA layers that were applied
    """
    adapter_path = Path(adapter_path)

    # Load adapter config
    adapter_config = LoRAAdapterConfig.load(adapter_path)

    # Unload existing adapter if requested
    if auto_unload and has_lora_adapter(model):
        logger.info("Unloading existing adapter before loading new one")
        unload_lora_adapter(model)

    # Load adapter weights
    weights_path = adapter_path / "adapter_weights.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Adapter weights not found at {weights_path}")

    lora_state = load_file(str(weights_path))
    logger.info(f"Loaded {len(lora_state)} LoRA tensors from {weights_path}")

    # Apply LoRA to matching layers
    lora_config = LoRAConfig(
        enabled=True,
        r=adapter_config.lora_r,
        alpha=adapter_config.lora_alpha,
        dropout=adapter_config.lora_dropout,
        target_modules=adapter_config.target_modules,
    )

    model, lora_layers = apply_lora_to_model(model, lora_config)

    # Load saved weights into LoRA layers
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_a_key = f"{name}.lora_A"
            lora_b_key = f"{name}.lora_B"

            if lora_a_key in lora_state and lora_b_key in lora_state:
                # Move loaded tensors to the same device as the module
                device = next(module.parameters()).device
                module.lora_A.data = lora_state[lora_a_key].to(device)
                module.lora_B.data = lora_state[lora_b_key].to(device)
                logger.debug(f"Loaded weights for {name}")
            else:
                logger.warning(f"No saved weights found for {name}")

    logger.info(f"Loaded LoRA adapter from {adapter_path}")
    return lora_layers


def unload_lora_adapter(model: nn.Module) -> int:
    """
    Remove all LoRA layers, restoring original Linear layers.

    Unlike remove_lora_from_model, this does NOT merge weights.
    Just removes LoRA layers entirely.

    Returns:
        Number of layers unloaded
    """
    count = 0

    def _get_parent_module(model: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
        """Get parent module and child name from full module path."""
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    # Collect all LoRA layers first (to avoid modifying dict during iteration)
    lora_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_layers.append((name, module))

    # Remove LoRA layers
    for name, lora_layer in lora_layers:
        parent, child_name = _get_parent_module(model, name)
        # Replace with original base_layer (no merge)
        setattr(parent, child_name, lora_layer.base_layer)
        count += 1
        logger.debug(f"Unloaded LoRA from {name}")

    if count > 0:
        logger.info(f"Unloaded {count} LoRA layers")

    return count


def has_lora_adapter(model: nn.Module) -> bool:
    """Check if model has LoRA layers applied."""
    for module in model.modules():
        if isinstance(module, LoRALayer):
            return True
    return False


def get_adapter_config(model: nn.Module) -> Optional[LoRAAdapterConfig]:
    """
    Get config of currently loaded adapter, if any.

    Note: This reconstructs config from LoRA layers.
    The actual adapter config is stored in model._adapter_config
    when loaded via model.load_adapter().
    """
    if not has_lora_adapter(model):
        return None

    # Extract config from first LoRA layer
    for module in model.modules():
        if isinstance(module, LoRALayer):
            target_modules = set()
            # Collect all target module groups (top-level modules)
            for name, m in model.named_modules():
                if isinstance(m, LoRALayer):
                    # Extract first level module name
                    parts = name.split(".")
                    if parts:
                        target_modules.add(parts[0])

            return LoRAAdapterConfig(
                adapter_type="lora",
                adapter_version="1.0",
                lora_r=module.r,
                lora_alpha=module.alpha,
                lora_dropout=(
                    module.lora_dropout.p if hasattr(module.lora_dropout, "p") else 0.0
                ),
                target_modules=sorted(list(target_modules)),
                created_at="",
            )

    return None
