"""
GLiNER2 World-Class Trainer
===========================

Production-grade training infrastructure with flexible data input.

Supported Data Formats:
-----------------------
1. Single JSONL file path (str or Path)
2. List of JSONL file paths
3. List of InputExample objects
4. TrainingDataset object
5. List of raw dict records ({"input": ..., "output": ...} format)

Basic Examples:
--------------
    >>> from gliner2.training.data import InputExample, TrainingDataset
    >>> from gliner2.training.trainer import TrainingConfig, GLiNER2Trainer
    >>>
>>> # 1. From list of InputExample
    >>> examples = [
    ...     InputExample(text="John works at Google.", entities={"person": ["John"], "company": ["Google"]}),
    ...     InputExample(text="Apple released iPhone.", entities={"company": ["Apple"], "product": ["iPhone"]}),
    ... ]
    >>> trainer = GLiNER2Trainer(model, config)
    >>> trainer.train(train_data=examples)
    >>>
>>> # 2. From JSONL file(s)
    >>> trainer.train(train_data="train.jsonl")
    >>> trainer.train(train_data=["train1.jsonl", "train2.jsonl"])
>>>
>>> # 3. From TrainingDataset
>>> dataset = TrainingDataset.load("train.jsonl")
>>> trainer.train(train_data=dataset)
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from tqdm.auto import tqdm

from gliner2.processor import SchemaTransformer, SamplingConfig
from gliner2.utils.device_errors import is_fatal_device_error
from gliner2.utils.sync_probe import count_cuda_syncs

# Import training data classes
from gliner2.training.data import (
    InputExample, TrainingDataset, DataValidationError,
    DataFormat, detect_data_format, DataLoader_Factory, TrainDataInput
)
from gliner2.training.sampler import (
    DistributedLengthGroupedSampler,
    LengthGroupedSampler,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    
    Parameters
    ----------
    output_dir : str
        Directory for saving checkpoints and logs.
    experiment_name : str
        Name of the experiment (used for logging).
    num_epochs : int
        Number of training epochs.
    max_steps : int
        Maximum training steps (-1 = determined by epochs).
    batch_size : int
        Training batch size per device.
    eval_batch_size : int
        Evaluation batch size.
    gradient_accumulation_steps : int
        Number of gradient accumulation steps.
    encoder_lr : float
        Learning rate for encoder parameters.
    task_lr : float
        Learning rate for task-specific parameters.
    weight_decay : float
        Weight decay for AdamW optimizer.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    scheduler_type : str
        LR scheduler type: "linear", "cosine", "cosine_restarts", "constant".
    warmup_ratio : float
        Warmup ratio (portion of total steps).
    warmup_steps : int
        Explicit warmup steps (overrides warmup_ratio if > 0).
    fp16 : bool
        Use FP16 mixed precision.
    bf16 : bool
        Use BF16 mixed precision.
    eval_strategy : str
        When to evaluate and save: "epoch", "steps", or "no".
    eval_steps : int
        Evaluate and save every N steps (if eval_strategy="steps").
    save_total_limit : int
        Maximum checkpoints to keep.
    save_best : bool
        Save best model based on metric.
    metric_for_best : str
        Metric to use for best model selection.
    greater_is_better : bool
        Whether higher metric is better.
    logging_steps : int
        Log every N steps (updates progress bar metrics).
    report_to_wandb : bool
        Enable Weights & Biases logging.
    wandb_project : str, optional
        W&B project name.
    early_stopping : bool
        Enable early stopping.
    early_stopping_patience : int
        Patience for early stopping.
    num_workers : int
        DataLoader workers.
    seed : int
        Random seed.
    validate_data : bool
        Validate training data before training.
    use_lora : bool
        Enable LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
    lora_r : int
        LoRA rank (bottleneck dimension). Higher = more parameters but better approximation.
        Typical values: 4, 8, 16, 32, 64.
    lora_alpha : float
        LoRA scaling factor. Final scaling is alpha/r. Typical: 2*r.
    lora_dropout : float
        Dropout probability for LoRA layers.
    lora_target_modules : List[str]
        Module groups to apply LoRA to. Options:
        - "encoder": All encoder layers (query, key, value, dense)
        - "encoder.query": Only query layers in encoder
        - "encoder.key": Only key layers in encoder
        - "encoder.value": Only value layers in encoder
        - "encoder.dense": Only dense (FFN) layers in encoder
        - "span_rep": All linear layers in span representation
        - "classifier": All linear layers in classifier head
        - "count_embed": All linear layers in count embedding
        - "count_pred": All linear layers in count prediction
        Default: All modules for maximum adaptation.
    save_adapter_only : bool
        When use_lora=True, save only adapter weights (not full model).
    """
    output_dir: str = "./output"
    experiment_name: str = "gliner2"
    num_epochs: int = 10
    max_steps: int = -1
    batch_size: int = 2
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    encoder_lr: float = 1e-5
    task_lr: float = 5e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    num_cycles: float = 0.5
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_total_limit: int = 3
    save_best: bool = True
    metric_for_best: str = "eval_loss"
    greater_is_better: bool = False
    logging_steps: int = 1
    logging_first_step: bool = True
    report_to_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    seed: int = 42
    deterministic: bool = False
    local_rank: int = -1
    debug: bool = False
    max_train_samples: int = -1
    max_eval_samples: int = -1
    validate_data: bool = True
    max_len: Optional[int] = None

    # Strict training invariants (boundary architecture; harmless for span).
    # When strict_training is True: processor/model exceptions propagate, a
    # non-finite loss raises (unless ignore_nonfinite_losses is enabled), and
    # batch-size discrepancies raise. ``skip_step_errors`` is an explicit
    # single-process recovery override for faulty micro-batches.
    strict_training: bool = True
    ignore_nonfinite_losses: bool = False
    skip_step_errors: bool = False
    allow_invalid_samples: bool = False
    log_proposal_metrics: bool = True
    gold_injection_start: float = 1.0
    gold_injection_end: float = 0.25
    gold_injection_hold_frac: float = 0.15
    diagnostics_every_n_steps: int = 0
    profile_first_n_steps: int = 0
    # Dump complete CPU-side preprocessed batches before these global steps.
    debug_global_steps: List[int] = field(default_factory=list)
    ddp_consensus_check: bool = True
    ddp_find_unused_parameters: bool = False
    ddp_static_graph: bool = True
    dry_run_recall_steps: int = 0
    gate_recall: float = 0.97
    gate_long_recall: float = 0.93
    # Gold-capacity overflow policy for boundary targets: "raise" (default,
    # no silent loss), "truncate_with_warning", or "skip_sample".
    on_capacity_exceeded: str = "raise"
    group_by_length: bool = True
    length_group_window_batches: int = 50
    compile_model: bool = False
    gradient_checkpointing: bool = False
    fused_optimizer: bool = True
    allow_tf32: bool = True
    float32_matmul_precision: str = "high"

    # LoRA Configuration (Parameter-Efficient Fine-Tuning)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_use_dora: bool = False
    lora_target_modules: List[str] = field(default_factory=lambda: ["encoder", "span_rep", "classifier", "count_embed", "count_pred"])
    save_adapter_only: bool = True  # Only applies when use_lora=True

    def __post_init__(self):
        self._precision_explicit = self.fp16 is not None or self.bf16 is not None
        if self.fp16 is None:
            self.fp16 = True
        if self.bf16 is None:
            self.bf16 = False
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")
        if not 0.0 <= self.gold_injection_start <= 1.0:
            raise ValueError("gold_injection_start must be in [0, 1]")
        if not 0.0 <= self.gold_injection_end <= 1.0:
            raise ValueError("gold_injection_end must be in [0, 1]")
        if not 0.0 <= self.gold_injection_hold_frac <= 1.0:
            raise ValueError("gold_injection_hold_frac must be in [0, 1]")
        if self.diagnostics_every_n_steps < 0:
            raise ValueError("diagnostics_every_n_steps must be >= 0")
        if self.profile_first_n_steps < 0:
            raise ValueError("profile_first_n_steps must be >= 0")
        if self.dry_run_recall_steps < 0:
            raise ValueError("dry_run_recall_steps must be >= 0")
        if not 0.0 <= self.gate_recall <= 1.0:
            raise ValueError("gate_recall must be in [0, 1]")
        if not 0.0 <= self.gate_long_recall <= 1.0:
            raise ValueError("gate_long_recall must be in [0, 1]")
        if self.bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but this CUDA device does not support it")
        
        # Validate logging_steps
        if self.logging_steps <= 0:
            raise ValueError(f"logging_steps must be > 0, got {self.logging_steps}")
        
        # Validate batch_size
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        
        if self.eval_batch_size <= 0:
            raise ValueError(f"eval_batch_size must be > 0, got {self.eval_batch_size}")
        
        # Validate gradient_accumulation_steps
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be > 0, got {self.gradient_accumulation_steps}")

        if self.on_capacity_exceeded not in ("raise", "truncate_with_warning", "skip_sample"):
            raise ValueError(
                "on_capacity_exceeded must be 'raise', 'truncate_with_warning', or "
                f"'skip_sample', got {self.on_capacity_exceeded!r}"
            )
        if self.length_group_window_batches <= 0:
            raise ValueError("length_group_window_batches must be > 0")
        if self.float32_matmul_precision not in ("highest", "high", "medium"):
            raise ValueError(
                "float32_matmul_precision must be 'highest', 'high', or 'medium'"
            )
        
        # Validate LoRA configuration
        if self.use_lora:
            if self.lora_r <= 0:
                raise ValueError(f"lora_r must be > 0, got {self.lora_r}")
            if self.lora_alpha <= 0:
                raise ValueError(f"lora_alpha must be > 0, got {self.lora_alpha}")
            if not 0 <= self.lora_dropout < 1:
                raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")
            if not self.lora_target_modules:
                raise ValueError("lora_target_modules cannot be empty when use_lora=True")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        with open(path) as f:
            return cls(**json.load(f))


# =============================================================================
# Dataset
# =============================================================================

class ExtractorDataset(Dataset):
    """
    Dataset for GLiNER2 training with multi-format support.

    Supports all formats through DataLoader_Factory:
    - JSONL file path(s)
    - List of InputExample objects
    - TrainingDataset object
    - List of raw dict records
    
    Examples
    --------
    >>> # From JSONL
    >>> dataset = ExtractorDataset("train.jsonl")
    
    >>> # From multiple JSONL files
    >>> dataset = ExtractorDataset(["train1.jsonl", "train2.jsonl"])
    
    >>> # From InputExample list
    >>> dataset = ExtractorDataset(examples)
    """

    def __init__(
            self,
            data: TrainDataInput,
            max_samples: int = -1,
            shuffle: bool = True,
            seed: int = 42,
            validate: bool = False,
    ):
        """
        Initialize dataset from various input formats.

        Parameters
        ----------
        data : TrainDataInput
            Training data in any supported format.
        max_samples : int, default=-1
            Maximum samples to use (-1 = all).
        shuffle : bool, default=True
            Whether to shuffle the data.
        seed : int, default=42
            Random seed for shuffling.
        validate : bool, default=False
            Whether to validate the data. Validation is always strict:
            checks that entity spans, relation values, and structure
            field values exist in the text.
        """
        self.data = DataLoader_Factory.load(
            data=data,
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed,
            validate=validate,
        )
        self.lengths = self._load_or_compute_lengths(data)

    @staticmethod
    def _record_text(record: Dict[str, Any]) -> str:
        return str(record.get("input", record.get("text", "")))

    def _length_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for record in self.data:
            text = self._record_text(record)
            digest.update(len(text).to_bytes(8, "little"))
            digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    def _load_or_compute_lengths(self, source: TrainDataInput) -> Tuple[int, ...]:
        """Return cached inexpensive token-count estimates for bucketing.

        A single JSONL source gets a ``.lengths.npy`` sidecar plus fingerprint
        metadata. In-memory and multi-source datasets are cached only for the
        lifetime of this object. Sidecar failures are deliberately non-fatal.
        """
        fingerprint = self._length_fingerprint()
        source_path = (
            Path(source).expanduser()
            if isinstance(source, (str, Path))
            else None
        )
        cache_path = (
            source_path.with_suffix(source_path.suffix + ".lengths.npy")
            if source_path is not None else None
        )
        fingerprint_path = (
            source_path.with_suffix(source_path.suffix + ".lengths.json")
            if source_path is not None else None
        )
        if cache_path is not None and fingerprint_path is not None:
            try:
                metadata = json.loads(fingerprint_path.read_text())
                cached = np.load(cache_path, allow_pickle=False)
                if (
                    metadata.get("fingerprint") == fingerprint
                    and cached.ndim == 1
                    and cached.shape[0] == len(self.data)
                ):
                    return tuple(int(value) for value in cached)
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        lengths = tuple(
            max(1, len(self._record_text(record).split()))
            for record in self.data
        )
        if cache_path is not None and fingerprint_path is not None:
            try:
                np.save(cache_path, np.asarray(lengths, dtype=np.int32))
                fingerprint_path.write_text(json.dumps({"fingerprint": fingerprint}))
            except OSError:
                logger.debug("Could not persist dataset length cache", exc_info=True)
        return lengths

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, Dict]:
        record = self.data[idx]
        # Handle both formats
        if "input" in record:
            return record["input"], record["output"]
        else:
            return record["text"], record["schema"]

    # Factory methods for explicit creation
    @classmethod
    def from_jsonl(cls, paths: Union[str, Path, List], **kwargs) -> 'ExtractorDataset':
        """Create from JSONL file(s)."""
        return cls(paths, **kwargs)

    @classmethod
    def from_examples(cls, examples: List[InputExample], **kwargs) -> 'ExtractorDataset':
        """Create from list of InputExample."""
        return cls(examples, **kwargs)

    @classmethod
    def from_training_dataset(cls, dataset: TrainingDataset, **kwargs) -> 'ExtractorDataset':
        """Create from TrainingDataset."""
        return cls(dataset, **kwargs)

    @classmethod
    def from_dicts(cls, dicts: List[Dict], **kwargs) -> 'ExtractorDataset':
        """Create from list of dicts."""
        return cls(dicts, **kwargs)


# =============================================================================
# Collator
# =============================================================================

class ExtractorCollator:
    """Data collator that converts raw records to model inputs."""

    def __init__(
            self, processor: SchemaTransformer, is_training: bool = True,
            max_len=None, architecture: str = "span",
            max_gold_per_query: Optional[int] = 32,
            build_targets: Optional[bool] = None,
            on_capacity_exceeded: str = "raise",
            allow_invalid_samples: bool = False,
    ):
        self.processor = processor
        self.is_training = is_training
        self.max_len = max_len
        self.architecture = architecture
        self.max_gold_per_query = max_gold_per_query
        # For an eval collator (``is_training=False``) set ``build_targets=True``
        # so a supervised eval loss can be computed while the model runs in eval
        # mode. Defaults to ``is_training`` (plain inference builds no targets).
        self.build_targets = build_targets
        # Gold-capacity overflow policy (raise | truncate_with_warning |
        # skip_sample); defaults to the no-silent-loss "raise".
        self.on_capacity_exceeded = on_capacity_exceeded
        self.allow_invalid_samples = allow_invalid_samples

    def __call__(self, batch: List[Tuple[str, Dict]]):
        """
        Convert batch of (text, schema) tuples to PreprocessedBatch.

        Args:
            batch: List of (text, schema) tuples from dataset

        Returns:
            PreprocessedBatch ready for model.forward()
        """
        if self.is_training:
            kwargs = dict(
                max_len=self.max_len, architecture=self.architecture,
                max_gold_per_query=self.max_gold_per_query,
                on_capacity_exceeded=self.on_capacity_exceeded,
            )
            if self.allow_invalid_samples:
                kwargs.update(
                    error_policy="skip",
                    ignore_missing_entities=True,
                )
            return self.processor.collate_fn_train(batch, **kwargs)
        else:
            kwargs = dict(
                max_len=self.max_len, architecture=self.architecture,
                build_targets=self.build_targets,
                max_gold_per_query=self.max_gold_per_query,
                on_capacity_exceeded=self.on_capacity_exceeded,
            )
            if self.allow_invalid_samples:
                kwargs.update(
                    error_policy="skip",
                    ignore_missing_entities=True,
                )
            return self.processor.collate_fn_inference(batch, **kwargs)


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float = 0.0
    classification_loss: float = 0.0
    structure_loss: float = 0.0
    count_loss: float = 0.0
    learning_rate: float = 0.0
    epoch: float = 0.0
    step: int = 0
    samples_seen: int = 0
    throughput: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# =============================================================================
# Scheduler Factory
# =============================================================================

def get_scheduler(optimizer, scheduler_type, num_training_steps, num_warmup_steps, num_cycles=0.5):
    """Create learning rate scheduler."""
    def lr_lambda_linear(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))

    def lr_lambda_cosine(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def lr_lambda_cosine_restarts(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((num_cycles * progress) % 1.0))))

    def lr_lambda_constant(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return 1.0

    schedulers = {
        "linear": lr_lambda_linear,
        "cosine": lr_lambda_cosine,
        "cosine_restarts": lr_lambda_cosine_restarts,
        "constant": lr_lambda_constant,
    }

    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    return LambdaLR(optimizer, schedulers[scheduler_type])


# =============================================================================
# Main Trainer
# =============================================================================

class ExtractorTrainer:
    """
    World-class trainer for GLiNER2 with flexible multi-format data input.

    Architecture-neutral: drives both the span and boundary architectures. The
    legacy name ``GLiNER2Trainer`` is preserved as an alias.

    Parameters
    ----------
    model : nn.Module
        The GLiNER2 model to train.
    config : TrainingConfig
        Training configuration.
    processor : SchemaTransformer, optional
        Schema processor. If None, uses model.processor.
    train_data : TrainDataInput, optional
        Training data (can be provided here or in train()).
    eval_data : TrainDataInput, optional
        Evaluation data.
    compute_metrics : Callable, optional
        Custom metrics function.

    Supported Data Formats
    ----------------------
    - Single JSONL file path (str or Path)
    - List of JSONL file paths
    - List of InputExample objects
    - TrainingDataset object
    - List of raw dict records

    Examples
    --------
    >>> # With InputExample list
    >>> examples = [InputExample(...), InputExample(...)]
    >>> trainer = GLiNER2Trainer(model, config)
    >>> trainer.train(train_data=examples)

    >>> # With JSONL file
    >>> trainer.train(train_data="train.jsonl")

    >>> # With multiple JSONL files
    >>> trainer.train(train_data=["train1.jsonl", "train2.jsonl"])

    >>> # With TrainingDataset
    >>> dataset = TrainingDataset.load("train.jsonl")
    >>> trainer.train(train_data=dataset)
    """

    def __init__(
            self,
            model: nn.Module,
            config: TrainingConfig,
            processor: SchemaTransformer = None,
            train_data: TrainDataInput = None,
            eval_data: TrainDataInput = None,
            compute_metrics: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        if (
            getattr(model, "architecture", "span") == "boundary"
            and not getattr(config, "_precision_explicit", True)
        ):
            config.fp16 = False
            config.bf16 = True
            logger.info("Boundary architecture defaulting to bf16 (fp16 disabled)")
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "The boundary architecture defaults to bf16, but this CUDA "
                    "device does not support it; choose fp32 explicitly."
                )
        torch.set_float32_matmul_precision(config.float32_matmul_precision)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
            torch.backends.cudnn.allow_tf32 = config.allow_tf32
        self.processor = processor or getattr(model, 'processor', None)
        if self.processor is None:
            raise ValueError("Processor must be provided or model must have .processor attribute")

        self.train_data = train_data
        self.eval_data = eval_data
        self.compute_metrics = compute_metrics

        self._setup_seed()
        self._setup_device()
        self._setup_output_dir()
        self._setup_logging()

        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf') if not config.greater_is_better else float('-inf')
        self.patience_counter = 0
        self.train_metrics_history = []
        self.eval_metrics_history = []

        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.wandb_run = None
        self.progress_bar = None
        
        # Most recent successful micro-batch outputs / grad norm, for logging.
        self._last_train_outputs = None
        self._last_grad_norm = None
        self._skip_counter = None
        self._loss_accum = None
        self._loss_finite_flag = None
        self._finite_grad_hook_handles = []

        # LoRA state
        self.lora_layers = {}
        self._setup_lora()
        base_model = self.model
        if config.gradient_checkpointing:
            encoder = getattr(base_model, "encoder", None)
            enable = getattr(encoder, "gradient_checkpointing_enable", None)
            if enable is None:
                raise ValueError("encoder does not support gradient checkpointing")
            enable()
        if config.compile_model:
            compile_method = getattr(base_model, "compile", None)
            if compile_method is None:
                raise ValueError("model does not support compile_model=True")
            compile_method(dynamic=True)
        self._install_finite_grad_hooks()

        self._setup_distributed()

    def _install_finite_grad_hooks(self) -> None:
        """Zero gradients from a device-detected non-finite loss before DDP."""
        def sanitize(gradient):
            flag = self._loss_finite_flag
            if flag is None:
                return gradient
            return torch.where(flag, gradient, torch.zeros_like(gradient))

        self._finite_grad_hook_handles = [
            parameter.register_hook(sanitize)
            for parameter in getattr(self.model, "parameters", lambda: ())()
            if parameter.requires_grad
        ]

    def _setup_seed(self):
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    def _setup_device(self):
        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device("cuda", self.config.local_rank)
            self.is_distributed = True
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
                logger.info(f"Initialized distributed training: rank {dist.get_rank()}/{dist.get_world_size()}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.is_distributed = False
        else:
            self.device = torch.device("cpu")
            self.is_distributed = False
            if self.config.fp16 or self.config.bf16:
                logger.warning("Mixed precision disabled on CPU")
                self.config.fp16 = False
                self.config.bf16 = False
        self.model.to(self.device)
        if not self.config.fp16 and not self.config.bf16:
            # Some encoder backends preserve a reduced parameter dtype when
            # loaded from a checkpoint. Full-precision training must normalize
            # every module before its activations reach FP32 task heads.
            self.model.float()
        logger.info(f"Using device: {self.device}")

    def _setup_output_dir(self):
        self.output_dir = Path(self.config.output_dir)
        self.logs_dir = self.output_dir / "logs"
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(exist_ok=True)
            self.config.save(str(self.output_dir / "training_config.json"))

    def _setup_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO if self.is_main_process else logging.WARNING,
        )
        
        # W&B setup (HuggingFace style)
        self.wandb_run = None
        if self.config.report_to_wandb and self.is_main_process:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project or self.config.experiment_name,
                    entity=self.config.wandb_entity,
                    name=self.config.wandb_run_name or f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=asdict(self.config),
                    tags=self.config.wandb_tags,
                    notes=self.config.wandb_notes,
                    dir=str(self.output_dir),
                )
                logger.info(f"W&B run: {self.wandb_run.url}")
            except ImportError:
                logger.warning("wandb not installed. Run: pip install wandb")
                self.config.report_to_wandb = False

    def _setup_lora(self):
        """Setup LoRA for parameter-efficient fine-tuning if enabled."""
        if not self.config.use_lora:
            logger.info("LoRA is disabled")
            return

        from gliner2.training.lora import _PeftLoraLayer

        for p in self.model.parameters():
            p.requires_grad = False
        logger.info("Froze all model parameters for LoRA training")

        self.model = self.model.apply_lora(
            r=self.config.lora_r, alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            targets=self.config.lora_target_modules,
            use_dora=self.config.lora_use_dora,
        )
        self.lora_layers = {n: m for n, m in self.model.named_modules() if isinstance(m, _PeftLoraLayer)}
        self.model._lora_layers = self.lora_layers

        lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        pct = (lora_params / total_params * 100) if total_params > 0 else 0.0
        logger.info(f"LoRA setup complete: {lora_params:,} trainable / {total_params:,} total ({pct:.2f}%)")

    def _setup_distributed(self):
        """Setup distributed training if enabled."""
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=self.config.ddp_find_unused_parameters,
                static_graph=self.config.ddp_static_graph,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
            )
            logger.info("Wrapped model in DistributedDataParallel")

    def _cleanup_distributed(self):
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        if self.is_distributed and hasattr(self.model, "module"):
            self.model = self.model.module
        self.is_distributed = False

    @property
    def is_main_process(self) -> bool:
        return not self.is_distributed or dist.get_rank() == 0

    @staticmethod
    def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safely divide two numbers, returning default if denominator is zero."""
        if denominator == 0:
            return default
        return numerator / denominator

    def _record_failed_batch(
        self,
        batch: Any,
        *,
        data_loader_step: int,
        error: BaseException,
    ) -> None:
        """Persist CPU-side sample contents without touching a failed device."""
        try:
            texts = list(getattr(batch, "original_texts", ()) or ())
            schemas = list(getattr(batch, "original_schemas", ()) or ())
            samples = []
            for sample_index in range(max(len(texts), len(schemas))):
                text = texts[sample_index] if sample_index < len(texts) else None
                schema = schemas[sample_index] if sample_index < len(schemas) else None
                canonical = json.dumps(
                    {"text": text, "schema": schema},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=repr,
                )
                samples.append({
                    "sample_index": sample_index,
                    "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                    "text": text,
                    "schema": schema,
                })

            rank = dist.get_rank() if self.is_distributed else 0
            record = {
                "timestamp": datetime.now().isoformat(),
                "rank": rank,
                "epoch": self.epoch,
                "global_step": self.global_step,
                "data_loader_step": data_loader_step,
                "error_type": type(error).__name__,
                "error": str(error),
                "samples": samples,
            }
            path = self.output_dir / (
                "failed_batches.jsonl"
                if not self.is_distributed
                else f"failed_batches.rank-{rank}.jsonl"
            )
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=repr))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            logger.error(
                "Recorded failing batch with %d sample(s) in %s",
                len(samples),
                path,
            )
        except Exception:
            # Diagnostics must never replace the original training exception.
            logger.exception("Could not record failing batch diagnostics")

    @classmethod
    def _diagnostic_json_value(cls, value: Any) -> Any:
        """Convert nested batch state, including tensors, to lossless JSON data."""
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            return {
                "__type__": "tensor",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "values": tensor.tolist(),
            }
        if isinstance(value, np.ndarray):
            return {
                "__type__": "ndarray",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "values": value.tolist(),
            }
        if isinstance(value, np.generic):
            return value.item()
        if is_dataclass(value) and not isinstance(value, type):
            result = {"__type__": type(value).__name__}
            result.update({
                item.name: cls._diagnostic_json_value(getattr(value, item.name))
                for item in dataclass_fields(value)
            })
            return result
        if isinstance(value, dict):
            return {
                str(key): cls._diagnostic_json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._diagnostic_json_value(item) for item in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if hasattr(value, "__dict__"):
            result = {"__type__": type(value).__name__}
            result.update({
                str(key): cls._diagnostic_json_value(item)
                for key, item in vars(value).items()
            })
            return result
        return repr(value)

    def _record_debug_batch(
        self,
        batch: Any,
        *,
        data_loader_step: int,
        micro_batch_in_window: int,
        is_last_micro: bool,
    ) -> None:
        """Dump a complete pre-GPU batch for a configured diagnostic step."""
        try:
            rank = dist.get_rank() if self.is_distributed else 0
            record = {
                "timestamp": datetime.now().isoformat(),
                "rank": rank,
                "epoch": self.epoch,
                "global_step_before_batch": self.global_step,
                "data_loader_step": data_loader_step,
                "micro_batch_in_window": micro_batch_in_window,
                "is_last_micro": is_last_micro,
                "batch": self._diagnostic_json_value(batch),
            }
            path = self.output_dir / (
                "debug_batches.jsonl"
                if not self.is_distributed
                else f"debug_batches.rank-{rank}.jsonl"
            )
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            logger.warning(
                "Recorded full pre-GPU batch for global step %d in %s",
                self.global_step,
                path,
            )
        except Exception:
            # Diagnostics must never introduce a new training failure.
            logger.exception("Could not record configured batch diagnostics")

    def _get_model_config(self) -> Any:
        """Return the underlying model config, handling DDP-wrapped models."""
        base_model = self.model.module if self.is_distributed and hasattr(self.model, "module") else self.model
        return getattr(base_model, "config", None)

    def _validate_training_setup(self, train_dataset: ExtractorDataset, eval_dataset: Optional[ExtractorDataset]):
        """Validate training setup and raise informative errors for edge cases."""
        # Check if dataset is empty
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty. Please provide at least one training example.")
        
        # Check if dataset is smaller than batch size
        if len(train_dataset) < self.config.batch_size:
            logger.warning(
                f"Training dataset size ({len(train_dataset)}) is smaller than batch_size "
                f"({self.config.batch_size}). Adjusting batch_size to {len(train_dataset)}."
            )
            # We'll handle this in _create_dataloader by adjusting drop_last
        
        # Check early stopping configuration
        if self.config.early_stopping:
            if eval_dataset is None:
                raise ValueError(
                    "early_stopping is enabled but no eval_data provided. "
                    "Please provide eval_data or disable early_stopping."
                )
            if len(eval_dataset) == 0:
                raise ValueError("Evaluation dataset is empty but early_stopping is enabled.")
        
        # Check eval strategy configuration
        if self.config.eval_strategy == "steps" and eval_dataset is None:
            logger.warning(
                "eval_strategy='steps' but no eval_data provided. "
                "Evaluation will be skipped."
            )
        
        # Warn about very small datasets
        if len(train_dataset) < self.config.gradient_accumulation_steps:
            logger.warning(
                f"Training dataset size ({len(train_dataset)}) is smaller than "
                f"gradient_accumulation_steps ({self.config.gradient_accumulation_steps}). "
                f"Training may not work as expected."
            )
    
    @staticmethod
    def _gold_injection_probability(
        progress: float, start: float, end: float, hold_fraction: float
    ) -> float:
        """Hold the initial injection rate, then linearly anneal to the end."""
        if progress <= hold_fraction:
            return float(start)
        fraction = min(
            max((progress - hold_fraction) / max(1.0 - hold_fraction, 1e-12), 0.0),
            1.0,
        )
        return float(start + (end - start) * fraction)

    @staticmethod
    def _soft_iou_anneal_scale(step: int, anneal_steps: int) -> float:
        """Linearly anneal soft-IoU supervision to exact zero."""
        if anneal_steps <= 0:
            return 0.0
        return max(1.0 - step / anneal_steps, 0.0)

    def _backward_one(
        self,
        batch,
        step: int,
        use_amp: bool,
        amp_dtype,
        *,
        is_last_micro: bool = True,
    ) -> torch.Tensor:
        """Run one micro-batch and always enter autograd/DDP collectives."""
        if self._skip_counter is None:
            self._skip_counter = torch.zeros(
                (), dtype=torch.long, device=self.device
            )
            self._loss_accum = torch.zeros(
                (), dtype=torch.float32, device=self.device
            )
        model = self.model.module if hasattr(self.model, "module") else self.model
        boundary_head = getattr(model, "boundary_head", None)
        if boundary_head is not None:
            planned = max(getattr(self, "_planned_max_steps", 1), 1)
            progress = self.global_step / planned
            injection_probability = self._gold_injection_probability(
                progress,
                self.config.gold_injection_start,
                self.config.gold_injection_end,
                self.config.gold_injection_hold_frac,
            )
            boundary_head.set_gold_injection_prob(injection_probability)
            warmup = getattr(
                getattr(boundary_head, "settings", None),
                "consistency_warmup_steps",
                0,
            )
            consistency_scale = (
                1.0 if warmup <= 0 else min(self.global_step / warmup, 1.0)
            )
            boundary_head.set_consistency_scale(consistency_scale)
            soft_iou_steps = getattr(
                getattr(boundary_head, "settings", None),
                "soft_iou_anneal_steps",
                0,
            )
            soft_iou_scale = self._soft_iou_anneal_scale(
                self.global_step, soft_iou_steps
            )
            boundary_head.set_soft_iou_scale(soft_iou_scale)
        diagnostics_interval = (
            self.config.diagnostics_every_n_steps or self.config.logging_steps
        )
        collect = bool(
            boundary_head is not None
            and self.config.log_proposal_metrics
            and (self.global_step + 1) % diagnostics_interval == 0
        )
        previous_collect = getattr(boundary_head, "collect_diagnostics", False)
        if boundary_head is not None:
            boundary_head.collect_diagnostics = collect
        sync_ctx = (
            contextlib.nullcontext()
            if is_last_micro or not self.is_distributed
            else self.model.no_sync()
        )
        try:
            with sync_ctx:
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=use_amp,
                    dtype=amp_dtype,
                ):
                    outputs = self.model(batch)
                    loss = outputs.get("total_loss")
                    if loss is None:
                        zero = getattr(model, "_zero_loss", None)
                        loss = (
                            zero(self.device)
                            if zero is not None
                            else sum(
                                (parameter.sum() * 0.0)
                                for parameter in model.parameters()
                                if parameter.requires_grad
                            )
                        )
                    elif not loss.requires_grad:
                        zero = getattr(model, "_zero_loss", None)
                        touch = (
                            zero(self.device)
                            if zero is not None
                            else sum(
                                (parameter.sum() * 0.0)
                                for parameter in model.parameters()
                                if parameter.requires_grad
                            )
                        )
                        loss = loss.detach() * 0.0 + touch

                    finite = torch.isfinite(loss.detach()).all()
                    if self.is_distributed and self.config.ddp_consensus_check:
                        bad = (~finite).to(dtype=torch.float32)
                        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
                        finite = bad == 0
                    self._loss_finite_flag = finite
                    loss = torch.where(finite, loss, torch.zeros_like(loss))
                    reported_loss = loss.detach()
                    if self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps

                if self.config.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
        finally:
            if boundary_head is not None:
                boundary_head.collect_diagnostics = previous_collect

        self._skip_counter.add_((~finite).to(self._skip_counter.dtype))
        self._loss_accum.add_(reported_loss.float())
        self._last_train_outputs = outputs
        return reported_loss

    def _renormalize_partial_accumulation(self, micro_batches: int) -> None:
        """Correct gradients from an incomplete accumulation window."""
        accumulation = self.config.gradient_accumulation_steps
        if not 0 < micro_batches < accumulation:
            return
        scale = accumulation / micro_batches
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)

    def _flush_delayed_counters(self) -> None:
        """Read device counters only at an explicit logging boundary."""
        if self._skip_counter is None:
            return
        skipped = int(self._skip_counter.item())
        self._skip_counter.zero_()
        if skipped:
            message = f"{skipped} non-finite micro-batch loss(es) were zeroed"
            if self.config.strict_training and not self.config.ignore_nonfinite_losses:
                raise FloatingPointError(message)
            logger.warning(message)

    @staticmethod
    def _proposal_metric_ratios(values: Dict[str, Any], prefix: str) -> Dict[str, float]:
        """Convert accumulated proposal diagnostic counts to public ratios."""
        def scalar(name: str) -> float:
            value = values.get(name)
            if value is None:
                return 0.0
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu())
            return float(value)

        gold_total = scalar("proposal_gold_total")
        boundary_total = scalar("boundary_total")
        valid_queries = scalar("valid_queries")
        result: Dict[str, float] = {}
        if gold_total > 0:
            result[f"{prefix}_proposal_oracle_recall"] = (
                scalar("proposal_gold_hit") / gold_total
            )
        if boundary_total > 0:
            result[f"{prefix}_start_recall"] = scalar("start_hit") / boundary_total
            result[f"{prefix}_end_recall"] = scalar("end_hit") / boundary_total
        if valid_queries > 0:
            result[f"{prefix}_candidates_per_query"] = (
                scalar("unique_candidates") / valid_queries
            )
        for label in ("1", "2", "3_4", "5_8", "9_plus"):
            total = scalar(f"length_{label}_total")
            if total > 0:
                result[f"{prefix}_recall_length_{label}"] = (
                    scalar(f"length_{label}_hit") / total
                )
        absent_total = scalar("absent_query_total")
        if absent_total > 0:
            result[f"{prefix}_absent_query_false_positive_rate"] = (
                scalar("absent_query_false_positive") / absent_total
            )
        return result

    def _optimizer_step(self) -> bool:
        """Apply one optimizer update over the accumulated gradients.

        Always clears the gradient buffers afterwards. Returns ``True`` when the
        parameters were actually updated. Under AMP, ``GradScaler.step`` skips
        the update on non-finite gradients (signalled by a decreased loss
        scale); in that case the scheduler is not advanced so the LR schedule
        stays aligned with the number of real optimizer steps.
        """
        if self.config.fp16:
            self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self._last_grad_norm = (
            grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else grad_norm
        )

        if self.config.fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Non-finite loss/gradient handling is device-side before this point.
        # Avoid GradScaler.get_scale(), which synchronizes every optimizer step.
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return True

    def _prepare_data(self, data: TrainDataInput, is_train: bool = True) -> ExtractorDataset:
        """Convert any supported data format to ExtractorDataset."""
        if data is None:
            return None

        if isinstance(data, ExtractorDataset):
            return data

        max_samples = self.config.max_train_samples if is_train else self.config.max_eval_samples

        return ExtractorDataset(
            data=data,
            max_samples=max_samples,
            shuffle=is_train,
            seed=self.config.seed,
            validate=self.config.validate_data if is_train else False
        )

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with appropriate parameters based on LoRA configuration."""
        optimizer_kwargs = {
            "betas": (self.config.adam_beta1, self.config.adam_beta2),
            "eps": self.config.adam_epsilon,
        }
        if self.device.type == "cuda" and self.config.fused_optimizer:
            optimizer_kwargs["fused"] = True
        else:
            optimizer_kwargs["foreach"] = True
        if self.config.use_lora:
            lora_params = [p for p in self.model.parameters() if p.requires_grad]
            if not lora_params:
                raise ValueError("No LoRA parameters found. Check LoRA configuration.")
            logger.info("Optimizer: LoRA params only = %d, LR=%s", len(lora_params), self.config.task_lr)
            return AdamW(
                [{"params": lora_params, "lr": self.config.task_lr, "weight_decay": self.config.weight_decay}],
                **optimizer_kwargs,
            )
        # Normal training: separate LRs for encoder and task-specific layers.
        encoder_params = []
        task_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                encoder_params.append(param)
            else:
                task_params.append(param)

        enc_ids = {id(p) for p in encoder_params}
        task_ids = {id(p) for p in task_params}
        assert not (enc_ids & task_ids), "encoder/task optimizer groups overlap"
        assert len(enc_ids | task_ids) == len(encoder_params) + len(task_params), (
            "duplicate parameters across optimizer groups"
        )
        return AdamW(
            [
                {"params": encoder_params, "lr": self.config.encoder_lr, "weight_decay": self.config.weight_decay},
                {"params": task_params, "lr": self.config.task_lr, "weight_decay": self.config.weight_decay},
            ],
            **optimizer_kwargs,
        )

    @staticmethod
    def recall_gate_exit_code(
        metrics: Dict[str, float],
        *,
        overall_gate: float = 0.97,
        long_gate: float = 0.93,
    ) -> int:
        """Return a process-style exit code for oracle-recall launch gates."""
        overall = metrics.get("dry_run_proposal_oracle_recall", 0.0)
        long_recall = metrics.get("dry_run_recall_length_9_plus", 1.0)
        return int(overall < overall_gate or long_recall < long_gate)

    def _run_recall_dry_run(self, train_loader) -> Dict[str, float]:
        """Measure injection-off proposal recall without updating parameters."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        if getattr(model, "architecture", "span") != "boundary":
            raise ValueError("dry_run_recall_steps requires architecture='boundary'")
        was_training = self.model.training
        self.model.eval()
        counts: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for step, batch in enumerate(train_loader):
                if step >= self.config.dry_run_recall_steps:
                    break
                outputs = self.model(
                    batch,
                    gold_injection_prob=0.0,
                    collect_diagnostics=True,
                )
                for key, value in (getattr(outputs, "metrics", None) or {}).items():
                    detached = value.detach()
                    counts[key] = counts.get(key, torch.zeros_like(detached)) + detached
        if was_training:
            self.model.train()
        metrics = self._proposal_metric_ratios(counts, "dry_run")
        active_pool = getattr(
            model.boundary_head.settings, "candidate_pool", "per_query"
        )
        metrics.update(
            self._proposal_metric_ratios(counts, f"dry_run_{active_pool}")
        )
        comparison_pool = "shared" if active_pool == "per_query" else "per_query"
        comparison_counts = {
            key[len(comparison_pool) + 1:]: value
            for key, value in counts.items()
            if key.startswith(f"{comparison_pool}_")
        }
        metrics.update(
            self._proposal_metric_ratios(
                comparison_counts, f"dry_run_{comparison_pool}"
            )
        )
        metrics["dry_run_candidate_budget"] = float(
            (
                model.boundary_head.settings.pool_size
                if active_pool == "shared"
                else model.boundary_head.settings.candidate_budget
            )
        )
        metrics["dry_run_per_query_candidate_budget"] = float(
            model.boundary_head.settings.candidate_budget
        )
        metrics["dry_run_shared_candidate_budget"] = float(
            getattr(
                model.boundary_head.settings,
                "pool_size",
                model.boundary_head.settings.candidate_budget,
            )
        )
        metrics["dry_run_absent_query_total"] = float(
            counts.get("absent_query_total", torch.zeros(())).cpu()
        )
        columns = (
            "dry_run_proposal_oracle_recall",
            "dry_run_start_recall",
            "dry_run_end_recall",
            "dry_run_recall_length_1",
            "dry_run_recall_length_2",
            "dry_run_recall_length_3_4",
            "dry_run_recall_length_5_8",
            "dry_run_recall_length_9_plus",
            "dry_run_candidates_per_query",
            "dry_run_candidate_budget",
            "dry_run_absent_query_total",
        )
        logger.info(
            "Oracle recall dry run (gold injection=0, no weight updates)\n%s",
            "\n".join(f"{key}: {metrics.get(key, 0.0):.6g}" for key in columns),
        )
        return metrics

    def _create_dataloader(self, dataset: ExtractorDataset, batch_size: int, shuffle: bool = True, is_training: bool = True) -> DataLoader:
        sampler = None
        base_model = (
            self.model.module
            if self.is_distributed and hasattr(self.model, "module")
            else self.model
        )
        architecture = getattr(base_model, "architecture", "span")
        use_length_groups = (
            architecture == "boundary"
            and is_training
            and shuffle
            and self.config.group_by_length
        )
        effective_batch_size = min(batch_size, len(dataset))
        if use_length_groups and self.is_distributed:
            sampler = DistributedLengthGroupedSampler(
                dataset.lengths,
                effective_batch_size,
                window_batches=self.config.length_group_window_batches,
                seed=self.config.seed,
            )
            shuffle = False
        elif use_length_groups:
            sampler = LengthGroupedSampler(
                dataset.lengths,
                effective_batch_size,
                window_batches=self.config.length_group_window_batches,
                seed=self.config.seed,
            )
            shuffle = False
        elif self.is_distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False

        model_config = self._get_model_config()
        max_len = self.config.max_len or getattr(model_config, "max_len", None)
        max_gold = getattr(model_config, "boundary_head", {}).get("max_gold_per_query", 32)
        # Eval collator builds gold targets (build_targets=True) so the eval
        # loss is supervised and finite even on extraction-only eval sets. Gold
        # injection into proposals stays gated on model.training, so eval is
        # unbiased.
        collator = ExtractorCollator(
            self.processor, is_training=is_training, max_len=max_len,
            architecture=architecture, max_gold_per_query=max_gold,
            build_targets=None if is_training else True,
            on_capacity_exceeded=self.config.on_capacity_exceeded,
            allow_invalid_samples=self.config.allow_invalid_samples,
        )

        # Fix Bug #1 & #9: Handle small datasets
        # If dataset is smaller than batch_size, adjust to prevent empty dataloader
        drop_last = is_training and len(dataset) > batch_size
        
        # Adjust num_workers for small datasets
        effective_num_workers = (
            self.config.num_workers if len(dataset) > self.config.num_workers else 0
        )
        # macOS spawn must pickle the tokenizer-bearing collator; fast
        # tokenizers may contain non-picklable cached callables.
        if self.device.type == "mps" or sys.platform == "darwin":
            effective_num_workers = 0

        return DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=effective_num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor if effective_num_workers > 0 else None,
            collate_fn=collator,
            drop_last=drop_last,
            persistent_workers=effective_num_workers > 0,
        )

    def train(
            self,
            train_data: TrainDataInput = None,
            eval_data: TrainDataInput = None,
    ) -> Dict[str, Any]:
        """
        Main training loop.

        Parameters
        ----------
        train_data : TrainDataInput, optional
            Training data. Supports all formats:
            - str/Path: JSONL file path
            - List[str/Path]: Multiple JSONL files
            - List[InputExample]: List of examples
            - TrainingDataset: Dataset object
            - List[Dict]: Raw records

        eval_data : TrainDataInput, optional
            Evaluation data (same formats supported).

        Returns
        -------
        Dict[str, Any]
            Training summary with metrics history.
        """
        # Prepare datasets
        train_data = train_data or self.train_data
        eval_data = eval_data or self.eval_data

        if train_data is None:
            raise ValueError("No training data provided")

        train_dataset = self._prepare_data(train_data, is_train=True)
        eval_dataset = self._prepare_data(eval_data, is_train=False) if eval_data else None

        # Fix Bug #7: Validate training setup
        self._validate_training_setup(train_dataset, eval_dataset)

        train_loader = self._create_dataloader(train_dataset, self.config.batch_size, shuffle=True, is_training=True)

        # Fix Bug #1: Check if dataloader is empty
        if len(train_loader) == 0:
            raise ValueError(
                f"Training dataloader is empty. Dataset size: {len(train_dataset)}, "
                f"Batch size: {self.config.batch_size}. Please reduce batch_size or add more data."
            )
        if self.config.dry_run_recall_steps > 0:
            metrics = self._run_recall_dry_run(train_loader)
            exit_code = self.recall_gate_exit_code(
                metrics,
                overall_gate=self.config.gate_recall,
                long_gate=self.config.gate_long_recall,
            )
            metrics["recall_gate_exit_code"] = exit_code
            if exit_code:
                raise RuntimeError(
                    "oracle-recall gate failed "
                    f"(overall>={self.config.gate_recall}, "
                    f"9_plus>={self.config.gate_long_recall})"
                )
            return metrics

        # Calculate steps
        num_update_steps_per_epoch = math.ceil(
            len(train_loader) / self.config.gradient_accumulation_steps
        )
        
        # Fix Bug #1: Handle case where num_update_steps_per_epoch is 0
        if num_update_steps_per_epoch == 0:
            # If gradient accumulation is larger than dataloader, we have at least the batches we can process
            num_update_steps_per_epoch = 1
            logger.warning(
                f"gradient_accumulation_steps ({self.config.gradient_accumulation_steps}) is larger than "
                f"batches per epoch ({len(train_loader)}). Setting to 1 update step per epoch."
            )
        
        if self.config.max_steps > 0:
            max_steps = self.config.max_steps
            num_epochs = math.ceil(max_steps / num_update_steps_per_epoch)
        else:
            max_steps = num_update_steps_per_epoch * self.config.num_epochs
            num_epochs = self.config.num_epochs
        self._planned_max_steps = max_steps

        warmup_steps = self.config.warmup_steps or int(max_steps * self.config.warmup_ratio)

        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = get_scheduler(self.optimizer, self.config.scheduler_type, max_steps, warmup_steps, self.config.num_cycles)

        # Mixed precision
        use_amp = self.config.fp16 or self.config.bf16
        amp_dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.config.fp16
        )

        # Logging
        logger.info("***** Running Training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num epochs = {num_epochs}")
        logger.info(f"  Batch size = {self.config.batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size = {self.config.effective_batch_size}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Warmup steps = {warmup_steps}")
        
        # Log trainable parameters. Uses the same manual-count expression in
        # both branches; the LoRA branch just labels the output differently.
        # (Previously called ``count_lora_parameters`` from ``gliner2.training.lora``, 
        # which was dropped from the import list during the PEFT migration but
        # left behind here, producing a NameError at the start of every LoRA run.)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        percentage = (trainable_params / total_params * 100) if total_params > 0 else 0.0
        if self.config.use_lora:
            logger.info(f"  LoRA enabled: {trainable_params:,} trainable / {total_params:,} total ({percentage:.2f}%)")
        else:
            logger.info(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({percentage:.2f}%)")

        # Training state
        self.model.train()
        self.processor.change_mode(is_training=True)
        self.global_step = 0
        self.epoch = 0
        tr_loss = torch.zeros((), device=self.device)

        start_time = time.time()
        samples_seen = 0

        self.progress_bar = tqdm(total=max_steps, desc="Training", disable=not self.is_main_process)

        should_stop = False
        for epoch in range(num_epochs):
            self.epoch = epoch

            set_epoch = getattr(train_loader.sampler, "set_epoch", None)
            if set_epoch is not None:
                set_epoch(epoch)

            epoch_loss = torch.zeros((), device=self.device)
            epoch_steps = 0
            # Successful, not-yet-applied micro-batches in the current window.
            # Only a successful backward advances it, so skipped/OOM batches can
            # never drop or misalign an optimizer step.
            micro = 0
            accum = self.config.gradient_accumulation_steps

            for step, batch in enumerate(train_loader):
                samples_seen += len(batch)
                is_last_micro = (
                    (micro + 1) % accum == 0 or step + 1 == len(train_loader)
                )
                if self.global_step in self.config.debug_global_steps:
                    self._record_debug_batch(
                        batch,
                        data_loader_step=step,
                        micro_batch_in_window=micro,
                        is_last_micro=is_last_micro,
                    )

                try:
                    profile_step = self.global_step < self.config.profile_first_n_steps
                    if profile_step:
                        activities = [torch.profiler.ProfilerActivity.CPU]
                        if torch.cuda.is_available():
                            activities.append(torch.profiler.ProfilerActivity.CUDA)
                            torch.cuda.reset_peak_memory_stats()
                        with count_cuda_syncs() as syncs, torch.profiler.profile(
                            activities=activities
                        ) as prof:
                            micro_loss = self._backward_one(
                                batch,
                                step,
                                use_amp,
                                amp_dtype,
                                is_last_micro=is_last_micro,
                            )
                        logger.info(
                            "step profile: syncs=%d peak_memory=%d\n%s",
                            syncs["n"],
                            (
                                torch.cuda.max_memory_allocated()
                                if torch.cuda.is_available() else 0
                            ),
                            prof.key_averages().table(
                                sort_by=(
                                    "self_cuda_time_total"
                                    if torch.cuda.is_available()
                                    else "self_cpu_time_total"
                                ),
                                row_limit=20,
                            ),
                        )
                    else:
                        micro_loss = self._backward_one(
                            batch,
                            step,
                            use_amp,
                            amp_dtype,
                            is_last_micro=is_last_micro,
                        )
                except torch.cuda.OutOfMemoryError as exc:
                    self._record_failed_batch(
                        batch, data_loader_step=step, error=exc
                    )
                    # Discard the whole in-flight window: its partial gradients
                    # are unusable, so zero them and restart accumulation. Log
                    # the discarded count so the loss is visible, not silent.
                    logger.warning(
                        "OOM at step %d; discarding %d accumulated micro-batch(es) "
                        "in the in-flight window. Consider reducing batch_size or "
                        "max sequence length.", step, micro
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.optimizer.zero_grad(set_to_none=True)
                    micro = 0
                    continue
                except Exception as exc:
                    self._record_failed_batch(
                        batch, data_loader_step=step, error=exc
                    )
                    if is_fatal_device_error(exc):
                        raise
                    if not self.config.skip_step_errors or self.is_distributed:
                        raise
                    # A failed forward/backward may have left partial gradients
                    # in the accumulation window. Drop the whole window before
                    # accepting another batch and retain the traceback in logs.
                    logger.exception(
                        "Error in micro-batch %d; discarding %d accumulated "
                        "micro-batch(es) and continuing because "
                        "skip_step_errors=True.",
                        step,
                        micro,
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    micro = 0
                    continue

                tr_loss += micro_loss
                epoch_loss += micro_loss
                epoch_steps += 1
                micro += 1

                if micro % accum != 0:
                    continue

                applied = self._optimizer_step()
                micro = 0
                if not applied:
                    # AMP skipped the update on non-finite gradients; the window
                    # was cleared and no optimizer step counts.
                    continue

                outputs = self._last_train_outputs
                self.global_step += 1

                if self.global_step % self.config.logging_steps == 0:
                    self._flush_delayed_counters()
                    elapsed = time.time() - start_time
                    # Fix Bug #2: Safe division for metrics
                    avg_loss = float(
                        (tr_loss / max(self.config.logging_steps, 1)).item()
                    )
                    # Fix Bug #5: Safe division for epoch progress
                    epoch_progress = self._safe_divide(step, len(train_loader), default=0.0)
                    metrics = TrainingMetrics(
                        loss=avg_loss,
                        classification_loss=outputs.get("classification_loss", torch.tensor(0)).item(),
                        structure_loss=outputs.get("structure_loss", torch.tensor(0)).item(),
                        count_loss=outputs.get("count_loss", torch.tensor(0)).item(),
                        learning_rate=self.scheduler.get_last_lr()[0],
                        epoch=epoch + epoch_progress,
                        step=self.global_step,
                        samples_seen=samples_seen,
                        throughput=self._safe_divide(samples_seen, elapsed, default=0.0),
                    )
                    logged_metrics = metrics.to_dict()
                    proposal_counts = getattr(outputs, "metrics", None)
                    if proposal_counts:
                        logged_metrics.update(
                            self._proposal_metric_ratios(proposal_counts, "train")
                        )
                    self._log_metrics(logged_metrics, prefix="train")
                    tr_loss.zero_()

                if self.config.eval_strategy == "steps" and self.global_step % self.config.eval_steps == 0:
                    if eval_dataset:
                        prev_best = self.best_metric
                        eval_metrics = self._evaluate(eval_dataset)
                        self.model.train()
                        self.processor.change_mode(is_training=True)
                        if self.config.early_stopping and self._check_early_stopping(eval_metrics, prev_best):
                            logger.info(f"Early stopping triggered at step {self.global_step}")
                            should_stop = True
                            break
                    self._save_checkpoint(f"checkpoint-{self.global_step}")

                self.progress_bar.update(1)

                if self.global_step >= max_steps:
                    break
            
            if should_stop:
                break

            # Fix Bug #6: Flush a trailing partial window so its gradients are
            # applied (or discarded) rather than leaking into the next epoch.
            if micro > 0:
                self._renormalize_partial_accumulation(micro)
                if self._optimizer_step():
                    self.global_step += 1
                    self.progress_bar.update(1)
                    logger.info(
                        "Applied incomplete gradient accumulation at end of epoch %d", epoch + 1
                    )
                micro = 0
            # Epoch end is also an explicit logging/synchronization boundary.
            self._flush_delayed_counters()

            # Fix Bug #3: Safe division for epoch loss
            avg_epoch_loss = (
                float((epoch_loss / epoch_steps).item()) if epoch_steps else 0.0
            )
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_epoch_loss:.4f}")

            if self.config.eval_strategy == "epoch":
                if eval_dataset:
                    prev_best = self.best_metric
                    eval_metrics = self._evaluate(eval_dataset)
                    self.model.train()
                    self.processor.change_mode(is_training=True)
                    if self.config.early_stopping and self._check_early_stopping(eval_metrics, prev_best):
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break
                self._save_checkpoint(f"checkpoint-epoch-{epoch + 1}")

            if self.global_step >= max_steps:
                break

        self.progress_bar.close()
        self.progress_bar = None

        if self.is_main_process:
            self._save_checkpoint("final")
            if self.config.report_to_wandb:
                import wandb
                wandb.summary["best_metric"] = self.best_metric
                wandb.summary["total_steps"] = self.global_step
                wandb.finish()

        total_time = time.time() - start_time

        self._cleanup_distributed()

        return {
            "total_steps": self.global_step,
            "total_epochs": self.epoch + 1,
            "total_time_seconds": total_time,
            "samples_per_second": samples_seen / total_time,
            "best_metric": self.best_metric,
            "train_metrics_history": self.train_metrics_history,
            "eval_metrics_history": self.eval_metrics_history,
        }

    def _evaluate(self, eval_dataset: ExtractorDataset) -> Dict[str, float]:
        logger.info("Running evaluation...")
        self.model.eval()
        self.processor.change_mode(is_training=False)

        eval_loader = self._create_dataloader(eval_dataset, self.config.eval_batch_size, shuffle=False, is_training=False)

        # Fix Bug #4: Check if eval dataloader is empty
        if len(eval_loader) == 0:
            logger.warning(
                f"Evaluation dataloader is empty. Dataset size: {len(eval_dataset)}, "
                f"Batch size: {self.config.eval_batch_size}. Skipping evaluation."
            )
            return {
                "eval_loss": 0.0,
                "eval_classification_loss": 0.0,
                "eval_structure_loss": 0.0,
                "eval_count_loss": 0.0,
                "step": self.global_step,
                "epoch": self.epoch,
            }

        total_loss = 0.0
        total_cls_loss = 0.0
        total_struct_loss = 0.0
        total_count_loss = 0.0
        num_batches = 0
        proposal_counts: Dict[str, torch.Tensor] = {}

        use_amp = self.config.fp16 or self.config.bf16
        amp_dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        model = self.model.module if hasattr(self.model, "module") else self.model
        boundary_head = getattr(model, "boundary_head", None)
        previous_collect = getattr(boundary_head, "collect_diagnostics", False)
        if boundary_head is not None and self.config.log_proposal_metrics:
            boundary_head.collect_diagnostics = True

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", disable=not self.is_main_process):
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=use_amp,
                    dtype=amp_dtype,
                ):
                    outputs = self.model(batch)
                output_metrics = getattr(outputs, "metrics", None)
                if output_metrics:
                    for key, value in output_metrics.items():
                        detached = value.detach()
                        proposal_counts[key] = (
                            proposal_counts[key] + detached
                            if key in proposal_counts else detached
                        )

                # Fix C (Finding 1): a batch with no supervision yields no loss.
                # Skip-and-warn rather than dereferencing None (which would crash
                # eval on, e.g., an unlabeled batch).
                batch_loss = outputs.get("total_loss")
                if batch_loss is None:
                    logger.warning(
                        "Skipping eval batch with no loss (no supervision present)"
                    )
                    continue

                # Fix Bug #10: Move tensors to CPU to prevent memory leak
                total_loss += batch_loss.detach().cpu().item()
                total_cls_loss += outputs.get("classification_loss", torch.tensor(0)).detach().cpu().item()
                total_struct_loss += outputs.get("structure_loss", torch.tensor(0)).detach().cpu().item()
                total_count_loss += outputs.get("count_loss", torch.tensor(0)).detach().cpu().item()
                num_batches += 1
        if boundary_head is not None:
            boundary_head.collect_diagnostics = previous_collect

        # Fix Bug #4: Safe division for evaluation metrics
        metrics = {
            "eval_loss": self._safe_divide(total_loss, num_batches, default=0.0),
            "eval_classification_loss": self._safe_divide(total_cls_loss, num_batches, default=0.0),
            "eval_structure_loss": self._safe_divide(total_struct_loss, num_batches, default=0.0),
            "eval_count_loss": self._safe_divide(total_count_loss, num_batches, default=0.0),
            "step": self.global_step,
            "epoch": self.epoch,
        }
        metrics.update(self._proposal_metric_ratios(proposal_counts, "eval"))

        if self.compute_metrics:
            metrics.update(self.compute_metrics(self.model, eval_dataset))

        self._log_metrics(metrics, prefix="eval")
        self.eval_metrics_history.append(metrics)

        metric_value = metrics.get(self.config.metric_for_best, metrics["eval_loss"])
        is_best = (
            (self.config.greater_is_better and metric_value > self.best_metric) or
            (not self.config.greater_is_better and metric_value < self.best_metric)
        )

        if is_best:
            self.best_metric = metric_value
            if self.config.save_best:
                self._save_checkpoint("best")
            logger.info(f"New best {self.config.metric_for_best}: {self.best_metric:.4f}")

        return metrics

    def _check_early_stopping(self, metrics: Dict[str, float], prev_best: Optional[float] = None) -> bool:
        metric_value = metrics.get(self.config.metric_for_best, metrics["eval_loss"])
        compare_against = prev_best if prev_best is not None else self.best_metric
        if self.config.greater_is_better:
            improved = metric_value > compare_against + self.config.early_stopping_threshold
        else:
            improved = metric_value < compare_against - self.config.early_stopping_threshold

        if improved:
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.config.early_stopping_patience

    def _log_metrics(self, metrics: Union[Dict, TrainingMetrics], prefix: str = ""):
        """Log metrics with safe handling of edge cases."""
        if isinstance(metrics, TrainingMetrics):
            metrics = metrics.to_dict()
        
        # Handle empty metrics gracefully
        if not metrics:
            logger.warning("Attempted to log empty metrics")
            return

        # Update progress bar with key metrics
        if self.is_main_process and self.progress_bar is not None:
            postfix = {}
            for key, value in metrics.items():
                if key in ["loss", "learning_rate", "throughput"]:
                    if isinstance(value, float):
                        if math.isnan(value):
                            postfix[key] = "NaN"
                        elif math.isinf(value):
                            postfix[key] = "Inf"
                        elif key == "learning_rate":
                            postfix["lr"] = f"{value:.2e}"
                        elif key == "throughput":
                            postfix["samples/s"] = f"{value:.1f}"
                        else:
                            postfix[key] = f"{value:.4f}"
            
            # Add epoch info if available
            if "epoch" in metrics:
                postfix["epoch"] = f"{metrics['epoch']:.1f}"
            
            if postfix:
                self.progress_bar.set_postfix(postfix)

        # W&B logging
        if self.config.report_to_wandb and self.is_main_process:
            try:
                import wandb
                # Filter out NaN and Inf values for wandb
                wandb_metrics = {
                    k: v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v))
                }
                if wandb_metrics:
                    wandb.log(wandb_metrics, step=self.global_step)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

        if prefix == "train":
            self.train_metrics_history.append(metrics)

    def _save_checkpoint(self, name: str):
        if not self.is_main_process:
            return

        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(exist_ok=True)

        save_start = time.time()

        if self.config.use_lora and self.config.save_adapter_only:
            # PEFT-native format only: ``self.model`` is a ``PeftModel``, so
            # ``save_pretrained`` writes ``adapter_config.json`` (with
            # ``peft_type: "LORA"``) + ``adapter_model.safetensors``, which is
            # what ``PeftModel.from_pretrained`` and every downstream PEFT
            # consumer expect. The pre-migration code also invoked
            # ``gliner2.training.lora.save_lora_adapter`` here to emit the
            # legacy ``adapter_weights.safetensors`` + gliner2 ``LoRAAdapterConfig``
            # alongside the PEFT files, but ``LoRAAdapterConfig.save`` writes
            # to the same ``adapter_config.json`` path and **clobbers** the
            # PEFT config (strips ``peft_type``), so any PEFT reader then blew
            # up with ``KeyError: 'peft_type'`` at ``PeftConfig._get_peft_type``.
            # Legacy callers that still need the gliner2-native directory
            # shape can invoke ``save_lora_adapter`` directly on a
            # checkpoint dir after training — the shim is preserved with
            # ``PendingDeprecationWarning`` in ``gliner2/training/lora.py``.
            self.model.save_pretrained(str(checkpoint_dir))
            checkpoint_type = "adapter"
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            # Full model save: merge LoRA weights if present. Import both
            # helpers together: ``merge_lora_weights`` was previously
            # referenced without an import, raising NameError on the first
            # full-checkpoint save under LoRA. The import stays inside the
            # LoRA branch because it needs peft, which non-LoRA training
            # does not.
            lora_was_merged = False
            if self.config.use_lora and self.lora_layers:
                from gliner2.training.lora import merge_lora_weights, unmerge_lora_weights

                first_lora_layer = next(iter(self.lora_layers.values()))
                if not first_lora_layer.merged:
                    num_merged = merge_lora_weights(self.model)
                    lora_was_merged = True

            # Save the model (with merged weights if LoRA was used)
            model = self.model.module if self.is_distributed else self.model
            model.save_pretrained(str(checkpoint_dir))
            
            # Unmerge weights after saving to continue training with LoRA
            if lora_was_merged:
                unmerge_lora_weights(self.model)
            
            # Save LoRA configuration if used
            if self.config.use_lora:
                lora_config_dict = {
                    "use_lora": True,
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                    "lora_dropout": self.config.lora_dropout,
                    "lora_target_modules": self.config.lora_target_modules,
                    "merged": True,
                }
                import json
                with open(checkpoint_dir / "lora_config.json", "w") as f:
                    json.dump(lora_config_dict, f, indent=2)
            
            checkpoint_type = "full"
            trainable_params = sum(p.numel() for p in self.model.parameters())
        
        save_time = time.time() - save_start
        checkpoint_size_mb = sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file()) / (1024 * 1024)
        
        # World-class logging
        logger.info(
            f"💾 Saved {checkpoint_type} checkpoint '{name}' | "
            f"step {self.global_step} | epoch {self.epoch + 1:.1f} | "
            f"{trainable_params:,} params | {checkpoint_size_mb:.1f}MB | {save_time:.1f}s"
        )

        # Save model artifacts to W&B for best and final checkpoints
        if self.config.report_to_wandb and name in ["best", "final"]:
            try:
                import wandb
                artifact = wandb.Artifact(
                    name=f"model-{self.config.experiment_name}-{name}",
                    type="model",
                    metadata={
                        "step": self.global_step,
                        "epoch": self.epoch,
                        "checkpoint_type": checkpoint_type,
                        "params": trainable_params,
                        "size_mb": checkpoint_size_mb,
                    }
                )
                artifact.add_dir(str(checkpoint_dir))
                wandb.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"W&B artifact upload failed: {e}")

        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        if self.config.save_total_limit <= 0:
            return

        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: x.stat().st_mtime,
        )
        protected = {"best", "final"}
        checkpoints = [c for c in checkpoints if c.name not in protected]

        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            shutil.rmtree(oldest)
            logger.info(f"Removed old checkpoint: {oldest.name}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from a checkpoint (adapter-only or full)."""
        checkpoint_dir = Path(checkpoint_path)
        is_adapter = (checkpoint_dir / "adapter_config.json").exists()

        if is_adapter:
            from gliner2.training.lora import PeftModel, _PeftLoraLayer

            logger.info("Loading LoRA adapter from %s", checkpoint_path)
            base = self.model.get_base_model() if isinstance(self.model, PeftModel) else self.model
            self.model = PeftModel.from_pretrained(base, str(checkpoint_dir))
            self.model.to(self.device)
            self.lora_layers = {n: m for n, m in self.model.named_modules() if isinstance(m, _PeftLoraLayer)}
            self.model._lora_layers = self.lora_layers
        else:
            self.model = self.model.__class__.from_pretrained(str(checkpoint_dir))
            # if not bf16 and not fp16, then we need to move the model to the device
            if not self.config.bf16 and not self.config.fp16:
                logger.info("Moving model to %s", self.device)
                self.model.float()
            self.model.to(self.device)
            if self.config.use_lora:
                logger.info("Applying LoRA to loaded model...")
                self.lora_layers = {}
                self._setup_lora()

        logger.info("Loaded checkpoint: %s", checkpoint_path)


# Backward-compatible alias: the trainer is architecture-neutral now.
GLiNER2Trainer = ExtractorTrainer


# =============================================================================
# Convenience Functions
# =============================================================================

def train_gliner2(
        model_path: str,
        train_data: TrainDataInput,
        output_dir: str = "./output",
        eval_data: TrainDataInput = None,
        **config_kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for training GLiNER2.

    Parameters
    ----------
    model_path : str
        Path to pretrained model.
    train_data : TrainDataInput
        Training data in any supported format:
        - JSONL path(s)
        - List of InputExample
        - TrainingDataset
        - List of dicts
    output_dir : str
        Output directory for checkpoints.
    eval_data : TrainDataInput, optional
        Evaluation data.
    **config_kwargs
        Additional TrainingConfig parameters.

    Returns
    -------
    Dict[str, Any]
        Training results.

    Examples
    --------
    >>> # Train with JSONL file
    >>> results = train_gliner2("model-path", "train.jsonl", num_epochs=10)

    >>> # Train with multiple JSONL files
    >>> results = train_gliner2("model-path", ["train1.jsonl", "train2.jsonl"])

    >>> # Train with InputExample list
    >>> examples = [InputExample(...), ...]
    >>> results = train_gliner2("model-path", examples)

    >>> # Train with TrainingDataset
    >>> dataset = TrainingDataset.load("train.jsonl")
    >>> results = train_gliner2("model-path", dataset)
    """
    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained(model_path)
    config = TrainingConfig(output_dir=output_dir, **config_kwargs)

    trainer = GLiNER2Trainer(model=model, config=config)
    return trainer.train(train_data=train_data, eval_data=eval_data)
