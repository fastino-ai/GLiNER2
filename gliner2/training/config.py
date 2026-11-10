"""Torch-free training configuration shared by planning and the native engine."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional


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
    ddp_static_graph: bool = False
    dry_run_recall_steps: int = 0
    gate_recall: float = 0.97
    gate_long_recall: float = 0.93
    # Gold-capacity overflow policy for boundary targets: "raise" (default,
    # no silent loss), "truncate_with_warning", or "skip_sample".
    on_capacity_exceeded: str = "raise"
    dedupe_contained_entity_spans: bool = True
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
