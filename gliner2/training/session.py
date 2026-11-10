"""Bounded operations over the same engine used by ``ExtractorTrainer.train``.

Supports single-process span and boundary extractors with full weights or LoRA.
It does not own queues, placement, remote storage or checkpoint identity.
Callers serialize operations and provide trusted, session-owned local paths.
"""
from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from peft import PeftModel

from gliner2.training.data import InputExample
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig, get_scheduler


class TrainingSession:
    """Public batch, update, state and native export operations.

    A process must not interleave sessions: PyTorch and processor RNGs are global.
    Distributed workers and graph compilation require separate qualification.
    Mixed precision requires CUDA; CPU requests fail rather than downgrade.
    """

    def __init__(self, model: torch.nn.Module, config: TrainingConfig, *, max_steps: int,
                 trainer_factory: Callable[..., ExtractorTrainer] = ExtractorTrainer):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if config.local_rank != -1 or config.compile_model:
            raise ValueError("sessions require uncompiled single-process execution")
        if getattr(model, "architecture", "span") not in {"span", "boundary"}:
            raise ValueError("unsupported extractor architecture")
        if (config.fp16 or config.bf16) and not torch.cuda.is_available():
            raise ValueError("mixed precision sessions require CUDA")
        if config.bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            raise ValueError("bfloat16 sessions require native GPU support, not emulation")
        if config.num_workers != 0:
            raise ValueError("session preprocessing requires num_workers=0 for replayable RNG")
        if config.skip_step_errors or config.on_capacity_exceeded != "raise" or config.dry_run_recall_steps:
            raise ValueError("sessions must not silently discard work or run recall-only training")
        if config.max_train_samples != -1 or config.max_eval_samples != -1 or config.allow_invalid_samples:
            raise ValueError("session batches must not silently truncate or skip samples")
        if not config.strict_training or config.ignore_nonfinite_losses:
            raise ValueError("session batches require strict finite-loss training")
        if isinstance(model, PeftModel):
            from gliner2.training.session_adapters import validate_loaded_adapter
            validate_loaded_adapter(model, config)
        self.trainer = trainer_factory(model=model, config=copy.deepcopy(config))
        config = self.trainer.config
        self.trainer._planned_max_steps = max_steps
        self.trainer.optimizer = self.trainer._create_optimizer()
        warmup = config.warmup_steps or int(max_steps * config.warmup_ratio)
        self.trainer.scheduler = get_scheduler(self.optimizer, config.scheduler_type, max_steps, warmup, config.num_cycles)
        self.trainer.scaler = torch.amp.GradScaler(self.trainer.device.type, enabled=config.fp16)
        self.optimizer.zero_grad(set_to_none=True)
        self._pending = 0
        self._broken = False

    @property
    def model(self) -> torch.nn.Module:
        return self.trainer.model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.trainer.optimizer

    @property
    def step(self) -> int:
        return self.trainer.global_step

    @property
    def config(self) -> TrainingConfig:
        return self.trainer.config

    @property
    def max_steps(self) -> int:
        return self.trainer._planned_max_steps

    @property
    def pending_batches(self) -> int:
        return self._pending

    def _ready(self) -> None:
        if self._broken:
            raise RuntimeError("session failed; restore a checkpoint into a fresh session")

    def _batch(self, rows: list[InputExample], *, training: bool):
        if not isinstance(rows, list) or not 0 < len(rows) <= self.trainer.config.batch_size:
            raise ValueError("batch must contain 1..batch_size examples")
        if not all(isinstance(row, InputExample) for row in rows):
            raise ValueError("batch must contain InputExample records")
        self.model.train(training)
        self.trainer.processor.change_mode(is_training=training)
        # Dataset shuffling is disabled; bounded calls retain caller ordering.
        data = self.trainer._prepare_data(rows, is_train=False)
        loader = self.trainer._create_dataloader(data, len(rows), shuffle=False, is_training=training)
        return next(iter(loader))

    def forward(self, rows: list[InputExample]) -> dict[str, float]:
        """Return supervised held-out loss without updating parameters/gradients."""
        self._ready()
        batch = self._batch(rows, training=False)
        with torch.no_grad(), torch.amp.autocast(
            self.trainer.device.type, enabled=self.trainer.config.fp16 or self.trainer.config.bf16,
            dtype=torch.bfloat16 if self.trainer.config.bf16 else torch.float16,
        ):
            loss = self.model(batch).get("total_loss")
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("forward produced no finite supervised loss")
        return {"loss": float(loss.detach().cpu()), "num_tokens": float(batch.attention_mask.sum().item())}

    def forward_backward(self, rows: list[InputExample]) -> dict[str, float]:
        """Accumulate one bounded batch using the managed trainer's backward."""
        self._ready()
        if self._pending >= self.trainer.config.gradient_accumulation_steps:
            raise ValueError("apply accumulated gradients before another batch")
        if self.step >= self.trainer._planned_max_steps:
            raise ValueError("session reached max_steps")
        batch = self._batch(rows, training=True)
        try:
            loss = self.trainer._backward_one(
                batch, self._pending,
                self.trainer.config.fp16 or self.trainer.config.bf16,
                torch.bfloat16 if self.trainer.config.bf16 else torch.float16,
            )
            self.trainer._flush_delayed_counters()
        except BaseException:
            # A partial backward must never be retried against mutable state.
            self._broken = True
            raise
        self._pending += 1
        return {"loss": float(loss.cpu()), "num_tokens": float(batch.attention_mask.sum().item())}

    def optim_step(self) -> int:
        """Apply accumulated gradients with the managed trainer's optimizer."""
        self._ready()
        if not self._pending:
            raise ValueError("no accumulated gradients")
        try:
            self.trainer._renormalize_partial_accumulation(self._pending)
            applied = self.trainer._optimizer_step()
            if not applied:
                raise FloatingPointError("optimizer skipped a non-finite update; restore and retry")
        except BaseException:
            self._broken = True
            raise
        self.trainer.global_step += 1
        self._pending = 0
        return self.step

    def _configuration(self) -> dict[str, Any]:
        config = asdict(self.trainer.config)
        config.pop("output_dir")
        adapter = None
        if isinstance(self.model, PeftModel):
            adapter = json.loads(json.dumps(self.model.peft_config["default"].to_dict(),
                                           default=lambda value: sorted(value) if isinstance(value, set) else value.value))
            adapter.pop("base_model_name_or_path", None)
        return {"training": config, "model": self.model.config.to_dict(),
                "adapter": adapter, "sampling": asdict(self.trainer.processor.sampling_config),
                "max_steps": self.max_steps}

    def save_state(self, directory: str | Path) -> None:
        """Save weights, Adam moments, schedule, scaler, gradients and RNGs.

        The final file replaces an incomplete write atomically. Storage and
        checkpoint services own remote commit markers and authorization.
        """
        self._ready()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        numpy_state = np.random.get_state()
        payload = {
            "version": 2, "configuration": self._configuration(),
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.trainer.scheduler.state_dict(), "scaler": self.trainer.scaler.state_dict(),
            "step": self.step, "pending": self._pending,
            "gradients": {n: p.grad for n, p in self.model.named_parameters() if p.grad is not None},
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.trainer.device.type == "cuda" else [],
            "device_type": self.trainer.device.type,
            "engine_type": f"{type(self.trainer).__module__}.{type(self.trainer).__qualname__}",
            "engine_state": (self.trainer.session_state_dict()
                             if hasattr(self.trainer, "session_state_dict") else None),
        }
        temporary = path / "training-state.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, path / "training-state.pt")

    def load_state(self, directory: str | Path) -> None:
        """Restore complete state into a fresh, identically configured session.

        Checkpoints are service-owned, never arbitrary client pickle uploads;
        ``weights_only`` also prevents executing arbitrary pickle globals.
        """
        self._ready()
        if self.step or self._pending:
            raise ValueError("restore requires a fresh session")
        payload = torch.load(Path(directory) / "training-state.pt", map_location="cpu", weights_only=True)
        if payload["version"] != 2 or payload["configuration"] != self._configuration():
            raise ValueError("checkpoint configuration does not match session")
        if payload["device_type"] != self.trainer.device.type:
            raise ValueError("cross-device continuation is not qualified")
        engine_type = f"{type(self.trainer).__module__}.{type(self.trainer).__qualname__}"
        if payload.get("engine_type", engine_type) != engine_type:
            raise ValueError("checkpoint trainer engine does not match session")
        if payload.get("engine_state") is not None and not hasattr(self.trainer, "load_session_state_dict"):
            raise ValueError("trainer engine cannot restore its checkpoint state")
        try:
            self.model.load_state_dict(payload["model"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer"])
            self.trainer.scheduler.load_state_dict(payload["scheduler"])
            self.trainer.scaler.load_state_dict(payload["scaler"])
            self.trainer.global_step = payload["step"]
            self._pending = payload["pending"]
            if payload.get("engine_state") is not None:
                self.trainer.load_session_state_dict(payload["engine_state"])
            for name, parameter in self.model.named_parameters():
                gradient = payload["gradients"].get(name)
                parameter.grad = None if gradient is None else gradient.to(parameter.device)
            random.setstate(payload["python_rng"])
            state = payload["numpy_rng"]
            np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
            torch.set_rng_state(payload["torch_rng"])
            if payload["cuda_rng"]:
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
        except BaseException:
            self._broken = True
            raise

    def load_weights(self, directory: str | Path) -> None:
        """Start from trained weights while preserving a fresh run's schedule/RNG.

        The model and adapter topology must match. Optimizer hyperparameters,
        schedule and batching can differ for the new training run.
        """
        self._ready()
        if self.step or self._pending or self.optimizer.state:
            raise ValueError("weights-only restore requires a fresh session")
        payload = torch.load(Path(directory) / "training-state.pt", map_location="cpu", weights_only=True)
        expected = self._configuration()
        if payload["version"] != 2 or any(payload["configuration"][key] != expected[key]
                                         for key in ("model", "adapter")):
            raise ValueError("checkpoint model topology does not match session")
        try:
            self.model.load_state_dict(payload["model"], strict=True)
        except BaseException:
            self._broken = True
            raise

    def _copy_model_to_cpu(self) -> torch.nn.Module:
        # Allocate detached export parameters on CPU directly; deepcopying a
        # CUDA model first would need a second full model on the training GPU.
        memo = {id(parameter): torch.nn.Parameter(parameter.detach().to("cpu", copy=True),
                                                requires_grad=parameter.requires_grad)
                for parameter in self.model.parameters()}
        memo.update({id(buffer): buffer.detach().to("cpu", copy=True) for buffer in self.model.buffers()})
        return copy.deepcopy(self.model, memo)

    def export_weights(self, directory: str | Path, *, checkpoint: str | Path | None = None) -> None:
        """Export the native full-model artifact accepted by AutoExtractor."""
        self._ready()
        if self._pending:
            raise ValueError("apply accumulated gradients before exporting weights")
        exported = self.model
        if checkpoint is not None:
            payload = torch.load(Path(checkpoint) / "training-state.pt", map_location="cpu", weights_only=True)
            if payload["version"] != 2 or payload["configuration"] != self._configuration():
                raise ValueError("checkpoint configuration does not match session")
            if payload["pending"]:
                raise ValueError("cannot export a checkpoint with pending gradients")
            exported = self._copy_model_to_cpu()
            exported.load_state_dict(payload["model"], strict=True)
        if isinstance(exported, PeftModel):
            # Merge a detached copy: export must not perturb live weights,
            # adapter topology, optimizer references or subsequent training.
            if exported is self.model:
                exported = self._copy_model_to_cpu()
            exported = exported.merge_and_unload(safe_merge=True)
            exported.save_pretrained(str(directory))
        else:
            exported.save_pretrained(str(directory))
