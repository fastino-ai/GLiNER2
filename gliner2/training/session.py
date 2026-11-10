"""Bounded operations over the same engine used by ``ExtractorTrainer.train``.

The first adapter supports a single-process, full-weight span extractor. It
does not own queues, epochs, placement, remote storage or checkpoint identity.
Callers serialize operations and provide trusted, session-owned local paths.
"""
from __future__ import annotations

import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gliner2.training.data import InputExample
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig, get_scheduler


class TrainingSession:
    """Public batch, update, state and export operations for a span extractor.

    This bounded proof deliberately rejects LoRA, distributed execution and
    boundary models until their state/export equivalence is qualified. A
    process must not interleave sessions: PyTorch and processor RNGs are global.
    """

    def __init__(self, model: torch.nn.Module, config: TrainingConfig, *, max_steps: int):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if config.use_lora or config.local_rank != -1 or getattr(model, "architecture", "span") != "span":
            raise ValueError("session proof supports single-process full-weight span models only")
        if config.fp16 or config.bf16:
            raise ValueError("session proof supports fp32 only")
        if config.max_train_samples != -1 or config.max_eval_samples != -1 or config.allow_invalid_samples:
            raise ValueError("session batches must not silently truncate or skip samples")
        if not config.strict_training or config.ignore_nonfinite_losses:
            raise ValueError("session batches require strict finite-loss training")
        self.trainer = ExtractorTrainer(model, config)
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
        return {"loss": float(loss.detach().cpu())}

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
        return {"loss": float(loss.cpu())}

    def optim_step(self) -> int:
        """Apply accumulated gradients with the managed trainer's optimizer."""
        self._ready()
        if not self._pending:
            raise ValueError("no accumulated gradients")
        try:
            self.trainer._renormalize_partial_accumulation(self._pending)
            self.trainer._optimizer_step()
        except BaseException:
            self._broken = True
            raise
        self.trainer.global_step += 1
        self._pending = 0
        return self.step

    def _configuration(self) -> dict[str, Any]:
        config = asdict(self.trainer.config)
        config.pop("output_dir")
        return {"training": config, "model": self.model.config.to_dict(),
                "max_steps": self.trainer._planned_max_steps}

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
            "version": 1, "configuration": self._configuration(),
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.trainer.scheduler.state_dict(), "scaler": self.trainer.scaler.state_dict(),
            "step": self.step, "pending": self._pending,
            "gradients": {n: p.grad for n, p in self.model.named_parameters() if p.grad is not None},
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.trainer.device.type == "cuda" else [],
            "device_type": self.trainer.device.type,
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
        if payload["version"] != 1 or payload["configuration"] != self._configuration():
            raise ValueError("checkpoint configuration does not match session")
        if payload["device_type"] != self.trainer.device.type:
            raise ValueError("cross-device continuation is not qualified")
        try:
            self.model.load_state_dict(payload["model"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer"])
            self.trainer.scheduler.load_state_dict(payload["scheduler"])
            self.trainer.scaler.load_state_dict(payload["scaler"])
            self.trainer.global_step = payload["step"]
            self._pending = payload["pending"]
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

    def export_weights(self, directory: str | Path) -> None:
        """Export the native full-model artifact accepted by AutoExtractor."""
        self._ready()
        if self._pending:
            raise ValueError("apply accumulated gradients before exporting weights")
        self.model.save_pretrained(str(directory))
