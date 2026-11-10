"""Deterministic dataset/epochs orchestration over public session operations.

The plan can also be consumed by a remote async dispatcher: enqueue its batch,
then enqueue an optimizer operation exactly when ``is_update`` is true. Neither
this module nor a remote driver implements a backward pass or optimizer.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Protocol

from gliner2.training.data import InputExample
from gliner2.training.sampler import LengthGroupedSampler
from gliner2.training.trainer import TrainingConfig


class SessionOperations(Protocol):
    config: TrainingConfig
    step: int
    max_steps: int
    pending_batches: int

    def forward(self, rows: list[InputExample]) -> dict[str, float]: ...
    def forward_backward(self, rows: list[InputExample]) -> dict[str, float]: ...
    def optim_step(self) -> int: ...
    def save_state(self, directory: str | Path) -> None: ...
    def load_state(self, directory: str | Path) -> None: ...
    def export_weights(self, directory: str | Path, *, checkpoint: str | Path | None = None) -> None: ...


class ManagedTrainingPlan:
    """Stable batches including the tail, accumulation boundaries and identity.

    The dataset is copied so caller mutations cannot change a running plan.
    ``max_steps`` overrides epochs like the native trainer. Unlike its legacy
    drop-last loader, every example belongs to one batch in a complete epoch.
    The final partial accumulation window is applied at each epoch boundary.
    """

    def __init__(self, examples: list[InputExample], config: TrainingConfig, *,
                 architecture: str = "span", shuffle: bool = True):
        if not examples or not all(isinstance(row, InputExample) for row in examples):
            raise ValueError("managed training requires nonempty InputExample records")
        if architecture not in {"span", "boundary"}:
            raise ValueError("unsupported extractor architecture")
        if config.num_epochs < 1 and config.max_steps <= 0:
            raise ValueError("num_epochs or max_steps must be positive")
        if config.max_train_samples != -1 or config.allow_invalid_samples:
            raise ValueError("managed training cannot discard input examples")
        self._examples = copy.deepcopy(examples)
        self.config = copy.deepcopy(config)
        self.architecture = architecture
        self.shuffle = shuffle
        self.batches_per_epoch = math.ceil(len(examples) / config.batch_size)
        updates = math.ceil(self.batches_per_epoch / config.gradient_accumulation_steps)
        self.max_steps = config.max_steps if config.max_steps > 0 else updates * config.num_epochs
        complete_epochs, remaining_updates = divmod(self.max_steps, updates)
        self.batch_count = complete_epochs * self.batches_per_epoch + min(
            remaining_updates * config.gradient_accumulation_steps, self.batches_per_epoch)
        self._cached_epoch = -1
        self._order: list[int] = []
        settings = asdict(config)
        settings.pop("output_dir")
        canonical = {"examples": [row.to_dict() for row in self._examples],
                     "config": settings, "architecture": architecture, "shuffle": shuffle}
        self.fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()

    def epoch(self, index: int) -> int:
        if not 0 <= index < self.batch_count:
            raise IndexError("batch index is outside the managed plan")
        return index // self.batches_per_epoch

    def is_epoch_end(self, index: int) -> bool:
        self.epoch(index)
        return (index + 1) % self.batches_per_epoch == 0

    def is_update(self, index: int) -> bool:
        self.epoch(index)
        within_epoch = index % self.batches_per_epoch
        return ((within_epoch + 1) % self.config.gradient_accumulation_steps == 0
                or self.is_epoch_end(index) or index + 1 == self.batch_count)

    def batch(self, index: int) -> list[InputExample]:
        epoch = self.epoch(index)
        if self._cached_epoch != epoch:
            if self.shuffle and self.architecture == "boundary" and self.config.group_by_length:
                sampler = LengthGroupedSampler(
                    [max(1, len(row.text.split())) for row in self._examples], self.config.batch_size,
                    window_batches=self.config.length_group_window_batches, seed=self.config.seed)
                sampler.set_epoch(epoch)
                self._order = list(sampler)
            else:
                self._order = list(range(len(self._examples)))
                if self.shuffle:
                    random.Random(self.config.seed + epoch).shuffle(self._order)
            self._cached_epoch = epoch
        start = (index % self.batches_per_epoch) * self.config.batch_size
        return copy.deepcopy([self._examples[i] for i in self._order[start:start + self.config.batch_size]])


    def to_manifest(self, *, eval_examples: list[InputExample] | None = None) -> dict:
        """Serialize the exact batch/update/evaluation plan for a durable dispatcher.

        A consumer executes this schedule rather than reimplementing shuffle or
        accumulation. The content digest includes validation and best/stop policy;
        changing any input therefore changes operation idempotency keys.
        """
        batches = []
        step = 0
        for index in range(self.batch_count):
            self.batch(index)  # Populate the same epoch ordering used by batch().
            start = (index % self.batches_per_epoch) * self.config.batch_size
            update = self.is_update(index)
            step += int(update)
            evaluate = ((self.config.eval_strategy == "epoch" and self.is_epoch_end(index))
                        or (self.config.eval_strategy == "steps" and update
                            and step % self.config.eval_steps == 0))
            batches.append({"indices": self._order[start:start + self.config.batch_size],
                            "epoch": self.epoch(index), "update": update, "evaluate": evaluate})
        config = asdict(self.config)
        config.pop("output_dir")
        manifest = {"version": 1, "source_fingerprint": self.fingerprint,
                    "architecture": self.architecture, "config": config,
                    "max_steps": self.max_steps,
                    "examples": [row.to_dict() for row in self._examples], "batches": batches,
                    "eval_examples": [row.to_dict() for row in eval_examples or []],
                    "eval_batch_size": min(self.config.eval_batch_size, self.config.batch_size),
                    "save_best": self.config.save_best, "greater_is_better": self.config.greater_is_better,
                    "early_stopping_patience": self.config.early_stopping_patience if self.config.early_stopping else None,
                    "early_stopping_threshold": self.config.early_stopping_threshold}
        manifest["fingerprint"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return manifest


class ManagedTraining:
    """In-process reference driver for the same plan used by remote dispatch.

    Call ``advance`` for one bounded microbatch or ``run`` for the whole plan.
    Progress callbacks run after committed updates; callers own reporting and
    native artifact publication. Save/load also persists the exact data cursor.
    """

    def __init__(self, session: SessionOperations, plan: ManagedTrainingPlan, *,
                 eval_examples: list[InputExample] | None = None, keep_best: bool | None = None):
        if session.step or session.pending_batches:
            raise ValueError("managed driver requires a fresh session or load_state")
        if session.max_steps != plan.max_steps:
            raise ValueError("session schedule does not match the managed plan")
        left, right = asdict(session.config), asdict(plan.config)
        left.pop("output_dir")
        right.pop("output_dir")
        if left != right:
            raise ValueError("session configuration does not match the managed plan")
        if session.config.report_to_wandb:
            raise ValueError("managed session callbacks own external reporting")
        if session.config.metric_for_best != "eval_loss":
            raise ValueError("managed sessions select best checkpoints by eval_loss")
        if session.config.eval_strategy not in {"steps", "epoch", "no"}:
            raise ValueError("unsupported evaluation strategy")
        if session.config.eval_steps < 1:
            raise ValueError("eval_steps must be positive")
        self.session = session
        self.plan = plan
        self.eval_examples = copy.deepcopy(eval_examples or [])
        self.cursor = 0
        self._loss_sum = 0.0
        self._samples = 0
        self.eval_history: list[dict[str, float]] = []
        self.train_history: list[dict[str, float]] = []
        self.best_metric: float | None = None
        self.best_step: int | None = None
        self.keep_best = session.config.save_best if keep_best is None else keep_best
        self._best_root = tempfile.TemporaryDirectory(prefix="gliner-session-best-")
        self._patience = 0
        self._stopped = False
        self._failed = False
        self._started = time.monotonic()
        self._previous_elapsed = 0.0

    @property
    def complete(self) -> bool:
        return self._stopped or self.cursor == self.plan.batch_count

    def evaluate(self) -> float:
        if not self.eval_examples:
            raise ValueError("no evaluation examples")
        size = min(self.session.config.eval_batch_size, self.session.config.batch_size)
        total = 0.0
        for start in range(0, len(self.eval_examples), size):
            batch = self.eval_examples[start:start + size]
            total += self.session.forward(batch)["loss"] * len(batch)
        return total / len(self.eval_examples)

    def advance(self) -> dict[str, float]:
        if self._failed:
            raise RuntimeError("managed training failed; restore a checkpoint into a fresh driver")
        if self.complete:
            raise StopIteration("managed training is complete")
        try:
            return self._advance()
        except BaseException:
            self._failed = True
            raise

    def _advance(self) -> dict[str, float]:
        if self.complete:
            raise StopIteration("managed training is complete")
        index = self.cursor
        batch = self.plan.batch(index)
        metrics = self.session.forward_backward(batch)
        update = self.plan.is_update(index)
        if update:
            self.session.optim_step()
        self._loss_sum += metrics["loss"] * len(batch)
        self._samples += len(batch)
        self.cursor += 1
        metrics = dict(metrics, step=float(self.session.step),
                       epoch=float(self.plan.epoch(index)), optimizer_update=float(update))
        cfg = self.session.config
        should_eval = ((cfg.eval_strategy == "epoch" and self.plan.is_epoch_end(index))
                       or (cfg.eval_strategy == "steps" and update and self.session.step % cfg.eval_steps == 0))
        if self.eval_examples and should_eval:
            metrics["eval_loss"] = self.evaluate()
            self.eval_history.append({"step": float(self.session.step), "epoch": metrics["epoch"], "eval_loss": metrics["eval_loss"]})
            self._select_best(metrics["eval_loss"])
        if update:
            self.train_history.append(dict(metrics))
        return metrics

    def _select_best(self, loss: float) -> None:
        cfg = self.session.config
        improvement = (float("inf") if self.best_metric is None else
                       loss - self.best_metric if cfg.greater_is_better else self.best_metric - loss)
        if improvement > cfg.early_stopping_threshold:
            self.best_metric = loss
            self.best_step = self.session.step
            self._patience = 0
            if self.keep_best:
                self.session.save_state(self._best_root.name)
        else:
            self._patience += 1
        if cfg.early_stopping and self._patience >= cfg.early_stopping_patience:
            self._stopped = True

    def export_weights(self, directory: str | Path) -> None:
        """Export selected best weights without corrupting live optimizer state."""
        checkpoint = self._best_root.name if self.keep_best and self.best_step is not None else None
        self.session.export_weights(directory, checkpoint=checkpoint)

    def close(self) -> None:
        self._best_root.cleanup()

    def run(self, on_step: Callable[[int, dict[str, float]], None] | None = None) -> dict:
        while not self.complete:
            metrics = self.advance()
            if on_step is not None and metrics["optimizer_update"]:
                on_step(self.session.step, metrics)
        elapsed = self._previous_elapsed + time.monotonic() - self._started
        return {"total_steps": self.session.step,
                "total_epochs": math.ceil(self.cursor / self.plan.batches_per_epoch),
                "total_time_seconds": elapsed, "samples_per_second": self._samples / max(elapsed, 1e-9),
                "best_metric": self.best_metric if self.best_metric is not None else float("nan"),
                "train_metrics_history": self.train_history, "eval_metrics_history": self.eval_history,
                "selected_step": self.best_step if self.keep_best and self.best_step is not None else self.session.step,
                "samples_seen": self._samples, "train_loss": self._loss_sum / self._samples}

    def save_state(self, directory: str | Path) -> None:
        """Atomically publish an immutable checkpoint including the data cursor."""
        if self._failed:
            raise RuntimeError("cannot checkpoint a failed managed operation")
        target = Path(directory)
        if target.exists():
            raise FileExistsError("managed checkpoint already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".managed-", dir=target.parent))
        try:
            self.session.save_state(staging)
            digest = hashlib.sha256((staging / "training-state.pt").read_bytes()).hexdigest()
            metadata = {"version": 1, "plan": self.plan.fingerprint, "cursor": self.cursor,
                        "state_sha256": digest, "loss_sum": self._loss_sum, "samples": self._samples,
                        "eval_history": self.eval_history, "train_history": self.train_history,
                        "best_metric": self.best_metric, "best_step": self.best_step,
                        "keep_best": self.keep_best, "patience": self._patience, "stopped": self._stopped,
                        "elapsed": self._previous_elapsed + time.monotonic() - self._started,
                        "eval_examples": [row.to_dict() for row in self.eval_examples]}
            if self.keep_best and self.best_step is not None:
                shutil.copytree(self._best_root.name, staging / "best")
            (staging / "managed.json").write_text(json.dumps(metadata, sort_keys=True))
            os.rename(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def load_state(self, directory: str | Path) -> None:
        if self.cursor:
            raise ValueError("managed restore requires a fresh driver")
        path = Path(directory)
        metadata = json.loads((path / "managed.json").read_text())
        digest = hashlib.sha256((path / "training-state.pt").read_bytes()).hexdigest()
        if (metadata["version"] != 1 or metadata["plan"] != self.plan.fingerprint
                or metadata["state_sha256"] != digest
                or metadata["keep_best"] != self.keep_best
                or metadata["eval_examples"] != [row.to_dict() for row in self.eval_examples]
                or not 0 <= metadata["cursor"] <= self.plan.batch_count):
            raise ValueError("managed checkpoint does not match its plan or state")
        self.session.load_state(path)
        self.cursor = metadata["cursor"]
        self._loss_sum = metadata["loss_sum"]
        self._samples = metadata["samples"]
        self.eval_history = metadata["eval_history"]
        self.train_history = metadata["train_history"]
        self.best_metric = metadata["best_metric"]
        self.best_step = metadata["best_step"]
        self._patience = metadata["patience"]
        self._stopped = metadata["stopped"]
        self._previous_elapsed = metadata["elapsed"]
        self._started = time.monotonic()
        if self.keep_best and self.best_step is not None:
            shutil.copytree(path / "best", self._best_root.name, dirs_exist_ok=True)
