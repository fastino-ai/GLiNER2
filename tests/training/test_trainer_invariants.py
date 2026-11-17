"""Trainer generalization: alias, strict config, disjoint optimizer groups."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from gliner2.training.trainer import (
    ExtractorTrainer,
    GLiNER2Trainer,
    TrainingConfig,
)
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model


def test_gliner2trainer_is_extractor_trainer_alias():
    assert GLiNER2Trainer is ExtractorTrainer


def test_strict_training_defaults():
    config = TrainingConfig()
    assert config.strict_training is True
    assert config.skip_step_errors is False
    assert config.allow_invalid_samples is False
    assert config.log_proposal_metrics is True
    assert config.debug_global_steps == []


def test_optimizer_groups_are_disjoint_and_complete(tmp_path):
    model = build_tiny_span_model()
    config = TrainingConfig(output_dir=str(tmp_path / "out"), num_workers=0, fp16=False)
    trainer = ExtractorTrainer(model=model, config=config)

    optimizer = trainer._create_optimizer()  # asserts disjoint/complete internally
    grouped = sum(len(g["params"]) for g in optimizer.param_groups)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert grouped == trainable

    # Encoder params land in the encoder LR group, task params in the task group.
    enc_group, task_group = optimizer.param_groups
    assert enc_group["lr"] == config.encoder_lr
    assert task_group["lr"] == config.task_lr


def test_failed_batch_records_cpu_side_samples(tmp_path):
    model = build_tiny_span_model()
    output_dir = tmp_path / "out"
    trainer = ExtractorTrainer(
        model=model,
        config=TrainingConfig(output_dir=str(output_dir), num_workers=0, fp16=False),
    )
    trainer.epoch = 2
    trainer.global_step = 476
    batch = SimpleNamespace(
        original_texts=["Alice works at Acme."],
        original_schemas=[{"entities": {"person": ["Alice"]}}],
    )

    trainer._record_failed_batch(
        batch,
        data_loader_step=953,
        error=RuntimeError("device-side assert triggered"),
    )

    record = json.loads((output_dir / "failed_batches.jsonl").read_text())
    assert record["epoch"] == 2
    assert record["global_step"] == 476
    assert record["data_loader_step"] == 953
    assert record["error_type"] == "RuntimeError"
    assert record["samples"][0]["text"] == "Alice works at Acme."
    assert len(record["samples"][0]["sha256"]) == 64


def test_debug_batch_records_complete_preprocessed_state(tmp_path):
    model = build_tiny_span_model()
    output_dir = tmp_path / "out"
    trainer = ExtractorTrainer(
        model=model,
        config=TrainingConfig(output_dir=str(output_dir), num_workers=0, fp16=False),
    )
    trainer.global_step = 477
    batch = SimpleNamespace(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        original_texts=["Alice works at Acme."],
        original_schemas=[{"entities": {"person": ["Alice"]}}],
    )

    trainer._record_debug_batch(
        batch,
        data_loader_step=954,
        micro_batch_in_window=0,
        is_last_micro=False,
    )

    record = json.loads((output_dir / "debug_batches.jsonl").read_text())
    assert record["global_step_before_batch"] == 477
    assert record["data_loader_step"] == 954
    assert record["batch"]["input_ids"] == {
        "__type__": "tensor",
        "dtype": "torch.int64",
        "shape": [1, 3],
        "values": [[1, 2, 3]],
    }
    assert record["batch"]["original_texts"] == ["Alice works at Acme."]


def test_boundary_head_encoders_receive_task_lr(tmp_path):
    from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model

    model = build_tiny_boundary_model()
    config = TrainingConfig(
        output_dir=str(tmp_path / "out"),
        encoder_lr=1e-5,
        task_lr=5e-4,
        num_workers=0,
        fp16=False,
        bf16=False,
    )
    trainer = ExtractorTrainer(model=model, config=config)
    optimizer = trainer._create_optimizer()

    id_to_name = {id(p): name for name, p in model.named_parameters()}
    enc_names = [id_to_name[id(p)] for p in optimizer.param_groups[0]["params"]]
    task_names = [id_to_name[id(p)] for p in optimizer.param_groups[1]["params"]]

    assert enc_names and all(n.startswith("encoder.") for n in enc_names)
    head_encoders = [
        n for n in task_names
        if n.startswith("boundary_head.boundary_encoder.") or n.startswith("boundary_head.candidate_encoder.")
    ]
    assert head_encoders, "expected boundary_encoder and candidate_encoder parameters in task_lr group"
