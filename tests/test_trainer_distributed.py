from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.nn as nn

from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


class _StubModel:
    def __init__(self):
        self.to = Mock(return_value=self)
        self.processor = Mock()


def _mock_distributed_runtime(monkeypatch, *, rank: int, world_size: int = 2):
    init_process_group = Mock()
    destroy_process_group = Mock()
    set_device = Mock()
    ddp_cls = Mock(
        side_effect=lambda model, **kwargs: SimpleNamespace(
            module=model,
            parameters=lambda: iter(()),
            **kwargs,
        )
    )

    monkeypatch.setattr("gliner2.training.trainer.torch.nn.parallel.DistributedDataParallel", ddp_cls)
    monkeypatch.setattr("gliner2.training.trainer.dist.init_process_group", init_process_group)
    is_initialized = Mock(side_effect=[False, True])
    monkeypatch.setattr("gliner2.training.trainer.dist.is_initialized", is_initialized)
    monkeypatch.setattr("gliner2.training.trainer.dist.get_world_size", lambda: world_size)
    monkeypatch.setattr("gliner2.training.trainer.dist.get_rank", lambda: rank)
    monkeypatch.setattr("gliner2.training.trainer.torch.cuda.set_device", set_device)
    monkeypatch.setattr("gliner2.training.trainer.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("gliner2.training.trainer.dist.destroy_process_group", destroy_process_group)

    return init_process_group, destroy_process_group, set_device, ddp_cls, is_initialized


class _SaveableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)
        self.processor = MagicMock()

    def save_pretrained(self, path: str):
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target / "pytorch_model.bin")


@pytest.fixture
def trainer_output_dir(tmp_path):
    return str(tmp_path / "out")


@pytest.fixture
def default_config(trainer_output_dir):
    return TrainingConfig(
        output_dir=trainer_output_dir,
        fp16=False,
        bf16=False,
    )


def _config_with_local_rank(config: TrainingConfig, *, local_rank: int) -> TrainingConfig:
    return replace(config, local_rank=local_rank)


def test_distributed_init_process_group_and_ddp_wrap(default_config, monkeypatch):
    init_process_group, _, set_device, ddp_cls, _ = _mock_distributed_runtime(monkeypatch, rank=0)

    trainer = GLiNER2Trainer(model=_StubModel(), config=_config_with_local_rank(default_config, local_rank=0))

    init_process_group.assert_called_once_with(backend="nccl", init_method="env://")
    set_device.assert_called_once_with(0)
    ddp_cls.assert_called_once()

    kwargs = ddp_cls.call_args.kwargs
    assert kwargs["device_ids"] == [0]
    assert kwargs["output_device"] == 0
    assert kwargs["find_unused_parameters"] is True
    assert trainer.is_distributed is True
    assert trainer.is_main_process is True


def test_non_zero_rank_and_cleanup(default_config, monkeypatch):
    _, destroy_process_group, set_device, ddp_cls, _ = _mock_distributed_runtime(monkeypatch, rank=1)

    trainer = GLiNER2Trainer(model=_StubModel(), config=_config_with_local_rank(default_config, local_rank=1))

    assert trainer.is_distributed is True
    assert trainer.is_main_process is False
    set_device.assert_called_once_with(1)
    ddp_cls.assert_called_once()
    assert ddp_cls.call_args.kwargs["device_ids"] == [1]

    wrapped_model = trainer.model
    base_model = wrapped_model.module

    trainer._cleanup_distributed()
    assert trainer.is_distributed is False
    assert trainer.model is base_model
    assert not hasattr(trainer.model, "module")
    destroy_process_group.assert_called_once()


def test_init_process_group_failure_propagates(default_config, monkeypatch):
    monkeypatch.setattr("gliner2.training.trainer.dist.is_initialized", lambda: False)
    monkeypatch.setattr(
        "gliner2.training.trainer.dist.init_process_group",
        Mock(side_effect=RuntimeError("failed to initialize process group")),
    )
    monkeypatch.setattr("gliner2.training.trainer.dist.get_world_size", lambda: 2)
    monkeypatch.setattr("gliner2.training.trainer.dist.get_rank", lambda: 0)
    monkeypatch.setattr("gliner2.training.trainer.torch.cuda.set_device", Mock())
    monkeypatch.setattr("gliner2.training.trainer.torch.cuda.is_available", lambda: True)

    with pytest.raises(RuntimeError, match="failed to initialize process group"):
        GLiNER2Trainer(model=_StubModel(), config=_config_with_local_rank(default_config, local_rank=0))


def test_save_checkpoint_uses_module_under_ddp(default_config, monkeypatch):
    _mock_distributed_runtime(monkeypatch, rank=0)

    trainer = GLiNER2Trainer(model=_StubModel(), config=_config_with_local_rank(default_config, local_rank=0))
    base_model = trainer.model.module
    base_model.save_pretrained = Mock()
    trainer._save_checkpoint("step_0")

    base_model.save_pretrained.assert_called_once()


def test_single_device_paths(default_config, trainer_output_dir, monkeypatch):
    monkeypatch.setattr("gliner2.training.trainer.torch.cuda.is_available", lambda: True)

    trainer = GLiNER2Trainer(model=_StubModel(), config=_config_with_local_rank(default_config, local_rank=-1))

    assert trainer.is_distributed is False

    monkeypatch.setattr("gliner2.training.trainer.torch.cuda.is_available", lambda: False)
    trainer = GLiNER2Trainer(model=_SaveableModel(), config=_config_with_local_rank(default_config, local_rank=-1))
    trainer._save_checkpoint("step_0")

    checkpoint_file = Path(trainer_output_dir) / "step_0" / "pytorch_model.bin"
    assert checkpoint_file.exists()
