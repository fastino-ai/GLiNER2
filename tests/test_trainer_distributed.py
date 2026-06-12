from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


def _mock_model() -> MagicMock:
    m = MagicMock()
    m.to = MagicMock(return_value=m)
    m.processor = MagicMock()
    return m


class _TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)
        self.processor = MagicMock()

    def save_pretrained(self, path: str):
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target / "pytorch_model.bin")


@pytest.fixture
def output_dir(tmp_path):
    return str(tmp_path / "out")


@pytest.fixture
def base_config(output_dir):
    return TrainingConfig(
        output_dir=output_dir,
        fp16=False,
        bf16=False,
    )


def _config(base: TrainingConfig, *, local_rank: int) -> TrainingConfig:
    return replace(base, local_rank=local_rank)


class TestDistributedTrainerSetup:
    @patch("gliner2.training.trainer.torch.nn.parallel.DistributedDataParallel")
    @patch("gliner2.training.trainer.dist.init_process_group")
    @patch("gliner2.training.trainer.dist.is_initialized", return_value=False)
    @patch(
        "gliner2.training.trainer.dist.get_world_size",
        return_value=2,
    )
    @patch("gliner2.training.trainer.dist.get_rank", return_value=0)
    @patch("gliner2.training.trainer.torch.cuda.set_device")
    @patch("gliner2.training.trainer.torch.cuda.is_available", return_value=True)
    def test_distributed_init_process_group_and_ddp_wrap(
        self,
        _cuda_avail,
        _set_device,
        _rank,
        _world,
        _is_init,
        _init_pg,
        mock_ddp_cls,
        base_config,
    ):
        config = _config(base_config, local_rank=0)
        inner = _mock_model()
        trainer = GLiNER2Trainer(model=inner, config=config)

        _init_pg.assert_called_once_with(backend="nccl", init_method="env://")
        _set_device.assert_called_once_with(0)
        mock_ddp_cls.assert_called_once()
        call_kw = mock_ddp_cls.call_args.kwargs
        assert call_kw["device_ids"] == [0]
        assert call_kw["output_device"] == 0
        assert call_kw["find_unused_parameters"] is True
        assert trainer.is_distributed is True
        assert trainer.is_main_process is True

    @patch("gliner2.training.trainer.torch.nn.parallel.DistributedDataParallel")
    @patch("gliner2.training.trainer.dist.init_process_group")
    @patch("gliner2.training.trainer.dist.is_initialized", return_value=False)
    @patch("gliner2.training.trainer.dist.get_world_size", return_value=2)
    @patch("gliner2.training.trainer.dist.get_rank", return_value=1)
    @patch("gliner2.training.trainer.torch.cuda.set_device")
    @patch("gliner2.training.trainer.torch.cuda.is_available", return_value=True)
    def test_non_zero_rank_not_main_process(
        self,
        _cuda_avail,
        _set_device,
        _rank,
        _world,
        _is_init,
        _init_pg,
        mock_ddp_cls,
        base_config,
    ):
        config = _config(base_config, local_rank=1)
        trainer = GLiNER2Trainer(model=_mock_model(), config=config)

        assert trainer.is_distributed is True
        assert trainer.is_main_process is False
        _set_device.assert_called_once_with(1)
        mock_ddp_cls.assert_called_once()
        assert mock_ddp_cls.call_args.kwargs["device_ids"] == [1]

    @patch("gliner2.training.trainer.torch.nn.parallel.DistributedDataParallel")
    @patch("gliner2.training.trainer.dist.init_process_group")
    @patch("gliner2.training.trainer.dist.is_initialized", return_value=True)
    @patch("gliner2.training.trainer.dist.get_world_size", return_value=2)
    @patch("gliner2.training.trainer.dist.get_rank", return_value=0)
    @patch("gliner2.training.trainer.torch.cuda.set_device")
    @patch("gliner2.training.trainer.torch.cuda.is_available", return_value=True)
    def test_skips_init_when_process_group_already_initialized(
        self,
        _cuda_avail,
        _set_device,
        _rank,
        _world,
        _is_init,
        _init_pg,
        _ddp,
        base_config,
    ):
        config = _config(base_config, local_rank=0)
        GLiNER2Trainer(model=_mock_model(), config=config)
        _init_pg.assert_not_called()

    @patch("gliner2.training.trainer.torch.nn.parallel.DistributedDataParallel")
    @patch("gliner2.training.trainer.dist.init_process_group")
    @patch("gliner2.training.trainer.dist.is_initialized", return_value=False)
    @patch("gliner2.training.trainer.dist.get_world_size", return_value=2)
    @patch("gliner2.training.trainer.dist.get_rank", return_value=0)
    @patch("gliner2.training.trainer.torch.cuda.set_device")
    @patch("gliner2.training.trainer.torch.cuda.is_available", return_value=True)
    def test_cleanup_distributed_destroys_process_group(
        self,
        _cuda_avail,
        _set_device,
        _rank,
        _world,
        _is_init,
        _init_pg,
        _ddp,
        base_config,
    ):
        with patch("gliner2.training.trainer.dist.destroy_process_group") as destroy:
            config = _config(base_config, local_rank=0)
            trainer = GLiNER2Trainer(model=_mock_model(), config=config)
            trainer._cleanup_distributed()
            destroy.assert_called_once()

    @patch("gliner2.training.trainer.torch.nn.parallel.DistributedDataParallel")
    @patch("gliner2.training.trainer.dist.init_process_group")
    @patch("gliner2.training.trainer.dist.is_initialized", return_value=False)
    @patch("gliner2.training.trainer.dist.get_world_size", return_value=2)
    @patch("gliner2.training.trainer.dist.get_rank", return_value=0)
    @patch("gliner2.training.trainer.torch.cuda.set_device")
    @patch("gliner2.training.trainer.torch.cuda.is_available", return_value=True)
    def test_save_checkpoint_uses_module_under_ddp(
        self,
        _cuda_avail,
        _set_device,
        _rank,
        _world,
        _is_init,
        _init_pg,
        mock_ddp_cls,
        base_config,
    ):
        mock_ddp = MagicMock()
        mock_ddp.module = MagicMock()
        mock_ddp.parameters = MagicMock(return_value=iter(()))
        mock_ddp_cls.return_value = mock_ddp

        config = _config(base_config, local_rank=0)
        trainer = GLiNER2Trainer(model=_mock_model(), config=config)
        trainer.model = mock_ddp

        trainer._save_checkpoint("step_0")

        mock_ddp.module.save_pretrained.assert_called_once()

    @patch("gliner2.training.trainer.torch.nn.parallel.DistributedDataParallel")
    @patch(
        "gliner2.training.trainer.dist.init_process_group",
        side_effect=RuntimeError("failed to initialize process group"),
    )
    @patch("gliner2.training.trainer.dist.is_initialized", return_value=False)
    @patch("gliner2.training.trainer.torch.cuda.set_device")
    @patch("gliner2.training.trainer.torch.cuda.is_available", return_value=True)
    def test_init_process_group_failure_propagates(
        self,
        _cuda_avail,
        _set_device,
        _is_init,
        _init_pg,
        mock_ddp_cls,
        base_config,
    ):
        config = _config(base_config, local_rank=0)

        with pytest.raises(RuntimeError, match="failed to initialize process group"):
            GLiNER2Trainer(model=_mock_model(), config=config)

        _set_device.assert_called_once_with(0)
        _init_pg.assert_called_once_with(backend="nccl", init_method="env://")
        mock_ddp_cls.assert_not_called()

    @patch("gliner2.training.trainer.torch.nn.parallel.DistributedDataParallel")
    @patch("gliner2.training.trainer.torch.cuda.is_available", return_value=True)
    def test_single_device_no_ddp_when_local_rank_negative(
        self, _cuda_avail, mock_ddp_cls, base_config
    ):
        config = _config(base_config, local_rank=-1)
        inner = _mock_model()
        GLiNER2Trainer(model=inner, config=config)
        mock_ddp_cls.assert_not_called()

    @patch("gliner2.training.trainer.torch.cuda.is_available", return_value=False)
    def test_save_checkpoint_with_real_nn_module_single_device(
        self, _cuda_avail, output_dir, base_config
    ):
        config = _config(base_config, local_rank=-1)
        trainer = GLiNER2Trainer(model=_TinyModule(), config=config)

        trainer._save_checkpoint("step_0")

        checkpoint_file = Path(output_dir) / "step_0" / "pytorch_model.bin"
        assert checkpoint_file.exists()
