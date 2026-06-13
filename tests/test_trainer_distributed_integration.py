from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from gliner2.training.data import InputExample
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


class _TestProcessor:
    def change_mode(self, is_training: bool) -> None:
        self.is_training = is_training

    def collate_fn_train(self, batch, max_len=None):
        return self._collate(batch)

    def collate_fn_inference(self, batch, max_len=None):
        return self._collate(batch)

    @staticmethod
    def _collate(batch):
        values = torch.tensor(
            [len(text) for text, _schema in batch],
            dtype=torch.float32,
        ).unsqueeze(1)
        return {"values": values}


class _TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"max_len": None})()
        self.layer = nn.Linear(1, 1)
        self.processor = _TestProcessor()

    def forward(self, batch):
        output = self.layer(batch["values"])
        loss = (output ** 2).mean()
        return {
            "total_loss": loss,
            "classification_loss": loss.detach(),
            "structure_loss": loss.detach(),
            "count_loss": loss.detach(),
        }

    def save_pretrained(self, path: str):
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target / "pytorch_model.bin")


def _run_ddp_worker(rank: int, world_size: int, init_file: str, out_dir: str):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29587"

    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    try:
        torch.cuda.set_device(rank)
        torch.manual_seed(7 + rank)

        model = _TestModel().cuda(rank)
        config = TrainingConfig(
            output_dir=out_dir,
            num_epochs=1,
            batch_size=2,
            eval_batch_size=2,
            fp16=False,
            bf16=False,
            eval_strategy="no",
            save_best=False,
            report_to_wandb=False,
            local_rank=rank,
            num_workers=0,
            pin_memory=False,
            validate_data=False,
        )

        trainer = GLiNER2Trainer(model=model, config=config, processor=model.processor)
        is_distributed = trainer.is_distributed
        is_main_process = trainer.is_main_process
        training_examples = [
            InputExample(
                text="Apple hired Tim Cook in Cupertino.",
                entities={"company": ["Apple"], "person": ["Tim Cook"], "location": ["Cupertino"]},
            ),
            InputExample(
                text="Google opened a new office in Mountain View.",
                entities={"company": ["Google"], "location": ["Mountain View"]},
            ),
            InputExample(
                text="Tesla built the Cybertruck prototype in Texas.",
                entities={"company": ["Tesla"], "product": ["Cybertruck"], "location": ["Texas"]},
            ),
            InputExample(
                text="OpenAI released GPT-4 after research work.",
                entities={"company": ["OpenAI"], "product": ["GPT-4"]},
            ),
        ]
        result = trainer.train(train_data=training_examples)
        torch.save(
            {
                "rank": rank,
                "is_distributed": is_distributed,
                "is_main_process": is_main_process,
                "total_steps": result["total_steps"],
            },
            Path(out_dir) / f"rank_{rank}_result.pt",
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="CUDA with at least 2 devices is required for the trainer's NCCL DDP path",
)
def test_two_process_ddp_training_saves_rank0_checkpoint(tmp_path):
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    world_size = 2
    out_dir = tmp_path / "trainer_ddp"
    out_dir.mkdir(parents=True, exist_ok=True)

    init_file = tmp_path / "trainer_ddp_init_file"
    init_file.touch()

    mp.spawn(
        _run_ddp_worker,
        args=(world_size, str(init_file), str(out_dir)),
        nprocs=world_size,
        join=True,
    )

    rank0 = torch.load(out_dir / "rank_0_result.pt")
    rank1 = torch.load(out_dir / "rank_1_result.pt")

    assert rank0["is_distributed"] is True
    assert rank1["is_distributed"] is True
    assert rank0["is_main_process"] is True
    assert rank1["is_main_process"] is False
    assert (out_dir / "final").is_dir()
    assert (out_dir / "final" / "pytorch_model.bin").exists()
    assert rank0["total_steps"] == rank1["total_steps"] == 1
