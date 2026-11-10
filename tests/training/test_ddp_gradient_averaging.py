"""Multi-GPU DDP must average gradients across ranks.

With DistributedDataParallel(static_graph=True) and a forward returning the plain ExtractorOutput dataclass, DDP's
backward sink is attached through torch.utils._pytree, which saw the dataclass as one opaque leaf: no gradient was
averaged, and ranks trained on their own data and drifted apart silently. The fix: ExtractorOutput is registered with
pytree, the trainer defaults to plain DDP, and under DDP every trainable parameter is touched each step (which
parameters get a gradient varies by batch, and plain DDP requires the same set every step).
"""
import tempfile

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from gliner2.models.outputs import ExtractorOutput
from gliner2.training.data import create_entity_example
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def test_extractor_output_flattens_to_its_tensors_and_round_trips():
    out = ExtractorOutput(loss=torch.tensor(1.0, requires_grad=True), total_loss=torch.tensor(2.0), batch_size=3)
    leaves, spec = tree_flatten(out)
    assert any(isinstance(x, torch.Tensor) and x.requires_grad for x in leaves)
    back = tree_unflatten(leaves, spec)
    assert type(back) is ExtractorOutput and back["total_loss"].item() == 2.0 and back.batch_size == 3


def test_the_trainer_defaults_to_plain_ddp():
    assert TrainingConfig.ddp_static_graph is False


def _grads_after_one_step(monkeypatch, distributed: bool):
    cfg = TrainingConfig(output_dir=tempfile.mkdtemp(), bf16=False, fp16=False, validate_data=False, batch_size=1,
                         num_workers=0, ddp_consensus_check=False)
    tr = ExtractorTrainer(build_tiny_boundary_model(), cfg)
    example = create_entity_example("apple released iphone .", {"company": ["apple"]}, {"company": "a company"})
    batch = next(iter(tr._create_dataloader(tr._prepare_data([example], True), 1, shuffle=False,
                                            is_training=True))).to(tr.device)
    if distributed:   # after the dataloader, which would otherwise need a process group for its sampler
        monkeypatch.setattr(type(tr), "is_distributed", property(lambda self: True), raising=False)
    tr.model.train()
    tr._backward_one(batch, 0, False, None)
    return {n: p.grad for n, p in tr.model.named_parameters() if p.requires_grad}


def test_an_entity_batch_leaves_parameters_without_gradient_outside_ddp(monkeypatch):
    assert any(g is None for g in _grads_after_one_step(monkeypatch, distributed=False).values())


def test_under_ddp_every_parameter_gets_a_gradient(monkeypatch):
    missing = [n for n, g in _grads_after_one_step(monkeypatch, distributed=True).items() if g is None]
    assert not missing, f"{len(missing)} parameters without gradient, e.g. {missing[:5]}"
