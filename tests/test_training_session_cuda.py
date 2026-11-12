"""Real GPU qualification; no mocks, downloads, private bases or publication.

Run with CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest -q tests/test_training_session_cuda.py.
A skip is not GPU qualification. The runner should reject any skipped GPU case
for an approved hardware profile except unsupported bfloat16 on pre-Ampere GPUs.
"""
from dataclasses import replace

import pytest
import torch

from gliner2 import AutoExtractor
from gliner2.training import ManagedTraining, ManagedTrainingPlan, TrainingSession
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model
from tests.test_training_session import ROWS, config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA GPU")


@pytest.mark.parametrize("builder,precision", [
    (build_tiny_span_model, "fp32"),
    (build_tiny_span_model, "bf16"),
    (build_tiny_boundary_model, "fp32"),
    (build_tiny_boundary_model, "bf16"),
])
@pytest.mark.parametrize("use_lora", [False, True])
def test_cuda_training_continuation_and_managed_equivalence(tmp_path, builder, precision, use_lora):
    if precision == "bf16" and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("hardware does not support bfloat16")
    cfg = config(tmp_path / "a", fp16=precision == "fp16", bf16=precision == "bf16",
                 use_lora=use_lora, lora_r=2, lora_alpha=4, lora_dropout=0.1,
                 encoder_lr=1e-5, task_lr=1e-4, gradient_checkpointing=precision != "fp16",
                 lora_target_modules=["encoder", "all_task_heads"],
                 gradient_accumulation_steps=2, num_epochs=2, deterministic=True,
                 allow_tf32=False, float32_matmul_precision="highest")
    plan = ManagedTrainingPlan(ROWS * 3, cfg, architecture=builder().architecture)
    session = TrainingSession(builder(), cfg, max_steps=plan.max_steps)
    assert next(session.model.parameters()).device.type == "cuda"
    managed = ManagedTraining(session, plan)
    managed.advance()
    if use_lora:
        gradients = [value.grad for name, value in session.model.named_parameters()
                     if ".encoder." in name and "lora_B" in name]
        assert gradients and any(value is not None and torch.count_nonzero(value) for value in gradients)
    managed.save_state(tmp_path / "pending")
    managed.run()
    expected = {name: value.detach().cpu().clone() for name, value in session.model.state_dict().items()}
    checkpoint = torch.load(tmp_path / "pending" / "training-state.pt", map_location="cpu", weights_only=True)
    resumed_session = TrainingSession(builder(seed=99), replace(cfg, output_dir=str(tmp_path / "b")), max_steps=plan.max_steps)
    resumed = ManagedTraining(resumed_session, plan)
    resumed.load_state(tmp_path / "pending")
    for name, parameter in resumed_session.model.named_parameters():
        torch.testing.assert_close(parameter.detach().cpu(), checkpoint["model"][name], rtol=0, atol=0)
        gradient = checkpoint["gradients"].get(name)
        if gradient is not None:
            torch.testing.assert_close(parameter.grad.cpu(), gradient, rtol=0, atol=0)
    assert torch.equal(torch.cuda.get_rng_state(), checkpoint["cuda_rng"][0])
    resumed.run()
    for name, value in resumed_session.model.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected[name], rtol=0, atol=0)
    direct = TrainingSession(builder(), replace(cfg, output_dir=str(tmp_path / "c")), max_steps=plan.max_steps)
    for index in range(plan.batch_count):
        direct.forward_backward(plan.batch(index))
        if plan.is_update(index):
            direct.optim_step()
    for name, value in direct.model.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected[name], rtol=0, atol=0)
    direct.export_weights(tmp_path / "native")
    loaded = AutoExtractor.from_pretrained(str(tmp_path / "native")).to("cuda")
    loaded.eval()
    assert loaded.extract_entities("john works at google", ["person"], threshold=0.0) is not None
    assert direct.trainer.scaler.is_enabled() == (precision == "fp16")
    for run in (managed, resumed):
        run.close()
