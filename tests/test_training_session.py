"""Real offline GLiNER training, continuation and native artifact loading."""
from dataclasses import replace

import pytest
import torch

from gliner2 import AutoExtractor
from gliner2.training import InputExample, TrainingConfig
from gliner2.training.session import TrainingSession
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model


ROWS = [InputExample(text="john works at google", entities={"person": ["john"]})]


def config(tmp_path, **kwargs):
    options = dict(output_dir=str(tmp_path), batch_size=1, num_workers=0,
                   fp16=False, bf16=False, use_lora=False, warmup_ratio=0,
                   scheduler_type="linear")
    return TrainingConfig(**(options | kwargs))


def stochastic_model(seed=13):
    model = build_tiny_span_model(seed=seed)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.2
    return model


def test_step_restore_continues_same_optimizer_and_exports_native_weights(tmp_path):
    cfg = config(tmp_path / "first")
    session = TrainingSession(stochastic_model(), cfg, max_steps=4)
    before = {k: v.clone() for k, v in session.model.state_dict().items()}
    assert session.forward_backward(ROWS)["loss"] > 0
    assert session.optim_step() == 1
    assert any(not torch.equal(before[k], v) for k, v in session.model.state_dict().items())
    session.save_state(tmp_path / "checkpoint")
    session.forward_backward(ROWS)
    session.optim_step()
    expected = {k: v.clone() for k, v in session.model.state_dict().items()}

    resumed = TrainingSession(stochastic_model(seed=99),
                              replace(cfg, output_dir=str(tmp_path / "second")), max_steps=4)
    resumed.load_state(tmp_path / "checkpoint")
    assert resumed.step == 1
    assert resumed.optimizer.state  # Adam moments, not only model weights.
    assert resumed.trainer.scheduler.last_epoch == 1
    resumed.forward_backward(ROWS)
    assert resumed.optim_step() == 2
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    held_out = resumed.forward(ROWS)
    resumed.export_weights(tmp_path / "export")
    loaded = AutoExtractor.from_pretrained(str(tmp_path / "export"))
    loaded.to("cpu")
    loaded.eval()
    resumed.model.eval()
    assert loaded.extract_entities("john works at google", ["person"], threshold=0.0) == resumed.model.extract_entities("john works at google", ["person"], threshold=0.0)
    assert resumed.forward(ROWS) == held_out


def test_restore_preserves_pending_accumulation_and_rejects_empty_step(tmp_path):
    cfg = config(tmp_path / "a", gradient_accumulation_steps=2)
    a = TrainingSession(build_tiny_span_model(), cfg, max_steps=3)
    with pytest.raises(ValueError, match="gradient"):
        a.optim_step()
    a.forward_backward(ROWS)
    a.save_state(tmp_path / "pending")
    a.forward_backward(ROWS)
    a.optim_step()
    expected = {k: v.clone() for k, v in a.model.state_dict().items()}
    b = TrainingSession(build_tiny_span_model(), replace(cfg, output_dir=str(tmp_path / "b")), max_steps=3)
    b.load_state(tmp_path / "pending")
    b.forward_backward(ROWS)
    b.optim_step()
    for name, value in b.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_session_rejects_oversized_batch_and_mismatched_restore(tmp_path):
    a = TrainingSession(build_tiny_span_model(), config(tmp_path / "a"), max_steps=2)
    with pytest.raises(ValueError, match="batch"):
        a.forward_backward(ROWS * 2)
    a.save_state(tmp_path / "saved")
    b = TrainingSession(build_tiny_span_model(), config(tmp_path / "b", task_lr=0.03), max_steps=2)
    with pytest.raises(ValueError, match="configuration"):
        b.load_state(tmp_path / "saved")


@pytest.mark.parametrize("options", [
    {"fp16": True}, {"bf16": True}, {"max_eval_samples": 1},
    {"strict_training": False}, {"ignore_nonfinite_losses": True},
])
def test_session_rejects_unqualified_or_lossy_configuration(tmp_path, options):
    cfg = config(tmp_path)
    for name, value in options.items():
        setattr(cfg, name, value)
    with pytest.raises(ValueError):
        TrainingSession(build_tiny_span_model(), cfg, max_steps=2)
