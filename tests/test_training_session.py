"""Real offline GLiNER training, continuation and native artifact loading."""
from dataclasses import replace

import pytest
import torch

from gliner2 import AutoExtractor
from gliner2.training import InputExample, TrainingConfig
from gliner2.training.session import TrainingSession
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model
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


@pytest.mark.parametrize("builder", [build_tiny_span_model, build_tiny_boundary_model])
@pytest.mark.parametrize("use_lora", [False, True])
def test_family_state_continuation_and_native_export(tmp_path, builder, use_lora):
    cfg = config(tmp_path / "a", use_lora=use_lora, lora_r=2, lora_alpha=4,
                 lora_target_modules=["encoder", "all_task_heads"], gradient_accumulation_steps=2)
    a = TrainingSession(builder(), cfg, max_steps=3)
    a.forward_backward(ROWS)
    a.optim_step()
    a.forward_backward(ROWS)
    a.save_state(tmp_path / "pending")
    a.forward_backward(ROWS)
    a.optim_step()
    expected = {k: v.clone() for k, v in a.model.state_dict().items()}
    b = TrainingSession(builder(seed=99), replace(cfg, output_dir=str(tmp_path / "b")), max_steps=3)
    b.load_state(tmp_path / "pending")
    b.forward_backward(ROWS)
    b.optim_step()
    for name, value in b.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    b.export_weights(tmp_path / "export")
    for name, value in b.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    loaded = AutoExtractor.from_pretrained(str(tmp_path / "export")).to("cpu")
    loaded.eval()
    b.model.eval()
    assert loaded.extract_entities("john works at google", ["person"], threshold=0.0) == b.model.extract_entities("john works at google", ["person"], threshold=0.0)
    # A trained full-model artifact can start another full-weight or LoRA run.
    continued = TrainingSession(loaded, replace(cfg, output_dir=str(tmp_path / "continued")), max_steps=2)
    assert continued.forward_backward(ROWS)["loss"] > 0
    assert continued.optim_step() == 1


def test_preloaded_adapter_continues_without_reinitialization(tmp_path):
    from peft import PeftModel
    cfg = config(tmp_path / "a", use_lora=True, lora_r=2, lora_alpha=4, lora_target_modules=["encoder"])
    base = build_tiny_span_model()
    base.save_pretrained(str(tmp_path / "base"))
    a = TrainingSession(base, cfg, max_steps=3)
    a.forward_backward(ROWS)
    a.optim_step()
    a.model.save_pretrained(tmp_path / "adapter")
    loaded = PeftModel.from_pretrained(AutoExtractor.from_pretrained(str(tmp_path / "base")), str(tmp_path / "adapter"), is_trainable=True)
    expected = {k: v.clone() for k, v in loaded.state_dict().items()}
    b = TrainingSession(loaded, replace(cfg, output_dir=str(tmp_path / "b")), max_steps=3)
    for name, value in b.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    b.forward_backward(ROWS)
    b.optim_step()
    assert any(not torch.equal(expected[k], v) for k, v in b.model.state_dict().items())
    with pytest.raises(ValueError, match="LoRA"):
        TrainingSession(loaded, replace(cfg, lora_r=8), max_steps=3)
    with pytest.raises(ValueError, match="LoRA"):
        TrainingSession(loaded, replace(cfg, use_lora=False), max_steps=3)


def test_native_amp_skip_keeps_schedule_and_weights_unchanged(tmp_path):
    session = TrainingSession(build_tiny_span_model(), config(tmp_path), max_steps=2)
    # Exercise the real GradScaler overflow branch on CPU. CUDA qualification
    # is separate; no fake scaler or mocked successful update is used here.
    trainer = session.trainer
    trainer.config.fp16 = True
    trainer.scaler = torch.amp.GradScaler("cpu", enabled=True)
    parameter = next(p for p in session.model.parameters() if p.requires_grad)
    trainer.scaler.scale(parameter.sum()).backward()
    parameter.grad.fill_(float("inf"))
    before = parameter.clone()
    scale = trainer.scaler.get_scale()
    assert trainer._optimizer_step() is False
    assert trainer.scaler.get_scale() < scale
    assert trainer.scheduler.last_epoch == 0
    assert not session.optimizer.state
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)


def test_restore_rejects_changed_label_sampling(tmp_path):
    a = TrainingSession(build_tiny_span_model(), config(tmp_path / "a"), max_steps=2)
    a.save_state(tmp_path / "state")
    changed = build_tiny_span_model()
    changed.processor.sampling_config.remove_entities_prob = 1.0
    b = TrainingSession(changed, config(tmp_path / "b"), max_steps=2)
    with pytest.raises(ValueError, match="configuration"):
        b.load_state(tmp_path / "state")


def test_batch_metrics_report_actual_preprocessed_tokens(tmp_path):
    session = TrainingSession(build_tiny_span_model(), config(tmp_path), max_steps=2)
    measured = session.forward(ROWS)
    assert measured.get("num_tokens", 0) > len(ROWS[0].text.split())
    batch = session._batch(ROWS, training=False)
    assert measured["num_tokens"] == int(batch.attention_mask.sum())
    assert session.forward_backward(ROWS)["num_tokens"] > 0
