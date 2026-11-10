"""Direct and managed training share exact batches, updates and resume state."""
from dataclasses import replace

import pytest
import torch

from gliner2 import AutoExtractor
from gliner2.training import ManagedTraining, ManagedTrainingPlan, TrainingSession
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model
from tests.test_training_session import ROWS, config, stochastic_model


@pytest.mark.parametrize("builder", [build_tiny_span_model, build_tiny_boundary_model])
@pytest.mark.parametrize("use_lora", [False, True])
def test_managed_direct_equivalence_and_partial_epoch_batches(tmp_path, builder, use_lora):
    cfg = config(tmp_path / "managed", batch_size=2, gradient_accumulation_steps=2,
                 num_epochs=2, use_lora=use_lora, lora_r=2, lora_alpha=4, lora_target_modules=["encoder", "all_task_heads"])
    plan = ManagedTrainingPlan(ROWS * 5, cfg, architecture=builder().architecture)
    assert plan.batch_count == 6 and plan.max_steps == 4
    managed_session = TrainingSession(builder(), cfg, max_steps=plan.max_steps)
    managed = ManagedTraining(managed_session, plan)
    observed = []
    result = managed.run(lambda step, metrics: observed.append(step))
    expected = {key: value.clone() for key, value in managed_session.model.state_dict().items()}
    assert observed == [1, 2, 3, 4]
    assert result["samples_seen"] == 10
    direct = TrainingSession(builder(), replace(cfg, output_dir=str(tmp_path / "direct")), max_steps=plan.max_steps)
    for index in range(plan.batch_count):
        direct.forward_backward(plan.batch(index))
        if plan.is_update(index):
            direct.optim_step()
    for key, value in direct.model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    managed.close()


def test_managed_resume_preserves_cursor_rng_pending_gradients_and_validation(tmp_path):
    cfg = config(tmp_path / "a", gradient_accumulation_steps=2, num_epochs=2, eval_strategy="epoch")
    plan = ManagedTrainingPlan(ROWS * 3, cfg)
    a = ManagedTraining(TrainingSession(stochastic_model(), cfg, max_steps=plan.max_steps), plan, eval_examples=ROWS)
    a.advance()  # Persist an incomplete accumulation window, not only a whole step.
    a.save_state(tmp_path / "checkpoint")
    expected_result = a.run()
    expected = {key: value.clone() for key, value in a.session.model.state_dict().items()}
    b = ManagedTraining(TrainingSession(stochastic_model(seed=99), cfg, max_steps=plan.max_steps), plan, eval_examples=ROWS)
    b.load_state(tmp_path / "checkpoint")
    assert b.cursor == 1 and b.session.pending_batches == 1
    actual_result = b.run()
    assert actual_result["eval_metrics_history"] == expected_result["eval_metrics_history"]
    for key, value in b.session.model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    b.export_weights(tmp_path / "best")
    loaded = AutoExtractor.from_pretrained(str(tmp_path / "best")).to("cpu")
    assert loaded.extract_entities("john works at google", ["person"], threshold=0.0) is not None
    b.save_state(tmp_path / "complete")
    c = ManagedTraining(TrainingSession(stochastic_model(), cfg, max_steps=plan.max_steps), plan, eval_examples=ROWS)
    c.load_state(tmp_path / "complete")
    assert c.complete and c.best_step == b.best_step
    c.export_weights(tmp_path / "best-resumed")
    for run in (a, b, c):
        run.close()


def test_best_checkpoint_and_early_stopping_are_preserved(tmp_path):
    cfg = config(tmp_path / "a", encoder_lr=0.0, task_lr=0.0, num_epochs=5,
                 early_stopping=True, early_stopping_patience=1, eval_strategy="epoch")
    plan = ManagedTrainingPlan(ROWS, cfg)
    a = ManagedTraining(TrainingSession(build_tiny_span_model(), cfg, max_steps=plan.max_steps), plan, eval_examples=ROWS)
    result = a.run()
    assert result["total_steps"] == 2
    assert result["selected_step"] == 1
    assert a.complete
    a.save_state(tmp_path / "early")
    b = ManagedTraining(TrainingSession(build_tiny_span_model(), cfg, max_steps=plan.max_steps), plan, eval_examples=ROWS)
    b.load_state(tmp_path / "early")
    assert b.complete and b.run()["selected_step"] == 1
    for run in (a, b):
        run.close()


def test_plan_and_checkpoint_fail_closed_on_changed_dataset(tmp_path):
    cfg = config(tmp_path / "a", num_epochs=1, max_steps=2)
    plan = ManagedTrainingPlan(ROWS, cfg)
    assert plan.batch_count == 2 and plan.max_steps == 2
    a = ManagedTraining(TrainingSession(build_tiny_span_model(), cfg, max_steps=2), plan)
    a.advance()
    a.save_state(tmp_path / "checkpoint")
    other = ManagedTrainingPlan(ROWS * 2, cfg)
    b = ManagedTraining(TrainingSession(build_tiny_span_model(), cfg, max_steps=2), other)
    with pytest.raises(ValueError, match="plan"):
        b.load_state(tmp_path / "checkpoint")
    with pytest.raises(FileExistsError):
        a.save_state(tmp_path / "checkpoint")
    for run in (a, b):
        run.close()


@pytest.mark.parametrize("architecture", ["span", "boundary"])
def test_manifest_is_the_same_exact_schedule_and_includes_validation(tmp_path, architecture):
    import json
    cfg = config(tmp_path / "managed", batch_size=2, gradient_accumulation_steps=2,
                 num_epochs=2, eval_strategy="epoch")
    plan = ManagedTrainingPlan(ROWS * 5, cfg, architecture=architecture)
    manifest = json.loads(json.dumps(plan.to_manifest(eval_examples=ROWS)))
    for index, batch in enumerate(manifest["batches"]):
        assert [manifest["examples"][i] for i in batch["indices"]] == [row.to_dict() for row in plan.batch(index)]
        assert batch["update"] == plan.is_update(index)
        assert batch["evaluate"] == plan.is_epoch_end(index)
    assert sum(batch["update"] for batch in manifest["batches"]) == plan.max_steps
    assert manifest == plan.to_manifest(eval_examples=ROWS)
    assert manifest["fingerprint"] != plan.to_manifest()["fingerprint"]
