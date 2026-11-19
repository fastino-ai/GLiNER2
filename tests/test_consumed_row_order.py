"""Only successful consumed batches contribute to a private row-order receipt."""

import pytest

from gliner2.training import ManagedTraining, ManagedTrainingPlan, TrainingSession
from gliner2.training.row_order import ConsumedRowOrder
from gliner2.training.trainer import ExtractorTrainer
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model
from tests.test_training_session import ROWS, config


def test_receipt_contains_no_rows_and_is_batch_order_sensitive():
    left = ConsumedRowOrder(attempt_id="attempt-1", seed=7, shuffle=True)
    right = ConsumedRowOrder(attempt_id="attempt-2", seed=7, shuffle=True)
    for receipt, order in ((left, (0, 1)), (right, (1, 0))):
        receipt.observe(epoch=0, ordinals=order)
    observed = left.receipt()
    assert observed["attempt_id"] == "attempt-1"
    assert observed["sample_count"] == 2
    assert observed["batch_count"] == 1
    assert observed["epoch_count"] == 1
    assert observed["seed"] == 7
    assert observed["shuffle"] is True
    assert observed["sha256"] != right.receipt()["sha256"]
    assert "ordinals" not in observed and "text" not in observed


def test_managed_receipt_matches_consumed_plan_across_epochs(tmp_path):
    cfg = config(tmp_path / "managed", num_epochs=2, batch_size=1)
    plan = ManagedTrainingPlan(ROWS * 2, cfg)
    driver = ManagedTraining(
        TrainingSession(build_tiny_span_model(), cfg, max_steps=plan.max_steps),
        plan,
        row_order_attempt_id="managed-attempt",
    )
    try:
        result = driver.run()
        expected = ConsumedRowOrder(attempt_id="managed-attempt", seed=cfg.seed, shuffle=True)
        for index in range(plan.batch_count):
            expected.observe(epoch=plan.epoch(index), ordinals=plan.batch_indices(index))
        assert result["consumed_row_order"] == expected.receipt()
    finally:
        driver.close()


def test_native_trainer_receipt_tracks_batches_returned_by_loader(tmp_path):
    cfg = config(tmp_path / "native", num_epochs=2, batch_size=1, pin_memory=False)
    trainer = ExtractorTrainer(build_tiny_span_model(), cfg)
    result = trainer.train(train_data=ROWS * 2, row_order_attempt_id="native-attempt")
    receipt = result["consumed_row_order"]
    assert receipt["attempt_id"] == "native-attempt"
    assert receipt["sample_count"] == 4
    assert receipt["batch_count"] == 4
    assert receipt["epoch_count"] == 2
    assert receipt["shuffle"] is True


def test_missing_consumption_fails_closed():
    with pytest.raises(ValueError, match="no training batch"):
        ConsumedRowOrder(attempt_id="attempt", seed=1, shuffle=False).receipt()
