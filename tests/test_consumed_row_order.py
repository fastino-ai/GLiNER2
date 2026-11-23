"""Consumed-row receipts hash private row identities in the order batches are consumed."""

import hashlib
import itertools
import random

import pytest

from gliner2.training import ManagedTraining, ManagedTrainingPlan, TrainingSession
from gliner2.training.data import DataLoader_Factory, InputExample
from gliner2.training.row_order import ConsumedRowOrder
from gliner2.training.trainer import ExtractorTrainer
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model
from tests.test_training_session import ROWS, config

BINDING = "sha256:" + "c" * 64
FIRST = InputExample(text="john works at google", entities={"person": ["john"]})
SECOND = InputExample(text="mary lives in paris", entities={"person": ["mary"]})
RECEIPT_OPTIONS = ("row_order_attempt_id", "row_order_identities", "row_order_source_binding")
INCOMPLETE_OPTIONS = [
    names for size in (1, 2) for names in itertools.combinations(RECEIPT_OPTIONS, size)
]


def _identity(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def _receipt(identities, *, attempt="attempt-1", seed=7):
    receipt = ConsumedRowOrder(
        attempt_id=attempt, seed=seed, shuffle=True,
        row_identities=identities, source_binding=BINDING,
    )
    receipt.bind_sampler("native.random")
    return receipt


def test_receipt_hashes_identities_so_prepared_order_changes_the_digest():
    a, b = _identity("a"), _identity("b")
    prepared_ab, prepared_ba = _receipt([a, b]), _receipt([b, a], attempt="attempt-2")
    for receipt in (prepared_ab, prepared_ba):
        receipt.observe(epoch=0, positions=(0, 1))
    assert prepared_ab.receipt()["sha256"] != prepared_ba.receipt()["sha256"]
    same_rows = _receipt([b, a], attempt="attempt-3")
    same_rows.observe(epoch=0, positions=(1, 0))
    assert same_rows.receipt()["sha256"] == prepared_ab.receipt()["sha256"]
    observed = prepared_ab.receipt()
    assert observed["source"] == "gliner2_consumed_training_batches_v2"
    assert observed["sampler"] == "native.random"
    assert observed["source_binding"] == BINDING
    assert (observed["sample_count"], observed["batch_count"], observed["epoch_count"]) == (2, 1, 1)
    assert a not in str(observed) and b not in str(observed)


def test_receipt_rejects_unbound_or_malformed_evidence():
    with pytest.raises(ValueError, match="64-hex identity"):
        _receipt(["row-1"])
    with pytest.raises(ValueError, match="source binding"):
        ConsumedRowOrder(attempt_id="a", seed=1, shuffle=True,
                         row_identities=[_identity("a")], source_binding="row-1")
    receipt = _receipt([_identity("a")])
    with pytest.raises(ValueError, match="source rows"):
        receipt.observe(epoch=0, positions=(1,))
    with pytest.raises(ValueError, match="cannot change"):
        receipt.bind_sampler("native.sequential")
    with pytest.raises(ValueError, match="no training batch"):
        receipt.receipt()


def test_indexed_load_keeps_the_legacy_shuffle_and_maps_each_row_to_its_source():
    records = [{"input": f"row {i}", "output": {"entities": {}}} for i in range(9)]
    legacy = list(records)
    random.seed(5)
    random.shuffle(legacy)
    loaded, sources = DataLoader_Factory.load_indexed(records, max_samples=6, shuffle=True, seed=5)
    assert loaded == legacy[:6]
    assert [records[index] for index in sources] == loaded
    assert DataLoader_Factory.load(records, max_samples=6, shuffle=True, seed=5) == loaded


def test_native_receipt_follows_rows_not_prepared_positions(tmp_path):
    a, b = _identity("first"), _identity("second")
    digests = []
    runs = (("ab", [FIRST, SECOND], [a, b]), ("ab-again", [FIRST, SECOND], [a, b]),
            ("ba", [SECOND, FIRST], [b, a]))
    for name, rows, identities in runs:
        cfg = config(tmp_path / name, num_epochs=1, batch_size=1, pin_memory=False, seed=11)
        trainer = ExtractorTrainer(build_tiny_span_model(), cfg)
        result = trainer.train(
            train_data=rows, row_order_attempt_id=f"native-{name}",
            row_order_identities=identities, row_order_source_binding=BINDING,
        )
        receipt = result["consumed_row_order"]
        assert (receipt["sample_count"], receipt["batch_count"]) == (2, 2)
        assert receipt["sampler"] == "native.random"
        digests.append(receipt["sha256"])
    assert digests[0] == digests[1]
    assert digests[0] != digests[2]


def test_native_receipt_requires_identities_for_every_row(tmp_path):
    cfg = config(tmp_path / "missing", num_epochs=1, batch_size=1, pin_memory=False)
    trainer = ExtractorTrainer(build_tiny_span_model(), cfg)
    with pytest.raises(ValueError, match="attempt, row identities and a source binding"):
        trainer.train(train_data=[FIRST, SECOND], row_order_attempt_id="native")
    with pytest.raises(ValueError, match="do not cover"):
        trainer.train(train_data=[FIRST, SECOND], row_order_attempt_id="native",
                      row_order_identities=[_identity("only")], row_order_source_binding=BINDING)


def _options(names, identities):
    values = {
        "row_order_attempt_id": "attempt",
        "row_order_identities": identities,
        "row_order_source_binding": BINDING,
    }
    return {name: values[name] for name in names}


@pytest.mark.parametrize("names", INCOMPLETE_OPTIONS, ids="+".join)
def test_native_rejects_every_incomplete_receipt_option_set(tmp_path, names):
    cfg = config(tmp_path / "partial", num_epochs=1, batch_size=1, pin_memory=False)
    trainer = ExtractorTrainer(build_tiny_span_model(), cfg)
    options = _options(names, [_identity("first"), _identity("second")])
    with pytest.raises(ValueError, match="attempt, row identities and a source binding"):
        trainer.train(train_data=[FIRST, SECOND], **options)


def test_native_without_receipt_options_trains_without_a_receipt(tmp_path):
    cfg = config(tmp_path / "none", num_epochs=1, batch_size=1, pin_memory=False)
    result = ExtractorTrainer(build_tiny_span_model(), cfg).train(train_data=[FIRST, SECOND])
    assert "consumed_row_order" not in result


@pytest.mark.parametrize("names", INCOMPLETE_OPTIONS, ids="+".join)
def test_managed_rejects_every_incomplete_receipt_option_set(tmp_path, names):
    cfg = config(tmp_path / "managed-partial", num_epochs=1, batch_size=1)
    plan = ManagedTrainingPlan([FIRST, SECOND], cfg)
    session = TrainingSession(build_tiny_span_model(), cfg, max_steps=plan.max_steps)
    options = _options(names, [_identity("first"), _identity("second")])
    with pytest.raises(ValueError, match="attempt, row identities and a source binding"):
        ManagedTraining(session, plan, **options)


def test_managed_without_receipt_options_reports_no_receipt(tmp_path):
    cfg = config(tmp_path / "managed-none", num_epochs=1, batch_size=1)
    plan = ManagedTrainingPlan([FIRST, SECOND], cfg)
    driver = ManagedTraining(TrainingSession(build_tiny_span_model(), cfg, max_steps=plan.max_steps), plan)
    try:
        assert "consumed_row_order" not in driver.run()
    finally:
        driver.close()


def test_managed_receipt_matches_consumed_plan_across_epochs(tmp_path):
    cfg = config(tmp_path / "managed", num_epochs=2, batch_size=1)
    examples = [FIRST, SECOND]
    identities = [_identity("first"), _identity("second")]
    plan = ManagedTrainingPlan(examples, cfg)
    driver = ManagedTraining(
        TrainingSession(build_tiny_span_model(), cfg, max_steps=plan.max_steps),
        plan,
        row_order_attempt_id="managed-attempt",
        row_order_identities=identities,
        row_order_source_binding=BINDING,
    )
    try:
        result = driver.run()
        expected = ConsumedRowOrder(attempt_id="managed-attempt", seed=cfg.seed, shuffle=True,
                                    row_identities=identities, source_binding=BINDING)
        expected.bind_sampler("managed.random")
        for index in range(plan.batch_count):
            expected.observe(epoch=plan.epoch(index), positions=plan.batch_indices(index))
        assert result["consumed_row_order"] == expected.receipt()
    finally:
        driver.close()


def test_managed_receipt_rejects_identities_for_other_rows(tmp_path):
    cfg = config(tmp_path / "managed-mismatch", num_epochs=1, batch_size=1)
    plan = ManagedTrainingPlan(ROWS, cfg)
    with pytest.raises(ValueError, match="match the managed plan"):
        ManagedTraining(
            TrainingSession(build_tiny_span_model(), cfg, max_steps=plan.max_steps), plan,
            row_order_attempt_id="managed", row_order_identities=[_identity("a"), _identity("b")],
            row_order_source_binding=BINDING,
        )
