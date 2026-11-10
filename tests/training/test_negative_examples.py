"""Negative NER examples: rows whose declared entity types have no mentions.

The training objective supports them: a declared type with no mentions yields
an all-zero target for that type, so every candidate span is trained as a
negative. What a negative row needs is its label set, which comes from
``entities`` (types mapped to empty lists) or from ``entity_descriptions``.
"""

from __future__ import annotations

import math

import pytest
import torch

from gliner2.training.data import (
    DataValidationError,
    InputExample,
    TrainingDataset,
    create_entity_example,
)
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model

DESCRIPTIONS = {
    "PRODUCT": "the vendor software product name",
    "VERSION": "a software version identifier such as v3.2.1",
    "ACCOUNT_ID": "a customer account identifier such as ACC-8814",
}


def test_described_types_without_mentions_become_negative_labels():
    example = create_entity_example("What are your support hours?", {}, DESCRIPTIONS)

    assert example.validate() == []
    assert example.to_dict()["output"]["entities"] == {
        "PRODUCT": [],
        "VERSION": [],
        "ACCOUNT_ID": [],
    }


def test_description_for_type_absent_from_row_is_accepted_as_negative():
    example = create_entity_example(
        "Vertex Pulse broke for ACC-8803.",
        {"PRODUCT": ["Vertex Pulse"], "ACCOUNT_ID": ["ACC-8803"]},
        DESCRIPTIONS,
    )

    assert example.validate() == []
    assert example.to_dict()["output"]["entities"] == {
        "PRODUCT": ["Vertex Pulse"],
        "ACCOUNT_ID": ["ACC-8803"],
        "VERSION": [],
    }


def test_declaring_described_types_does_not_mutate_callers_dict():
    entities = {"PRODUCT": ["Vertex Pulse"]}

    create_entity_example("Vertex Pulse broke.", entities, DESCRIPTIONS)

    assert entities == {"PRODUCT": ["Vertex Pulse"]}


def test_dataset_mixing_positives_and_negatives_validates():
    dataset = TrainingDataset()
    dataset.add_many([
        create_entity_example(
            "Beacon Flow (v1.0.4) still throws for ACC-6278.",
            {"PRODUCT": ["Beacon Flow"], "VERSION": ["v1.0.4"], "ACCOUNT_ID": ["ACC-6278"]},
            DESCRIPTIONS,
        ),
        create_entity_example("I meant the vertex pulse in the graph editor.", {}, DESCRIPTIONS),
        InputExample(text="Please remove my colleague.", entities={"PRODUCT": [], "VERSION": []}),
    ])

    report = dataset.validate(raise_on_error=True)

    assert report["valid"] == 3 and report["invalid"] == 0


def test_row_with_no_task_and_no_label_set_fails_with_negative_guidance():
    dataset = TrainingDataset([InputExample(text="What are your support hours?", entities={})])

    with pytest.raises(DataValidationError) as excinfo:
        dataset.validate(raise_on_error=True)

    message = str(excinfo.value)
    assert "at least one task" in message
    assert "empty mention lists" in message
    assert "entity_descriptions" in message


def test_sanitize_keeps_described_negative_but_not_a_type_dropped_for_bad_mention():
    example = InputExample(
        text="Vertex Pulse broke.",
        entities={"PRODUCT": ["Nimbus Ledger"]},
        entity_descriptions={"PRODUCT": "product", "VERSION": "version"},
    )

    _, still_valid = example.sanitize()

    assert still_valid
    assert example.entities == {"VERSION": []}
    assert example.entity_descriptions == {"VERSION": "version"}


def _config(tmp_path):
    return TrainingConfig(
        output_dir=str(tmp_path / "out"),
        batch_size=1,
        num_epochs=1,
        eval_strategy="no",
        fp16=False,
        bf16=False,
        num_workers=0,
        logging_steps=10_000,
        warmup_ratio=0.0,
        scheduler_type="constant",
        local_rank=-1,
    )


@pytest.mark.parametrize(
    "build_model", [build_tiny_span_model, build_tiny_boundary_model], ids=["span", "boundary"]
)
def test_training_on_positive_and_negative_rows_yields_finite_supervised_loss(
    tmp_path, build_model
):
    descriptions = {"company": "a company", "product": "a product"}
    examples = [
        create_entity_example(
            "apple released iphone .", {"company": ["apple"], "product": ["iphone"]}, descriptions
        ),
        create_entity_example("the cat sat on the mat .", {}, descriptions),
        InputExample(text="a dog ran in the park .", entities={"company": [], "product": []}),
    ]
    trainer = GLiNER2Trainer(build_model(), _config(tmp_path))
    losses = []
    original_backward = trainer._backward_one

    def recording_backward(*args, **kwargs):
        reported = original_backward(*args, **kwargs)
        total = trainer._last_train_outputs["total_loss"]
        losses.append((float(total.detach()), total.requires_grad))
        return reported

    trainer._backward_one = recording_backward
    trainer.train(train_data=examples)

    assert len(losses) == len(examples), "a negative row was dropped before training"
    for value, requires_grad in losses:
        assert math.isfinite(value)
        assert value > 0.0 and requires_grad, "a row produced no supervision"


@pytest.mark.parametrize(
    "build_model", [build_tiny_span_model, build_tiny_boundary_model], ids=["span", "boundary"]
)
def test_negative_only_row_trains_its_declared_types_toward_no_mentions(tmp_path, build_model):
    torch.manual_seed(0)
    examples = [InputExample(text="the cat sat on the mat .", entities={"company": []})] * 4
    trainer = GLiNER2Trainer(build_model(), _config(tmp_path))
    grads = []
    original_backward = trainer._backward_one

    def recording_backward(*args, **kwargs):
        reported = original_backward(*args, **kwargs)
        grads.append(
            sum(
                float(p.grad.abs().sum())
                for p in trainer.model.parameters()
                if p.grad is not None
            )
        )
        return reported

    trainer._backward_one = recording_backward
    trainer.train(train_data=examples)

    assert grads and all(g > 0.0 for g in grads)
