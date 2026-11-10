"""Real-backend parity tests for the optional DisentangledFlash extra."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import DebertaV2Config

pytest.importorskip("disentangled_flash", minversion="1.1")

from gliner2 import ExtractorConfig
from gliner2.inference.engine import BoundaryExtractor, GLiNER2
from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD
from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer


def _tiny_deberta_config(vocab_size: int) -> DebertaV2Config:
    return DebertaV2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
        max_position_embeddings=128,
        relative_attention=True,
        pos_att_type=["c2p", "p2c"],
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        conv_kernel_size=0,
    )


def _build_extractor(architecture: str):
    tokenizer = build_tiny_tokenizer()
    encoder_config = _tiny_deberta_config(len(tokenizer))
    if architecture == "span":
        config = ExtractorConfig(
            model_name="tiny-deberta",
            max_width=8,
            token_pooling="first",
            attn_implementation="eager",
        )
        model_class = GLiNER2
    else:
        config = ExtractorConfig(
            model_name="tiny-deberta",
            architecture="boundary",
            boundary_head=dict(TINY_BOUNDARY_HEAD),
            token_pooling="first",
            attn_implementation="eager",
        )
        model_class = BoundaryExtractor

    torch.manual_seed(7)
    return model_class(
        config,
        encoder_config=encoder_config,
        tokenizer=tokenizer,
    )


@pytest.mark.parametrize("architecture", ["span", "boundary"])
def test_full_extraction_parity_with_torch_backend(architecture):
    baseline = _build_extractor(architecture).eval()
    optimized = copy.deepcopy(baseline).eval()
    state_keys = set(optimized.state_dict())
    optimized.enable_disentangled_flash(backend="torch")

    texts = [
        "apple acquired microsoft in nyc .",
        "elon musk founded spacex .",
    ]
    options = {
        "batch_size": 2,
        "threshold": 0.1,
        "include_confidence": True,
        "include_spans": True,
    }
    with torch.inference_mode():
        expected = baseline.batch_extract_entities(
            texts,
            ["person", "organization", "location"],
            **options,
        )
        actual = optimized.batch_extract_entities(
            texts,
            ["person", "organization", "location"],
            **options,
        )

    assert actual == expected
    assert set(optimized.state_dict()) == state_keys
    assert optimized._disentangled_flash_mode == "inference"
    assert optimized._disentangled_flash_backend == "torch"


def test_training_backend_preserves_outputs_gradients_and_parameter_names():
    baseline = _build_extractor("span").train()
    optimized = copy.deepcopy(baseline).train()
    parameter_names = tuple(name for name, _ in optimized.encoder.named_parameters())
    optimized.enable_disentangled_flash(backend="torch", inference=False)

    input_ids = torch.randint(0, baseline.encoder.config.vocab_size, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    expected = baseline.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    actual = optimized.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    expected.square().mean().backward()
    actual.square().mean().backward()

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    baseline_gradients = dict(baseline.encoder.named_parameters())
    for name, parameter in optimized.encoder.named_parameters():
        expected_gradient = baseline_gradients[name].grad
        assert expected_gradient is not None
        assert parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad,
            expected_gradient,
            rtol=1e-5,
            atol=1e-6,
        )
    assert tuple(name for name, _ in optimized.encoder.named_parameters()) == parameter_names
    assert optimized._disentangled_flash_mode == "training"


_MIXED_LENGTH_TEXTS = [
    "apple acquired microsoft in nyc .",
    "elon musk founded spacex .",
    "apple .",
    "microsoft in nyc acquired apple and elon musk founded spacex in nyc .",
]


@pytest.mark.parametrize("architecture", ["span", "boundary"])
@pytest.mark.parametrize("packed", [True, "auto"])
def test_packed_extraction_matches_padded(architecture, packed):
    baseline = _build_extractor(architecture).eval()
    optimized = copy.deepcopy(baseline).eval()
    optimized.enable_disentangled_flash(
        backend="torch",
        packed=packed,
        packed_min_padding=0.1,
        packed_min_work=0,
    )
    labels = ["person", "organization", "location"]
    options = {
        "batch_size": 4,
        "threshold": 0.1,
        "include_confidence": True,
        "include_spans": True,
    }
    with torch.inference_mode():
        expected = baseline.batch_extract_entities(_MIXED_LENGTH_TEXTS, labels, **options)
        actual = optimized.batch_extract_entities(_MIXED_LENGTH_TEXTS, labels, **options)

    assert optimized._disentangled_flash_batches["packed"] >= 1
    for expected_row, actual_row in zip(expected, actual, strict=True):
        for label in labels:
            expected_entities = expected_row["entities"][label]
            actual_entities = actual_row["entities"][label]
            assert [(e["text"], e["start"], e["end"]) for e in actual_entities] == [
                (e["text"], e["start"], e["end"]) for e in expected_entities
            ]
            for e, a in zip(expected_entities, actual_entities, strict=True):
                assert a["confidence"] == pytest.approx(e["confidence"], abs=1e-5)


def test_packed_encoder_matches_padded_on_real_tokens():
    baseline = _build_extractor("span").eval()
    optimized = copy.deepcopy(baseline).eval()
    optimized.enable_disentangled_flash(backend="torch", packed=True)
    input_ids = torch.randint(0, baseline.encoder.config.vocab_size, (3, 12))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 7:] = 0
    attention_mask[2, 3:] = 0

    with torch.inference_mode():
        expected = baseline.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        actual = optimized.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

    real = attention_mask.bool()
    torch.testing.assert_close(actual[real], expected[real], rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(actual[~real]) == 0
    assert optimized._disentangled_flash_batches == {"packed": 1, "padded": 0}


def test_packed_training_matches_padded_gradients():
    baseline = _build_extractor("span").train()
    optimized = copy.deepcopy(baseline).train()
    optimized.enable_disentangled_flash(backend="torch", inference=False, packed=True)
    input_ids = torch.randint(0, baseline.encoder.config.vocab_size, (3, 12))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 7:] = 0
    attention_mask[2, 3:] = 0
    weight = attention_mask[..., None]

    for model in (baseline, optimized):
        hidden = model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        (hidden * weight).square().mean().backward()

    baseline_parameters = dict(baseline.encoder.named_parameters())
    for name, parameter in optimized.encoder.named_parameters():
        expected_gradient = baseline_parameters[name].grad
        assert expected_gradient is not None and parameter.grad is not None
        torch.testing.assert_close(parameter.grad, expected_gradient, rtol=1e-4, atol=1e-6)
    assert optimized._disentangled_flash_batches == {"packed": 1, "padded": 0}
