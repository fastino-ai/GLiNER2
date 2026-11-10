"""CPU-only tests for warnings emitted while loading the encoder."""

import warnings

import pytest
import torch

from gliner2.models import base


class PlainConfig:
    """Config that does not trigger the optional FlashDeBERTa backend."""


def _reject_non_eager(monkeypatch, *, emit_jit_warning=False):
    standard_encoder = torch.nn.Linear(2, 2)

    def fake_from_config(config, **kwargs):
        if emit_jit_warning:
            warnings.warn(
                "`torch.jit.script` is not supported in Python 3.14+ and may break.",
                FutureWarning,
            )
        if kwargs.get("attn_implementation", "eager") != "eager":
            raise ValueError("does not support an attention implementation")
        return standard_encoder

    monkeypatch.setattr(base.AutoModel, "from_config", fake_from_config)
    return standard_encoder


def test_default_sdpa_fallback_is_silent(monkeypatch):
    standard_encoder = _reject_non_eager(monkeypatch, emit_jit_warning=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        encoder = base.BaseExtractorModel._load_encoder(
            "unused",
            encoder_config=PlainConfig(),
            attn_implementation="sdpa",
            use_flashdeberta=False,
        )

    assert encoder is standard_encoder


def test_explicit_flash_attention_fallback_still_warns(monkeypatch):
    standard_encoder = _reject_non_eager(monkeypatch)

    with pytest.warns(RuntimeWarning, match="flash_attention_2"):
        encoder = base.BaseExtractorModel._load_encoder(
            "unused",
            encoder_config=PlainConfig(),
            attn_implementation="flash_attention_2",
            use_flashdeberta=False,
        )

    assert encoder is standard_encoder
