"""CPU-only tests for optional FlashDeBERTa backend selection."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from gliner2.models import base
from gliner2.models.boundary import model as boundary_model
from gliner2.models.span import model as span_model


class DebertaV2Config:
    """Minimal config whose class name triggers the optional backend."""


class FakeFlashDebertaV2Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.ones(1))


def _install_fake_flashdeberta(monkeypatch, model_class=FakeFlashDebertaV2Model):
    module = types.ModuleType("flashdeberta")
    module.FlashDebertaV2Model = model_class
    monkeypatch.setitem(sys.modules, "flashdeberta", module)
    monkeypatch.setattr(
        base.importlib.util,
        "find_spec",
        lambda name: object() if name == "flashdeberta" else None,
    )


def test_environment_enables_flashdeberta_when_option_is_omitted(monkeypatch):
    _install_fake_flashdeberta(monkeypatch)
    monkeypatch.setenv("USE_FLASHDEBERTA", "1")

    encoder = base.BaseExtractorModel._load_encoder(
        "unused",
        encoder_config=DebertaV2Config(),
        attn_implementation="eager",
    )

    assert isinstance(encoder, FakeFlashDebertaV2Model)


def test_explicit_false_ignores_flashdeberta_environment(monkeypatch):
    _install_fake_flashdeberta(monkeypatch)
    monkeypatch.setenv("USE_FLASHDEBERTA", "1")
    standard_encoder = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        base.AutoModel,
        "from_config",
        lambda config, **kwargs: standard_encoder,
    )

    encoder = base.BaseExtractorModel._load_encoder(
        "unused",
        encoder_config=DebertaV2Config(),
        attn_implementation="eager",
        use_flashdeberta=False,
    )

    assert encoder is standard_encoder


def test_flashdeberta_initialization_failure_falls_back(monkeypatch):
    class BrokenFlashDebertaV2Model:
        def __init__(self, config):
            raise RuntimeError("incompatible flash kernel")

    _install_fake_flashdeberta(monkeypatch, BrokenFlashDebertaV2Model)
    standard_encoder = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        base.AutoModel,
        "from_config",
        lambda config, **kwargs: standard_encoder,
    )

    with pytest.warns(RuntimeWarning, match="could not initialize"):
        encoder = base.BaseExtractorModel._load_encoder(
            "unused",
            encoder_config=DebertaV2Config(),
            attn_implementation="eager",
            use_flashdeberta=True,
        )

    assert encoder is standard_encoder


@pytest.mark.parametrize(
    ("model_class", "model_module"),
    [
        (span_model.SpanExtractorModel, span_model),
        (boundary_model.BoundaryExtractorModel, boundary_model),
    ],
)
def test_from_pretrained_threads_flashdeberta_option(
    monkeypatch, tmp_path, model_class, model_module
):
    captured = {}
    config = SimpleNamespace(_name_or_path=None)

    def fake_init(
        self,
        received_config,
        encoder_config=None,
        tokenizer=None,
        use_flashdeberta=None,
        word_splitter=None,
    ):
        torch.nn.Module.__init__(self)
        self.config = received_config
        captured.update(
            encoder_config=encoder_config,
            tokenizer=tokenizer,
            use_flashdeberta=use_flashdeberta,
        )

    monkeypatch.setattr(model_class, "__init__", fake_init)
    monkeypatch.setattr(
        model_module.AutoConfig,
        "from_pretrained",
        lambda path: "encoder-config",
    )
    monkeypatch.setattr(
        model_module,
        "load_extractor_tokenizer",
        lambda path: "tokenizer",
    )
    monkeypatch.setattr(
        model_module,
        "load_checkpoint_state_dict",
        lambda path, hub_kwargs: {},
    )

    model_class.from_pretrained(
        str(tmp_path),
        config=config,
        use_flashdeberta=True,
    )

    assert captured == {
        "encoder_config": "encoder-config",
        "tokenizer": "tokenizer",
        "use_flashdeberta": True,
    }
