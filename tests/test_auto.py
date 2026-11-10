"""Tests for ``AutoExtractor`` registry dispatch and safe loading (PR 2)."""

from __future__ import annotations

import pytest

from gliner2.auto import (
    AutoExtractor,
    UnknownArchitectureError,
    ArchitectureMismatchError,
    ArchitectureRegistrationError,
)
from gliner2.configuration import ExtractorConfig
from tests.fixtures.tiny_boundary_checkpoint import save_tiny_boundary_checkpoint
from tests.fixtures.tiny_span_checkpoint import save_tiny_span_checkpoint


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_span_is_registered_after_import():
    import gliner2.inference.engine  # noqa: F401
    assert "span" in AutoExtractor._registry


def test_register_conflict_raises_without_exist_ok():
    class Dummy:
        pass

    AutoExtractor.register("span", Dummy, exist_ok=True)  # temporarily override
    try:
        with pytest.raises(ArchitectureRegistrationError):
            AutoExtractor.register("span", object)  # different class, no exist_ok
    finally:
        # Restore the real span class.
        from gliner2.inference.engine import SpanExtractor
        AutoExtractor.register("span", SpanExtractor, exist_ok=True)


def test_register_rejects_unknown_architecture_name():
    with pytest.raises(ValueError):
        AutoExtractor.register("nonsense", object)


# ---------------------------------------------------------------------------
# Resolution / unknown architecture
# ---------------------------------------------------------------------------

def test_resolve_span_returns_span_class():
    cls = AutoExtractor._resolve_class("span")
    from gliner2.inference.engine import SpanExtractor
    assert cls is SpanExtractor


def test_resolve_unregistered_architecture_raises(monkeypatch):
    """A known-but-unregistered architecture must raise UnknownArchitectureError."""
    import gliner2.inference.engine  # noqa: F401  (ensure builtins registered)
    registry_without_boundary = {
        k: v for k, v in AutoExtractor._registry.items() if k != "boundary"
    }
    monkeypatch.setattr(AutoExtractor, "_registry", registry_without_boundary)
    # Prevent _ensure_registered from re-adding boundary.
    monkeypatch.setattr("gliner2.auto._ensure_registered", lambda: None)
    with pytest.raises(UnknownArchitectureError):
        AutoExtractor._resolve_class("boundary")


# ---------------------------------------------------------------------------
# Dispatch via from_config
# ---------------------------------------------------------------------------

def test_from_config_dispatches_span(tiny_tokenizer, tiny_encoder_config):
    cfg = ExtractorConfig(model_name="tiny-bert-fixture", max_width=8)
    model = AutoExtractor.from_config(
        cfg, encoder_config=tiny_encoder_config, tokenizer=tiny_tokenizer
    )
    from gliner2.inference.engine import SpanExtractor
    assert isinstance(model, SpanExtractor)
    assert model.architecture == "span"


# ---------------------------------------------------------------------------
# Dispatch via from_pretrained (round-trip on a saved tiny checkpoint)
# ---------------------------------------------------------------------------

def test_from_pretrained_dispatches_span(tmp_path):
    save_tiny_span_checkpoint(tmp_path)
    model = AutoExtractor.from_pretrained(str(tmp_path))
    from gliner2.inference.engine import SpanExtractor
    assert isinstance(model, SpanExtractor)
    assert model.architecture == "span"


def test_from_pretrained_dispatches_boundary(tmp_path):
    save_tiny_boundary_checkpoint(tmp_path)
    model = AutoExtractor.from_pretrained(str(tmp_path))
    from gliner2.inference.engine import BoundaryExtractor
    assert isinstance(model, BoundaryExtractor)
    assert model.architecture == "boundary"


def test_from_pretrained_architecture_mismatch_raises(tmp_path):
    save_tiny_span_checkpoint(tmp_path)
    with pytest.raises(ArchitectureMismatchError) as exc_info:
        AutoExtractor.from_pretrained(str(tmp_path), architecture="boundary")
    assert "from_span_checkpoint" not in str(exc_info.value)
    assert "automatic architecture conversion is not supported" in str(exc_info.value)


def test_from_pretrained_rejects_unknown_load_option():
    with pytest.raises(TypeError, match="does not accept.*beam_size"):
        AutoExtractor.from_pretrained("unused", beam_size=4)


def test_hub_options_are_used_for_config_but_not_forwarded(monkeypatch):
    import gliner2.auto as auto_module

    captured = {}

    class DummySpan:
        @classmethod
        def from_pretrained(cls, path, *args, **kwargs):
            captured["model"] = (path, args, kwargs)
            return object()

    config = ExtractorConfig(model_name="unused")

    def fake_load_config(path, hub_kwargs):
        captured["config"] = (path, hub_kwargs)
        return config

    monkeypatch.setattr(auto_module, "_load_config", fake_load_config)
    monkeypatch.setattr(AutoExtractor, "_registry", {"span": DummySpan})
    monkeypatch.setattr(auto_module, "_ensure_registered", lambda: None)

    AutoExtractor.from_pretrained(
        "org/repo",
        revision="release",
        token="secret",
        map_location="cpu",
        attention_backend="disentangled_flash",
    )

    assert captured["config"] == (
        "org/repo",
        {"revision": "release", "token": "secret"},
    )
    assert captured["model"] == (
        "org/repo",
        (),
        {
            "config": config,
            "map_location": "cpu",
            "attention_backend": "disentangled_flash",
        },
    )


def test_from_config_forwards_word_splitter(tiny_tokenizer, tiny_encoder_config):
    from gliner2.processor import CharLevelSplitter, WhitespaceTokenSplitter

    cfg = ExtractorConfig(model_name="tiny-bert-fixture", max_width=8)
    default = AutoExtractor.from_config(
        cfg, encoder_config=tiny_encoder_config, tokenizer=tiny_tokenizer
    )
    char_model = AutoExtractor.from_config(
        cfg,
        encoder_config=tiny_encoder_config,
        tokenizer=tiny_tokenizer,
        word_splitter="char",
    )
    assert isinstance(default.processor.word_splitter, WhitespaceTokenSplitter)
    assert isinstance(char_model.processor.word_splitter, CharLevelSplitter)


def test_from_pretrained_forwards_word_splitter(tmp_path):
    from gliner2.processor import CharLevelSplitter, WhitespaceTokenSplitter

    save_tiny_span_checkpoint(tmp_path)
    default = AutoExtractor.from_pretrained(str(tmp_path))
    char_model = AutoExtractor.from_pretrained(str(tmp_path), word_splitter="char")
    assert isinstance(default.processor.word_splitter, WhitespaceTokenSplitter)
    assert isinstance(char_model.processor.word_splitter, CharLevelSplitter)


def test_from_pretrained_boundary_forwards_word_splitter(tmp_path):
    from gliner2.processor import CharLevelSplitter

    save_tiny_boundary_checkpoint(tmp_path)
    model = AutoExtractor.from_pretrained(str(tmp_path), word_splitter="char")
    assert isinstance(model.processor.word_splitter, CharLevelSplitter)


def test_set_word_splitter_after_load(tmp_path):
    from gliner2.processor import CharLevelSplitter, WhitespaceTokenSplitter

    save_tiny_span_checkpoint(tmp_path)
    model = AutoExtractor.from_pretrained(str(tmp_path))
    assert isinstance(model.processor.word_splitter, WhitespaceTokenSplitter)
    returned = model.set_word_splitter("char")
    assert returned is model
    assert isinstance(model.processor.word_splitter, CharLevelSplitter)
    model.set_word_splitter(WhitespaceTokenSplitter())
    assert isinstance(model.processor.word_splitter, WhitespaceTokenSplitter)
