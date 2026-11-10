"""Contract tests for optional DisentangledFlash inference support."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from gliner2.models.base import BaseExtractorModel
from gliner2.models.loading import (
    apply_post_load_options,
    pop_attention_backend,
    pop_disentangled_flash_options,
    split_load_kwargs,
)


class DebertaV2Config:
    """Class name intentionally matches the Transformers config."""


class FakeOptimizedEncoder(torch.nn.Module):
    def __init__(self, backend="torch"):
        super().__init__()
        self.backend = backend
        self._prepared_plans = {}
        self.active_length = None
        self.prepare_calls = []

    def prepare_for_inference(self, lengths):
        lengths = tuple(int(length) for length in lengths)
        self.prepare_calls.append(lengths)
        self._prepared_plans = {length: object() for length in lengths}

    def activate_shape(self, length):
        if int(length) not in self._prepared_plans:
            raise ValueError("unprepared length")
        self.active_length = int(length)


class FakeBackbone(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or DebertaV2Config()
        self.encoder = torch.nn.Identity()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids=None, attention_mask=None):
        return input_ids


class FakeExtractor:
    def __init__(self, config=None):
        self.training = False
        self.encoder = FakeBackbone(config).eval()


def _install_fake_disentangled_flash(monkeypatch):
    calls = []
    module = types.ModuleType("disentangled_flash")
    module.DebertaV2OptimizedEncoder = FakeOptimizedEncoder
    module.PackedSequenceInfo = tuple

    class KernelTuningOptions:
        def __init__(self, mode="auto"):
            self.mode = mode

    module.KernelTuningOptions = KernelTuningOptions

    def enable_deberta_inference(
        model,
        *,
        backend="auto",
        inference=True,
        sequence_lengths=None,
        fp32_precision="strict",
        tuning=None,
    ):
        selected_backend = "torch" if backend == "auto" else backend
        calls.append(
            {
                "model": model,
                "backend": backend,
                "inference": inference,
                "sequence_lengths": sequence_lengths,
                "fp32_precision": fp32_precision,
                "tuning": tuning,
            }
        )
        model.encoder = FakeOptimizedEncoder(selected_backend)
        model.encoder.inference = inference
        return model

    module.optimize_deberta = enable_deberta_inference
    monkeypatch.setitem(sys.modules, "disentangled_flash", module)
    return calls


def test_enable_requires_eval_mode(monkeypatch):
    _install_fake_disentangled_flash(monkeypatch)
    model = FakeExtractor()
    model.training = True

    with pytest.raises(RuntimeError, match="requires eval mode"):
        BaseExtractorModel.enable_disentangled_flash(model, inference=True)


def test_enable_rejects_non_deberta_backbone(monkeypatch):
    _install_fake_disentangled_flash(monkeypatch)
    model = FakeExtractor(config=object())

    with pytest.raises(TypeError, match="DeBERTa-v2/v3"):
        BaseExtractorModel.enable_disentangled_flash(model)


def test_enable_validates_backend_before_import(monkeypatch):
    model = FakeExtractor()

    with pytest.raises(ValueError, match="backend must be"):
        BaseExtractorModel.enable_disentangled_flash(model, backend="cuda")


def test_auto_backend_selection(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert (
        BaseExtractorModel._disentangled_flash_backend_for_device(
            torch.device("cuda")
        )
        == "triton"
    )
    assert (
        BaseExtractorModel._disentangled_flash_backend_for_device(
            torch.device("cpu")
        )
        == "torch"
    )
    assert (
        BaseExtractorModel._disentangled_flash_backend_for_device(
            torch.device("mps")
        )
        == "torch"
    )
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    assert (
        BaseExtractorModel._disentangled_flash_backend_for_device(
            torch.device("cuda")
        )
        == "torch"
    )


def test_enable_installs_backend_and_is_idempotent(monkeypatch):
    calls = _install_fake_disentangled_flash(monkeypatch)
    model = FakeExtractor()

    result = BaseExtractorModel.enable_disentangled_flash(model)
    second_result = BaseExtractorModel.enable_disentangled_flash(model)

    assert result is model
    assert second_result is model
    assert calls == [
        {
            "model": model.encoder,
            "backend": "torch",
            "inference": True,
            "sequence_lengths": None,
            "fp32_precision": "strict",
            "tuning": None,
        }
    ]
    assert model._disentangled_flash_backend == "torch"
    assert model._disentangled_flash_hook_handle is not None


def test_forward_hook_prepares_new_lengths_and_reuses_them(monkeypatch):
    _install_fake_disentangled_flash(monkeypatch)
    model = FakeExtractor()
    BaseExtractorModel.enable_disentangled_flash(model, backend="torch")
    optimized = model.encoder.encoder

    model.encoder(input_ids=torch.ones((2, 8), dtype=torch.long))
    model.encoder(input_ids=torch.ones((1, 8), dtype=torch.long))
    model.encoder(input_ids=torch.ones((1, 12), dtype=torch.long))

    assert optimized.prepare_calls == [(8,), (8, 12)]
    assert optimized.active_length == 12


def test_forward_hook_rejects_training_after_enable(monkeypatch):
    _install_fake_disentangled_flash(monkeypatch)
    model = FakeExtractor()
    BaseExtractorModel.enable_disentangled_flash(model)
    model.training = True

    with pytest.raises(RuntimeError, match="cannot run in training mode"):
        model.encoder(input_ids=torch.ones((1, 8), dtype=torch.long))


def test_training_mode_uses_differentiable_backend_without_hook(monkeypatch):
    calls = _install_fake_disentangled_flash(monkeypatch)
    model = FakeExtractor()
    model.training = True
    model.encoder.train()

    result = BaseExtractorModel.enable_disentangled_flash(
        model,
        backend="torch",
        inference=False,
        packed=False,
    )

    assert result is model
    assert calls[0]["inference"] is False
    assert model._disentangled_flash_mode == "training"
    assert model._disentangled_flash_backend == "torch"
    assert not hasattr(model, "_disentangled_flash_hook_handle")


def test_standard_load_option_enables_backend_after_device_and_precision(monkeypatch):
    events = []

    class LoadableModel:
        def to(self, device):
            events.append(("to", device))
            return self

        def quantize(self):
            events.append(("quantize",))
            return self

        def eval(self):
            events.append(("eval",))
            return self

        def enable_disentangled_flash(self, **options):
            events.append(("disentangled_flash", options))
            return self

    model = LoadableModel()
    result = apply_post_load_options(
        model,
        map_location="cuda",
        quantize=True,
        attention_backend="disentangled_flash",
    )

    assert result is model
    assert events == [
        ("to", "cuda"),
        ("quantize",),
        ("eval",),
        ("disentangled_flash", {}),
    ]


def test_standard_load_option_is_recognized():
    model_options, hub_options = split_load_kwargs(
        {"attention_backend": "disentangled_flash", "revision": "main"}
    )

    assert model_options == {"attention_backend": "disentangled_flash"}
    assert hub_options == {"revision": "main"}


def test_standard_load_option_rejects_compile_combination():
    with pytest.raises(ValueError, match="cannot be enabled together"):
        apply_post_load_options(
            object(),
            compile_model=True,
            attention_backend="disentangled_flash",
        )


@pytest.mark.parametrize(
    ("requested", "expected_flash"),
    [
        ("standard", False),
        ("flashdeberta", True),
        ("disentangled_flash", False),
    ],
)
def test_unified_attention_backend_selection(requested, expected_flash):
    options = {"attention_backend": requested}

    backend, use_flashdeberta = pop_attention_backend(options)

    assert backend == requested
    assert use_flashdeberta is expected_flash
    assert options == {}


def test_unified_backend_rejects_conflicting_legacy_alias():
    with pytest.raises(ValueError, match="conflicts"):
        pop_attention_backend(
            {
                "attention_backend": "disentangled_flash",
                "use_flashdeberta": True,
            }
        )


def _right_padded_mask(lengths, sequence_length):
    positions = torch.arange(sequence_length)
    return (positions[None, :] < torch.tensor(lengths)[:, None]).long()


def test_packing_layout_auto_uses_padding_threshold():
    layout = BaseExtractorModel._disentangled_flash_packing_layout
    # 4 x 10 slots, 22 real tokens -> 45% padding.
    mask = _right_padded_mask([10, 6, 4, 2], 10)

    packed = layout(mask, "auto", 0.4)
    assert packed is not None
    boolean_mask, lengths = packed
    assert lengths == (10, 6, 4, 2)
    assert boolean_mask.dtype == torch.bool
    assert layout(mask, "auto", 0.5) is None


def test_packing_layout_auto_falls_back_for_unpackable_batches():
    layout = BaseExtractorModel._disentangled_flash_packing_layout
    left_padded = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    empty_row = _right_padded_mask([4, 0], 4)

    assert layout(None, "auto", 0.0) is None
    assert layout(left_padded, "auto", 0.0) is None
    assert layout(empty_row, "auto", 0.0) is None


def test_packing_layout_forced_packs_unpadded_batches_and_rejects_bad_ones():
    layout = BaseExtractorModel._disentangled_flash_packing_layout
    assert layout(_right_padded_mask([4, 4], 4), True, 0.9)[1] == (4, 4)
    with pytest.raises(ValueError, match="right-padded"):
        layout(torch.tensor([[0, 1], [1, 1]]), True, 0.0)
    with pytest.raises(ValueError, match="attention_mask"):
        layout(None, True, 0.0)


@pytest.mark.parametrize(
    ("packed", "min_padding", "error"),
    [
        ("always", None, ValueError),
        ("auto", 1.5, ValueError),
        ("auto", "0.2", TypeError),
        ("auto", True, TypeError),
    ],
)
def test_enable_validates_packed_options(monkeypatch, packed, min_padding, error):
    _install_fake_disentangled_flash(monkeypatch)
    with pytest.raises(error):
        BaseExtractorModel.enable_disentangled_flash(
            FakeExtractor(),
            packed=packed,
            packed_min_padding=min_padding,
        )


def test_padded_batches_keep_prepared_plan_path(monkeypatch):
    _install_fake_disentangled_flash(monkeypatch)
    model = FakeExtractor()
    BaseExtractorModel.enable_disentangled_flash(
        model, backend="torch", packed_min_padding=0.5
    )
    optimized = model.encoder.encoder

    with torch.inference_mode():
        model.encoder(
            input_ids=torch.ones((2, 8), dtype=torch.long),
            attention_mask=_right_padded_mask([8, 6], 8),
        )

    assert optimized.prepare_calls == [(8,)]
    assert model._disentangled_flash_batches == {"packed": 0, "padded": 1}


def test_packed_load_options_are_forwarded_to_enable():
    model_options, _ = split_load_kwargs(
        {
            "attention_backend": "disentangled_flash",
            "disentangled_flash_packed": True,
            "disentangled_flash_min_padding": 0.3,
            "disentangled_flash_min_work": 2**20,
            "disentangled_flash_tuning": "heuristic",
        }
    )
    options = pop_disentangled_flash_options(model_options)
    assert options == {
        "packed": True,
        "packed_min_padding": 0.3,
        "packed_min_work": 2**20,
        "tuning": "heuristic",
    }
    assert model_options == {"attention_backend": "disentangled_flash"}

    received = {}

    class LoadableModel:
        def eval(self):
            return self

        def enable_disentangled_flash(self, **kwargs):
            received.update(kwargs)
            return self

    apply_post_load_options(
        LoadableModel(),
        attention_backend="disentangled_flash",
        disentangled_flash_options=options,
    )
    assert received == options


def test_packed_load_options_require_disentangled_flash_backend():
    with pytest.raises(ValueError, match="require attention_backend"):
        apply_post_load_options(
            object(),
            attention_backend="standard",
            disentangled_flash_options={"packed": True},
        )


def test_tuning_mode_name_is_forwarded(monkeypatch):
    calls = _install_fake_disentangled_flash(monkeypatch)
    BaseExtractorModel.enable_disentangled_flash(FakeExtractor(), tuning="heuristic")
    assert calls[0]["tuning"].mode == "heuristic"


def test_tuning_rejects_unknown_type(monkeypatch):
    _install_fake_disentangled_flash(monkeypatch)
    with pytest.raises(TypeError, match="tuning must be"):
        BaseExtractorModel.enable_disentangled_flash(FakeExtractor(), tuning=3)


def test_packing_layout_auto_requires_enough_attention_work():
    layout = BaseExtractorModel._disentangled_flash_packing_layout
    mask = _right_padded_mask([10, 6, 4, 2], 10)  # B * L**2 = 400

    assert layout(mask, "auto", 0.4, min_work=400) is not None
    assert layout(mask, "auto", 0.4, min_work=401) is None
    # Forced packing ignores both thresholds.
    assert layout(mask, True, 0.9, min_work=10**9) is not None


def test_default_packing_thresholds_match_measured_break_even():
    assert BaseExtractorModel.DISENTANGLED_FLASH_PACKED_MIN_PADDING == 0.15
    assert BaseExtractorModel.DISENTANGLED_FLASH_PACKED_MIN_WORK == 2**21
    layout = BaseExtractorModel._disentangled_flash_packing_layout
    defaults = (
        BaseExtractorModel.DISENTANGLED_FLASH_PACKED_MIN_PADDING,
        BaseExtractorModel.DISENTANGLED_FLASH_PACKED_MIN_WORK,
    )
    # 16 x 256 never paid off on the H200, even at 70% padding.
    small = _right_padded_mask([256] + [40] * 15, 256)
    assert layout(small, "auto", *defaults) is None
    # 8 x 512 paid off from 15% padding.
    large = _right_padded_mask([512] * 4 + [300] * 4, 512)
    assert layout(large, "auto", *defaults) is not None


@pytest.mark.parametrize(("min_work", "error"), [(-1, ValueError), (1.5, TypeError)])
def test_enable_validates_packed_min_work(monkeypatch, min_work, error):
    _install_fake_disentangled_flash(monkeypatch)
    with pytest.raises(error):
        BaseExtractorModel.enable_disentangled_flash(
            FakeExtractor(), packed_min_work=min_work
        )
