"""CompileSafeGRU hoist: one x-side matmul, loop only on the recurrent h-side."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gliner2.layers import CompileSafeGRU, CountLSTM


def _loop_gru(module: CompileSafeGRU, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Pre-hoist reference: F.linear(x[t], ...) inside the T loop."""
    seq_len = x.shape[0]
    if seq_len == 0:
        return x.new_empty(0, h.shape[0], module.hidden_size)
    outputs = []
    for t in range(seq_len):
        gi = F.linear(x[t], module.weight_ih_l0, module.bias_ih_l0)
        gh = F.linear(h, module.weight_hh_l0, module.bias_hh_l0)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h = (1 - z) * n + z * h
        outputs.append(h)
    return torch.stack(outputs, dim=0)


def test_compile_safe_gru_matches_per_step_linear():
    torch.manual_seed(0)
    gru = CompileSafeGRU(input_size=32, hidden_size=48)
    x = torch.randn(7, 5, 32)
    h = torch.randn(5, 48)
    out = gru(x, h)
    ref = _loop_gru(gru, x, h)
    assert torch.allclose(out, ref, atol=1e-6, rtol=0.0)


def test_compile_safe_gru_matches_nn_gru():
    torch.manual_seed(1)
    hidden, inp, seq, batch = 64, 64, 11, 4
    safe = CompileSafeGRU(input_size=inp, hidden_size=hidden)
    native = nn.GRU(input_size=inp, hidden_size=hidden, batch_first=False)
    native.weight_ih_l0.data.copy_(safe.weight_ih_l0.data)
    native.weight_hh_l0.data.copy_(safe.weight_hh_l0.data)
    native.bias_ih_l0.data.copy_(safe.bias_ih_l0.data)
    native.bias_hh_l0.data.copy_(safe.bias_hh_l0.data)
    x = torch.randn(seq, batch, inp)
    h = torch.randn(batch, hidden)
    out_safe = safe(x, h)
    out_native, _ = native(x, h.unsqueeze(0))
    assert torch.allclose(out_safe, out_native, atol=1e-5, rtol=1e-5)


def test_compile_safe_gru_empty_sequence():
    gru = CompileSafeGRU(input_size=8, hidden_size=16)
    x = torch.zeros(0, 3, 8)
    h = torch.zeros(3, 16)
    out = gru(x, h)
    assert out.shape == (0, 3, 16)


def test_count_lstm_still_runs():
    torch.manual_seed(2)
    layer = CountLSTM(hidden_size=24, max_count=6)
    fields = torch.randn(3, 24)
    out = layer(fields, gold_count_val=4)
    assert out.shape == (4, 3, 24)
