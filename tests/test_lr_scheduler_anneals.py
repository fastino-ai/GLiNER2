"""
Test: a decaying schedule must reach ~0 by the final step.

`lr_lambda_cosine_restarts` computes `cos(pi * ((num_cycles * progress) % 1.0))`.
With `num_cycles = 0.5` the argument only ever reaches 0.5, so it never wraps:
no restart happens, and the learning rate stops at half of base with the final
epoch still taking full-size steps.

    num_cycles=0.5  ->  progress 0.50: 0.854   progress 1.00: 0.500
    num_cycles=1.0  ->  progress 0.50: 0.500   progress 1.00: 0.000

Run:
    pytest tests/test_lr_scheduler_anneals.py
"""

import math

import pytest
import torch

from gliner2.training.trainer import TrainingConfig, get_scheduler

TOTAL, WARMUP = 10_000, 500
DECAYING = ("linear", "cosine", "cosine_restarts")


def _lr_lambda(scheduler_type, num_cycles):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = get_scheduler(optimizer, scheduler_type, TOTAL, WARMUP, num_cycles)
    return scheduler.lr_lambdas[0]


def _default_cycles():
    return TrainingConfig.__dataclass_fields__["num_cycles"].default


@pytest.mark.parametrize("scheduler_type", DECAYING)
def test_decaying_schedules_reach_zero_at_the_last_step(scheduler_type):
    lr = _lr_lambda(scheduler_type, _default_cycles())
    assert lr(TOTAL - 1) < 0.01, f"{scheduler_type} ends at {lr(TOTAL - 1):.3f} of base"


def test_warmup_ramps_to_full_lr():
    lr = _lr_lambda("cosine_restarts", _default_cycles())
    assert lr(0) == pytest.approx(0.0)
    assert lr(WARMUP // 2) == pytest.approx(0.5, abs=0.01)
    assert lr(WARMUP) == pytest.approx(1.0)


def test_default_cosine_restarts_matches_plain_cosine_on_real_steps():
    # With one full cycle the two schedules agree everywhere training runs; they
    # differ only at step == TOTAL, which is never executed (steps are 0..TOTAL-1).
    restarts = _lr_lambda("cosine_restarts", 1.0)
    cosine = _lr_lambda("cosine", 1.0)
    for step in (WARMUP, TOTAL // 2, int(TOTAL * 0.9), TOTAL - 1):
        assert math.isclose(restarts(step), cosine(step), abs_tol=1e-9)


def test_half_a_cycle_still_degenerates():
    # Not a recommendation -- a record of what the old default did, for anyone who
    # sets it deliberately: no restart, and the LR stops at half of base.
    lr = _lr_lambda("cosine_restarts", 0.5)
    assert lr(TOTAL - 1) == pytest.approx(0.5, abs=0.01)
