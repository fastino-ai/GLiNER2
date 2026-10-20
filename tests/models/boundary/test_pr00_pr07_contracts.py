"""CPU contracts added by boundary PR-00 through PR-07."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from gliner2.models.boundary.constants import MASK_LOGIT
from gliner2.models.boundary.model import BoundaryExtractorModel
from gliner2.models.boundary.proposal import select_top_boundaries
from gliner2.processing.targets import MentionTarget, TargetGraph, pad_target_graphs
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
from scripts.preflight_gold_capacity import build_capacity_report


ROOT = Path(__file__).resolve().parents[3]


def test_golden_shape_invariant(golden_batch):
    head = golden_batch["head"]
    with torch.no_grad():
        head(
            golden_batch["token_states"],
            golden_batch["text_mask"],
            golden_batch["query_states"],
            golden_batch["query_mask"],
            golden_batch["targets"],
        )
    stats = head._last_proposal_stats
    b, q = 4, 3
    ks = head.settings.start_top_k
    length = max(golden_batch["lengths"])
    assert stats.max_materialized_pair_elements <= b * q * ks * (length + 1)


def test_benchmark_cpu_smoke(tmp_path):
    output = tmp_path / "bench.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "bench_boundary_head.py"),
            "--device", "cpu",
            "--iters", "2",
            "--batch", "1",
            "--length", "12",
            "--queries", "2",
            "--budget", "8",
            "--output", str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    row = json.loads(output.read_text())
    assert row["device"] == "cpu"
    assert row["forward_median_ms"] > 0
    assert row["forward_backward_median_ms"] > 0


def test_capacity_report_exact_p999_and_max():
    rows = [
        {"sample_id": i, "gold_counts": [value], "token_length": 10 + i}
        for i, value in enumerate(range(1, 1001))
    ]
    report = build_capacity_report(rows)
    assert report["gold_count_quantiles"]["p99.9"] == 999
    assert report["gold_count_quantiles"]["max"] == 1000


def test_preflight_dynamic_capacity_allocates_only_observed_gold():
    targets = pad_target_graphs(
        [
            TargetGraph(tuple(MentionTarget(0, i, i + 1) for i in range(3))),
            TargetGraph((MentionTarget(1, 0, 2),)),
        ],
        [1, 2],
        [4, 4],
        None,
        build_dense=False,
    )
    assert targets.mention_pairs.shape == (2, 2, 3, 2)
    assert targets.mention_mask.sum().item() == 4


def test_mask_logit_and_validity_are_finite_and_explicit():
    logits = torch.tensor([[[0.0, -60.0, 2.0]]], dtype=torch.float16)
    mask = torch.tensor([[[True, False, True]]])
    scores, indices, valid = select_top_boundaries(logits, mask, 3)
    assert MASK_LOGIT == -1.0e4
    assert torch.isfinite(scores).all()
    assert valid.sum() == 2
    assert not valid[0, 0, -1]
    assert indices[0, 0, -1] == 0


def test_recall_gate_exit_code_contract():
    good = {
        "dry_run_proposal_oracle_recall": 0.98,
        "dry_run_recall_length_9_plus": 0.94,
    }
    assert ExtractorTrainer.recall_gate_exit_code(good) == 0
    assert ExtractorTrainer.recall_gate_exit_code(
        {**good, "dry_run_proposal_oracle_recall": 0.96}
    ) == 1


def test_recall_dry_run_disables_injection_and_preserves_weights():
    class ProbeModel(nn.Module):
        architecture = "boundary"

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(3.0))
            self.boundary_head = SimpleNamespace(
                settings=SimpleNamespace(candidate_budget=8)
            )
            self.calls = []

        def forward(self, batch, **kwargs):
            self.calls.append(kwargs)
            one = self.weight.detach().new_tensor(1.0)
            return SimpleNamespace(
                metrics={
                    "proposal_gold_hit": one,
                    "proposal_gold_total": one,
                    "start_hit": one,
                    "end_hit": one,
                    "boundary_total": one,
                    "unique_candidates": one,
                    "valid_queries": one,
                    "absent_query_total": one.new_zeros(()),
                }
            )

    model = ProbeModel()
    trainer = object.__new__(ExtractorTrainer)
    trainer.model = model
    trainer.config = TrainingConfig(
        fp16=False, bf16=False, dry_run_recall_steps=1
    )
    before = model.weight.detach().clone()
    metrics = trainer._run_recall_dry_run([object()])
    assert model.calls == [
        {"gold_injection_prob": 0.0, "collect_diagnostics": True}
    ]
    assert torch.equal(model.weight, before)
    assert metrics["dry_run_proposal_oracle_recall"] == 1.0


def test_new_training_config_contracts():
    config = TrainingConfig(fp16=False, bf16=False)
    assert config.ddp_consensus_check
    assert not config.ddp_static_graph  # plain DDP: see tests/training/test_ddp_gradient_averaging.py
    assert not config.ddp_find_unused_parameters
    assert config.profile_first_n_steps == 0
    assert config.dry_run_recall_steps == 0


def test_backward_never_skips_and_nonfinite_grads_are_zero():
    class LossModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))
            self.loss = None

        def _zero_loss(self, device):
            return self.weight.sum() * 0.0

        def forward(self, batch):
            return {"total_loss": self.loss}

    model = LossModel()
    trainer = object.__new__(ExtractorTrainer)
    trainer.model = model
    trainer.config = TrainingConfig(fp16=False, bf16=False, strict_training=False)
    trainer.device = torch.device("cpu")
    trainer.is_distributed = False
    trainer.global_step = 0
    trainer._planned_max_steps = 1
    trainer._skip_counter = None
    trainer._loss_accum = None
    trainer._loss_finite_flag = None
    trainer._finite_grad_hook_handles = []
    trainer._install_finite_grad_hooks()

    trainer._backward_one(None, 0, False, torch.float16)
    assert model.weight.grad is not None
    assert model.weight.grad.item() == 0.0

    model.zero_grad(set_to_none=True)
    model.loss = model.weight * torch.tensor(float("nan"))
    trainer._backward_one(None, 1, False, torch.float16)
    assert torch.isfinite(model.weight.grad)
    assert model.weight.grad.item() == 0.0


def test_ignore_nonfinite_losses_keeps_strict_training_running():
    trainer = object.__new__(ExtractorTrainer)
    trainer.config = TrainingConfig(
        fp16=False,
        bf16=False,
        strict_training=True,
        ignore_nonfinite_losses=True,
    )
    trainer._skip_counter = torch.tensor(3)

    trainer._flush_delayed_counters()

    assert trainer._skip_counter.item() == 0


def test_optional_head_touch_produces_zero_gradients():
    holder = nn.Module()
    holder.record_decoder = nn.Linear(3, 2)
    holder.relation_scorer = nn.Linear(2, 1)
    value = BoundaryExtractorModel._head_touch(holder, torch.device("cpu"))
    assert value.item() == 0.0
    value.backward()
    for parameter in holder.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0
