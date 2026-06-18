"""
Test: span_mask and spans_idx are trimmed to match span_rep in batched computation.

Regression test for the bug where compute_span_rep_batched returned untrimmed
span_mask/spans_idx (padded to max_text_len * max_width) while span_rep was
correctly trimmed to each sample's actual text_len. This caused tensor shape
mismatches in compute_struct_loss when batch_size > 1 and samples had different
text lengths.

Run with: pytest tests/test_batch_span_mask_trim.py -v
"""

import io
import sys
import tempfile

import pytest
import torch

from gliner2 import GLiNER2
from gliner2.training.data import InputExample
from gliner2.training.trainer import TrainingConfig, GLiNER2Trainer


@pytest.fixture(scope="module")
def model():
    """Load model once for all tests in this module."""
    m = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Unit tests: span_rep shape correctness
# ---------------------------------------------------------------------------

class TestSpanMaskTrim:
    """Verify span_mask and spans_idx are trimmed to match span_rep per sample."""

    def test_batched_span_mask_matches_span_rep(self, model):
        """span_mask length == text_len * max_width for each sample in a batch."""
        device = next(model.parameters()).device
        max_width = model.max_width

        # Variable lengths to trigger different padding amounts
        lengths = [3, 12, 7, 1]
        embs_list = [
            torch.randn(l, model.hidden_size, device=device) for l in lengths
        ]

        results = model.compute_span_rep_batched(embs_list)

        assert len(results) == len(lengths)
        for i, (result, tl) in enumerate(zip(results, lengths)):
            expected_spans = tl * max_width
            span_rep = result["span_rep"]
            span_mask = result["span_mask"]
            spans_idx = result["spans_idx"]

            # span_rep shape: (text_len, max_width, hidden_size)
            assert span_rep.shape[0] == tl, (
                f"Sample {i}: span_rep dim0 should be {tl}, got {span_rep.shape[0]}"
            )

            # span_mask shape: (1, n_spans) — n_spans must equal text_len * max_width
            assert span_mask.shape[1] == expected_spans, (
                f"Sample {i}: span_mask should have {expected_spans} spans, "
                f"got {span_mask.shape[1]}"
            )

            # spans_idx shape: (1, n_spans, 2) — must also match
            assert spans_idx.shape[1] == expected_spans, (
                f"Sample {i}: spans_idx should have {expected_spans} spans, "
                f"got {spans_idx.shape[1]}"
            )

    def test_batched_span_mask_consistent_with_single(self, model):
        """Batched span_mask/spans_idx shapes match single-sample computation."""
        device = next(model.parameters()).device

        lengths = [5, 15, 2]
        embs_list = [
            torch.randn(l, model.hidden_size, device=device) for l in lengths
        ]

        batched_results = model.compute_span_rep_batched(embs_list)
        single_results = [model.compute_span_rep(e) for e in embs_list]

        for i in range(len(lengths)):
            b_mask = batched_results[i]["span_mask"]
            s_mask = single_results[i]["span_mask"]
            assert b_mask.shape == s_mask.shape, (
                f"Sample {i}: batched span_mask shape {b_mask.shape} != "
                f"single span_mask shape {s_mask.shape}"
            )

            b_idx = batched_results[i]["spans_idx"]
            s_idx = single_results[i]["spans_idx"]
            assert b_idx.shape == s_idx.shape, (
                f"Sample {i}: batched spans_idx shape {b_idx.shape} != "
                f"single spans_idx shape {s_idx.shape}"
            )

    def test_uniform_lengths_no_mismatch(self, model):
        """When all samples have the same length, no trimming issue arises."""
        device = next(model.parameters()).device
        max_width = model.max_width

        length = 8
        embs_list = [
            torch.randn(length, model.hidden_size, device=device)
            for _ in range(4)
        ]

        results = model.compute_span_rep_batched(embs_list)

        for i, result in enumerate(results):
            expected_spans = length * max_width
            assert result["span_mask"].shape[1] == expected_spans
            assert result["spans_idx"].shape[1] == expected_spans


# ---------------------------------------------------------------------------
# Training integration tests
# ---------------------------------------------------------------------------

# Samples with deliberately varying text lengths and 9+ entity labels
TRAIN_EXAMPLES = [
    InputExample(
        text="John works at Google in NYC.",
        entities={"person": ["John"], "company": ["Google"], "city": ["NYC"]},
    ),
    InputExample(
        text="A very long address like 123 Main Street, Apartment 5B, Springfield, Illinois, 62704, United States of America is hard to parse correctly.",
        entities={
            "street": ["Main Street"],
            "postal code": ["62704"],
            "state": ["Illinois"],
            "country": ["United States of America"],
        },
    ),
    InputExample(
        text="Dr. Jane Smith published a paper on quantum computing at MIT in September 2024.",
        entities={
            "person": ["Dr. Jane Smith"],
            "field": ["quantum computing"],
            "organization": ["MIT"],
            "date": ["September 2024"],
        },
    ),
    InputExample(
        text="OK.",
        entities={"person": []},
    ),
    InputExample(
        text="The price of Bitcoin hit $97,000 on the Nasdaq exchange during the morning session in Tokyo, Japan.",
        entities={
            "currency": ["Bitcoin"],
            "price": ["$97,000"],
            "exchange": ["Nasdaq"],
            "city": ["Tokyo"],
            "country": ["Japan"],
        },
    ),
    InputExample(
        text="CEO Satya Nadella announced Microsoft Azure's new GPU cluster in Redmond.",
        entities={
            "person": ["Satya Nadella"],
            "company": ["Microsoft"],
            "product": ["Azure"],
            "location": ["Redmond"],
        },
    ),
    InputExample(
        text="Hi.",
        entities={"greeting": ["Hi"]},
    ),
    InputExample(
        text="The European Central Bank raised interest rates by 25 basis points to combat inflation across the Eurozone, effective January 15, 2025.",
        entities={
            "organization": ["European Central Bank"],
            "metric": ["25 basis points"],
            "region": ["Eurozone"],
            "date": ["January 15, 2025"],
        },
    ),
]


class TestTrainingBatchCollation:
    """Integration tests: training with batch_size > 1 and varying text lengths."""

    def test_train_batch4_no_sample_errors(self, model):
        """Train with batch_size=4, 9+ entity labels, varying lengths.

        Verifies no 'Error processing sample' messages appear in stdout,
        which would indicate the tensor mismatch bug.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = TrainingConfig(
                output_dir=tmp_dir,
                batch_size=4,
                num_epochs=1,
                eval_strategy="no",
                logging_steps=9999,  # suppress normal logging
                fp16=False,
            )

            trainer = GLiNER2Trainer(model, config)

            # Capture stdout to check for error messages
            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                trainer.train(train_data=TRAIN_EXAMPLES)
            finally:
                sys.stdout = old_stdout

            output = captured.getvalue()
            assert "Error processing sample" not in output, (
                f"Training produced sample errors:\n{output}"
            )

    def test_all_samples_contribute_to_loss(self, model):
        """Verify valid_samples == batch_size for every batch (no silent drops).

        Uses return_individual_losses to inspect per-sample loss output and
        confirm every sample in the batch was processed without error.
        """
        from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

        dataset = ExtractorDataset(TRAIN_EXAMPLES, validate=False)
        collator = ExtractorCollator(
            model.processor, is_training=True, max_len=model.config.max_len
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, collate_fn=collator, shuffle=False
        )

        device = next(model.parameters()).device
        model.train()

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            outputs = model(batch, return_individual_losses=True)

            batch_size_actual = len(batch.task_types)
            valid_samples = outputs["batch_size"]
            assert valid_samples == batch_size_actual, (
                f"Batch {batch_idx}: only {valid_samples}/{batch_size_actual} "
                f"samples were valid"
            )

            # No individual sample should have an error key
            for i, ind in enumerate(outputs["individual_losses"]):
                assert "error" not in ind, (
                    f"Batch {batch_idx}, sample {i} errored: {ind['error']}"
                )

        model.eval()

    def test_batch1_vs_batch4_loss_similarity(self, model):
        """Training loss with batch_size=1 and batch_size=4 should be comparable.

        Both should produce non-zero, finite losses. With the bug, batch_size=4
        would silently skip most samples yielding near-zero loss.
        """
        losses = {}

        for bs in [1, 4]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = TrainingConfig(
                    output_dir=tmp_dir,
                    batch_size=bs,
                    num_epochs=1,
                    eval_strategy="no",
                    logging_steps=9999,
                    fp16=False,
                )

                trainer = GLiNER2Trainer(model, config)
                result = trainer.train(train_data=TRAIN_EXAMPLES)

                # Collect final epoch loss from metrics history
                history = result.get("train_metrics_history", [])
                if history:
                    losses[bs] = history[-1].loss
                else:
                    # Fall back: run a manual forward pass to get the loss
                    from gliner2.training.trainer import (
                        ExtractorCollator,
                        ExtractorDataset,
                    )
                    dataset = ExtractorDataset(TRAIN_EXAMPLES, validate=False)
                    collator = ExtractorCollator(
                        model.processor,
                        is_training=True,
                        max_len=model.config.max_len,
                    )
                    loader = torch.utils.data.DataLoader(
                        dataset, batch_size=bs, collate_fn=collator
                    )
                    device = next(model.parameters()).device
                    model.train()
                    total_loss = 0.0
                    n_batches = 0
                    for batch in loader:
                        batch = batch.to(device)
                        outputs = model(batch)
                        total_loss += outputs["total_loss"].item()
                        n_batches += 1
                    losses[bs] = total_loss / max(n_batches, 1)
                    model.eval()

        loss_bs1 = losses[1]
        loss_bs4 = losses[4]

        # Both losses must be positive and finite
        assert loss_bs1 > 0, f"batch_size=1 loss is {loss_bs1}, expected > 0"
        assert loss_bs4 > 0, f"batch_size=4 loss is {loss_bs4}, expected > 0"
        assert torch.isfinite(torch.tensor(loss_bs1)), "batch_size=1 loss is not finite"
        assert torch.isfinite(torch.tensor(loss_bs4)), "batch_size=4 loss is not finite"

        # Losses should be in the same ballpark (within 10x).
        # With the bug, batch_size=4 loss would be near zero because most
        # samples are skipped and contribute zero loss.
        ratio = max(loss_bs1, loss_bs4) / min(loss_bs1, loss_bs4)
        assert ratio < 10, (
            f"Loss ratio too large: bs1={loss_bs1:.4f}, bs4={loss_bs4:.4f}, "
            f"ratio={ratio:.1f}x — suggests samples are being silently dropped"
        )
