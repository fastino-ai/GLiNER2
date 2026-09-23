"""Test harness for the classification module.

Extends the ``joint_ie/tests/test_engine.py`` fakes with a classification path.
Critical additions over the joint IE version: ``[L]`` markers in
``schema_tokens_list``, ``task_types == ["classifications"]``, and a
``classifier`` attribute on the model.

The point of the ``planted`` fixture is that every scorer and decoder test can
state its inputs as *logits per label* and assert on outputs, with zero neural
indirection.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gliner2.models.base import BaseExtractorModel


def _task_tokens(name, labels):
    """Encode one task the way the processor would: ( [P] name ( [L] a [L] b ) )."""
    tokens = ["(", "[P]", name, "("]
    for label in labels:
        tokens += ["[L]", label]
    tokens += [")", ")"]
    return tokens


class FakeClsBatch:
    """Mirrors PreprocessedBatch's read surface for classification only."""

    def __init__(self, texts, schemas, tasks):
        # tasks: list[(task_name, [label, ...])] shared by every row.
        self.input_ids = torch.ones((len(texts), 4), dtype=torch.long)
        self.attention_mask = torch.ones_like(self.input_ids)
        self.text_tokens = [[t.lower(), "."] for t in texts]
        self.start_mappings = [[0, len(t)] for t in texts]
        self.end_mappings = [[len(t), len(t) + 1] for t in texts]
        self.original_texts = [t + "." for t in texts]
        self.original_schemas = schemas
        self.task_types = [["classifications"] * len(tasks) for _ in texts]
        self.schema_tokens_list = [[
            _task_tokens(name, labels) for name, labels in tasks
        ] for _ in texts]
        self._tasks = tasks

    def __len__(self):
        return self.input_ids.shape[0]

    def to(self, device, dtype=None):
        return self


class FakeClsProcessor:
    def __init__(self, tasks, logits):
        self.tasks, self.logits, self.calls = tasks, logits, 0
        self.is_training = False

    def change_mode(self, is_training):
        self.is_training = is_training

    def collate_fn_inference(self, rows, max_len=None, architecture="span"):
        self.calls += 1
        texts, schemas = zip(*rows)
        return FakeClsBatch(list(texts), list(schemas), self.tasks)

    def extract_embeddings_from_batch(self, encoded, input_ids, batch):
        # One row per marker: [P] first, then one per [L] label. Encode the
        # intended logit in the embedding so FakeClsModel can read it back.
        token = [torch.zeros((2, 4)) for _ in range(len(batch))]
        schema = []
        for _ in range(len(batch)):
            per_task = []
            for name, labels in self.tasks:
                rows = [torch.zeros(4)]                              # [P]
                rows += [torch.full((4,), self.logits[name][l]) for l in labels]
                per_task.append(rows)
            schema.append(per_task)
        return token, schema


class FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=torch.zeros((*input_ids.shape, 4)))


class FakeClsModel(torch.nn.Module):
    """model.classifier(embs).squeeze(-1) must reproduce the planted logits."""

    encode_tokens = BaseExtractorModel.encode_tokens

    def __init__(self, tasks, logits):
        super().__init__()
        self.encoder = FakeEncoder()
        self.processor = FakeClsProcessor(tasks, logits)

    def classifier(self, embeds):        # (n_labels, hidden) -> (n_labels, 1)
        return embeds[:, :1]


@pytest.fixture
def planted():
    """Factory: planted(tasks, logits) -> FakeClsModel with known label logits."""
    def make(tasks, logits):
        return FakeClsModel(tasks, logits)
    return make
