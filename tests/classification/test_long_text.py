"""Phase 9 gate: long-text classification aggregates logits BEFORE decoding, so
a constraint satisfied within each chunk but violated in aggregate is enforced.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gliner2.classification import constraints as C
from gliner2.classification.engine import ClassificationConfig, Classifier
from gliner2.classification.long_text import aggregate_scores
from gliner2.classification.schema import ClassificationSchema
from gliner2.classification.compiler import compile_schema
from gliner2.classification.scoring import ClassificationScores
from gliner2.models.base import BaseExtractorModel


# A fake whose per-label logits depend on the chunk text, so different chunks
# yield different evidence (unlike the shared-logit conftest fake).
class _TextKeyedProcessor:
    def __init__(self, tasks, logits_by_key):
        self.tasks = tasks
        self.logits_by_key = logits_by_key  # substring -> {task: {label: logit}}
        self.is_training = False

    def change_mode(self, is_training):
        self.is_training = is_training

    def _key(self, text):
        for key in self.logits_by_key:
            if key in text:
                return key
        return next(iter(self.logits_by_key))

    def collate_fn_inference(self, rows, max_len=None, architecture="span"):
        texts = [t for t, _ in rows]
        batch = SimpleNamespace()
        batch.input_ids = torch.ones((len(texts), 4), dtype=torch.long)
        batch.attention_mask = torch.ones_like(batch.input_ids)
        batch.task_types = [["classifications"] * len(self.tasks) for _ in texts]
        batch.schema_tokens_list = [[
            ["(", "[P]", name, "("] + sum(([["[L]", l][k] for k in (0, 1)]
                                           for l in labels), []) + [")", ")"]
            for name, labels in self.tasks
        ] for _ in texts]
        batch._texts = texts
        batch.to = lambda *a, **k: batch
        return batch

    def extract_embeddings_from_batch(self, encoded, input_ids, batch):
        token = [torch.zeros((2, 4)) for _ in batch._texts]
        schema = []
        for text in batch._texts:
            key = self._key(text)
            per_task = []
            for name, labels in self.tasks:
                rows = [torch.zeros(4)]
                rows += [torch.full((4,), self.logits_by_key[key][name][l]) for l in labels]
                per_task.append(rows)
            schema.append(per_task)
        return token, schema


class _FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=torch.zeros((*input_ids.shape, 4)))


class _FakeModel(torch.nn.Module):
    encode_tokens = BaseExtractorModel.encode_tokens

    def __init__(self, tasks, logits_by_key):
        super().__init__()
        self.encoder = _FakeEncoder()
        self.processor = _TextKeyedProcessor(tasks, logits_by_key)

    def classifier(self, embeds):
        return embeds[:, :1]


# ---- T-L1 : aggregation precedes decoding ------------------------------

def test_aggregation_precedes_decoding():
    tasks = [("flag", ["a", "b"])]
    # chunk with "alpha": a high, b low; chunk with "beta": b high, a low.
    logits = {
        "alpha": {"flag": {"a": 3.0, "b": -3.0}},
        "beta": {"flag": {"a": -3.0, "b": 3.0}},
    }
    schema = (ClassificationSchema()
              .multi("flag", ["a", "b"], min_labels=0)
              .constrain(C.excludes(("flag", "a"), ("flag", "b"))))
    clf = Classifier(_FakeModel(tasks, logits))
    text = "alpha " * 400 + "beta " * 400  # forces >1 chunk
    result = clf.classify_long(text, schema, aggregate="max",
                               config=ClassificationConfig(decoder="exact",
                                                           candidate_threshold=0.0),
                               chunk_size=384, chunk_overlap=0)
    selected = result.selected("flag")
    # aggregate max makes both a and b look strong; excludes forbids both.
    assert not ({"a", "b"} <= set(selected))
    assert len(selected) <= 1


# ---- T-L2 : aggregate modes --------------------------------------------

def test_aggregate_modes():
    schema = ClassificationSchema().single("s", ["x", "y"])
    compiled = compile_schema(schema)

    def scores(val):
        return ClassificationScores("c", {"s": {"x": val, "y": 0.0}},
                                    compiled.fingerprint,
                                    {sp.name: sp for sp in compiled.task_specs})

    chunk_scores = [scores(4.0), scores(0.0)]
    agg_max = aggregate_scores(chunk_scores, compiled, "t", "max")
    agg_mean = aggregate_scores(chunk_scores, compiled, "t", "mean")
    agg_first = aggregate_scores(chunk_scores, compiled, "t", "first")
    assert agg_max.tasks["s"]["x"] == 4.0
    assert agg_mean.tasks["s"]["x"] == 2.0
    assert agg_first.tasks["s"]["x"] == 4.0
    with pytest.raises(ValueError):
        aggregate_scores(chunk_scores, compiled, "t", "median")


# ---- T-L3 : single short chunk == classify -----------------------------

def test_single_chunk_matches_classify():
    tasks = [("s", ["x", "y"])]
    logits = {"": {"s": {"x": 2.0, "y": -1.0}}}
    schema = ClassificationSchema().single("s", ["x", "y"])
    clf = Classifier(_FakeModel(tasks, logits))
    short = "hello world"
    direct = clf.classify(short, schema)
    chunked = clf.classify_long(short, schema)
    assert direct.value("s") == chunked.value("s")


# ---- T-L4 : chunking honors chunk_size (interaction with max_len) ------

def test_chunking_produces_multiple_chunks():
    from gliner2.inference.chunking import split_text_into_chunks
    text = "word " * 1000
    chunks = split_text_into_chunks(text, chunk_size=384, chunk_overlap=0)
    assert len(chunks) > 1  # documents chunk_size behavior for the long path


def test_classify_long_uses_model_word_splitter(monkeypatch):
    from gliner2.inference.chunking import TextChunk

    sentinel = object()
    captured = {}

    def fake_split(text, chunk_size=384, chunk_overlap=64, word_splitter=None):
        captured["word_splitter"] = word_splitter
        return [
            TextChunk(
                text=text,
                start_char=0,
                end_char=len(text),
                start_word=0,
                end_word=1,
            )
        ]

    monkeypatch.setattr(
        "gliner2.classification.long_text.split_text_into_chunks", fake_split
    )
    tasks = [("s", ["x", "y"])]
    logits = {"": {"s": {"x": 2.0, "y": -1.0}}}
    model = _FakeModel(tasks, logits)
    model.processor.word_splitter = sentinel
    clf = Classifier(model)
    schema = ClassificationSchema().single("s", ["x", "y"])
    clf.classify_long("hello world", schema)
    assert captured["word_splitter"] is sentinel
