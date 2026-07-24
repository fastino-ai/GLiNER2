from types import SimpleNamespace

import pytest
import torch

from gliner2.joint_ie import JointIE, JointIEConfig, RawScorer
from gliner2.joint_ie.schema import JointSchema


class FakeBatch:
    def __init__(self, texts, schemas):
        self.input_ids = torch.ones((len(texts), 4), dtype=torch.long)
        self.attention_mask = torch.ones_like(self.input_ids)
        self.start_mappings = [[0, len(text)] for text in texts]
        self.end_mappings = [[len(text), len(text) + 1] for text in texts]
        self.text_tokens = [[text.lower(), "."] for text in texts]
        self.original_texts = [text + "." for text in texts]
        self.original_schemas = schemas
        self.task_types = [["entities", "relations"] for _ in texts]
        self.schema_tokens_list = [[
            ["(", "[P]", "entities", "[E]", "person", "[E]", "org", ")"],
            ["(", "[P]", "works_for", "[R]", "head", "[R]", "tail", ")"],
        ] for _ in texts]

    def __len__(self):
        return len(self.input_ids)

    def to(self, device, dtype=None):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class FakeProcessor:
    def __init__(self):
        self.calls = 0

    def change_mode(self, is_training):
        self.is_training = is_training

    def collate_fn_inference(self, rows, max_len=None):
        self.calls += 1
        texts, schemas = zip(*rows)
        return FakeBatch(list(texts), list(schemas))

    def extract_embeddings_from_batch(self, encoded, input_ids, batch):
        token = [torch.zeros((2, 4)) for _ in range(len(batch))]
        schemas = [[
            [torch.ones(4), torch.ones(4), torch.ones(4) * 2],
            [torch.ones(4), torch.ones(4), torch.ones(4) * 2],
        ] for _ in range(len(batch))]
        return token, schemas


class FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=torch.zeros((*input_ids.shape, 4)))


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FakeEncoder()
        self.processor = FakeProcessor()

    def compute_span_rep_batched(self, embeddings):
        # Dense L=2, W=2; the second token is processor-appended punctuation.
        return [{"span_rep": torch.ones((2, 2, 4))} for _ in embeddings]

    def count_pred(self, embedding):
        return torch.tensor([[0.0, 3.0, 2.0, -1.0]])

    def count_embed(self, fields, count):
        return fields.unsqueeze(0).expand(count, -1, -1)


def test_raw_scorer_batches_encoder_once_and_preserves_caller_offsets():
    model = FakeModel()
    scorer = RawScorer(model)
    lattices = scorer.batch_score(["Ada", "Bob"], {"entities": {}},
                                  batch_size=8, count_top_k=2)

    assert model.encoder.calls == 1
    assert len(lattices) == 2
    assert lattices[0].text == "Ada"
    assert lattices[0].start_mappings == (0,)
    assert lattices[0].end_mappings == (3,)
    assert lattices[0].valid_span_mask.tolist() == [[True, False]]

    entity, relation = lattices[0].tasks
    assert [hyp.count for hyp in entity.count_hypotheses] == [1]
    assert entity.count_hypotheses[0].role_logits.shape == (1, 2, 1, 2)
    assert [hyp.count for hyp in relation.count_hypotheses] == [1, 2]
    assert relation.count_hypotheses[1].role_logits.shape == (2, 2, 1, 2)
    assert torch.isfinite(relation.count_hypotheses[0].role_logits[..., 0, 0]).all()
    assert torch.isneginf(relation.count_hypotheses[0].role_logits[..., 0, 1]).all()


def test_batch_score_supports_per_document_schemas_and_one_pass_per_batch():
    model = FakeModel()
    scorer = RawScorer(model)
    schemas = [{"entities": {"person": ""}}, {"entities": {"org": ""}},
               {"entities": {"place": ""}}]
    lattices = scorer.batch_score(["A", "B", "C"], schemas, batch_size=2)
    assert model.encoder.calls == 2
    assert [item.schema for item in lattices] == schemas


def test_engine_compiles_shared_schema_once_and_runs_complete_pipeline():
    model = FakeModel()
    engine = JointIE(model)
    schema = JointSchema().entities(["person", "org"]).relation(
        "works_for", "person", "org"
    )
    calls = 0
    original = engine.compiler.compile

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    engine.compiler.compile = counted
    results = engine.batch_extract(["Ada", "Bob"], schema,
                                   config=JointIEConfig(batch_size=2))
    assert calls == 1
    assert model.encoder.calls == 1
    assert len(results) == 2
    assert all(result.text in {"Ada", "Bob"} for result in results)


def test_from_pretrained_splits_wrapper_and_model_options(monkeypatch):
    import gliner2

    captured = {}

    class Loader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured.update(path=path, kwargs=kwargs)
            return FakeModel()

    monkeypatch.setattr(gliner2, "GLiNER2", Loader)
    engine = JointIE.from_pretrained("repo", quantize=True, map_location="cpu")
    assert captured == {"path": "repo", "kwargs": {"quantize": True, "map_location": "cpu"}}
    assert engine.model is not None
    with pytest.raises(TypeError, match="JointIEConfig"):
        JointIE.from_pretrained("repo", beam_size=4)


def test_score_public_inspection_and_extraction_serialization_flags():
    engine = JointIE(FakeModel())
    schema = JointSchema().entities(["person", "org"]).relation("works_for", "person", "org")
    lattice = engine.score("Ada", engine.compile_schema(schema))
    span, probability = lattice.top_entities("person", 1)[0]
    assert (span.char_start, span.char_end, span.text) == (0, 3, "Ada")
    assert 0 <= probability <= 1
    assert lattice.top_heads("works_for", 0, 1)[0][0].text == "Ada"
    result = engine.extract("Ada", schema, config=JointIEConfig(
        include_confidence=False, include_spans=False))
    assert all("start" not in entity and "confidence" not in entity
               for entity in result.to_dict()["entities"])
    assert all(entity.start >= 0 and entity.end > entity.start for entity in result.entities)
    assert all("start" in entity for entity in result.to_dict(include_spans=True)["entities"])


def test_public_all_is_exact():
    import gliner2.joint_ie as joint_ie
    assert joint_ie.__all__ == ["JointIEEngine", "JointSchema", "JointResult"]


def test_public_score_compiles_joint_schema_for_processor():
    model = FakeModel()
    received = []
    original = model.processor.collate_fn_inference
    def capture(rows, max_len=None):
        rows = list(rows)
        received.extend(schema for _, schema in rows)
        return original(rows, max_len=max_len)
    model.processor.collate_fn_inference = capture
    engine = JointIE(model)
    schema = JointSchema().entity("person")
    engine.score("Ada", schema)
    assert isinstance(received[0], dict)
    assert received[0]["entities"] == {"person": ""}


def test_public_batch_score_compiles_shared_schema_once():
    model = FakeModel()
    engine = JointIE(model)
    schema = JointSchema().entity("person")
    calls = 0
    original = engine.compiler.compile
    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)
    engine.compiler.compile = counted
    lattices = engine.batch_score(["Ada", "Bob"], schema,
                                  config=JointIEConfig(batch_size=2))
    assert calls == 1
    assert model.encoder.calls == 1
    assert len(lattices) == 2
    with pytest.raises(ValueError, match="same length"):
        engine.batch_score(["Ada"], [schema, schema])


def test_prediction_config_is_per_call_and_model_identity_is_stable(monkeypatch):
    model = FakeModel()
    engine = JointIE(model)
    schema = JointSchema().entities(["person", "org"]).relation(
        "works_for", "person", "org")
    seen = []
    original = engine._make_optimizer

    def capture(config):
        value = original(config)
        seen.append((type(value).__name__, getattr(value, "beam_width", None)))
        return value

    monkeypatch.setattr(engine, "_make_optimizer", capture)
    first = engine.extract("Ada", schema, config=JointIEConfig(
        optimizer="greedy", count_top_k=1, include_confidence=False,
        include_spans=False))
    second = engine.extract("Ada", schema, config=JointIEConfig(
        optimizer="beam", beam_size=7, count_top_k=3,
        include_confidence=True, include_spans=True))

    assert engine.model is model
    assert seen == [("GreedyOptimizer", None), ("BeamOptimizer", 7)]
    assert first.default_include_confidence is False
    assert first.default_include_spans is False
    assert second.default_include_confidence is True
    assert second.default_include_spans is True


def test_count_top_k_is_call_scoped():
    engine = JointIE(FakeModel())
    schema = JointSchema().entities(["person", "org"]).relation(
        "works_for", "person", "org")
    one = engine.score("Ada", schema, config=JointIEConfig(count_top_k=1))
    three = engine.score("Ada", schema, config=JointIEConfig(count_top_k=3))
    relation_one = next(task for task in one.tasks if task.task_type == "relations")
    relation_three = next(task for task in three.tasks if task.task_type == "relations")
    assert len(relation_one.count_hypotheses) == 1
    assert len(relation_three.count_hypotheses) == 3


def test_only_joint_ie_config_class_exists():
    import ast
    from pathlib import Path
    root = Path(__file__).parents[1]
    names = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        names.extend(node.name for node in ast.walk(tree)
                     if isinstance(node, ast.ClassDef) and node.name.endswith("Config"))
    assert names == ["JointIEConfig"]
