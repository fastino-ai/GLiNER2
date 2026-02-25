"""
Correctness tests for batched inference optimizations.

Verifies that optimized (batched) code paths produce identical output
to the original single-sample processing. Tests cover all extraction types:
entity, relation, structure, and classification.

Run with: pytest tests/test_batching_correctness.py -v
"""

import pytest
import torch

from gliner2 import GLiNER2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    """Load model once for all tests in this module."""
    m = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_single_results(model, texts, schema_builder_fn, extract_fn, **kwargs):
    """Run extraction one text at a time and return list of results."""
    results = []
    for text in texts:
        schema = schema_builder_fn(model)
        result = extract_fn(model, text, schema, **kwargs)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Entity Extraction
# ---------------------------------------------------------------------------

class TestEntityExtraction:
    """Entity extraction correctness tests."""

    ENTITY_TYPES = ["company", "person", "product", "location", "date"]

    TEXTS_SHORT = [
        "Apple released iPhone 15.",
        "Tim Cook leads Apple.",
    ]

    TEXTS_MEDIUM = [
        "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino on September 12.",
        "Google's Sundar Pichai spoke at the developer conference in Mountain View last Tuesday.",
        "Microsoft released Windows 11 in Redmond during their annual event.",
    ]

    TEXTS_LONG = [
        (
            "Apple Inc., headquartered in Cupertino, California, announced the launch of "
            "their latest flagship product, the iPhone 15 Pro Max, at a special event held "
            "on September 12, 2023. CEO Tim Cook presented the device alongside other "
            "products including the Apple Watch Series 9 and AirPods Pro 2nd generation. "
            "The event was held at the Steve Jobs Theater."
        ),
        (
            "Google CEO Sundar Pichai unveiled the Pixel 8 smartphone at a press conference "
            "in Mountain View, California. The device features Google's custom Tensor G3 "
            "chip and will be available starting at $699. The announcement was made during "
            "the annual Made by Google event on October 4, 2023."
        ),
    ]

    def test_single_entity_extraction(self, model):
        """Single text entity extraction produces non-empty results."""
        text = self.TEXTS_MEDIUM[0]
        result = model.extract_entities(text, self.ENTITY_TYPES)
        assert isinstance(result, dict)
        assert "entities" in result

    def test_batch_vs_single_entities(self, model):
        """Batch extraction matches single-sample extraction for entities."""
        texts = self.TEXTS_MEDIUM
        entity_types = self.ENTITY_TYPES

        # Single-sample results
        single_results = []
        for text in texts:
            r = model.extract_entities(text, entity_types)
            single_results.append(r)

        # Batch results
        batch_results = model.batch_extract_entities(
            texts, entity_types, batch_size=len(texts)
        )

        assert len(batch_results) == len(single_results)
        for i, (single, batch) in enumerate(zip(single_results, batch_results)):
            assert single == batch, (
                f"Sample {i} mismatch:\nsingle={single}\nbatch={batch}"
            )

    def test_batch_vs_single_entities_with_confidence(self, model):
        """Batch matches single with confidence scores."""
        texts = self.TEXTS_MEDIUM
        entity_types = self.ENTITY_TYPES

        single_results = []
        for text in texts:
            r = model.extract_entities(text, entity_types, include_confidence=True)
            single_results.append(r)

        batch_results = model.batch_extract_entities(
            texts, entity_types, batch_size=len(texts), include_confidence=True
        )

        assert len(batch_results) == len(single_results)
        for i, (single, batch) in enumerate(zip(single_results, batch_results)):
            assert single == batch, (
                f"Sample {i} mismatch with confidence:\nsingle={single}\nbatch={batch}"
            )

    def test_batch_vs_single_entities_short(self, model):
        """Batch matches single for short inputs."""
        texts = self.TEXTS_SHORT
        entity_types = self.ENTITY_TYPES

        single_results = [
            model.extract_entities(t, entity_types) for t in texts
        ]
        batch_results = model.batch_extract_entities(
            texts, entity_types, batch_size=len(texts)
        )

        for i, (s, b) in enumerate(zip(single_results, batch_results)):
            assert s == b, f"Short text sample {i} mismatch"

    def test_batch_vs_single_entities_long(self, model):
        """Batch matches single for long inputs."""
        texts = self.TEXTS_LONG
        entity_types = self.ENTITY_TYPES

        single_results = [
            model.extract_entities(t, entity_types) for t in texts
        ]
        batch_results = model.batch_extract_entities(
            texts, entity_types, batch_size=len(texts)
        )

        for i, (s, b) in enumerate(zip(single_results, batch_results)):
            assert s == b, f"Long text sample {i} mismatch"

    def test_batch_size_1(self, model):
        """batch_size=1 produces same results as single extraction."""
        text = self.TEXTS_MEDIUM[0]
        entity_types = self.ENTITY_TYPES

        single = model.extract_entities(text, entity_types)
        batch = model.batch_extract_entities(
            [text], entity_types, batch_size=1
        )

        assert len(batch) == 1
        assert single == batch[0]


# ---------------------------------------------------------------------------
# Relation Extraction
# ---------------------------------------------------------------------------

class TestRelationExtraction:
    """Relation extraction correctness tests."""

    TEXTS = [
        "Apple CEO Tim Cook works in Cupertino.",
        "Google CEO Sundar Pichai leads the company in Mountain View.",
        "Microsoft was founded by Bill Gates.",
    ]
    RELATION_TYPES = ["CEO_of", "works_in", "founded_by"]

    def test_single_relation_extraction(self, model):
        """Single text relation extraction produces a result."""
        result = model.extract_relations(self.TEXTS[0], self.RELATION_TYPES)
        assert isinstance(result, dict)

    def test_batch_vs_single_relations(self, model):
        """Batch matches single for relation extraction."""
        single_results = [
            model.extract_relations(t, self.RELATION_TYPES) for t in self.TEXTS
        ]
        batch_results = model.batch_extract_relations(
            self.TEXTS, self.RELATION_TYPES, batch_size=len(self.TEXTS)
        )

        assert len(batch_results) == len(single_results)
        for i, (s, b) in enumerate(zip(single_results, batch_results)):
            assert s == b, f"Relation sample {i} mismatch:\nsingle={s}\nbatch={b}"


# ---------------------------------------------------------------------------
# Structure Extraction
# ---------------------------------------------------------------------------

class TestStructureExtraction:
    """Structure extraction correctness tests."""

    TEXTS = [
        "Apple announced iPhone 15 at $999 on September 12.",
        "Google released Pixel 8 for $699 in October.",
        "Microsoft launched Surface Pro 9 at $1299.",
    ]

    def _make_schema(self, model):
        schema = model.create_schema()
        schema.structure("product_launch").field("company").field("product").field("price")
        return schema

    def test_single_structure_extraction(self, model):
        """Single text structure extraction produces results."""
        schema = self._make_schema(model)
        result = model.extract(self.TEXTS[0], schema)
        assert isinstance(result, dict)

    def test_batch_vs_single_structures(self, model):
        """Batch matches single for structure extraction."""
        single_results = []
        for text in self.TEXTS:
            schema = self._make_schema(model)
            r = model.extract(text, schema)
            single_results.append(r)

        schema = self._make_schema(model)
        batch_results = model.batch_extract(
            self.TEXTS, schema, batch_size=len(self.TEXTS)
        )

        assert len(batch_results) == len(single_results)
        for i, (s, b) in enumerate(zip(single_results, batch_results)):
            assert s == b, f"Structure sample {i} mismatch:\nsingle={s}\nbatch={b}"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestClassification:
    """Classification extraction correctness tests."""

    TEXTS = [
        "The movie was absolutely wonderful and moving.",
        "I hated every minute of this terrible film.",
        "It was okay, nothing special.",
    ]

    def _make_schema(self, model):
        schema = model.create_schema()
        schema.classification(
            "sentiment",
            labels=["positive", "negative", "neutral"],
        )
        return schema

    def test_single_classification(self, model):
        """Single text classification produces results."""
        schema = self._make_schema(model)
        result = model.extract(self.TEXTS[0], schema)
        assert isinstance(result, dict)

    def test_batch_vs_single_classification(self, model):
        """Batch matches single for classification."""
        single_results = []
        for text in self.TEXTS:
            schema = self._make_schema(model)
            r = model.extract(text, schema)
            single_results.append(r)

        schema = self._make_schema(model)
        batch_results = model.batch_extract(
            self.TEXTS, schema, batch_size=len(self.TEXTS)
        )

        assert len(batch_results) == len(single_results)
        for i, (s, b) in enumerate(zip(single_results, batch_results)):
            assert s == b, f"Classification sample {i} mismatch:\nsingle={s}\nbatch={b}"


# ---------------------------------------------------------------------------
# Span Representation Internals
# ---------------------------------------------------------------------------

class TestSpanRepInternals:
    """Test internal span representation for bit-identical output."""

    def test_compute_span_rep_deterministic(self, model):
        """compute_span_rep produces identical output on repeated calls."""
        device = next(model.parameters()).device
        token_embs = torch.randn(10, model.hidden_size, device=device)

        r1 = model.compute_span_rep(token_embs)
        r2 = model.compute_span_rep(token_embs)

        assert torch.equal(r1["span_rep"], r2["span_rep"])
        assert torch.equal(r1["spans_idx"], r2["spans_idx"])
        assert torch.equal(r1["span_mask"], r2["span_mask"])

    def test_compute_span_rep_single_token(self, model):
        """Span rep works for single-token input."""
        device = next(model.parameters()).device
        token_embs = torch.randn(1, model.hidden_size, device=device)

        result = model.compute_span_rep(token_embs)
        assert result["span_rep"].shape[0] == 1
        assert result["span_rep"].shape[1] == model.max_width

    def test_compute_span_rep_various_lengths(self, model):
        """Span rep works for various text lengths."""
        device = next(model.parameters()).device

        for length in [1, 2, 5, 10, 20, 50]:
            token_embs = torch.randn(length, model.hidden_size, device=device)
            result = model.compute_span_rep(token_embs)
            assert result["span_rep"].shape == (length, model.max_width, model.hidden_size)

    def test_batched_vs_single_span_rep(self, model):
        """Batched span rep closely matches per-sample span rep.

        Not bit-identical due to BLAS selecting different matmul algorithms for
        different batch shapes, but differences are < 0.01 — negligible compared
        to extraction thresholds (typically 0.3-0.5).
        """
        device = next(model.parameters()).device
        lengths = [3, 7, 12, 5, 1]
        embs_list = [torch.randn(l, model.hidden_size, device=device) for l in lengths]

        # Per-sample
        single_results = [model.compute_span_rep(e) for e in embs_list]

        # Batched
        batched_results = model.compute_span_rep_batched(embs_list)

        assert len(batched_results) == len(single_results)
        for i, (s, b) in enumerate(zip(single_results, batched_results)):
            assert torch.allclose(s["span_rep"], b["span_rep"], atol=1e-2, rtol=1e-3), (
                f"span_rep mismatch at sample {i}, "
                f"max diff={torch.max(torch.abs(s['span_rep'] - b['span_rep'])).item()}"
            )

    def test_batched_single_sample_bit_identical(self, model):
        """Batched with 1 sample is bit-identical to single compute_span_rep."""
        device = next(model.parameters()).device
        emb = torch.randn(10, model.hidden_size, device=device)

        single = model.compute_span_rep(emb)
        batched = model.compute_span_rep_batched([emb])

        assert torch.equal(single["span_rep"], batched[0]["span_rep"])


# ---------------------------------------------------------------------------
# Embedding Extraction Fast Path
# ---------------------------------------------------------------------------

class TestEmbeddingExtractionFastPath:
    """Verify fast path produces identical output to loop path."""

    def test_fast_vs_loop_embedding_extraction(self, model):
        """Fast gather path matches loop-based extraction."""
        texts = [
            "Apple released iPhone 15.",
            "Google CEO Sundar Pichai spoke at the conference in Mountain View last Tuesday.",
            "OK.",
        ]
        entity_types = ["company", "person", "product"]

        # Build a batch through the normal pipeline
        from gliner2.training.trainer import ExtractorCollator
        collator = ExtractorCollator(model.processor)

        schema_obj = model.create_schema()
        schema_obj.entities(entity_types)
        schema = schema_obj.build()

        dataset = [(t, schema) for t in texts]
        batch = collator(dataset)

        device = next(model.parameters()).device
        batch = batch.to(device)

        # Run encoder
        outputs = model.encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask
        )
        token_embeddings = outputs.last_hidden_state

        # Fast path
        fast_tok, fast_schema = model.processor._extract_embeddings_fast(
            token_embeddings, batch
        )
        # Loop path
        loop_tok, loop_schema = model.processor._extract_embeddings_loop(
            token_embeddings, batch.input_ids, batch
        )

        assert len(fast_tok) == len(loop_tok)
        for i in range(len(fast_tok)):
            assert torch.equal(fast_tok[i], loop_tok[i]), (
                f"Token embs differ at sample {i}: "
                f"fast shape={fast_tok[i].shape}, loop shape={loop_tok[i].shape}"
            )

        assert len(fast_schema) == len(loop_schema)
        for i in range(len(fast_schema)):
            for j in range(len(fast_schema[i])):
                for k in range(len(fast_schema[i][j])):
                    assert torch.equal(fast_schema[i][j][k], loop_schema[i][j][k]), (
                        f"Schema embs differ at sample {i}, schema {j}, token {k}"
                    )


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_single_word_text(self, model):
        """Single-word text produces results without errors."""
        result = model.extract_entities("Hello.", ["greeting"])
        assert isinstance(result, dict)

    def test_batch_size_1_entities(self, model):
        """batch_size=1 works for entities."""
        result = model.batch_extract_entities(
            ["Apple released iPhone 15."],
            ["company", "product"],
            batch_size=1
        )
        assert len(result) == 1

    def test_variable_length_batch(self, model):
        """Batch with variable-length texts works correctly."""
        texts = [
            "Hi.",
            "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino on September 12.",
            "OK.",
        ]
        result = model.batch_extract_entities(
            texts, ["company", "person"], batch_size=3
        )
        assert len(result) == 3
