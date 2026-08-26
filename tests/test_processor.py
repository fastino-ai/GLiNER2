"""Comprehensive tests for gliner2.processor.

Covers:
- WhitespaceTokenSplitter correctness and offset integrity
- Token alignment (word -> subword -> embedding position mapping)
- Special token extraction (schema marker positions)
- SchemaTransformer end-to-end transform
- Collate batch padding and routing indices
- Classification prefix and selection wrapping
- Edge cases (empty text, multi-schema, truncation)
"""

from __future__ import annotations

import random

import pytest
import torch

from gliner2.processor import (
    CharLevelSplitter,
    PreprocessedBatch,
    SchemaTransformer,
    SamplingConfig,
    WhitespaceTokenSplitter,
    resolve_word_splitter,
)
from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    extra = [
        "[sep_struct]",
        "[sep_text]",
        "[p]",
        "[c]",
        "[e]",
        "[r]",
        "[l]",
        "[example]",
        "[output]",
        "[description]",
        "person",
        "location",
        "organization",
        "sentiment",
        "head",
        "tail",
        "relation",
        "field",
        "1",
        "2",
        "3",
        "4",
        "positive",
        "negative",
        "neutral",
        "hello",
        "world",
        "is",
        "great",
        "runs",
        "new",
        "york",
        "city",
        "lives",
    ]
    return build_tiny_tokenizer(extra_words=extra)


@pytest.fixture
def processor(tokenizer):
    return SchemaTransformer(tokenizer=tokenizer, token_pooling="first")


@pytest.fixture
def processor_no_sampling(tokenizer):
    cfg = SamplingConfig(
        remove_json_structure_prob=0.0,
        shuffle_json_fields=False,
        remove_json_field_prob=0.0,
        remove_entities_prob=0.0,
        shuffle_entities=False,
        remove_entity_prob=0.0,
        synthetic_entity_label_prob=0.0,
        remove_relations_prob=0.0,
        swap_head_tail_prob=0.0,
        remove_classification_prob=0.0,
        shuffle_classification_labels=False,
        remove_classification_label_prob=0.0,
        synthetic_label_prob=0.0,
        include_true_label_prob=1.0,
    )
    return SchemaTransformer(
        tokenizer=tokenizer, sampling_config=cfg, token_pooling="first"
    )


# ===========================================================================
# WhitespaceTokenSplitter
# ===========================================================================


class TestWhitespaceTokenSplitter:
    def test_basic_split(self):
        splitter = WhitespaceTokenSplitter()
        tokens = list(splitter("Hello World", lower=True))
        assert tokens[0] == ("hello", 0, 5)
        assert tokens[1] == ("world", 6, 11)

    def test_offsets_index_original_text(self):
        splitter = WhitespaceTokenSplitter()
        text = "The cat sat on the mat."
        for tok, start, end in splitter(text, lower=False):
            assert text[start:end] == tok

    def test_preserves_case_when_lower_false(self):
        splitter = WhitespaceTokenSplitter()
        tokens = list(splitter("NYC Apple", lower=False))
        assert tokens[0][0] == "NYC"
        assert tokens[1][0] == "Apple"

    def test_lowercases_when_lower_true(self):
        splitter = WhitespaceTokenSplitter()
        tokens = list(splitter("NYC Apple", lower=True))
        assert tokens[0][0] == "nyc"
        assert tokens[1][0] == "apple"

    def test_empty_string(self):
        splitter = WhitespaceTokenSplitter()
        assert list(splitter("")) == []

    def test_punctuation_separate(self):
        splitter = WhitespaceTokenSplitter()
        tokens = list(splitter("a. b!", lower=True))
        words = [t[0] for t in tokens]
        assert "a" in words
        assert "b" in words
        assert "." in words
        assert "!" in words

    def test_url_single_token(self):
        splitter = WhitespaceTokenSplitter()
        tokens = list(splitter("visit https://example.com today", lower=True))
        urls = [t[0] for t in tokens if "https" in t[0]]
        assert len(urls) == 1

    def test_email_single_token(self):
        splitter = WhitespaceTokenSplitter()
        tokens = list(splitter("email foo@bar.com now", lower=True))
        emails = [t[0] for t in tokens if "@" in t[0] and "." in t[0]]
        assert len(emails) == 1


class TestCharLevelSplitter:
    def test_keeps_latin_words_together(self):
        splitter = CharLevelSplitter()
        tokens = list(splitter("Hello 世界", lower=False))
        assert tokens[0] == ("Hello", 0, 5)
        assert [t[0] for t in tokens[1:]] == ["世", "界"]

    def test_chinese_character_boundaries(self):
        splitter = CharLevelSplitter()
        text = "我爱北京Tiananmen"
        tokens = list(splitter(text, lower=False))
        words = [t[0] for t in tokens]
        assert words == ["我", "爱", "北", "京", "Tiananmen"]
        for tok, start, end in tokens:
            assert text[start:end] == tok

    def test_does_not_lowercase_source_before_matching(self):
        splitter = CharLevelSplitter()
        text = "İA"
        tokens = list(splitter(text, lower=True))
        assert text[tokens[0][1] : tokens[0][2]] == "İ"
        assert text[tokens[1][1] : tokens[1][2]] == "A"


class TestResolveWordSplitter:
    def test_none_and_whitespace_name_are_default(self):
        default = resolve_word_splitter(None)
        named = resolve_word_splitter("whitespace")
        assert isinstance(default, WhitespaceTokenSplitter)
        assert isinstance(named, WhitespaceTokenSplitter)

    def test_char_name_and_class(self):
        assert isinstance(resolve_word_splitter("char"), CharLevelSplitter)
        assert isinstance(resolve_word_splitter(CharLevelSplitter), CharLevelSplitter)

    def test_unknown_name_lists_supported_values(self):
        with pytest.raises(ValueError, match="Supported names"):
            resolve_word_splitter("bytes")

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError, match="callable"):
            resolve_word_splitter(123)

    def test_custom_callable_is_returned(self):
        def custom(text, lower=True):
            yield text, 0, len(text)

        assert resolve_word_splitter(custom) is custom


class TestSchemaTransformerWordSplitter:
    def test_default_is_whitespace(self, tokenizer):
        processor = SchemaTransformer(tokenizer=tokenizer)
        assert isinstance(processor.word_splitter, WhitespaceTokenSplitter)

    def test_char_name_injection(self, tokenizer):
        processor = SchemaTransformer(tokenizer=tokenizer, word_splitter="char")
        assert isinstance(processor.word_splitter, CharLevelSplitter)

    def test_callable_injection(self, tokenizer):
        processor = SchemaTransformer(
            tokenizer=tokenizer, word_splitter=CharLevelSplitter()
        )
        assert isinstance(processor.word_splitter, CharLevelSplitter)


# ===========================================================================
# Token Alignment
# ===========================================================================


class TestTokenAlignment:
    """Verify word tokens align correctly with subword positions."""

    def test_text_word_first_positions_count(self, processor):
        """Each text word should get exactly one entry in text_word_first_positions."""
        text = "John Smith lives in New York City."
        schema = {"entities": {"person": [], "location": []}}
        record = processor.transform_and_format(text, schema)

        assert len(record.text_word_first_positions) == len(record.text_tokens)

    def test_text_word_positions_are_increasing(self, processor):
        """First-subword positions must be strictly increasing."""
        text = "The quick brown fox jumps over the lazy dog."
        schema = {"entities": {"entity": []}}
        record = processor.transform_and_format(text, schema)

        positions = record.text_word_first_positions
        for i in range(1, len(positions)):
            assert positions[i] > positions[i - 1], (
                f"Position {i} ({positions[i]}) <= position {i - 1} ({positions[i - 1]})"
            )

    def test_start_end_idx_map_to_original_text(self, processor):
        """start_token_idx / end_token_idx should map back to spans in the original text."""
        text = "Apple acquired Google."
        schema = {"entities": {"company": ["Apple", "Google"]}}
        record = processor.transform_and_format(text, schema)

        for i, tok in enumerate(record.text_tokens):
            start = record.start_token_idx[i]
            end = record.end_token_idx[i]
            # The lowercased word should match the span from text (case-insensitive)
            assert text[start:end].lower() == tok.lower()

    def test_batch_word_indices_valid_range(self, processor):
        """text_word_indices values must be valid positions within seq_len."""
        batch_data = [
            ("The cat sat.", {"entities": {"animal": ["cat"]}}),
            ("A dog ran fast.", {"entities": {"animal": ["dog"]}}),
        ]
        processor.is_training = False
        batch = processor.collate_fn_inference(batch_data)

        seq_len = batch.input_ids.shape[1]
        for i in range(len(batch)):
            n = batch.text_word_counts[i]
            indices = batch.text_word_indices[i, :n]
            assert (indices >= 0).all()
            assert (indices < seq_len).all()

    def test_mapped_indices_cover_full_sequence(self, processor):
        """Every input_id should have a corresponding mapping entry."""
        text = "Hello world."
        schema = {"entities": {"entity": ["Hello"]}}
        record = processor.transform_and_format(text, schema)

        assert len(record.mapped_indices) == len(record.input_ids)

    def test_mapped_indices_segment_types(self, processor):
        """Mappings should contain only 'schema', 'sep', and 'text' segments."""
        text = "Hello world."
        schema = {"entities": {"entity": ["Hello"]}}
        record = processor.transform_and_format(text, schema)

        seg_types = {m[0] for m in record.mapped_indices}
        assert seg_types <= {"schema", "sep", "text"}


# ===========================================================================
# Special Token Extraction
# ===========================================================================


class TestSpecialTokenExtraction:
    """Verify schema special token positions are correctly identified."""

    def test_schema_special_positions_count_entities(self, processor):
        """For entities schema, special positions = [P] + one [E] per entity label."""
        text = "John lives in NYC."
        schema = {"entities": {"person": ["John"], "location": ["NYC"]}}
        record = processor.transform_and_format(text, schema)

        # One schema group for entities: [P] + 2 [E] tokens = 3 special positions
        assert len(record.schema_special_positions) == 1
        assert len(record.schema_special_positions[0]) == 3  # [P], [E], [E]

    def test_schema_special_positions_count_relations(self, processor_no_sampling):
        """Relations schema should have [P] + one [R] per field."""
        text = "John founded SpaceX."
        schema = {"relations": [{"founded_by": {"head": "SpaceX", "tail": "John"}}]}
        processor_no_sampling.is_training = False
        record = processor_no_sampling.transform_and_format(text, schema)

        # [P] + 2 [R] tokens (head, tail)
        assert len(record.schema_special_positions) == 1
        assert len(record.schema_special_positions[0]) == 3

    def test_record_json_fields_preserve_declaration_order(
        self,
        processor_no_sampling,
    ):
        schema = {
            "json_structures": [
                {
                    "order": {
                        "order_id": "",
                        "quantity": "",
                        "item": "",
                        "total": "",
                    },
                }
            ],
            "record_metadata": {
                "order": {"mode": "natural", "anchor": "order_id"},
            },
        }
        transformed, labels, types = [], [], []
        processor_no_sampling.is_training = False

        processor_no_sampling._process_json_structures(
            schema,
            transformed,
            labels,
            types,
            sampling=None,
        )

        tokens = transformed[0]
        fields = [
            tokens[index + 1]
            for index, token in enumerate(tokens[:-1])
            if token == processor_no_sampling.C_TOKEN
        ]
        assert fields == ["order_id", "quantity", "item", "total"]

    def test_relation_description_is_encoded_in_parent_prompt(
        self,
        processor_no_sampling,
    ):
        schema = {
            "relations": [{"acquired": {"head": "", "tail": ""}}],
            "relation_descriptions": {
                "acquired": "completed purchase of a company",
            },
        }
        transformed, labels, types = [], [], []
        processor_no_sampling.is_training = False

        processor_no_sampling._process_relations(
            schema,
            transformed,
            labels,
            types,
            sampling=None,
        )

        assert transformed[0][2] == ("acquired: completed purchase of a company")

    def test_schema_special_positions_multi_schema(self, processor_no_sampling):
        """Multiple schema groups should each have their own positions list."""
        text = "Apple is great."
        schema = {
            "entities": {"company": ["Apple"]},
            "classifications": [
                {
                    "task": "sentiment",
                    "labels": ["positive", "negative"],
                    "true_label": ["positive"],
                }
            ],
        }
        processor_no_sampling.is_training = False
        record = processor_no_sampling.transform_and_format(text, schema)

        assert len(record.schema_special_positions) == 2
        # Each group must have at least a [P] token
        for group in record.schema_special_positions:
            assert len(group) >= 1

    def test_special_positions_point_to_special_tokens(self, processor):
        """Positions in schema_special_positions should map to special token IDs."""
        text = "The cat sat."
        schema = {"entities": {"animal": ["cat"]}}
        record = processor.transform_and_format(text, schema)

        special_ids = processor._special_ids
        for pos in record.schema_special_positions[0]:
            token_id = record.input_ids[pos]
            assert token_id in special_ids, (
                f"Position {pos} has id {token_id}, not a special token"
            )

    def test_query_marker_indices_match_schema_positions(self, processor):
        """Batch query_marker_indices should reflect schema_special_positions (minus [P])."""
        batch_data = [
            ("The cat sat.", {"entities": {"animal": ["cat"], "color": []}}),
        ]
        processor.is_training = False
        batch = processor.collate_fn_inference(batch_data)

        # For entities: [E] markers (not [P]) end up in query_marker_indices
        n_markers = batch.query_marker_mask[0].sum().item()
        # 2 entity labels => 2 [E] markers
        assert n_markers == 2

    def test_cls_marker_indices_for_classification(self, processor_no_sampling):
        """Classification [L] markers should appear in cls_marker_indices."""
        batch_data = [
            (
                "Hello world.",
                {
                    "classifications": [
                        {
                            "task": "sentiment",
                            "labels": ["positive", "negative", "neutral"],
                            "true_label": ["positive"],
                        }
                    ],
                },
            ),
        ]
        processor_no_sampling.is_training = False
        batch = processor_no_sampling.collate_fn_inference(batch_data)

        n_cls = batch.cls_marker_mask[0].sum().item()
        assert n_cls == 3  # 3 labels


# ===========================================================================
# SchemaTransformer End-to-End
# ===========================================================================


class TestSchemaTransformerE2E:
    def test_transform_entities_basic(self, processor):
        text = "John lives in NYC."
        schema = {"entities": {"person": ["John"], "location": ["NYC"]}}
        record = processor.transform_and_format(text, schema)

        assert record.text == text
        assert len(record.task_types) == 1
        assert record.task_types[0] == "entities"
        assert record.num_schemas == 1

    def test_transform_classification(self, processor_no_sampling):
        text = "This is great."
        schema = {
            "classifications": [
                {
                    "task": "sentiment",
                    "labels": ["positive", "negative"],
                    "true_label": ["positive"],
                }
            ]
        }
        processor_no_sampling.is_training = False
        record = processor_no_sampling.transform_and_format(text, schema)
        assert record.task_types[0] == "classifications"
        assert record.structure_labels[0] == [0, 0]
        gold = processor_no_sampling.transform_and_format(
            text, schema, build_targets=True
        )
        assert gold.structure_labels[0] == [1, 0]

    def test_collate_padding(self, processor):
        """Shorter sequences should be zero-padded to the longest."""
        batch_data = [
            ("short.", {"entities": {"x": []}}),
            ("a much longer sentence with many words in it.", {"entities": {"x": []}}),
        ]
        processor.is_training = False
        batch = processor.collate_fn_inference(batch_data)

        assert batch.input_ids.shape[0] == 2
        # Both have same padded length
        assert batch.input_ids.shape[1] == max(batch.original_lengths)
        # Attention mask zeros where padded
        for i in range(2):
            orig_len = batch.original_lengths[i]
            assert batch.attention_mask[i, :orig_len].sum() == orig_len
            if orig_len < batch.input_ids.shape[1]:
                assert batch.attention_mask[i, orig_len:].sum() == 0

    def test_collate_empty_batch(self, processor):
        processor.is_training = False
        batch = processor.collate_fn_inference([])
        assert len(batch) == 0

    def test_max_len_truncation(self, processor):
        """max_len should limit the number of text words."""
        text = "one two three four five six seven eight nine ten."
        schema = {"entities": {"number": []}}
        processor.is_training = False
        batch = processor.collate_fn_inference([(text, schema)], max_len=3)

        # Only 3 text words kept
        assert len(batch.text_tokens[0]) == 3
        assert batch.text_word_counts[0] == 3

    def test_punctuation_appended(self, processor):
        """Texts without trailing punctuation get a '.' appended."""
        batch_data = [("hello world", {"entities": {"x": []}})]
        processor.is_training = False
        batch = processor.collate_fn_inference(batch_data)

        # The original text stored should end with "."
        assert batch.original_texts[0].endswith(".")


# ===========================================================================
# Classification Prefix
# ===========================================================================


class TestClassificationPrefix:
    def test_prefix_creates_choice_tokens(self, processor_no_sampling):
        """JSON structures with choices should produce a prefix."""
        schema = {
            "json_structures": [
                {
                    "report": {
                        "sentiment": {
                            "value": "positive",
                            "choices": ["positive", "negative"],
                        },
                        "text": "Hello",
                    }
                }
            ]
        }
        prefix = processor_no_sampling._build_classification_prefix(schema)
        assert len(prefix) > 0
        assert "positive" in prefix or "negative" in prefix

    def test_selection_wrapping(self, processor_no_sampling):
        """Values with choices should be wrapped with [selection] prefix."""
        schema = {
            "json_structures": [
                {
                    "report": {
                        "mood": {"value": "happy", "choices": ["happy", "sad"]},
                    }
                }
            ]
        }
        processor_no_sampling._wrap_classification_fields(schema, ["dummy"])
        val = schema["json_structures"][0]["report"]["mood"]
        assert val == "[selection]happy"


# ===========================================================================
# Batch Device Transfer
# ===========================================================================


class TestBatchDeviceTransfer:
    def test_to_preserves_shape(self, processor):
        batch_data = [("The cat.", {"entities": {"animal": ["cat"]}})]
        processor.is_training = False
        batch = processor.collate_fn_inference(batch_data)

        moved = batch.to(torch.device("cpu"))
        assert moved.input_ids.shape == batch.input_ids.shape

    @pytest.mark.parametrize("floating_dtype", [torch.float16, torch.bfloat16])
    def test_to_preserves_every_integer_and_boolean_dtype(
        self, processor, floating_dtype
    ):
        processor.is_training = False
        batch = processor.collate_fn_inference(
            [("The cat.", {"entities": {"animal": ["cat"]}})]
        )

        moved = batch.to(torch.device("cpu"), floating_dtype)
        tensor_fields = [
            "input_ids",
            "attention_mask",
            "text_word_indices",
            "text_word_mask",
            "query_marker_indices",
            "query_marker_mask",
            "query_group_index",
            "cls_marker_indices",
            "cls_marker_mask",
            "cls_group_index",
        ]
        for field_name in tensor_fields:
            original = getattr(batch, field_name)
            transferred = getattr(moved, field_name)
            assert transferred.dtype == original.dtype, field_name

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="pin_memory requires CUDA"
    )
    def test_pin_memory(self, processor):
        batch_data = [("The cat.", {"entities": {"animal": ["cat"]}})]
        processor.is_training = False
        batch = processor.collate_fn_inference(batch_data)

        pinned = batch.pin_memory()
        assert pinned.input_ids.shape == batch.input_ids.shape


# ===========================================================================
# Error Policies
# ===========================================================================


class TestErrorPolicies:
    def test_skip_policy_drops_bad_record(self, processor, monkeypatch):
        """error_policy='skip' should silently drop malformed records."""
        calls = [0]
        orig = processor._transform_record

        def fail_first(record, max_len=None, **kwargs):
            calls[0] += 1
            if calls[0] == 1:
                raise ValueError("bad record")
            return orig(record, max_len=max_len, **kwargs)

        monkeypatch.setattr(processor, "_transform_record", fail_first)
        processor.is_training = False

        batch_data = [
            ("bad text.", {"entities": {"x": []}}),
            ("good text.", {"entities": {"x": []}}),
        ]
        batch = processor.collate_fn_inference(batch_data, error_policy="skip")
        assert len(batch) == 1

    def test_raise_policy_propagates(self, processor, monkeypatch):
        """error_policy='raise' should propagate exceptions."""
        monkeypatch.setattr(
            processor,
            "_transform_record",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("boom")),
        )
        processor.is_training = False
        with pytest.raises(ValueError, match="boom"):
            processor.collate_fn_inference(
                [("x.", {"entities": {"x": []}})], error_policy="raise"
            )


# ===========================================================================
# Public single-record transform + tokenization cache
# ===========================================================================


class TestTransformRecord:
    def test_transform_record_matches_private_path(self, processor):
        text = "John Smith lives in New York City."
        schema = {"entities": {"person": [], "location": []}}
        public = processor.transform_record(text, schema)
        private = processor._transform_record({"text": text, "schema": schema.copy()})
        assert public.input_ids == private.input_ids
        assert public.text_tokens == private.text_tokens
        assert public.task_types == private.task_types

    def test_transform_record_honors_max_len(self, processor):
        text = "one two three four five six seven eight nine ten."
        schema = {"entities": {"number": []}}
        record = processor.transform_record(text, schema, max_len=3)
        assert len(record.text_tokens) == 3

    def test_transform_and_format_delegates(self, processor):
        text = "hello world."
        schema = {"entities": {"x": []}}
        a = processor.transform_and_format(text, schema)
        b = processor.transform_record(text, schema)
        assert a.input_ids == b.input_ids


class TestTokenizationCache:
    def test_repeated_schema_hits_tokenize_cache(self, processor):
        schema = {"entities": {"person": [], "location": []}}
        processor._tokenize_cached.cache_clear()
        processor.transform_record("Alice met Bob.", schema)
        after_first = processor._tokenize_cached.cache_info()
        processor.transform_record("Carol met Dave.", schema)
        after_second = processor._tokenize_cached.cache_info()
        assert after_second.hits > after_first.hits
        assert after_second.currsize >= after_first.currsize
