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
import re

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
    return SchemaTransformer(tokenizer=tokenizer, sampling_config=cfg, token_pooling="first")


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

    @pytest.mark.parametrize(
        "text,words",
        [
            ("प्रधानमंत्री नरेंद्र मोदी ने कहा।", ["प्रधानमंत्री", "नरेंद्र", "मोदी", "ने", "कहा", "।"]),
            ("কলকাতা শহরে", ["কলকাতা", "শহরে"]),
            ("சென்னையில் மோடி", ["சென்னையில்", "மோடி"]),
            ("హైదరాబాద్ నగరం", ["హైదరాబాద్", "నగరం"]),
            ("ذَهَبَ مُحَمَّدٌ", ["ذَهَبَ", "مُحَمَّدٌ"]),
        ],
    )
    def test_combining_marks_stay_inside_words(self, text, words):
        splitter = WhitespaceTokenSplitter()
        tokens = list(splitter(text, lower=False))
        assert [t[0] for t in tokens] == words
        for tok, start, end in tokens:
            assert text[start:end] == tok

    def test_zero_width_joiners_stay_inside_words(self):
        splitter = WhitespaceTokenSplitter()
        text = "क्‍ष और र‌ा"
        assert [t[0] for t in splitter(text, lower=False)] == ["क्‍ष", "और", "र‌ा"]

    def test_decomposed_latin_accents_stay_inside_words(self):
        splitter = WhitespaceTokenSplitter()
        text = "Café Nguyễn"
        assert [t[0] for t in splitter(text, lower=False)] == ["Café", "Nguyễn"]

    @pytest.mark.parametrize(
        "text",
        [
            "Dr. Smith-Jones met @ann at foo@bar.com, https://x.org/a?b=1 (2024)!",
            "Größe café naïve São Paulo русский",
            "snake_case and kebab-case-words -leading trailing- 3.14",
        ],
    )
    def test_text_without_combining_marks_splits_as_before(self, text):
        previous = re.compile(
            r"""(?:https?://[^\s]+|www\.[^\s]+)
            |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
            |@[a-z0-9_]+
            |\w+(?:[-_]\w+)*
            |\S""",
            re.VERBOSE | re.IGNORECASE,
        )
        expected = [(m.group().lower(), m.start(), m.end()) for m in previous.finditer(text)]
        assert list(WhitespaceTokenSplitter()(text, lower=True)) == expected


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
        processor = SchemaTransformer(tokenizer=tokenizer, word_splitter=CharLevelSplitter())
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

    def test_record_json_fields_preserve_declaration_order(self, processor_no_sampling):
        schema = {
            "json_structures": [
                {"order": {"order_id": "", "quantity": "", "item": "", "total": ""}}
            ],
            "record_metadata": {"order": {"mode": "natural", "anchor": "order_id"}},
        }
        transformed, labels, types = [], [], []
        processor_no_sampling.is_training = False

        processor_no_sampling._process_json_structures(
            schema, transformed, labels, types, sampling=None
        )

        tokens = transformed[0]
        fields = [
            tokens[index + 1]
            for index, token in enumerate(tokens[:-1])
            if token == processor_no_sampling.C_TOKEN
        ]
        assert fields == ["order_id", "quantity", "item", "total"]

    def test_relation_description_is_encoded_in_parent_prompt(self, processor_no_sampling):
        schema = {
            "relations": [{"acquired": {"head": "", "tail": ""}}],
            "relation_descriptions": {"acquired": "completed purchase of a company"},
        }
        transformed, labels, types = [], [], []
        processor_no_sampling.is_training = False

        processor_no_sampling._process_relations(schema, transformed, labels, types, sampling=None)

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
            assert token_id in special_ids, f"Position {pos} has id {token_id}, not a special token"

    def test_query_marker_indices_match_schema_positions(self, processor):
        """Batch query_marker_indices should reflect schema_special_positions (minus [P])."""
        batch_data = [("The cat sat.", {"entities": {"animal": ["cat"], "color": []}})]
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
                    ]
                },
            )
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
        assert record.structure_labels[0] == [1, 0]
        gold = processor_no_sampling.transform_and_format(text, schema, build_targets=True)
        assert gold.structure_labels[0] == [1, 0]
        plain = processor_no_sampling.transform_and_format(text, schema, build_targets=False)
        assert plain.structure_labels[0] == [0, 0]

    def test_transform_classification_without_gold(self, processor_no_sampling):
        schema = {"classifications": [{"task": "sentiment", "labels": ["positive", "negative"]}]}
        processor_no_sampling.is_training = False
        record = processor_no_sampling.transform_and_format("This is great.", schema)
        assert record.structure_labels[0] == [0, 0]

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
# Explicit gold spans
# ===========================================================================


# token offsets: Pushkin 0, street 1, runs 2, past 3, the 4, Pushkin 5, monument 6
SPAN_TEXT = "Pushkin street runs past the Pushkin monument."


class TestGoldSpans:
    @staticmethod
    def _entity_labels(processor, mentions, **kwargs):
        schema = {"entities": {"street": mentions}}
        return processor.transform_record(SPAN_TEXT, schema, **kwargs).structure_labels

    def test_span_pins_one_occurrence_of_a_repeated_surface(self, processor_no_sampling):
        p = processor_no_sampling
        assert self._entity_labels(p, ["Pushkin"]) == [[1, [[[(0, 0), (5, 5)]]]]]
        assert self._entity_labels(
            p, [{"text": "Pushkin", "start": 0, "end": 7}]
        ) == [[1, [[[(0, 0)]]]]]
        assert self._entity_labels(
            p, [{"text": "Pushkin", "start": 29, "end": 36}]
        ) == [[1, [[[(5, 5)]]]]]

    def test_span_must_cover_the_text_it_claims(self, processor_no_sampling):
        with pytest.raises(ValueError, match="not 'Pushkin'"):
            self._entity_labels(
                processor_no_sampling, [{"text": "Pushkin", "start": 1, "end": 8}]
            )

    def test_span_past_max_len_is_not_found(self, processor_no_sampling):
        p = processor_no_sampling
        far = {"text": "Pushkin", "start": 29, "end": 36}
        assert self._entity_labels(p, [far], max_len=3) == [[1, [[[(-1, -1)]]]]]
        assert self._entity_labels(p, [far], max_len=50) == [[1, [[[(5, 5)]]]]]
        # a span straddling the cut keeps the tokens that survived it
        assert self._entity_labels(
            p, [{"text": "street runs past", "start": 8, "end": 24}], max_len=3
        ) == [[1, [[[(1, 2)]]]]]

    def test_spans_and_surfaces_mix_within_one_structure(self, processor_no_sampling):
        schema = {
            "json_structures": [
                {"place": {"name": {"text": "Pushkin", "start": 0, "end": 7}}},
                {"place": {"name": "monument"}},
            ]
        }
        record = processor_no_sampling.transform_record(SPAN_TEXT, schema)
        assert record.structure_labels == [[2, [[[(0, 0)]], [[(6, 6)]]]]]

    def test_span_is_rejected_for_a_choice_field(self, processor_no_sampling):
        schema = {
            "json_structures": [
                {
                    "place": {
                        "kind": {
                            "value": {"text": "Pushkin", "start": 0, "end": 7},
                            "choices": ["street", "monument"],
                        }
                    }
                }
            ]
        }
        with pytest.raises(ValueError, match="cannot be pinned to a span"):
            processor_no_sampling.transform_record(SPAN_TEXT, schema)


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
                        "sentiment": {"value": "positive", "choices": ["positive", "negative"]},
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
                {"report": {"mood": {"value": "happy", "choices": ["happy", "sad"]}}}
            ]
        }
        processor_no_sampling._wrap_classification_fields(schema, ["dummy"])
        val = schema["json_structures"][0]["report"]["mood"]
        assert val == "[selection]happy"


class TestClassificationSyntheticLabels:
    """Synthetic renaming must keep the gold label a positive under its new name."""

    @pytest.fixture
    def processor_synthetic(self, tokenizer):
        cfg = SamplingConfig(
            remove_classification_prob=0.0,
            shuffle_classification_labels=True,
            remove_classification_label_prob=1.0,
            synthetic_label_prob=1.0,
            include_true_label_prob=1.0,
        )
        return SchemaTransformer(tokenizer=tokenizer, sampling_config=cfg, token_pooling="first")

    @pytest.mark.parametrize("true_label", [["neutral"], "neutral", ["positive", "neutral"]])
    def test_reinserted_true_label_uses_synthetic_name(self, processor_synthetic, true_label):
        labels = ["positive", "negative", "neutral"]
        gold = true_label if isinstance(true_label, list) else [true_label]
        expected = {f"label {labels.index(t) + 1}" for t in gold}
        random.seed(0)
        for _ in range(200):
            schema = {
                "classifications": [
                    {"task": "sentiment", "labels": list(labels), "true_label": true_label}
                ]
            }
            processor_synthetic._process_classifications(
                schema, [], [], [], processor_synthetic.sampling_config
            )
            item = schema["classifications"][0]
            assert not set(labels) & set(item["labels"])
            assert set(item["true_label"]) == expected
            assert expected <= set(item["labels"])


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
    def test_to_preserves_every_integer_and_boolean_dtype(self, processor, floating_dtype):
        processor.is_training = False
        batch = processor.collate_fn_inference([("The cat.", {"entities": {"animal": ["cat"]}})])

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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
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
            processor.collate_fn_inference([("x.", {"entities": {"x": []}})], error_policy="raise")


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

    @pytest.mark.parametrize("text", ["John Smith lives in New York City", "", "hello world!"])
    def test_transform_record_matches_collate_fn_inference(self, processor, text):
        schema = {
            "entities": {"person": [], "location": []},
            "classifications": [{"task": "sentiment", "labels": ["positive", "negative"]}],
        }
        processor.is_training = True
        record = processor.transform_record(text, schema)
        assert processor.is_training is False
        batch = processor.collate_fn_inference([(text, schema)], error_policy="raise")
        assert record.input_ids == batch.input_ids[0, : batch.original_lengths[0]].tolist()
        assert record.text == batch.original_texts[0]
        assert record.text_tokens == batch.text_tokens[0]
        assert record.schema_tokens_list == batch.schema_tokens_list[0]
        assert record.structure_labels == batch.structure_labels[0]


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


# ===========================================================================
# find_unalignable_entities (Bug 1: pre-flight alignment check)
# ===========================================================================


class TestFindUnalignableEntities:
    """Regression coverage for the "ABS" vs hyphen-compound-filename bug.

    "ABS" is a real, exact character substring of the surrounding text, but
    ``WhitespaceTokenSplitter``'s hyphen-continuation rule merges the whole
    hyphen-joined run into a single token, so a standalone-token search for
    "abs" finds nothing. A naive substring check (like
    ``InputExample.validate()``'s ``mention.lower() in text.lower()``) would
    say this is fine; ``find_unalignable_entities`` must not, because it
    reuses the exact tokenizer/search training uses.
    """

    def test_hyphen_compound_filename_is_reported_unalignable(self, processor):
        text = "See IMG Bosch-eBike-LEDRemote-ABS-BES3-MY2023.png for wiring."
        result = processor.find_unalignable_entities(text, {"PartCode": "ABS"})
        assert result == [
            {"entity_type": "PartCode", "value": "ABS", "value_index": None}
        ]

    def test_naive_substring_check_would_have_missed_this(self, processor):
        text = "See IMG Bosch-eBike-LEDRemote-ABS-BES3-MY2023.png for wiring."
        # The exact "gap" this utility closes: a plain substring check thinks
        # this annotation is fine, but it is not token-alignable.
        assert "abs" in text.lower()
        assert processor.find_unalignable_entities(text, {"PartCode": "ABS"})

    def test_fully_alignable_schema_returns_empty_list(self, processor):
        text = "John Smith lives in New York City."
        result = processor.find_unalignable_entities(
            text, {"person": ["John Smith"], "location": ["New York City"]}
        )
        assert result == []

    def test_list_valued_field_reports_only_the_failing_value_with_its_index(
        self, processor
    ):
        text = "See IMG Bosch-eBike-LEDRemote-ABS-BES3-MY2023.png for wiring. New York is nice."
        result = processor.find_unalignable_entities(
            text, {"tag": ["New York", "ABS"]}
        )
        assert result == [{"entity_type": "tag", "value": "ABS", "value_index": 1}]

    def test_reused_search_matches_actual_training_behavior(self, processor):
        """The utility must never drift from ``collate_fn_train``'s own check.

        A value reported as alignable here must not raise during training,
        and a value reported as unalignable here must raise during training.
        """
        text = "See IMG Bosch-eBike-LEDRemote-ABS-BES3-MY2023.png for wiring."
        schema = {"entities": {"PartCode": "ABS"}}
        assert processor.find_unalignable_entities(text, schema["entities"]) == [
            {"entity_type": "PartCode", "value": "ABS", "value_index": None}
        ]
        with pytest.raises(ValueError, match="was not found"):
            processor.collate_fn_train(
                [(text, schema)], architecture="boundary", error_policy="raise"
            )

    def test_empty_and_none_values_are_skipped_not_reported(self, processor):
        text = "John Smith lives in New York."
        result = processor.find_unalignable_entities(
            text, {"person": ["John Smith", ""], "ghost": None, "empty_list": []}
        )
        assert result == []

    def test_selection_prefixed_values_are_out_of_scope(self, processor):
        # [selection]-wrapped values are matched against the classification
        # prefix, not document text; this utility only reasons about document
        # alignment, so it must not misreport these as unalignable.
        text = "John Smith lives in New York."
        result = processor.find_unalignable_entities(
            text, {"status": "[selection]anything"}
        )
        assert result == []
