"""
GLiNER2 Schema Transformer with Optimized Batch Processing

This module handles all preprocessing for GLiNER2, with efficient batching
via DataLoader collate functions for parallel preprocessing.
"""

import copy
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Iterator, List
import torch
from transformers import AutoTokenizer


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TransformedRecord:
    """Single transformed record ready for batching."""
    input_ids: List[int]
    mapped_indices: List[Tuple[str, int, int]]
    schema_tokens_list: List[List[str]]
    text_tokens: List[str]
    structure_labels: List[Any]
    task_types: List[str]
    start_token_idx: List[int]
    end_token_idx: List[int]
    text: str
    schema: Dict[str, Any]
    num_schemas: int = field(init=False)

    def __post_init__(self):
        self.num_schemas = len(self.schema_tokens_list)


@dataclass
class PreprocessedBatch:
    """GPU-ready batch for training/inference."""
    input_ids: torch.Tensor  # (batch, max_seq_len)
    attention_mask: torch.Tensor  # (batch, max_seq_len)
    mapped_indices: List[List[Tuple]]  # Per-sample token mappings
    schema_counts: List[int]  # Number of schemas per sample
    original_lengths: List[int]  # Original sequence lengths
    structure_labels: List[List[Any]]  # Ground truth labels
    task_types: List[List[str]]  # Task types per schema
    text_tokens: List[List[str]]  # Original text tokens
    schema_tokens_list: List[List[List[str]]]  # Schema tokens per sample
    start_mappings: List[List[int]]  # Char position start mappings
    end_mappings: List[List[int]]  # Char position end mappings
    original_texts: List[str]  # For result formatting
    original_schemas: List[Dict]  # For result formatting

    def to(self, device: torch.device) -> 'PreprocessedBatch':
        """Move tensors to device."""
        return PreprocessedBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            mapped_indices=self.mapped_indices,
            schema_counts=self.schema_counts,
            original_lengths=self.original_lengths,
            structure_labels=self.structure_labels,
            task_types=self.task_types,
            text_tokens=self.text_tokens,
            schema_tokens_list=self.schema_tokens_list,
            start_mappings=self.start_mappings,
            end_mappings=self.end_mappings,
            original_texts=self.original_texts,
            original_schemas=self.original_schemas,
        )

    def pin_memory(self) -> 'PreprocessedBatch':
        """Pin tensors to memory for faster GPU transfer."""
        return PreprocessedBatch(
            input_ids=self.input_ids.pin_memory(),
            attention_mask=self.attention_mask.pin_memory(),
            mapped_indices=self.mapped_indices,
            schema_counts=self.schema_counts,
            original_lengths=self.original_lengths,
            structure_labels=self.structure_labels,
            task_types=self.task_types,
            text_tokens=self.text_tokens,
            schema_tokens_list=self.schema_tokens_list,
            start_mappings=self.start_mappings,
            end_mappings=self.end_mappings,
            original_texts=self.original_texts,
            original_schemas=self.original_schemas,
        )

    def __contains__(self, key: str) -> bool:
        """Check if key is a field name. Required for HuggingFace Trainer compatibility."""
        return hasattr(self, key)

    def __iter__(self):
        """Iterate over field names. Required for HuggingFace Trainer compatibility."""
        return iter(self.__dataclass_fields__.keys())

    def __getitem__(self, key):
        """Get field by name. Required for HuggingFace Trainer compatibility."""
        if isinstance(key, str):
            return getattr(self, key)
        raise KeyError(f"PreprocessedBatch does not support integer indexing: {key}")

    def __len__(self) -> int:
        return self.input_ids.shape[0]


# =============================================================================
# Tokenizer
# =============================================================================

class WhitespaceTokenSplitter:
    """Fast regex-based tokenizer for text splitting."""
    __slots__ = ()

    _PATTERN = re.compile(
        r"""(?:https?://[^\s]+|www\.[^\s]+)
        |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
        |@[a-z0-9_]+
        |\w+(?:[-_]\w+)*
        |\S""",
        re.VERBOSE | re.IGNORECASE,
    )

    def __call__(self, text: str, lower: bool = True) -> Iterator[Tuple[str, int, int]]:
        if lower:
            text = text.lower()
        for m in self._PATTERN.finditer(text):
            yield m.group(), m.start(), m.end()


# =============================================================================
# Sampling Configuration
# =============================================================================

@dataclass
class SamplingConfig:
    """Configuration for stochastic sampling during training."""
    # JSON Structures
    remove_json_structure_prob: float = 0.2
    shuffle_json_fields: bool = True
    remove_json_field_prob: float = 0.2
    # Entities
    remove_entities_prob: float = 0.0
    shuffle_entities: bool = False
    remove_entity_prob: float = 0.0
    synthetic_entity_label_prob: float = 0.2
    # Relations
    remove_relations_prob: float = 0.2
    swap_head_tail_prob: float = 0.2
    # Classifications
    remove_classification_prob: float = 0.0
    shuffle_classification_labels: bool = True
    remove_classification_label_prob: float = 0.5
    synthetic_label_prob: float = 0.5
    include_true_label_prob: float = 0.5
    max_num_labels: int = 1000


# =============================================================================
# Main Processor Class
# =============================================================================

class SchemaTransformer:
    """
    Schema-based text transformer for GLiNER2.

    Provides efficient batch preprocessing via collate functions
    for parallel DataLoader preprocessing.
    """

    # Special tokens
    SEP_STRUCT = "[SEP_STRUCT]"
    SEP_TEXT = "[SEP_TEXT]"
    P_TOKEN = "[P]"
    C_TOKEN = "[C]"
    E_TOKEN = "[E]"
    R_TOKEN = "[R]"
    L_TOKEN = "[L]"
    EXAMPLE_TOKEN = "[EXAMPLE]"
    OUTPUT_TOKEN = "[OUTPUT]"
    DESC_TOKEN = "[DESCRIPTION]"

    SPECIAL_TOKENS = [
        SEP_STRUCT, SEP_TEXT, P_TOKEN, C_TOKEN, E_TOKEN,
        R_TOKEN, L_TOKEN, EXAMPLE_TOKEN, OUTPUT_TOKEN, DESC_TOKEN
    ]

    def __init__(
            self,
            model_name: str = None,
            tokenizer=None,
            sampling_config: SamplingConfig = None,
            token_pooling: str = "first"
    ):
        if model_name is None and tokenizer is None:
            raise ValueError("Either model_name or tokenizer must be provided.")

        self.token_pooling = token_pooling if token_pooling in ["first", "mean", "max"] else "first"
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.word_splitter = WhitespaceTokenSplitter()
        self.sampling_config = sampling_config or SamplingConfig()
        self.is_training = False

        # Add special tokens
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": self.SPECIAL_TOKENS
        })

    def change_mode(self, is_training: bool):
        """Switch between training and inference mode."""
        self.is_training = is_training

    # =========================================================================
    # Main Public API: Collate Functions
    # =========================================================================

    def collate_fn_train(
            self,
            batch: List[Tuple[str, Dict]]
    ) -> PreprocessedBatch:
        """
        Collate function for training DataLoader.

        Use this with DataLoader for parallel preprocessing:

            loader = DataLoader(
                dataset,
                batch_size=32,
                collate_fn=processor.collate_fn_train,
                num_workers=8
            )

        Args:
            batch: List of (text, schema) tuples from dataset

        Returns:
            PreprocessedBatch ready for model.forward()
        """
        self.is_training = True
        return self._collate_batch(batch)

    def collate_fn_inference(
            self,
            batch: List[Tuple[str, Any]]
    ) -> PreprocessedBatch:
        """
        Collate function for inference DataLoader.

        Args:
            batch: List of (text, schema) tuples

        Returns:
            PreprocessedBatch for batch_extract
        """
        self.is_training = False
        return self._collate_batch(batch)

    def transform_and_format(
            self,
            text: str,
            schema: Dict[str, Any]
    ) -> TransformedRecord:
        """
        Transform and format a single record.

        This is the main preprocessing entry point for single records.
        For batch processing, use collate_fn_train/collate_fn_inference.

        Args:
            text: Input text
            schema: Schema dictionary

        Returns:
            TransformedRecord ready for batching
        """
        record = {"text": text, "schema": schema}
        return self._transform_record(record)

    # =========================================================================
    # Internal: Batch Processing
    # =========================================================================

    def _collate_batch(
            self,
            batch: List[Tuple[str, Any]]
    ) -> PreprocessedBatch:
        """Internal collate implementation."""
        transformed_records = []

        for text, schema in batch:
            # Handle Schema objects
            if hasattr(schema, 'build'):
                schema = schema.build()
            elif hasattr(schema, 'schema'):
                schema = schema.schema

            # Ensure text ends with punctuation
            if text and not text.endswith(('.', '!', '?')):
                text = text + "."
            elif not text:
                text = "."

            record = {"text": text, "schema": copy.deepcopy(schema)}

            try:
                transformed = self._transform_record(record)
                transformed_records.append(transformed)
            except Exception as e:
                # Create minimal fallback record
                transformed_records.append(self._create_fallback_record(text, schema))

        return self._pad_batch(transformed_records)

    def _transform_record(self, record: Dict[str, Any]) -> TransformedRecord:
        """Transform a single record (internal)."""
        record_ = copy.deepcopy(record)
        text, schema = record_["text"], record_["schema"]

        # Build classification prefix
        prefix = self._build_classification_prefix(schema)

        # Handle classification field wrapping
        if prefix:
            self._wrap_classification_fields(schema, prefix)

        # Tokenize text
        text_tokens = []
        start_idx_map = []
        end_idx_map = []
        for tkn, start, end in self.word_splitter(text, lower=True):
            text_tokens.append(tkn)
            start_idx_map.append(start)
            end_idx_map.append(end)

        if prefix:
            text_tokens = prefix + text_tokens
        len_prefix = len(prefix)

        # Infer schema
        processed = self._infer_from_json(schema)

        # Build outputs
        results = self._build_outputs(
            processed, schema, text_tokens, len_prefix
        )

        # Format input
        schema_tokens_list = [r["schema_tokens"] for r in results]
        format_result = self._format_input_with_mapping(schema_tokens_list, text_tokens)

        return TransformedRecord(
            input_ids=format_result["input_ids"],
            mapped_indices=format_result["mapped_indices"],
            schema_tokens_list=schema_tokens_list,
            text_tokens=text_tokens,
            structure_labels=[r["output"] for r in results],
            task_types=[r["task_type"] for r in results],
            start_token_idx=start_idx_map,
            end_token_idx=end_idx_map,
            text=text,
            schema=schema,
        )

    def _pad_batch(
            self,
            records: List[TransformedRecord]
    ) -> PreprocessedBatch:
        """Pad transformed records into a batch."""
        if not records:
            return self._empty_batch()

        max_len = max(len(r.input_ids) for r in records)
        batch_size = len(records)

        # Pre-allocate tensors
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        original_lengths = []

        for i, rec in enumerate(records):
            seq_len = len(rec.input_ids)
            input_ids[i, :seq_len] = torch.tensor(rec.input_ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            original_lengths.append(seq_len)

        return PreprocessedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mapped_indices=[r.mapped_indices for r in records],
            schema_counts=[r.num_schemas for r in records],
            original_lengths=original_lengths,
            structure_labels=[r.structure_labels for r in records],
            task_types=[r.task_types for r in records],
            text_tokens=[r.text_tokens for r in records],
            schema_tokens_list=[r.schema_tokens_list for r in records],
            start_mappings=[r.start_token_idx for r in records],
            end_mappings=[r.end_token_idx for r in records],
            original_texts=[r.text for r in records],
            original_schemas=[r.schema for r in records],
        )

    def _empty_batch(self) -> PreprocessedBatch:
        """Create empty batch for edge cases."""
        return PreprocessedBatch(
            input_ids=torch.zeros((0, 0), dtype=torch.long),
            attention_mask=torch.zeros((0, 0), dtype=torch.long),
            mapped_indices=[],
            schema_counts=[],
            original_lengths=[],
            structure_labels=[],
            task_types=[],
            text_tokens=[],
            schema_tokens_list=[],
            start_mappings=[],
            end_mappings=[],
            original_texts=[],
            original_schemas=[],
        )

    def _create_fallback_record(self, text: str, schema: Dict) -> TransformedRecord:
        """Create minimal valid record for failed transformations."""
        dummy_tokens = [
            "(", "[P]", "dummy", "(", "[E]", "entity", ")", ")"
        ]
        format_result = self._format_input_with_mapping([dummy_tokens], ["."])

        return TransformedRecord(
            input_ids=format_result["input_ids"],
            mapped_indices=format_result["mapped_indices"],
            schema_tokens_list=[dummy_tokens],
            text_tokens=["."],
            structure_labels=[[1, [[(0, 0)]]]],
            task_types=["entities"],
            start_token_idx=[0],
            end_token_idx=[1],
            text=text or ".",
            schema=schema or {},
        )

    # =========================================================================
    # Internal: Schema Processing
    # =========================================================================

    def _build_classification_prefix(self, schema: Dict[str, Any]) -> List[str]:
        """Build classification prefix tokens."""
        prefix_tokens = []

        for struct in schema.get("json_structures", []):
            for parent, fields in struct.items():
                cls_fields = [
                    (fname, fval) for fname, fval in fields.items()
                    if isinstance(fval, dict) and "value" in fval and "choices" in fval
                ]

                if self.is_training:
                    random.shuffle(cls_fields)

                inner = []
                for fname, fval in cls_fields:
                    choices = fval["choices"].copy()
                    if self.is_training:
                        random.shuffle(choices)

                    choice_tokens = []
                    for i, c in enumerate(choices):
                        if i > 0:
                            choice_tokens.append('|')
                        choice_tokens.append(c)

                    inner.extend([fname, '('] + choice_tokens + [')', ','])

                if inner:
                    inner = inner[:-1]
                    prefix_tokens.extend(['(', f"{parent}:", *inner, ')'])

        return prefix_tokens

    def _wrap_classification_fields(self, schema: Dict, prefix: List[str]):
        """Wrap classification field values with [selection] prefix."""

        def wrap(val):
            if isinstance(val, list):
                return [f"[selection]{v}" for v in val]
            return f"[selection]{val}"

        cls_keys = {
            f"{parent}.{fname}"
            for struct in schema.get("json_structures", [])
            for parent, fields in struct.items()
            for fname, fval in fields.items()
            if isinstance(fval, dict) and {"value", "choices"} <= fval.keys()
        }

        for struct in schema.get("json_structures", []):
            for parent, fields in struct.items():
                for fname in list(fields):
                    key = f"{parent}.{fname}"
                    if key not in cls_keys:
                        continue
                    fval = fields[fname]
                    raw = fval["value"] if isinstance(fval, dict) else fval
                    fields[fname] = wrap(raw)

    def _infer_from_json(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Infer schemas and labels from JSON schema."""
        schemas = []
        labels = []
        types = []

        sampling = self.sampling_config if self.is_training else None

        # Process JSON structures
        self._process_json_structures(schema, schemas, labels, types, sampling)

        # Process entities
        self._process_entities(schema, schemas, labels, types, sampling)

        # Process relations
        self._process_relations(schema, schemas, labels, types, sampling)

        # Process classifications
        self._process_classifications(schema, schemas, labels, types, sampling)

        # Shuffle task order during training
        if sampling:
            order = list(range(len(types)))
            random.shuffle(order)
            schemas = [schemas[i] for i in order]
            labels = [labels[i] for i in order]
            types = [types[i] for i in order]

        return {
            "schemas": schemas,
            "structure_labels": labels,
            "task_types": types,
            "new_schema": schema
        }

    def _process_json_structures(self, schema, schemas, labels, types, sampling):
        """Process JSON structure schemas."""
        if "json_structures" not in schema:
            return

        json_descs = schema.get("json_descriptions", {})
        groups = {}

        for item in schema["json_structures"]:
            for parent, fields in item.items():
                groups.setdefault(parent, []).append(fields)

        for parent, occurrences in groups.items():
            if sampling and random.random() < sampling.remove_json_structure_prob:
                continue

            all_fields = set()
            for occ in occurrences:
                all_fields.update(occ.keys())
            common = list(all_fields)

            if sampling and sampling.shuffle_json_fields:
                random.shuffle(common)

            chosen = [f for f in common if not (
                    sampling and random.random() < sampling.remove_json_field_prob
            )]
            if not chosen:
                continue

            # Handle synthetic labeling
            real2syn = {}
            descs = json_descs.get(parent, {})
            example_modes = ["none", "descriptions"]

            if sampling and random.random() < sampling.synthetic_entity_label_prob:
                example_modes.remove("none")
                synthetic = []
                for i, real in enumerate(chosen, 1):
                    syn = f"field {i}"
                    real2syn[real] = syn
                    synthetic.append(syn)
                descs = {real2syn.get(k, k): descs.get(k, k) for k in chosen}
                chosen = synthetic

            # Build spans
            spans = []
            for occ in occurrences:
                span = [occ.get(f) for f in chosen]
                spans.append(span)

            # Dedup
            uniq = []
            seen = set()
            for s in spans:
                key = tuple(tuple(x) if isinstance(x, list) else x for x in s)
                if key not in seen:
                    uniq.append(s)
                    seen.add(key)

            # Check for empty
            if all(all(c is None or c == "" for c in span) for span in uniq):
                count = 0
                uniq = []
            else:
                count = len(uniq)

            labels.append([count, uniq])

            mode = random.choice(example_modes) if self.is_training else (
                "descriptions" if descs else "none"
            )

            schemas.append(self._transform_schema(
                parent, chosen, self.C_TOKEN, label_descriptions=descs, example_mode=mode
            ))
            types.append("json_structures")

    def _process_entities(self, schema, schemas, labels, types, sampling):
        """Process entity schemas."""
        if "entities" not in schema:
            return

        if sampling and random.random() < sampling.remove_entities_prob:
            return

        entity_fields = list(schema["entities"].keys())
        descs = schema.get("entity_descriptions", {})
        example_modes = ["none", "descriptions"]

        real2syn = {}
        if sampling and random.random() < sampling.synthetic_entity_label_prob:
            example_modes.remove("none")
            synthetic = []
            for i, real in enumerate(entity_fields, 1):
                syn = f"entity {i}"
                real2syn[real] = syn
                synthetic.append(syn)
            descs = {real2syn.get(k, k): v for k, v in descs.items()}
            schema["entities"] = {real2syn.get(k, k): v for k, v in schema["entities"].items()}
            entity_fields = synthetic

        if sampling and sampling.shuffle_entities:
            random.shuffle(entity_fields)

        chosen = [e for e in entity_fields if not (
                sampling and random.random() < sampling.remove_entity_prob
        )]

        if chosen:
            span = [schema["entities"][e] for e in chosen]
            labels.append([1, [span]])

            mode = random.choice(example_modes) if self.is_training else (
                "descriptions" if descs else "none"
            )

            schemas.append(self._transform_schema(
                "entities", chosen, self.E_TOKEN, label_descriptions=descs, example_mode=mode
            ))
            types.append("entities")

    def _process_relations(self, schema, schemas, labels, types, sampling):
        """Process relation schemas."""
        if "relations" not in schema:
            return

        groups = {}
        for item in schema["relations"]:
            if sampling and random.random() < sampling.remove_relations_prob:
                continue
            for parent, fields in item.items():
                groups.setdefault(parent, []).append(fields)

        for parent, occurrences in groups.items():
            field_names = list(occurrences[0].keys())

            if sampling and "head" in field_names and "tail" in field_names:
                if random.random() < sampling.swap_head_tail_prob:
                    idx_h = field_names.index("head")
                    idx_t = field_names.index("tail")
                    field_names[idx_h], field_names[idx_t] = field_names[idx_t], field_names[idx_h]

            spans = []
            for occ in occurrences:
                if all(f in occ for f in field_names):
                    spans.append([occ[f] for f in field_names])

            if not spans:
                continue

            # Dedup
            seen = set()
            uniq = []
            for span in spans:
                t = tuple(tuple(s) if isinstance(s, list) else s for s in span)
                if t not in seen:
                    seen.add(t)
                    uniq.append(span)

            labels.append([len(uniq), uniq])
            schemas.append(self._transform_schema(parent, field_names, self.R_TOKEN))
            types.append("relations")

    def _process_classifications(self, schema, schemas, labels, types, sampling):
        """Process classification schemas."""
        if "classifications" not in schema:
            return

        for idx, item in enumerate(schema["classifications"]):
            if sampling and random.random() < sampling.remove_classification_prob:
                continue

            cls_labels = item["labels"].copy()
            examples = item.get("examples", [])
            descs = item.get("label_descriptions", {}) or {}

            real2syn = {}
            example_modes = ["few_shot", "descriptions", "both", "none"] if self.is_training else ["both"]

            if sampling and random.random() < sampling.synthetic_label_prob:
                example_modes = [m for m in example_modes if m != "none"]
                synthetic = []
                for i, real in enumerate(cls_labels, 1):
                    syn = f"label {i}"
                    real2syn[real] = syn
                    synthetic.append(syn)
                cls_labels = synthetic
                descs = {real2syn.get(k, k): descs.get(k, k) for k in item["labels"]}
                examples = [(inp, real2syn.get(out, out)) for inp, out in examples]

            mode = random.choice(example_modes) if example_modes else "none"

            # Label dropping
            if sampling and hasattr(sampling, "remove_classification_label_prob"):
                drop_frac = random.betavariate(1, 1) * sampling.remove_classification_label_prob
                num_remove = int(len(cls_labels) * drop_frac)
                if num_remove > 0:
                    cls_labels = random.sample(cls_labels, len(cls_labels) - num_remove)

                max_labels = sampling.max_num_labels // 2 if mode in ["few_shot", "both",
                                                                      "descriptions"] else sampling.max_num_labels
                if len(cls_labels) > max_labels:
                    cls_labels = cls_labels[:max_labels]

                if random.random() < sampling.include_true_label_prob:
                    true_label = item.get("true_label", [])
                    if isinstance(true_label, list):
                        for tl in true_label:
                            if tl not in cls_labels:
                                cls_labels.append(tl)
                    elif true_label not in cls_labels:
                        cls_labels.append(true_label)

            if sampling and sampling.shuffle_classification_labels:
                random.shuffle(cls_labels)

            schemas.append(self._transform_schema(
                item["task"], cls_labels, self.L_TOKEN,
                prompt=item.get("prompt"), examples=examples,
                label_descriptions=descs, example_mode=mode
            ))
            types.append("classifications")

            # Update schema
            schema["classifications"][idx]["labels"] = cls_labels
            true_label = schema["classifications"][idx]["true_label"].copy()
            schema["classifications"][idx]["true_label"] = [real2syn.get(i, i) for i in true_label]
            labels.append([])

    def _transform_schema(
            self,
            parent: str,
            fields: List[str],
            child_prefix: str,
            prompt: str = None,
            examples: List[Tuple[str, str]] = None,
            label_descriptions: Dict[str, str] = None,
            example_mode: str = "both"
    ) -> List[str]:
        """Transform schema into token sequence."""
        prompt_str = parent
        if prompt:
            prompt_str = f"{parent}: {prompt}"

        if example_mode in ["descriptions", "both"] and label_descriptions:
            descs = [(l, d) for l, d in label_descriptions.items() if l in fields]
            if self.is_training:
                random.shuffle(descs)
            for label, desc in descs:
                prompt_str += f" {self.DESC_TOKEN} {label}: {desc}"

        if example_mode in ["few_shot", "both"] and examples:
            if self.is_training:
                random.shuffle(examples)
            for inp, out in examples:
                if out in fields:
                    out_str = out if isinstance(out, str) else ', '.join(out)
                    prompt_str += f" {self.EXAMPLE_TOKEN} {inp} {self.OUTPUT_TOKEN} {out_str}"

        tokens = ["(", self.P_TOKEN, prompt_str, "("]
        for field in fields:
            tokens.extend([child_prefix, field])
        tokens.extend([")", ")"])

        return tokens

    def _build_outputs(
            self,
            processed: Dict,
            schema: Dict,
            text_tokens: List[str],
            len_prefix: int
    ) -> List[Dict]:
        """Build output labels for each schema."""
        results = []

        for schema_tokens, task_type, struct_label in zip(
                processed["schemas"],
                processed["task_types"],
                processed["structure_labels"]
        ):
            if task_type != "classifications":
                count, spans = struct_label
                transformed = []

                for span in spans:
                    positions = []
                    for field in span:
                        if isinstance(field, list):
                            nested = []
                            for sub in field:
                                if str(sub).startswith("[selection]"):
                                    pos = self._find_sublist([str(sub)[11:]], text_tokens[:len_prefix])
                                else:
                                    pos = self._find_sublist(
                                        self._tokenize_text(str(sub)), text_tokens
                                    )
                                nested.extend(pos)
                            positions.append(nested)
                        else:
                            if str(field).startswith("[selection]"):
                                pos = self._find_sublist([str(field)[11:]], text_tokens[:len_prefix])
                            else:
                                pos = self._find_sublist(
                                    self._tokenize_text(str(field)), text_tokens
                                )
                            positions.append(pos)
                    transformed.append(positions)

                results.append({
                    "task_type": task_type,
                    "schema_tokens": schema_tokens,
                    "output": [count, transformed]
                })
            else:
                cls_item = next(
                    (c for c in schema["classifications"] if schema_tokens[2].startswith(c["task"])),
                    None
                )
                if cls_item is None:
                    raise ValueError(f"Missing classification for: {schema_tokens[2]}")

                bool_labels = [1 if l in cls_item["true_label"] else 0 for l in cls_item["labels"]]
                results.append({
                    "task_type": task_type,
                    "schema_tokens": schema_tokens,
                    "output": bool_labels
                })

        return results

    def _find_sublist(self, sub: List[str], lst: List[str]) -> List[Tuple[int, int]]:
        """Find all occurrences of sublist in list."""
        if not sub or all(t == "" for t in sub):
            return [(-1, -1)]

        sub_len = len(sub)
        matches = [
            (i, i + sub_len - 1)
            for i in range(len(lst) - sub_len + 1)
            if lst[i:i + sub_len] == sub
        ]
        return matches or [(-1, -1)]

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return [tok for tok, _, _ in self.word_splitter(text, lower=True)]

    # =========================================================================
    # Internal: Input Formatting
    # =========================================================================

    def _format_input_with_mapping(
            self,
            schema_tokens_list: List[List[str]],
            text_tokens: List[str]
    ) -> Dict[str, Any]:
        """Format input and create token mappings."""
        # Build combined tokens
        combined = []
        for struct in schema_tokens_list:
            combined.extend(struct)
            combined.append(self.SEP_STRUCT)
        if combined:
            combined.pop()
        combined.append(self.SEP_TEXT)
        combined.extend(text_tokens)

        # Build subword list and mappings
        subwords = []
        mappings = []

        num_schemas = len(schema_tokens_list)
        text_schema_idx = num_schemas
        current_schema = 0
        found_sep = False

        for orig_idx, token in enumerate(combined):
            if token == self.SEP_TEXT:
                seg_type = "sep"
                schema_idx = text_schema_idx
                found_sep = True
            elif not found_sep:
                seg_type = "schema"
                schema_idx = current_schema
                if token == self.SEP_STRUCT:
                    current_schema += 1
            else:
                seg_type = "text"
                schema_idx = text_schema_idx

            sub_tokens = self.tokenizer.tokenize(token)
            subwords.extend(sub_tokens)
            mappings.extend([(seg_type, orig_idx, schema_idx)] * len(sub_tokens))

        input_ids = self.tokenizer.convert_tokens_to_ids(subwords)

        return {
            "input_ids": input_ids,
            "mapped_indices": mappings,
            "subword_list": subwords
        }

    # =========================================================================
    # Embedding Extraction (Called by Model)
    # =========================================================================

    def extract_embeddings_from_batch(
            self,
            token_embeddings: torch.Tensor,
            input_ids: torch.Tensor,
            batch: PreprocessedBatch
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Extract token and schema embeddings from encoded batch.

        Args:
            token_embeddings: (batch, seq_len, hidden) from encoder
            input_ids: (batch, seq_len) input token IDs
            batch: PreprocessedBatch with metadata

        Returns:
            - all_token_embs: List of (text_len, hidden) per sample
            - all_schema_embs: List of schema embeddings per sample
        """
        all_token_embs = []
        all_schema_embs = []

        special_set = {self.P_TOKEN, self.C_TOKEN, self.E_TOKEN, self.R_TOKEN, self.L_TOKEN}

        for i in range(len(batch)):
            seq_len = batch.original_lengths[i]
            embs = token_embeddings[i, :seq_len, :]
            ids = input_ids[i, :seq_len].tolist()
            mappings = batch.mapped_indices[i][:seq_len]
            num_schemas = batch.schema_counts[i]

            schema_embs = [[] for _ in range(num_schemas)]
            word_embs = []
            bucket = []
            last_orig = None

            for j, tid in enumerate(ids):
                seg_type, orig_idx, schema_idx = mappings[j]
                emb = embs[j]

                if seg_type == "schema":
                    tok = self.tokenizer.convert_ids_to_tokens(tid)
                    if tok in special_set:
                        schema_embs[schema_idx].append(emb)

                elif seg_type == "text":
                    if last_orig is not None and orig_idx != last_orig and bucket:
                        word_embs.append(self._aggregate(bucket))
                        bucket = []
                    bucket.append(emb)
                    last_orig = orig_idx

            if bucket:
                word_embs.append(self._aggregate(bucket))

            all_token_embs.append(
                torch.stack(word_embs) if word_embs else torch.empty(0, embs.shape[-1], device=embs.device)
            )
            all_schema_embs.append(schema_embs)

        return all_token_embs, all_schema_embs

    def _aggregate(self, pieces: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate subword embeddings."""
        if self.token_pooling == "first":
            return pieces[0]
        stack = torch.stack(pieces)
        if self.token_pooling == "mean":
            return stack.mean(dim=0)
        if self.token_pooling == "max":
            return stack.max(dim=0).values
        return pieces[0]