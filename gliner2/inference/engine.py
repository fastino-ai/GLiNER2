"""
GLiNER2 - Advanced Information Extraction Engine

This module provides the main GLiNER2 class with optimized batch processing
using DataLoader-based parallel preprocessing.

Example:
    >>> from gliner2 import GLiNER2
    >>>
    >>> extractor = GLiNER2.from_pretrained("model-repo")
    >>>
    >>> # Simple extraction
    >>> results = extractor.extract_entities(
    ...     "Apple released iPhone 15.",
    ...     ["company", "product"]
    ... )
    >>>
    >>> # Batch extraction (parallel preprocessing)
    >>> results = extractor.batch_extract_entities(
    ...     texts_list,
    ...     ["company", "product"],
    ...     batch_size=32,
    ...     num_workers=4
    ... )
"""

from __future__ import annotations

import re
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING, Pattern, Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gliner2.model import Extractor
from gliner2.processor import PreprocessedBatch

if TYPE_CHECKING:
    from gliner2.api_client import GLiNER2API


# =============================================================================
# Validators
# =============================================================================

@dataclass
class RegexValidator:
    """Regex-based span filter for post-processing."""
    pattern: str | Pattern[str]
    mode: Literal["full", "partial"] = "full"
    exclude: bool = False
    flags: int = re.IGNORECASE
    _compiled: Pattern[str] = field(init=False, repr=False)

    def __post_init__(self):
        if self.mode not in {"full", "partial"}:
            raise ValueError(f"mode must be 'full' or 'partial', got {self.mode!r}")
        try:
            compiled = (
                self.pattern if isinstance(self.pattern, re.Pattern)
                else re.compile(self.pattern, self.flags)
            )
        except re.error as err:
            raise ValueError(f"Invalid regex: {self.pattern!r}") from err
        object.__setattr__(self, "_compiled", compiled)

    def __call__(self, text: str) -> bool:
        return self.validate(text)

    def validate(self, text: str) -> bool:
        matcher = self._compiled.fullmatch if self.mode == "full" else self._compiled.search
        matched = matcher(text) is not None
        return not matched if self.exclude else matched


# =============================================================================
# Schema Builder
# =============================================================================

class StructureBuilder:
    """Builder for structured data schemas."""

    def __init__(self, schema: 'Schema', parent: str):
        self.schema = schema
        self.parent = parent
        self.fields = OrderedDict()
        self.descriptions = OrderedDict()
        self.field_order = []
        self._finished = False

    def field(
        self,
        name: str,
        dtype: Literal["str", "list"] = "list",
        choices: Optional[List[str]] = None,
        description: Optional[str] = None,
        threshold: Optional[float] = None,
        validators: Optional[List[RegexValidator]] = None
    ) -> 'StructureBuilder':
        """Add a field to the structure."""
        self.fields[name] = {"value": "", "choices": choices} if choices else ""
        self.field_order.append(name)

        if description:
            self.descriptions[name] = description

        self.schema._store_field_metadata(self.parent, name, dtype, threshold, choices, validators)
        return self

    def _auto_finish(self):
        if not self._finished:
            self.schema._store_field_order(self.parent, self.field_order)
            self.schema.schema["json_structures"].append({self.parent: self.fields})

            if self.descriptions:
                if "json_descriptions" not in self.schema.schema:
                    self.schema.schema["json_descriptions"] = {}
                self.schema.schema["json_descriptions"][self.parent] = self.descriptions

            self._finished = True

    def __getattr__(self, name):
        if hasattr(self.schema, name):
            self._auto_finish()
            return getattr(self.schema, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class Schema:
    """Schema builder for extraction tasks."""

    def __init__(self):
        self.schema = {
            "json_structures": [],
            "classifications": [],
            "entities": OrderedDict(),
            "relations": [],
            "json_descriptions": {},
            "entity_descriptions": OrderedDict()
        }
        self._field_metadata = {}
        self._entity_metadata = {}
        self._relation_metadata = {}
        self._field_orders = {}
        self._entity_order = []
        self._relation_order = []
        self._active_builder = None

    def _store_field_metadata(self, parent, field, dtype, threshold, choices, validators=None):
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be 0-1, got {threshold}")
        self._field_metadata[f"{parent}.{field}"] = {
            "dtype": dtype, "threshold": threshold, "choices": choices,
            "validators": validators or []
        }

    def _store_entity_metadata(self, entity, dtype, threshold):
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be 0-1, got {threshold}")
        self._entity_metadata[entity] = {"dtype": dtype, "threshold": threshold}

    def _store_field_order(self, parent, order):
        self._field_orders[parent] = order

    def structure(self, name: str) -> StructureBuilder:
        """Start building a structure schema."""
        if self._active_builder:
            self._active_builder._auto_finish()
        self._active_builder = StructureBuilder(self, name)
        return self._active_builder

    def classification(
        self,
        task: str,
        labels: Union[List[str], Dict[str, str]],
        multi_label: bool = False,
        cls_threshold: float = 0.5,
        **kwargs
    ) -> 'Schema':
        """Add classification task."""
        if self._active_builder:
            self._active_builder._auto_finish()
            self._active_builder = None

        label_names = list(labels.keys()) if isinstance(labels, dict) else labels
        label_descs = labels if isinstance(labels, dict) else None

        config = {
            "task": task, "labels": label_names,
            "multi_label": multi_label, "cls_threshold": cls_threshold,
            "true_label": ["N/A"], **kwargs
        }
        if label_descs:
            config["label_descriptions"] = label_descs

        self.schema["classifications"].append(config)
        return self

    def entities(
        self,
        entity_types: Union[str, List[str], Dict[str, Union[str, Dict]]],
        dtype: Literal["str", "list"] = "list",
        threshold: Optional[float] = None
    ) -> 'Schema':
        """Add entity extraction task."""
        if self._active_builder:
            self._active_builder._auto_finish()
            self._active_builder = None

        entities = self._parse_entity_input(entity_types)

        for name, config in entities.items():
            self.schema["entities"][name] = ""
            if name not in self._entity_order:
                self._entity_order.append(name)

            self._store_entity_metadata(
                name,
                config.get("dtype", dtype),
                config.get("threshold", threshold)
            )

            if "description" in config:
                self.schema["entity_descriptions"][name] = config["description"]

        return self

    def _parse_entity_input(self, entity_types):
        if isinstance(entity_types, str):
            return {entity_types: {}}
        elif isinstance(entity_types, list):
            return {name: {} for name in entity_types}
        elif isinstance(entity_types, dict):
            result = {}
            for name, config in entity_types.items():
                if isinstance(config, str):
                    result[name] = {"description": config}
                elif isinstance(config, dict):
                    result[name] = config
                else:
                    result[name] = {}
            return result
        raise ValueError("Invalid entity_types format")

    def relations(
        self,
        relation_types: Union[str, List[str], Dict[str, Union[str, Dict]]],
        threshold: Optional[float] = None
    ) -> 'Schema':
        """Add relation extraction task."""
        if self._active_builder:
            self._active_builder._auto_finish()
            self._active_builder = None

        if isinstance(relation_types, str):
            relations = {relation_types: {}}
        elif isinstance(relation_types, list):
            relations = {name: {} for name in relation_types}
        elif isinstance(relation_types, dict):
            relations = {}
            for name, config in relation_types.items():
                relations[name] = {"description": config} if isinstance(config, str) else (config if isinstance(config, dict) else {})
        else:
            raise ValueError("Invalid relation_types format")

        for name, config in relations.items():
            self.schema["relations"].append({name: {"head": "", "tail": ""}})
            if name not in self._relation_order:
                self._relation_order.append(name)
            self._field_orders[name] = ["head", "tail"]

            rel_threshold = config.get("threshold", threshold)
            if rel_threshold is not None and not 0 <= rel_threshold <= 1:
                raise ValueError(f"Threshold must be 0-1, got {rel_threshold}")
            self._relation_metadata[name] = {"threshold": rel_threshold}

        return self

    def build(self) -> Dict[str, Any]:
        """Build final schema dictionary."""
        if self._active_builder:
            self._active_builder._auto_finish()
            self._active_builder = None
        return self.schema


# =============================================================================
# Main GLiNER2 Class
# =============================================================================

class GLiNER2(Extractor):
    """
    GLiNER2 Information Extraction Model.

    Provides efficient batch extraction with parallel preprocessing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._schema_cache = {}

    @classmethod
    def from_api(cls, api_key: str = None, api_base_url: str = None,
                 timeout: float = 30.0, max_retries: int = 3) -> 'GLiNER2API':
        """Load from API instead of local model."""
        from gliner2.api_client import GLiNER2API
        return GLiNER2API(api_key=api_key, api_base_url=api_base_url,
                         timeout=timeout, max_retries=max_retries)

    def create_schema(self) -> Schema:
        """Create a new schema builder."""
        return Schema()

    # =========================================================================
    # Main Batch Extraction
    # =========================================================================

    @torch.no_grad()
    def batch_extract(
        self,
        texts: List[str],
        schemas: Union[Schema, List[Schema], Dict, List[Dict]],
        batch_size: int = 8,
        threshold: float = 0.5,
        num_workers: int = 0,
        format_results: bool = True,
        include_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extract from multiple texts with parallel preprocessing.

        Args:
            texts: List of input texts
            schemas: Single schema or list of schemas
            batch_size: Batch size for processing
            threshold: Confidence threshold
            num_workers: Workers for parallel preprocessing
            format_results: Format output nicely
            include_confidence: Include confidence scores

        Returns:
            List of extraction results
        """
        if not texts:
            return []

        self.eval()
        self.processor.change_mode(is_training=False)

        # Normalize schemas
        if isinstance(schemas, list):
            if len(schemas) != len(texts):
                raise ValueError(f"Schema count ({len(schemas)}) != text count ({len(texts)})")
            schema_list = schemas
        else:
            schema_list = [schemas] * len(texts)

        # Build schema dicts and metadata
        schema_dicts = []
        metadata_list = []

        for schema in schema_list:
            if hasattr(schema, 'build'):
                schema_dict = schema.build()
                metadata = {
                    "field_metadata": schema._field_metadata,
                    "entity_metadata": schema._entity_metadata,
                    "relation_metadata": getattr(schema, '_relation_metadata', {}),
                    "field_orders": schema._field_orders,
                    "entity_order": schema._entity_order,
                    "relation_order": getattr(schema, '_relation_order', [])
                }
            else:
                schema_dict = schema
                metadata = {
                    "field_metadata": {}, "entity_metadata": {},
                    "relation_metadata": {}, "field_orders": {},
                    "entity_order": [], "relation_order": []
                }

            # Ensure classifications have true_label
            for cls_config in schema_dict.get("classifications", []):
                cls_config.setdefault("true_label", ["N/A"])

            schema_dicts.append(schema_dict)
            metadata_list.append(metadata)

        # Normalize texts
        normalized = []
        for text in texts:
            if not text:
                text = "."
            elif not text.endswith(('.', '!', '?')):
                text = text + "."
            normalized.append(text)

        # Create dataset and loader
        dataset = list(zip(normalized, schema_dicts))

        from gliner2.trainer import ExtractorDataCollator
        collator = ExtractorDataCollator(self.processor, is_training=False)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Process batches
        all_results = []
        sample_idx = 0
        device = next(self.parameters()).device

        for batch in loader:
            batch = batch.to(device)
            batch_results = self._extract_from_batch(
                batch, threshold, metadata_list[sample_idx:sample_idx + len(batch)],
                include_confidence
            )

            if format_results:
                for i, result in enumerate(batch_results):
                    meta = metadata_list[sample_idx + i]
                    requested_relations = meta.get("relation_order", [])
                    batch_results[i] = self.format_results(
                        result, include_confidence, requested_relations
                    )

            all_results.extend(batch_results)
            sample_idx += len(batch)

        return all_results

    def _extract_from_batch(
        self,
        batch: PreprocessedBatch,
        threshold: float,
        metadata_list: List[Dict],
        include_confidence: bool
    ) -> List[Dict[str, Any]]:
        """Extract from preprocessed batch."""
        # Encode batch
        all_token_embs, all_schema_embs = self.processor.extract_embeddings_from_batch(
            self.encoder(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask
            ).last_hidden_state,
            batch.input_ids,
            batch
        )

        results = []

        for i in range(len(batch)):
            try:
                sample_result = self._extract_sample(
                    token_embs=all_token_embs[i],
                    schema_embs=all_schema_embs[i],
                    schema_tokens_list=batch.schema_tokens_list[i],
                    task_types=batch.task_types[i],
                    text_tokens=batch.text_tokens[i],
                    original_text=batch.original_texts[i],
                    schema=batch.original_schemas[i],
                    start_mapping=batch.start_mappings[i],
                    end_mapping=batch.end_mappings[i],
                    threshold=threshold,
                    metadata=metadata_list[i],
                    include_confidence=include_confidence
                )
                results.append(sample_result)
            except Exception as e:
                print(f"Error extracting sample {i}: {e}")
                results.append({})

        return results

    def _extract_sample(
        self,
        token_embs: torch.Tensor,
        schema_embs: List[List[torch.Tensor]],
        schema_tokens_list: List[List[str]],
        task_types: List[str],
        text_tokens: List[str],
        original_text: str,
        schema: Dict,
        start_mapping: List[int],
        end_mapping: List[int],
        threshold: float,
        metadata: Dict,
        include_confidence: bool
    ) -> Dict[str, Any]:
        """Extract from single sample."""
        results = {}

        # Compute span representations if needed
        has_span_task = any(t != "classifications" for t in task_types)
        span_info = None
        if has_span_task and token_embs.numel() > 0:
            span_info = self.compute_span_rep(token_embs)

        # Build classification field map
        cls_fields = {}
        for struct in schema.get("json_structures", []):
            for parent, fields in struct.items():
                for fname, fval in fields.items():
                    if isinstance(fval, dict) and "choices" in fval:
                        cls_fields[f"{parent}.{fname}"] = fval["choices"]

        text_len = len(self.processor._tokenize_text(original_text))

        for i, (schema_tokens, task_type) in enumerate(zip(schema_tokens_list, task_types)):
            if len(schema_tokens) < 4 or not schema_embs[i]:
                continue

            schema_name = schema_tokens[2].split(" [DESCRIPTION] ")[0]
            embs = torch.stack(schema_embs[i])

            if task_type == "classifications":
                self._extract_classification_result(
                    results, schema_name, schema, embs, schema_tokens
                )
            else:
                self._extract_span_result(
                    results, schema_name, task_type, embs, span_info,
                    schema_tokens, text_tokens, text_len, original_text,
                    start_mapping, end_mapping, threshold, metadata,
                    cls_fields, include_confidence
                )

        return results

    def _extract_classification_result(
        self,
        results: Dict,
        schema_name: str,
        schema: Dict,
        embs: torch.Tensor,
        schema_tokens: List[str]
    ):
        """Extract classification result."""
        cls_config = next(
            c for c in schema["classifications"]
            if schema_tokens[2].startswith(c["task"])
        )

        cls_embeds = embs[1:]
        logits = self.classifier(cls_embeds).squeeze(-1)

        activation = cls_config.get("class_act", "auto")
        is_multi = cls_config.get("multi_label", False)

        if activation == "sigmoid":
            probs = torch.sigmoid(logits)
        elif activation == "softmax":
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = torch.sigmoid(logits) if is_multi else torch.softmax(logits, dim=-1)

        labels = cls_config["labels"]
        cls_threshold = cls_config.get("cls_threshold", 0.5)

        if is_multi:
            chosen = [(labels[j], probs[j].item()) for j in range(len(labels)) if probs[j].item() >= cls_threshold]
            if not chosen:
                best = int(torch.argmax(probs).item())
                chosen = [(labels[best], probs[best].item())]
            results[schema_name] = chosen
        else:
            best = int(torch.argmax(probs).item())
            results[schema_name] = (labels[best], probs[best].item())

    def _extract_span_result(
        self,
        results: Dict,
        schema_name: str,
        task_type: str,
        embs: torch.Tensor,
        span_info: Dict,
        schema_tokens: List[str],
        text_tokens: List[str],
        text_len: int,
        original_text: str,
        start_mapping: List[int],
        end_mapping: List[int],
        threshold: float,
        metadata: Dict,
        cls_fields: Dict,
        include_confidence: bool
    ):
        """Extract span-based results."""
        # Get field names
        field_names = []
        for j in range(len(schema_tokens) - 1):
            if schema_tokens[j] in ("[E]", "[C]", "[R]"):
                field_names.append(schema_tokens[j + 1])

        if not field_names:
            results[schema_name] = [] if schema_name == "entities" else {}
            return

        # Predict count
        count_logits = self.count_pred(embs[0].unsqueeze(0))
        pred_count = int(count_logits.argmax(dim=1).item())

        if pred_count <= 0 or span_info is None:
            if schema_name == "entities":
                results[schema_name] = []
            elif task_type == "relations":
                results[schema_name] = []
            else:
                results[schema_name] = {}
            return

        # Get span scores
        struct_proj = self.count_embed(embs[1:], pred_count)
        span_scores = torch.sigmoid(
            torch.einsum("lkd,bpd->bplk", span_info["span_rep"], struct_proj)
        )

        # Extract based on type
        if schema_name == "entities":
            results[schema_name] = self._extract_entities(
                field_names, span_scores, text_len, text_tokens,
                original_text, start_mapping, end_mapping,
                threshold, metadata, include_confidence
            )
        elif task_type == "relations":
            results[schema_name] = self._extract_relations(
                schema_name, field_names, span_scores, pred_count,
                text_len, text_tokens, original_text, start_mapping, end_mapping,
                threshold, metadata
            )
        else:
            results[schema_name] = self._extract_structures(
                schema_name, field_names, span_scores, pred_count,
                text_len, text_tokens, original_text, start_mapping, end_mapping,
                threshold, metadata, cls_fields, include_confidence
            )

    def _extract_entities(
        self,
        entity_names: List[str],
        span_scores: torch.Tensor,
        text_len: int,
        text_tokens: List[str],
        text: str,
        start_map: List[int],
        end_map: List[int],
        threshold: float,
        metadata: Dict,
        include_confidence: bool
    ) -> List[Dict]:
        """Extract entity results."""
        scores = span_scores[0, :, -text_len:]
        entity_results = OrderedDict()

        for name in metadata.get("entity_order", entity_names):
            if name not in entity_names:
                continue

            idx = entity_names.index(name)
            meta = metadata.get("entity_metadata", {}).get(name, {})
            ent_threshold = meta.get("threshold") or threshold
            dtype = meta.get("dtype", "list")

            spans = self._find_spans(
                scores[idx], ent_threshold, text_len, text,
                start_map, end_map
            )

            if dtype == "list":
                entity_results[name] = self._format_spans(spans, include_confidence)
            else:
                if include_confidence and spans:
                    entity_results[name] = spans[0]
                else:
                    entity_results[name] = spans[0][0] if spans else ""

        return [entity_results] if entity_results else []

    def _extract_relations(
        self,
        rel_name: str,
        field_names: List[str],
        span_scores: torch.Tensor,
        count: int,
        text_len: int,
        text_tokens: List[str],
        text: str,
        start_map: List[int],
        end_map: List[int],
        threshold: float,
        metadata: Dict
    ) -> List[Tuple[str, str]]:
        """Extract relation results."""
        instances = []

        rel_threshold = threshold
        if rel_name in metadata.get("relation_metadata", {}):
            rel_threshold = metadata["relation_metadata"][rel_name].get("threshold") or threshold

        ordered_fields = metadata.get("field_orders", {}).get(rel_name, field_names)

        for inst in range(count):
            scores = span_scores[inst, :, -text_len:]
            values = []

            for fname in ordered_fields:
                if fname not in field_names:
                    continue
                fidx = field_names.index(fname)
                spans = self._find_spans(
                    scores[fidx], rel_threshold, text_len, text,
                    start_map, end_map
                )
                values.append(spans[0][0] if spans else None)

            if len(values) == 2 and values[0] and values[1]:
                instances.append((values[0], values[1]))

        return instances

    def _extract_structures(
        self,
        struct_name: str,
        field_names: List[str],
        span_scores: torch.Tensor,
        count: int,
        text_len: int,
        text_tokens: List[str],
        text: str,
        start_map: List[int],
        end_map: List[int],
        threshold: float,
        metadata: Dict,
        cls_fields: Dict,
        include_confidence: bool
    ) -> List[Dict]:
        """Extract structure results."""
        instances = []
        ordered_fields = metadata.get("field_orders", {}).get(struct_name, field_names)

        for inst in range(count):
            scores = span_scores[inst, :, -text_len:]
            instance = OrderedDict()

            for fname in ordered_fields:
                if fname not in field_names:
                    continue

                fidx = field_names.index(fname)
                field_key = f"{struct_name}.{fname}"
                meta = metadata.get("field_metadata", {}).get(field_key, {})
                field_threshold = meta.get("threshold") or threshold
                dtype = meta.get("dtype", "list")
                validators = meta.get("validators", [])

                if field_key in cls_fields:
                    # Classification field
                    choices = cls_fields[field_key]
                    prefix_scores = span_scores[inst, fidx, :-text_len]

                    if dtype == "list":
                        selected = []
                        seen = set()
                        for choice in choices:
                            if choice in seen:
                                continue
                            idx = self._find_choice_idx(choice, text_tokens[:-text_len])
                            if idx >= 0 and idx < prefix_scores.shape[0]:
                                if prefix_scores[idx, 0].item() >= field_threshold:
                                    selected.append(choice)
                                    seen.add(choice)
                        instance[fname] = selected
                    else:
                        best = None
                        best_score = -1.0
                        for choice in choices:
                            idx = self._find_choice_idx(choice, text_tokens[:-text_len])
                            if idx >= 0 and idx < prefix_scores.shape[0]:
                                score = prefix_scores[idx, 0].item()
                                if score > best_score:
                                    best_score = score
                                    best = choice
                        instance[fname] = best if best and best_score >= field_threshold else None
                else:
                    # Regular span field
                    spans = self._find_spans(
                        scores[fidx], field_threshold, text_len, text,
                        start_map, end_map
                    )

                    if validators:
                        spans = [s for s in spans if all(v.validate(s[0]) for v in validators)]

                    if dtype == "list":
                        instance[fname] = self._format_spans(spans, include_confidence)
                    else:
                        if include_confidence and spans:
                            instance[fname] = spans[0]
                        else:
                            instance[fname] = spans[0][0] if spans else None

            # Only add if has content
            if any(v is not None and v != [] for v in instance.values()):
                instances.append(instance)

        return instances

    def _find_spans(
        self,
        scores: torch.Tensor,
        threshold: float,
        text_len: int,
        text: str,
        start_map: List[int],
        end_map: List[int]
    ) -> List[Tuple[str, float, int, int]]:
        """Find valid spans above threshold."""
        valid = torch.where(scores >= threshold)
        starts, widths = valid

        spans = []
        for start, width in zip(starts.tolist(), widths.tolist()):
            end = start + width + 1
            if 0 <= start < text_len and end <= text_len:
                try:
                    char_start = start_map[start]
                    char_end = end_map[end - 1]
                    text_span = text[char_start:char_end].strip()
                except (IndexError, KeyError):
                    continue

                if text_span:
                    conf = scores[start, width].item()
                    spans.append((text_span, conf, start, end))

        return spans

    def _format_spans(
        self,
        spans: List[Tuple],
        include_confidence: bool
    ) -> Union[List[str], List[Tuple]]:
        """Format spans with overlap removal."""
        if not spans:
            return []

        sorted_spans = sorted(spans, key=lambda x: x[1], reverse=True)
        selected = []

        for text, conf, start, end in sorted_spans:
            overlap = any(not (end <= s[2] or start >= s[3]) for s in selected)
            if not overlap:
                selected.append((text, conf, start, end))

        if include_confidence:
            return selected
        return [s[0] for s in selected]

    def _find_choice_idx(self, choice: str, tokens: List[str]) -> int:
        """Find index of choice in tokens."""
        choice_lower = choice.lower()
        for i, tok in enumerate(tokens):
            if tok.lower() == choice_lower or choice_lower in tok.lower():
                return i
        return -1

    # =========================================================================
    # Result Formatting
    # =========================================================================

    def format_results(
        self,
        results: Dict,
        include_confidence: bool = False,
        requested_relations: List[str] = None
    ) -> Dict[str, Any]:
        """Format extraction results."""
        formatted = {}
        relations = {}

        for key, value in results.items():
            if isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], tuple) and len(value[0]) == 2:
                    relations[key] = value
                elif len(value) == 0:
                    if requested_relations and key in requested_relations:
                        relations[key] = []
                    elif key == "entities":
                        formatted[key] = {}
                    else:
                        formatted[key] = value
                elif len(value) > 0 and isinstance(value[0], dict):
                    if key == "entities":
                        formatted[key] = self._format_entity_dict(value[0], include_confidence)
                    else:
                        formatted[key] = [self._format_struct(v, include_confidence) for v in value]
                elif isinstance(value[0], tuple):
                    if include_confidence:
                        formatted[key] = [{"label": l, "confidence": c} for l, c in value]
                    else:
                        formatted[key] = [l for l, _ in value]
                else:
                    formatted[key] = value
            elif isinstance(value, tuple):
                label, conf = value
                formatted[key] = {"label": label, "confidence": conf} if include_confidence else label
            elif isinstance(value, dict):
                formatted[key] = self._format_struct(value, include_confidence)
            else:
                formatted[key] = value

        # Add relations
        if requested_relations:
            for rel in requested_relations:
                if rel not in relations:
                    relations[rel] = []

        if relations:
            formatted["relation_extraction"] = relations

        return formatted

    def _format_entity_dict(self, entities: Dict, include_confidence: bool) -> Dict:
        formatted = {}
        for name, spans in entities.items():
            if isinstance(spans, list):
                unique = []
                seen = set()
                for span in spans:
                    if isinstance(span, tuple):
                        text, conf, start, end = span
                        if text and text.lower() not in seen:
                            seen.add(text.lower())
                            unique.append({"text": text, "confidence": conf} if include_confidence else text)
                    else:
                        if span and span.lower() not in seen:
                            seen.add(span.lower())
                            unique.append(span)
                formatted[name] = unique
            elif isinstance(spans, tuple):
                text, conf, _, _ = spans
                formatted[name] = {"text": text, "confidence": conf} if include_confidence and text else text
            else:
                formatted[name] = spans or None
        return formatted

    def _format_struct(self, struct: Dict, include_confidence: bool) -> Dict:
        formatted = {}
        for field, value in struct.items():
            if isinstance(value, list):
                unique = []
                seen = set()
                for v in value:
                    if isinstance(v, tuple):
                        text, conf, _, _ = v
                        if text and text.lower() not in seen:
                            seen.add(text.lower())
                            unique.append({"text": text, "confidence": conf} if include_confidence else text)
                    else:
                        if v and v.lower() not in seen:
                            seen.add(v.lower())
                            unique.append(v)
                formatted[field] = unique
            elif isinstance(value, tuple):
                text, conf, _, _ = value
                formatted[field] = {"text": text, "confidence": conf} if include_confidence and text else text
            elif value:
                formatted[field] = value
            else:
                formatted[field] = None
        return formatted

    # =========================================================================
    # Convenience Methods (route through batch)
    # =========================================================================

    def extract(self, text: str, schema, threshold: float = 0.5,
                format_results: bool = True, include_confidence: bool = False) -> Dict:
        """Extract from single text."""
        return self.batch_extract([text], schema, 1, threshold, 0, format_results, include_confidence)[0]

    def extract_entities(self, text: str, entity_types, threshold: float = 0.5,
                        format_results: bool = True, include_confidence: bool = False) -> Dict:
        """Extract entities from text."""
        schema = self.create_schema().entities(entity_types)
        return self.extract(text, schema, threshold, format_results, include_confidence)

    def batch_extract_entities(self, texts: List[str], entity_types, batch_size: int = 8,
                               threshold: float = 0.5, format_results: bool = True,
                               include_confidence: bool = False) -> List[Dict]:
        """Batch extract entities."""
        schema = self.create_schema().entities(entity_types)
        return self.batch_extract(texts, schema, batch_size, threshold, 0, format_results, include_confidence)

    def classify_text(self, text: str, tasks: Dict, threshold: float = 0.5,
                     format_results: bool = True, include_confidence: bool = False) -> Dict:
        """Classify text."""
        schema = self.create_schema()
        for name, config in tasks.items():
            if isinstance(config, dict) and "labels" in config:
                cfg = config.copy()
                labels = cfg.pop("labels")
                schema.classification(name, labels, **cfg)
            else:
                schema.classification(name, config)
        return self.extract(text, schema, threshold, format_results, include_confidence)

    def batch_classify_text(self, texts: List[str], tasks: Dict, batch_size: int = 8,
                           threshold: float = 0.5, format_results: bool = True,
                           include_confidence: bool = False) -> List[Dict]:
        """Batch classify texts."""
        schema = self.create_schema()
        for name, config in tasks.items():
            if isinstance(config, dict) and "labels" in config:
                cfg = config.copy()
                labels = cfg.pop("labels")
                schema.classification(name, labels, **cfg)
            else:
                schema.classification(name, config)
        return self.batch_extract(texts, schema, batch_size, threshold, 0, format_results, include_confidence)

    def extract_json(self, text: str, structures: Dict, threshold: float = 0.5,
                    format_results: bool = True, include_confidence: bool = False) -> Dict:
        """Extract structured data."""
        schema = self.create_schema()
        for parent, fields in structures.items():
            builder = schema.structure(parent)
            for spec in fields:
                name, dtype, choices, desc = self._parse_field_spec(spec)
                builder.field(name, dtype=dtype, choices=choices, description=desc)
        return self.extract(text, schema, threshold, format_results, include_confidence)

    def batch_extract_json(self, texts: List[str], structures: Dict, batch_size: int = 8,
                          threshold: float = 0.5, format_results: bool = True,
                          include_confidence: bool = False) -> List[Dict]:
        """Batch extract structured data."""
        schema = self.create_schema()
        for parent, fields in structures.items():
            builder = schema.structure(parent)
            for spec in fields:
                name, dtype, choices, desc = self._parse_field_spec(spec)
                builder.field(name, dtype=dtype, choices=choices, description=desc)
        return self.batch_extract(texts, schema, batch_size, threshold, 0, format_results, include_confidence)

    def extract_relations(self, text: str, relation_types, threshold: float = 0.5,
                         format_results: bool = True, include_confidence: bool = False) -> Dict:
        """Extract relations."""
        schema = self.create_schema().relations(relation_types)
        return self.extract(text, schema, threshold, format_results, include_confidence)

    def batch_extract_relations(self, texts: List[str], relation_types, batch_size: int = 8,
                               threshold: float = 0.5, format_results: bool = True,
                               include_confidence: bool = False) -> List[Dict]:
        """Batch extract relations."""
        schema = self.create_schema().relations(relation_types)
        return self.batch_extract(texts, schema, batch_size, threshold, 0, format_results, include_confidence)

    def _parse_field_spec(self, spec: str) -> Tuple[str, str, Optional[List[str]], Optional[str]]:
        """Parse field specification string."""
        parts = spec.split('::', 2)
        name = parts[0]
        dtype, choices, desc = "list", None, None

        if len(parts) == 1:
            return name, dtype, choices, desc

        for part in parts[1:]:
            if part in ['str', 'list']:
                dtype = part
            elif part.startswith('[') and part.endswith(']'):
                choices = [c.strip() for c in part[1:-1].split('|')]
                dtype = "str"
            else:
                desc = part

        return name, dtype, choices, desc


# Aliases
BuilderExtractor = GLiNER2
SchemaBuilder = Schema
JsonStructBuilder = StructureBuilder