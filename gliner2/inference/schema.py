"""
Torch-free schema and validator primitives.

This module contains the Schema builder, StructureBuilder, and RegexValidator
classes that have no dependency on torch, transformers, or any ML libraries.
They can be used standalone for schema construction, validation, and
serialization without installing the ``[local]`` extras.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Pattern, Union

from gliner2.inference.schema_model import SchemaInput


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Schema':
        """Create a Schema from a dictionary.

        Args:
            data: Dictionary with optional keys: entities, structures,
                  classifications, relations

        Returns:
            Schema: Constructed schema instance

        Raises:
            ValidationError: If the input data is invalid

        Example:
            >>> schema_dict = {
            ...     "entities": ["company", "person"],
            ...     "structures": {
            ...         "product_info": {
            ...             "fields": [
            ...                 {"name": "company", "dtype": "str"},
            ...                 {"name": "product"}
            ...             ]
            ...         }
            ...     },
            ...     "classifications": [
            ...         {"task": "sentiment", "labels": ["positive", "negative"]}
            ...     ],
            ...     "relations": ["works_for", "founded_by"]
            ... }
            >>> schema = Schema.from_dict(schema_dict)
        """
        validated = SchemaInput(**data)

        schema = cls()

        if validated.entities is not None:
            schema.entities(validated.entities)

        if validated.structures is not None:
            for struct_name, struct_input in validated.structures.items():
                builder = schema.structure(struct_name)
                for field_input in struct_input.fields:
                    builder.field(
                        name=field_input.name,
                        dtype=field_input.dtype,
                        choices=field_input.choices,
                        description=field_input.description
                    )
                builder._auto_finish()

        if validated.classifications is not None:
            for cls_input in validated.classifications:
                schema.classification(
                    task=cls_input.task,
                    labels=cls_input.labels,
                    multi_label=cls_input.multi_label
                )

        if validated.relations is not None:
            schema.relations(validated.relations)

        return schema

    @classmethod
    def from_json(cls, json_str: str) -> 'Schema':
        """Create a Schema from a JSON string.

        Args:
            json_str: JSON string with schema definition

        Returns:
            Schema: Constructed schema instance

        Raises:
            ValidationError: If the input data is invalid
            json.JSONDecodeError: If the JSON is malformed

        Example:
            >>> schema_json = '''
            ... {
            ...     "entities": ["company", "person"],
            ...     "classifications": [
            ...         {"task": "sentiment", "labels": ["positive", "negative"]}
            ...     ]
            ... }
            ... '''
            >>> schema = Schema.from_json(schema_json)
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to user-friendly dictionary format.

        Returns:
            Dict: Schema in dictionary format compatible with from_dict()

        Example:
            >>> schema = Schema()
            >>> schema.entities(["company", "person"])
            >>> schema_dict = schema.to_dict()
            >>> # schema_dict can be used with Schema.from_dict()
        """
        result = {}

        if self.schema["entities"]:
            if self.schema["entity_descriptions"]:
                result["entities"] = dict(self.schema["entity_descriptions"])
            else:
                result["entities"] = list(self.schema["entities"].keys())

        if self.schema["json_structures"]:
            result["structures"] = {}
            for struct_dict in self.schema["json_structures"]:
                for struct_name, struct_fields in struct_dict.items():
                    fields = []
                    field_order = self._field_orders.get(struct_name, [])

                    for field_name in field_order:
                        if field_name not in struct_fields:
                            continue

                        field_key = f"{struct_name}.{field_name}"
                        metadata = self._field_metadata.get(field_key, {})

                        field_def = {"name": field_name}

                        dtype = metadata.get("dtype", "list")
                        if dtype != "list":
                            field_def["dtype"] = dtype

                        choices = metadata.get("choices")
                        if choices:
                            field_def["choices"] = choices

                        desc = self.schema.get("json_descriptions", {}).get(struct_name, {}).get(field_name)
                        if desc:
                            field_def["description"] = desc

                        fields.append(field_def)

                    result["structures"][struct_name] = {"fields": fields}

        if self.schema["classifications"]:
            result["classifications"] = []
            for cls_config in self.schema["classifications"]:
                cls_def = {
                    "task": cls_config["task"],
                    "labels": cls_config["labels"]
                }
                if cls_config.get("multi_label", False):
                    cls_def["multi_label"] = True
                result["classifications"].append(cls_def)

        if self.schema["relations"]:
            result["relations"] = self._relation_order if self._relation_order else [
                list(rel_dict.keys())[0] for rel_dict in self.schema["relations"]
            ]

        return result
