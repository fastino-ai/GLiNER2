"""
GLiNER2 Training Data Creation & Validation Module

This module provides intuitive classes for creating, validating, and managing
training data for GLiNER2 models.

Quick Examples
--------------
Create entity examples:
    >>> example = InputExample(
    ...     text="John works at Google in NYC.",
    ...     entities={"person": ["John"], "company": ["Google"], "location": ["NYC"]}
    ... )

Create classification examples:
    >>> example = InputExample(
    ...     text="This movie is amazing!",
    ...     classifications=[
    ...         Classification(task="sentiment", labels=["positive", "negative"], true_label="positive")
    ...     ]
    ... )

Create structured data examples:
    >>> example = InputExample(
    ...     text="iPhone 15 costs $999",
    ...     structures=[
    ...         Structure("product", name="iPhone 15", price="$999")
    ...     ]
    ... )

Create relation examples:
    >>> example = InputExample(
    ...     text="Elon Musk founded SpaceX.",
    ...     relations=[
    ...         Relation("founded", head="Elon Musk", tail="SpaceX")
    ...     ]
    ... )

Build and validate dataset:
    >>> dataset = TrainingDataset(examples)
    >>> dataset.validate()  # Raises ValidationError if invalid
    >>> dataset.save("train.jsonl")

Load from JSONL:
    >>> dataset = TrainingDataset.load("train.jsonl")
    >>> # Or load multiple files
    >>> dataset = TrainingDataset.load(["train1.jsonl", "train2.jsonl"])
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator, TYPE_CHECKING
from collections import Counter
from tqdm import tqdm

if TYPE_CHECKING:
    # Forward declarations for type checking only
    pass


class ValidationError(Exception):
    """Raised when training data validation fails."""

    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or []

    def __str__(self):
        if self.errors:
            error_list = "\n  - ".join(self.errors[:10])
            suffix = (
                f"\n  ... and {len(self.errors) - 10} more errors"
                if len(self.errors) > 10
                else ""
            )
            return f"{self.args[0]}\n  - {error_list}{suffix}"
        return self.args[0]


# =============================================================================
# Data Format Detection & Loading
# =============================================================================


class DataFormat:
    """Enum-like class for supported data formats."""

    JSONL = "jsonl"
    JSONL_LIST = "jsonl_list"
    INPUT_EXAMPLE_LIST = "input_example_list"
    TRAINING_DATASET = "training_dataset"
    DICT_LIST = "dict_list"
    EXTRACTOR_DATASET = "extractor_dataset"


def detect_data_format(data: Any) -> str:
    """
    Detect the format of input data.

    Parameters
    ----------
    data : Any
        Input data in any supported format.

    Returns
    -------
    str
        The detected format name from DataFormat.
    """
    # String path
    if isinstance(data, str):
        return DataFormat.JSONL

    # Path object
    if isinstance(data, Path):
        return DataFormat.JSONL

    # List types
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, (str, Path)):
            return DataFormat.JSONL_LIST
        if isinstance(first, InputExample):
            return DataFormat.INPUT_EXAMPLE_LIST
        if isinstance(first, dict):
            return DataFormat.DICT_LIST

    # Empty list - default to dict list
    if isinstance(data, list) and len(data) == 0:
        return DataFormat.DICT_LIST

    # TrainingDataset
    if isinstance(data, TrainingDataset):
        return DataFormat.TRAINING_DATASET

    # ExtractorDataset (internal) - forward reference
    if type(data).__name__ == "ExtractorDataset":
        return DataFormat.EXTRACTOR_DATASET

    raise ValueError(f"Unsupported data format: {type(data)}")


class DataLoader_Factory:
    """
    Factory for loading data from various formats into a unified internal format.

    All loaders convert data to List[Dict] format where each dict has:
    - "input": str (the text)
    - "output": Dict (the schema/annotations)

    Or alternatively:
    - "text": str
    - "schema": Dict
    """

    @staticmethod
    def load(
        data: Any,
        max_samples: int = -1,
        shuffle: bool = True,
        seed: int = 42,
        validate: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Load data from any supported format.

        Parameters
        ----------
        data : Any
            Input data in any supported format.
        max_samples : int, default=-1
            Maximum samples to load (-1 = all).
        shuffle : bool, default=True
            Whether to shuffle the data.
        seed : int, default=42
            Random seed for shuffling.
        validate : bool, default=False
            Whether to validate the data. Validation is always strict:
            checks that entity spans, relation values, and structure
            field values exist in the text.

        Returns
        -------
        List[Dict[str, Any]]
            List of records in unified format.
        """
        fmt = detect_data_format(data)

        # Load based on format
        if fmt == DataFormat.JSONL:
            records = DataLoader_Factory._load_jsonl(data)
        elif fmt == DataFormat.JSONL_LIST:
            records = DataLoader_Factory._load_jsonl_list(data)
        elif fmt == DataFormat.INPUT_EXAMPLE_LIST:
            records = DataLoader_Factory._load_input_examples(data)
        elif fmt == DataFormat.TRAINING_DATASET:
            records = DataLoader_Factory._load_training_dataset(data)
        elif fmt == DataFormat.DICT_LIST:
            records = DataLoader_Factory._load_dict_list(data)
        elif fmt == DataFormat.EXTRACTOR_DATASET:
            records = data.data.copy()
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")

        # Validate if requested
        if validate and records:
            valid_indices, invalid_info = DataLoader_Factory._validate_records(records)

            if invalid_info:
                total_records = len(records)
                num_invalid = len(invalid_info)
                num_valid = len(valid_indices)

                print(
                    f"\nValidation: Found {num_invalid} invalid record(s) out of {total_records} total"
                )
                print("Removed invalid records:")

                # Print first 5 invalid records
                for idx, (record_idx, record, errors) in enumerate(invalid_info[:5]):
                    # Print first error for this record
                    error_msg = errors[0] if errors else "Unknown error"
                    print(f"  Record {record_idx}: {error_msg}")

                if num_invalid > 5:
                    print(f"  ... and {num_invalid - 5} more invalid record(s)")

                print(f"Kept {num_valid} valid record(s)\n")

                # Filter records to keep only valid ones
                records = [records[i] for i in valid_indices]

        # Shuffle
        if shuffle and records:
            random.seed(seed)
            random.shuffle(records)

        # Limit samples
        if max_samples > 0 and len(records) > max_samples:
            records = records[:max_samples]

        return records

    @staticmethod
    def _load_jsonl(path: Union[str, Path]) -> List[Dict]:
        """Load from single JSONL file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in {path} line {line_num}: {e}")

        return records

    @staticmethod
    def _load_jsonl_list(paths: List[Union[str, Path]]) -> List[Dict]:
        """Load from multiple JSONL files."""
        records = []
        for path in paths:
            records.extend(DataLoader_Factory._load_jsonl(path))
        return records

    @staticmethod
    def _load_input_examples(examples: List[InputExample]) -> List[Dict]:
        """Load from list of InputExample objects."""
        return [ex.to_dict() for ex in examples]

    @staticmethod
    def _load_training_dataset(dataset: TrainingDataset) -> List[Dict]:
        """Load from TrainingDataset object."""
        return dataset.to_records()

    @staticmethod
    def _load_dict_list(dicts: List[Dict]) -> List[Dict]:
        """Load from list of dicts."""
        if not dicts:
            return []

        first = dicts[0]

        # Check format
        if "input" in first and "output" in first:
            # Already in correct format
            return dicts
        elif "text" in first and "schema" in first:
            # Alternative format - keep as is (handled in __getitem__)
            return dicts
        elif "text" in first:
            # Maybe has entities/classifications at top level - try to convert
            records = []
            for d in dicts:
                output = {}
                if "entities" in d:
                    output["entities"] = d["entities"]
                if "classifications" in d:
                    output["classifications"] = d["classifications"]
                if "relations" in d:
                    output["relations"] = d["relations"]
                if "json_structures" in d:
                    output["json_structures"] = d["json_structures"]
                records.append({"input": d["text"], "output": output})
            return records
        else:
            raise ValueError(
                f"Unknown dict format. Expected keys like 'input'/'output', 'text'/'schema', "
                f"or 'text' with annotation keys. Got: {list(first.keys())}"
            )

    @staticmethod
    def _validate_records(
        records: List[Dict],
    ) -> Tuple[List[int], List[Tuple[int, Dict, List[str]]]]:
        """
        Validate and sanitize all records, removing only invalid parts.

        Uses granular sanitization:
        - Entities: Drop entity type if any mention not found
        - Classifications: Drop individual invalid classifications
        - Structures: Remove invalid fields, drop if all invalid
        - Relations: Drop relation if any field invalid
        - Record: Drop only if no valid tasks remain

        Returns
        -------
        Tuple[List[int], List[Tuple[int, Dict, List[str]]]]
            - First element: List of valid record indices (sanitized records replace originals)
            - Second element: List of (index, original_record, warning_messages) for dropped records
        """
        valid_indices = []
        invalid_info = []

        for i, record in tqdm(
            enumerate(records),
            total=len(records),
            desc="Validating records",
            unit="record",
        ):
            warnings = []
            try:
                example = InputExample.from_dict(record)
                sanitize_warnings, is_valid = example.sanitize()

                if is_valid:
                    # Replace record with sanitized version
                    records[i] = example.to_dict()
                    valid_indices.append(i)
                    if sanitize_warnings:
                        # Record was sanitized but still valid
                        warnings.extend(sanitize_warnings)
                else:
                    # No valid content remains
                    warnings.extend(sanitize_warnings)
                    invalid_info.append((i, record, warnings))
            except Exception as e:
                warnings.append(f"Failed to parse - {e}")
                invalid_info.append((i, record, warnings))

        return valid_indices, invalid_info


# Type alias for flexible data input
TrainDataInput = Union[
    str,  # Single JSONL path
    Path,  # Single JSONL path
    List[str],  # Multiple JSONL paths
    List[Path],  # Multiple JSONL paths
    List[Dict[str, Any]],  # Raw records
    "TrainingDataset",  # TrainingDataset (forward reference)
    "List[InputExample]",  # List of InputExample (forward reference)
    "ExtractorDataset",  # Legacy dataset (forward reference)
]


# =============================================================================
# Training Data Classes
# =============================================================================


@dataclass
class Classification:
    """
    A classification task definition.

    Parameters
    ----------
    task : str
        Name of the classification task (e.g., "sentiment", "category").
    labels : List[str]
        All possible labels for this task.
    true_label : str or List[str]
        The correct label(s) for this example.
    multi_label : bool, default=False
        Whether multiple labels can be selected.
    prompt : str, optional
        Custom prompt for the task.
    examples : List[Tuple[str, str]], optional
        Few-shot examples as (input, output) pairs.
    label_descriptions : Dict[str, str], optional
        Descriptions for each label.
    """

    task: str
    labels: List[str]
    true_label: Union[str, List[str]]
    multi_label: bool = False
    prompt: Optional[str] = None
    examples: Optional[List[Tuple[str, str]]] = None
    label_descriptions: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if isinstance(self.true_label, str):
            self._true_label_list = [self.true_label]
        else:
            self._true_label_list = list(self.true_label)

        # Auto-infer multi_label=True when multiple true labels are provided
        if len(self._true_label_list) > 1:
            self.multi_label = True

    def validate(self) -> List[str]:
        """Validate this classification and return list of errors."""
        errors = []
        if not self.task:
            errors.append("Classification task name cannot be empty")
        if not self.labels:
            errors.append(f"Classification '{self.task}' has no labels")
        for label in self._true_label_list:
            if label not in self.labels:
                errors.append(
                    f"True label '{label}' not in labels list for task '{self.task}'"
                )
        if len(self._true_label_list) > 1 and not self.multi_label:
            errors.append(
                f"Multiple true labels provided for '{self.task}' but multi_label=False"
            )
        if self.label_descriptions:
            for key in self.label_descriptions:
                if key not in self.labels:
                    errors.append(
                        f"Label description key '{key}' not in labels for task '{self.task}'"
                    )
        if self.examples:
            for i, ex in enumerate(self.examples):
                if not isinstance(ex, (list, tuple)) or len(ex) != 2:
                    errors.append(
                        f"Example {i} for task '{self.task}' must be (input, output) pair"
                    )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to training format dictionary."""
        result = {
            "task": self.task,
            "labels": self.labels,
            "true_label": self._true_label_list,
        }
        if self.multi_label:
            result["multi_label"] = True
        if self.prompt:
            result["prompt"] = self.prompt
        if self.examples:
            result["examples"] = [list(ex) for ex in self.examples]
        if self.label_descriptions:
            result["label_descriptions"] = self.label_descriptions
        return result


@dataclass
class ChoiceField:
    """
    A field with predefined choices (classification within structure).

    Parameters
    ----------
    value : str
        The selected value.
    choices : List[str]
        All possible choices.
    """

    value: str
    choices: List[str]

    def validate(self, field_name: str) -> List[str]:
        errors = []
        if self.value not in self.choices:
            errors.append(
                f"Choice value '{self.value}' not in choices {self.choices} for field '{field_name}'"
            )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "choices": self.choices}


@dataclass
class Structure:
    """
    A structured data extraction definition.

    Parameters
    ----------
    struct_name : str
        Name of the structure (e.g., "product", "contact").
    **fields : Any
        Field names and values. Values can be:
        - str: Single string value
        - List[str]: Multiple values
        - ChoiceField: Classification-style field with choices

    Examples
    --------
    >>> struct = Structure("product", name="iPhone", price="$999")
    >>> struct = Structure("contact", name="John", email="john@example.com")
    """

    struct_name: str
    _fields: Dict[str, Any] = field(default_factory=dict)
    descriptions: Optional[Dict[str, str]] = None

    def __init__(
        self, struct_name: str, _descriptions: Dict[str, str] = None, **fields
    ):
        self.struct_name = struct_name
        self._fields = fields
        self.descriptions = _descriptions

    def validate(self, text: str) -> List[str]:
        """
        Validate this structure.

        Parameters
        ----------
        text : str
            The text to validate against. Field values must exist in this text.

        Returns
        -------
        List[str]
            List of validation errors.
        """
        errors = []
        if not self.struct_name:
            errors.append("Structure name cannot be empty")
        if not self._fields:
            errors.append(f"Structure '{self.struct_name}' has no fields")
        for field_name, value in self._fields.items():
            if isinstance(value, ChoiceField):
                errors.extend(value.validate(f"{self.struct_name}.{field_name}"))
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    if v and v.lower() not in text.lower():
                        errors.append(
                            f"List value '{v}' at index {i} in '{self.struct_name}.{field_name}' not found in text"
                        )
            elif isinstance(value, str):
                if value and value.lower() not in text.lower():
                    errors.append(
                        f"Value '{value}' for '{self.struct_name}.{field_name}' not found in text"
                    )
        return errors

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        fields_dict = {}
        for field_name, value in self._fields.items():
            if isinstance(value, ChoiceField):
                fields_dict[field_name] = value.to_dict()
            else:
                fields_dict[field_name] = value
        return {self.struct_name: fields_dict}

    def get_descriptions(self) -> Optional[Dict[str, Dict[str, str]]]:
        if self.descriptions:
            return {self.struct_name: self.descriptions}
        return None


@dataclass
class Relation:
    """
    A relation extraction definition.

    Parameters
    ----------
    name : str
        Name of the relation (e.g., "works_for", "founded").
    head : str, optional
        The source/subject entity.
    tail : str, optional
        The target/object entity.
    **fields : Any
        Custom field names and values (use instead of head/tail).
    """

    name: str
    head: Optional[str] = None
    tail: Optional[str] = None
    _fields: Dict[str, str] = field(default_factory=dict)

    def __init__(self, name: str, head: str = None, tail: str = None, **fields):
        self.name = name
        self.head = head
        self.tail = tail
        if fields:
            self._fields = fields
        elif head is not None and tail is not None:
            self._fields = {"head": head, "tail": tail}
        else:
            self._fields = {}
            if head is not None:
                self._fields["head"] = head
            if tail is not None:
                self._fields["tail"] = tail

    def validate(self, text: str) -> List[str]:
        """
        Validate this relation.

        Parameters
        ----------
        text : str
            The text to validate against. Field values must exist in this text.

        Returns
        -------
        List[str]
            List of validation errors.
        """
        errors = []
        if not self.name:
            errors.append("Relation name cannot be empty")
        if not self._fields:
            errors.append(f"Relation '{self.name}' has no fields")
        for field_name, value in self._fields.items():
            if isinstance(value, str) and value:
                if value.lower() not in text.lower():
                    errors.append(
                        f"Relation value '{value}' for '{self.name}.{field_name}' not found in text"
                    )
        return errors

    def get_field_names(self) -> List[str]:
        return list(self._fields.keys())

    def to_dict(self) -> Dict[str, Dict[str, str]]:
        return {self.name: self._fields}


@dataclass
class InputExample:
    """
    A single training example for GLiNER2.

    Parameters
    ----------
    text : str
        The input text for this example.
    entities : Dict[str, List[str]], optional
        Entity type to mentions mapping.
    entity_descriptions : Dict[str, str], optional
        Descriptions for entity types.
    classifications : List[Classification], optional
        Classification tasks for this example.
    structures : List[Structure], optional
        Structured data extractions for this example.
    relations : List[Relation], optional
        Relation extractions for this example.

    Examples
    --------
    >>> example = InputExample(
    ...     text="John Smith works at Google.",
    ...     entities={"person": ["John Smith"], "company": ["Google"]}
    ... )
    """

    text: str
    entities: Optional[Dict[str, List[str]]] = None
    entity_descriptions: Optional[Dict[str, str]] = None
    classifications: Optional[List[Classification]] = None
    structures: Optional[List[Structure]] = None
    relations: Optional[List[Relation]] = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = {}
        if self.classifications is None:
            self.classifications = []
        if self.structures is None:
            self.structures = []
        if self.relations is None:
            self.relations = []

    def validate(self) -> List[str]:
        """
        Validate this example.

        Validation is always strict: checks that entity mentions, relation values,
        and structure field values exist in the text (case-insensitive).

        Returns
        -------
        List[str]
            List of validation errors. Empty list means valid.
        """
        errors = []
        if not self.text or not self.text.strip():
            errors.append("Text cannot be empty")
            return errors

        if self.entities:
            for entity_type, mentions in self.entities.items():
                if not entity_type:
                    errors.append("Entity type cannot be empty")
                for mention in mentions:
                    if mention and mention.lower() not in self.text.lower():
                        errors.append(
                            f"Entity '{mention}' (type: {entity_type}) not found in text"
                        )

        if self.entity_descriptions and self.entities:
            for desc_type in self.entity_descriptions:
                if desc_type not in self.entities:
                    errors.append(
                        f"Entity description for '{desc_type}' but no entities of that type"
                    )

        for cls in self.classifications:
            errors.extend(cls.validate())

        for struct in self.structures:
            errors.extend(struct.validate(self.text))

        relation_fields = {}
        for rel in self.relations:
            errors.extend(rel.validate(self.text))
            field_names = tuple(sorted(rel.get_field_names()))
            if rel.name in relation_fields:
                if relation_fields[rel.name] != field_names:
                    errors.append(
                        f"Relation '{rel.name}' has inconsistent fields: {relation_fields[rel.name]} vs {field_names}"
                    )
            else:
                relation_fields[rel.name] = field_names

        has_content = (
            bool(self.entities)
            or bool(self.classifications)
            or bool(self.structures)
            or bool(self.relations)
        )
        if not has_content:
            errors.append(
                "Example must have at least one task (entities, classifications, structures, or relations)"
            )

        return errors

    def is_valid(self) -> bool:
        """Check if this example is valid."""
        return len(self.validate()) == 0

    def sanitize(self) -> Tuple[List[str], bool]:
        """
        Remove invalid parts from this example, keeping only valid content.
        Mutates self in-place.

        Granular removal strategy:
        - Entities: Drop entire entity type if ANY mention is not found in text
        - Classifications: Drop individual classifications that have errors
        - Structures: Remove invalid fields; drop structure only if ALL fields become invalid
        - Relations: Drop the specific relation if ANY field has an error
        - Example: Mark as invalid only if no valid tasks remain

        Returns
        -------
        Tuple[List[str], bool]
            - List of warning messages about what was removed
            - bool: True if example still has valid content, False if should be dropped
        """
        warnings = []

        if not self.text or not self.text.strip():
            warnings.append("Text is empty")
            return warnings, False

        # 1. Sanitize entities - drop entity type if any mention not found
        if self.entities:
            types_to_remove = []
            for entity_type, mentions in self.entities.items():
                if not entity_type:
                    types_to_remove.append(entity_type)
                    warnings.append("Entity type is empty")
                    continue

                # Check if any mention is not in text
                has_invalid = False
                for mention in mentions:
                    if mention and mention.lower() not in self.text.lower():
                        has_invalid = True
                        warnings.append(
                            f"Entity '{mention}' (type: {entity_type}) not found in text - dropping entity type"
                        )
                        break

                if has_invalid:
                    types_to_remove.append(entity_type)

            # Remove invalid entity types
            for entity_type in types_to_remove:
                del self.entities[entity_type]

            # Clean up entity descriptions for removed types
            if self.entity_descriptions:
                desc_to_remove = [
                    desc_type
                    for desc_type in self.entity_descriptions
                    if desc_type not in self.entities
                ]
                for desc_type in desc_to_remove:
                    del self.entity_descriptions[desc_type]

        # 2. Sanitize classifications - drop individual invalid ones
        if self.classifications:
            valid_classifications = []
            for cls in self.classifications:
                cls_errors = cls.validate()
                if cls_errors:
                    warnings.append(
                        f"Classification '{cls.task}' has errors - dropping: {cls_errors[0]}"
                    )
                else:
                    valid_classifications.append(cls)
            self.classifications = valid_classifications

        # 3. Sanitize structures - remove invalid fields, drop if all invalid
        if self.structures:
            valid_structures = []
            for struct in self.structures:
                if not struct.struct_name:
                    warnings.append("Structure has empty name - dropping")
                    continue

                if not struct._fields:
                    warnings.append(
                        f"Structure '{struct.struct_name}' has no fields - dropping"
                    )
                    continue

                # Filter out invalid fields
                valid_fields = {}
                for field_name, value in struct._fields.items():
                    is_valid = True

                    if isinstance(value, ChoiceField):
                        field_errors = value.validate(
                            f"{struct.struct_name}.{field_name}"
                        )
                        if field_errors:
                            warnings.append(
                                f"Field '{struct.struct_name}.{field_name}' invalid - dropping field"
                            )
                            is_valid = False
                    elif isinstance(value, list):
                        for v in value:
                            if v and v.lower() not in self.text.lower():
                                warnings.append(
                                    f"List value '{v}' in '{struct.struct_name}.{field_name}' not found - dropping field"
                                )
                                is_valid = False
                                break
                    elif isinstance(value, str):
                        if value and value.lower() not in self.text.lower():
                            warnings.append(
                                f"Value '{value}' for '{struct.struct_name}.{field_name}' not found - dropping field"
                            )
                            is_valid = False

                    if is_valid:
                        valid_fields[field_name] = value

                # Only keep structure if it has at least one valid field
                if valid_fields:
                    struct._fields = valid_fields
                    valid_structures.append(struct)
                else:
                    warnings.append(
                        f"Structure '{struct.struct_name}' has no valid fields - dropping"
                    )

            self.structures = valid_structures

        # 4. Sanitize relations - drop entire relation if any field is invalid
        if self.relations:
            valid_relations = []
            for rel in self.relations:
                if not rel.name:
                    warnings.append("Relation has empty name - dropping")
                    continue

                if not rel._fields:
                    warnings.append(f"Relation '{rel.name}' has no fields - dropping")
                    continue

                # Check if any field value is invalid
                has_invalid = False
                for field_name, value in rel._fields.items():
                    if isinstance(value, str) and value:
                        if value.lower() not in self.text.lower():
                            warnings.append(
                                f"Relation '{rel.name}' field '{field_name}' value '{value}' not found - dropping relation"
                            )
                            has_invalid = True
                            break

                if not has_invalid:
                    valid_relations.append(rel)

            self.relations = valid_relations

        # Check if example still has any valid content
        has_content = (
            bool(self.entities)
            or bool(self.classifications)
            or bool(self.structures)
            or bool(self.relations)
        )

        if not has_content:
            warnings.append("No valid tasks remain after sanitization")
            return warnings, False

        return warnings, True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to GLiNER2 training format."""
        output = {}
        if self.entities:
            output["entities"] = self.entities
        if self.entity_descriptions:
            output["entity_descriptions"] = self.entity_descriptions
        if self.classifications:
            output["classifications"] = [cls.to_dict() for cls in self.classifications]
        if self.structures:
            output["json_structures"] = [struct.to_dict() for struct in self.structures]
            all_descriptions = {}
            for struct in self.structures:
                desc = struct.get_descriptions()
                if desc:
                    all_descriptions.update(desc)
            if all_descriptions:
                output["json_descriptions"] = all_descriptions
        if self.relations:
            output["relations"] = [rel.to_dict() for rel in self.relations]
        return {"input": self.text, "output": output}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputExample":
        """Create InputExample from training format dictionary."""
        text = data["input"]
        output = data["output"]

        entities = output.get("entities")
        entity_descriptions = output.get("entity_descriptions")

        classifications = []
        for cls_data in output.get("classifications", []):
            classifications.append(
                Classification(
                    task=cls_data["task"],
                    labels=cls_data["labels"],
                    true_label=cls_data["true_label"],
                    multi_label=cls_data.get("multi_label", False),
                    prompt=cls_data.get("prompt"),
                    examples=[tuple(ex) for ex in cls_data.get("examples", [])] or None,
                    label_descriptions=cls_data.get("label_descriptions"),
                )
            )

        structures = []
        json_descriptions = output.get("json_descriptions", {})
        for struct_data in output.get("json_structures", []):
            for struct_name, fields in struct_data.items():
                parsed_fields = {}
                for field_name, value in fields.items():
                    if (
                        isinstance(value, dict)
                        and "value" in value
                        and "choices" in value
                    ):
                        parsed_fields[field_name] = ChoiceField(
                            value["value"], value["choices"]
                        )
                    else:
                        parsed_fields[field_name] = value
                structures.append(
                    Structure(
                        struct_name,
                        _descriptions=json_descriptions.get(struct_name),
                        **parsed_fields,
                    )
                )

        relations = []
        for rel_data in output.get("relations", []):
            for rel_name, fields in rel_data.items():
                if "head" in fields and "tail" in fields and len(fields) == 2:
                    relations.append(
                        Relation(rel_name, head=fields["head"], tail=fields["tail"])
                    )
                else:
                    relations.append(Relation(rel_name, **fields))

        return cls(
            text=text,
            entities=entities,
            entity_descriptions=entity_descriptions,
            classifications=classifications if classifications else None,
            structures=structures if structures else None,
            relations=relations if relations else None,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "InputExample":
        return cls.from_dict(json.loads(json_str))


class TrainingDataset:
    """
    A collection of InputExamples for training GLiNER2.

    Can be created from:
    - List of InputExample objects
    - JSONL file path(s)
    - Raw dict data

    Parameters
    ----------
    examples : List[InputExample], optional
        Initial list of examples.

    Examples
    --------
    >>> # From InputExample list
    >>> dataset = TrainingDataset([example1, example2])
    >>>
    >>> # From JSONL file
    >>> dataset = TrainingDataset.load("train.jsonl")
    >>>
    >>> # From multiple JSONL files
    >>> dataset = TrainingDataset.load(["train1.jsonl", "train2.jsonl"])
    """

    def __init__(self, examples: List[InputExample] = None):
        self.examples: List[InputExample] = examples or []

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> InputExample:
        return self.examples[idx]

    def __iter__(self) -> Iterator[InputExample]:
        return iter(self.examples)

    def add(self, example: InputExample) -> "TrainingDataset":
        self.examples.append(example)
        return self

    def add_many(self, examples: List[InputExample]) -> "TrainingDataset":
        self.examples.extend(examples)
        return self

    def validate(self, raise_on_error: bool = True) -> Dict[str, Any]:
        """
        Validate all examples in the dataset.

        Validation is always strict: checks that entity mentions, relation values,
        and structure field values exist in the text (case-insensitive).

        Parameters
        ----------
        raise_on_error : bool, default=True
            If True, raises ValidationError when invalid examples are found.
            If False, returns validation report without raising.

        Returns
        -------
        Dict[str, Any]
            Validation report with counts and error details.
        """
        all_errors = []
        valid_count = 0
        invalid_indices = []

        for i, example in enumerate(self.examples):
            errors = example.validate()
            if errors:
                invalid_indices.append(i)
                for error in errors:
                    all_errors.append(f"Example {i}: {error}")
            else:
                valid_count += 1

        report = {
            "valid": valid_count,
            "invalid": len(invalid_indices),
            "total": len(self.examples),
            "invalid_indices": invalid_indices,
            "errors": all_errors,
        }

        if all_errors and raise_on_error:
            raise ValidationError(
                f"Dataset validation failed: {len(invalid_indices)} invalid examples",
                all_errors,
            )

        return report

    def validate_relation_consistency(self) -> List[str]:
        """Validate that relation field structures are consistent across the dataset."""
        errors = []
        relation_fields: Dict[str, Tuple[int, Tuple[str, ...]]] = {}

        for i, example in enumerate(self.examples):
            for rel in example.relations:
                field_names = tuple(sorted(rel.get_field_names()))
                if rel.name in relation_fields:
                    first_idx, first_fields = relation_fields[rel.name]
                    if first_fields != field_names:
                        errors.append(
                            f"Relation '{rel.name}' field inconsistency: Example {first_idx} has {list(first_fields)}, but Example {i} has {list(field_names)}"
                        )
                else:
                    relation_fields[rel.name] = (i, field_names)
        return errors

    def stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        stats = {
            "total_examples": len(self.examples),
            "entity_types": Counter(),
            "entity_mentions": 0,
            "classification_tasks": Counter(),
            "classification_labels": {},
            "structure_types": Counter(),
            "relation_types": Counter(),
            "text_lengths": [],
            "task_distribution": {
                "entities_only": 0,
                "classifications_only": 0,
                "structures_only": 0,
                "relations_only": 0,
                "multi_task": 0,
                "empty": 0,
            },
        }

        for example in self.examples:
            stats["text_lengths"].append(len(example.text))
            for entity_type, mentions in example.entities.items():
                stats["entity_types"][entity_type] += len(mentions)
                stats["entity_mentions"] += len(mentions)
            for cls in example.classifications:
                stats["classification_tasks"][cls.task] += 1
                if cls.task not in stats["classification_labels"]:
                    stats["classification_labels"][cls.task] = Counter()
                for label in cls._true_label_list:
                    stats["classification_labels"][cls.task][label] += 1
            for struct in example.structures:
                stats["structure_types"][struct.struct_name] += 1
            for rel in example.relations:
                stats["relation_types"][rel.name] += 1

            has_entities = bool(example.entities)
            has_cls = bool(example.classifications)
            has_struct = bool(example.structures)
            has_rel = bool(example.relations)
            task_count = sum([has_entities, has_cls, has_struct, has_rel])

            if task_count == 0:
                stats["task_distribution"]["empty"] += 1
            elif task_count > 1:
                stats["task_distribution"]["multi_task"] += 1
            elif has_entities:
                stats["task_distribution"]["entities_only"] += 1
            elif has_cls:
                stats["task_distribution"]["classifications_only"] += 1
            elif has_struct:
                stats["task_distribution"]["structures_only"] += 1
            elif has_rel:
                stats["task_distribution"]["relations_only"] += 1

        if stats["text_lengths"]:
            lengths = stats["text_lengths"]
            stats["text_length_stats"] = {
                "min": min(lengths),
                "max": max(lengths),
                "mean": sum(lengths) / len(lengths),
                "median": sorted(lengths)[len(lengths) // 2],
            }

        stats["entity_types"] = dict(stats["entity_types"])
        stats["classification_tasks"] = dict(stats["classification_tasks"])
        stats["classification_labels"] = {
            k: dict(v) for k, v in stats["classification_labels"].items()
        }
        stats["structure_types"] = dict(stats["structure_types"])
        stats["relation_types"] = dict(stats["relation_types"])

        return stats

    def print_stats(self):
        """Print formatted statistics."""
        s = self.stats()
        print(f"\n{'='*60}")
        print("GLiNER2 Training Dataset Statistics")
        print(f"{'='*60}")
        print(f"Total examples: {s['total_examples']}")

        if s.get("text_length_stats"):
            tls = s["text_length_stats"]
            print(
                f"\nText lengths: min={tls['min']}, max={tls['max']}, mean={tls['mean']:.1f}"
            )

        print("\nTask Distribution:")
        for task, count in s["task_distribution"].items():
            if count > 0:
                print(f"  {task}: {count} ({100*count/s['total_examples']:.1f}%)")

        if s["entity_types"]:
            print(f"\nEntity Types ({s['entity_mentions']} total mentions):")
            for etype, count in sorted(s["entity_types"].items(), key=lambda x: -x[1]):
                print(f"  {etype}: {count}")

        if s["classification_tasks"]:
            print("\nClassification Tasks:")
            for task, count in s["classification_tasks"].items():
                print(f"  {task}: {count} examples")
                if task in s["classification_labels"]:
                    for label, lcount in s["classification_labels"][task].items():
                        print(f"    - {label}: {lcount}")

        if s["structure_types"]:
            print("\nStructure Types:")
            for stype, count in s["structure_types"].items():
                print(f"  {stype}: {count}")

        if s["relation_types"]:
            print("\nRelation Types:")
            for rtype, count in s["relation_types"].items():
                print(f"  {rtype}: {count}")

        print(f"{'='*60}\n")

    def to_jsonl(self) -> str:
        return "\n".join(example.to_json() for example in self.examples)

    def to_records(self) -> List[Dict[str, Any]]:
        """Convert to list of record dicts for trainer."""
        return [ex.to_dict() for ex in self.examples]

    def save(self, path: Union[str, Path], validate_first: bool = True):
        """Save dataset to JSONL file."""
        if validate_first:
            self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for example in self.examples:
                f.write(example.to_json() + "\n")
        print(f"Saved {len(self.examples)} examples to {path}")

    @classmethod
    def load(
        cls,
        paths: Union[str, Path, List[Union[str, Path]]],
        shuffle: bool = False,
        seed: int = 42,
    ) -> "TrainingDataset":
        """
        Load dataset from JSONL file(s).

        Parameters
        ----------
        paths : str, Path, or List
            Single file path or list of file paths.
        shuffle : bool, default=False
            Whether to shuffle the loaded examples.
        seed : int, default=42
            Random seed for shuffling.

        Returns
        -------
        TrainingDataset
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]

        examples = []
        for path in paths:
            path = Path(path)
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            examples.append(InputExample.from_dict(data))
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"Invalid JSON in {path} line {line_num}: {e}"
                            )
                        except Exception as e:
                            raise ValueError(
                                f"Error parsing {path} line {line_num}: {e}"
                            )
            print(f"Loaded {len(examples)} examples from {path}")

        if shuffle:
            random.seed(seed)
            random.shuffle(examples)

        return cls(examples)

    @classmethod
    def from_records(cls, records: List[Dict[str, Any]]) -> "TrainingDataset":
        """Create dataset from list of record dicts."""
        examples = [InputExample.from_dict(r) for r in records]
        return cls(examples)

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple["TrainingDataset", "TrainingDataset", "TrainingDataset"]:
        """Split dataset into train/val/test sets."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")

        indices = list(range(len(self.examples)))
        if shuffle:
            random.seed(seed)
            random.shuffle(indices)

        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        return (
            TrainingDataset([self.examples[i] for i in indices[:train_end]]),
            TrainingDataset([self.examples[i] for i in indices[train_end:val_end]]),
            TrainingDataset([self.examples[i] for i in indices[val_end:]]),
        )

    def filter(self, predicate) -> "TrainingDataset":
        """Filter examples based on a predicate function."""
        return TrainingDataset([ex for ex in self.examples if predicate(ex)])

    def sample(self, n: int, seed: int = 42) -> "TrainingDataset":
        """Random sample of examples."""
        random.seed(seed)
        return TrainingDataset(random.sample(self.examples, min(n, len(self.examples))))


# Convenience functions
def create_entity_example(
    text: str, entities: Dict[str, List[str]], descriptions: Dict[str, str] = None
) -> InputExample:
    """Create an entity extraction example."""
    return InputExample(text=text, entities=entities, entity_descriptions=descriptions)


def create_classification_example(
    text: str,
    task: str,
    labels: List[str],
    true_label: Union[str, List[str]],
    multi_label: bool = False,
    **kwargs,
) -> InputExample:
    """Create a classification example."""
    return InputExample(
        text=text,
        classifications=[
            Classification(
                task=task,
                labels=labels,
                true_label=true_label,
                multi_label=multi_label,
                **kwargs,
            )
        ],
    )


def create_structure_example(text: str, structure_name: str, **fields) -> InputExample:
    """Create a structured data example."""
    return InputExample(text=text, structures=[Structure(structure_name, **fields)])


def create_relation_example(
    text: str, relation_name: str, head: str = None, tail: str = None, **fields
) -> InputExample:
    """Create a relation extraction example."""
    return InputExample(
        text=text, relations=[Relation(relation_name, head=head, tail=tail, **fields)]
    )
