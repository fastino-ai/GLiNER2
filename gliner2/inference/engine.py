"""
GLiNER2 - Advanced Information Extraction Engine with Batch Processing

This module provides an intuitive schema-based interface for extracting
structured information from text using the GLiNER2 model, with efficient
batch processing capabilities.

Quick Examples
--------------
Extract entities:
    >>> extractor = GLiNER2.from_pretrained("your-model")
    >>> results = extractor.extract_entities(
    ...     "Apple released iPhone 15 in September 2023.",
    ...     ["company", "product", "date"]
    ... )
    >>> # {'entities': {'company': ['Apple'], 'pxxroduct': ['iPhone 15'], 'date': ['September 2023']}}

Batch processing:
    >>> results = extractor.batch_extract_entities(
    ...     ["Text 1 about Apple.", "Text 2 about Google.", "Text 3 about Microsoft."],
    ...     ["company", "product", "person"],
    ...     batch_size=8
    ... )
    >>> # Returns list of results, one per text

Extract structured data:
    >>> results = extractor.extract_json(
    ...     "Contact John Doe at john@email.com or call 555-1234.",
    ...     {"contact": ["name:str", "email:str", "phone"]}
    ... )
    >>> # {'contact': [{'name': 'John Doe', 'email': 'john@email.com', 'phone': ['555-1234']}]}

Complex multi-task extraction:
    >>> schema = (extractor.create_schema()
    ...     .entities(["person", "location"])
    ...     .classification("sentiment", ["positive", "negative", "neutral"])
    ...     .structure("product_review")
    ...         .field("product", dtype="str")
    ...         .field("rating", dtype="str", choices=["1", "2", "3", "4", "5"])
    ...         .field("pros", dtype="list")
    ...         .field("cons", dtype="list")
    ... )
    >>> results = extractor.extract(review_text, schema)
"""

from __future__ import annotations

import re
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING
from typing import Pattern, Literal

import torch
import torch.nn.functional as F
from gliner2.model import Extractor

if TYPE_CHECKING:
    from gliner2.api_client import GLiNER2API


@dataclass
class RegexValidator:
    """
    Regex-based span filter.

    Parameters
    ----------
    pattern : str | Pattern[str]
        The regex to apply. If a pre-compiled Pattern is supplied, its flags
        are preserved.
    mode : Literal["full", "partial"], default="full"
        * "full":  `re.fullmatch`
        * "partial": `re.search`
    exclude : bool, default=False
        * False → keep only spans that match.
        * True  → drop spans that match.
    flags : int, default=re.IGNORECASE
        Standard `re` flags to apply **when pattern is a string**.
    """
    pattern: str | Pattern[str]
    mode: Literal["full", "partial"] = "full"
    exclude: bool = False
    flags: int = re.IGNORECASE
    _compiled: Pattern[str] = field(init=False, repr=False)

    # --------------------------------------------------------------------- #
    # Dunders
    # --------------------------------------------------------------------- #
    def __post_init__(self) -> None:  # noqa: D401
        if self.mode not in {"full", "partial"}:
            raise ValueError(f"mode must be 'full' or 'partial', got {self.mode!r}")
        try:
            compiled = (
                self.pattern
                if isinstance(self.pattern, re.Pattern)
                else re.compile(self.pattern, self.flags)
            )
        except re.error as err:
            raise ValueError(f"Invalid regex: {self.pattern!r}") from err

        object.__setattr__(self, "_compiled", compiled)

    def __call__(self, text: str) -> bool:  # noqa: D401
        """Alias for `validate` so you can use the instance like a function."""
        return self.validate(text)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def validate(self, text: str) -> bool:  # noqa: D401
        """Return **True** if the span passes this validator."""
        matcher = (
            self._compiled.fullmatch if self.mode == "full" else self._compiled.search
        )
        matched = matcher(text) is not None
        return not matched if self.exclude else matched

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def describe(self) -> str:
        neg = "exclude" if self.exclude else "include"
        return f"<RegexValidator {neg} /{self._compiled.pattern}/ ({self.mode})>"

    def __repr__(self) -> str:  # noqa: D401
        return self.describe()


class StructureBuilder:
    """
    Builder for structured data schemas - auto-finishes when parent method is called.

    This builder allows fluent construction of structured data with multiple fields.
    It automatically completes when any Schema method is called, ensuring
    proper schema construction without explicit finish() calls.
    """

    def __init__(self, schema: 'Schema', parent: str):
        self.schema = schema
        self.parent = parent
        self.fields = OrderedDict()  # Preserve field order
        self.descriptions = OrderedDict()  # Preserve description order
        self.field_order = []  # Track field addition order
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
        """
        Add a field to the structured data.

        Parameters
        ----------
        name : str
            The name of the field to extract.
        dtype : {"str", "list"}, default="list"
            The data type of the field:
            - "str": Extract a single value (best match)
            - "list": Extract multiple values
        choices : List[str], optional
            If provided, the field becomes a classification field where values
            must be selected from these choices. For dtype="str", selects the
            best matching choice. For dtype="list", can select multiple choices.
        description : str, optional
            Human-readable description of what this field represents. Used to
            improve extraction accuracy through better context understanding.
        threshold : float, optional
            Custom confidence threshold for this field (overrides default).
            Values must be between 0 and 1.
        validators : List[RegexValidator], optional
            List of regex validators to filter extracted spans.
            All validators must pass for a span to be included.

        Returns
        -------
        StructureBuilder
            Returns self for method chaining.
        """
        # Store field in schema format
        self.fields[name] = {"value": "", "choices": choices} if choices else ""
        self.field_order.append(name)  # Track order

        if description:
            self.descriptions[name] = description

        # Store metadata including threshold and validators
        self.schema._store_field_metadata(self.parent, name, dtype, threshold, choices, validators)
        return self

    def _auto_finish(self):
        """Automatically finish this structure when needed."""
        if not self._finished:
            # Store field order for this structure
            self.schema._store_field_order(self.parent, self.field_order)

            self.schema.schema["json_structures"].append({self.parent: self.fields})

            if self.descriptions:
                if "json_descriptions" not in self.schema.schema:
                    self.schema.schema["json_descriptions"] = {}
                self.schema.schema["json_descriptions"][self.parent] = self.descriptions

            self._finished = True

    def __getattr__(self, name):
        """Auto-finish when any schema method is called."""
        if hasattr(self.schema, name):
            self._auto_finish()
            return getattr(self.schema, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class Schema:
    """Main schema builder for extraction tasks."""

    def __init__(self):
        self.schema = {
            "json_structures": [],
            "classifications": [],
            "entities": OrderedDict(),  # Preserve entity order
            "relations": [],  # Relations list
            "json_descriptions": {},
            "entity_descriptions": OrderedDict()  # Preserve description order
        }
        # Store metadata for thresholds and types
        self._field_metadata = {}  # "parent.field" -> {dtype, threshold, choices, validators}
        self._entity_metadata = {}  # "entity_name" -> {dtype, threshold}
        self._relation_metadata = {}  # "relation_name" -> {threshold}
        self._field_orders = {}  # "parent" -> [field1, field2, ...]
        self._entity_order = []  # [entity1, entity2, ...]
        self._relation_order = []  # [relation1, relation2, ...]
        self._active_structure_builder = None  # Track active structure builder

    def _store_field_metadata(self, parent: str, field: str, dtype: str, threshold: Optional[float],
                              choices: Optional[List[str]], validators: Optional[List[RegexValidator]] = None):
        """Store field configuration."""
        # Validate threshold if provided
        if threshold is not None:
            if not 0 <= threshold <= 1:
                raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        key = f"{parent}.{field}"
        self._field_metadata[key] = {
            "dtype": dtype,
            "threshold": threshold,
            "choices": choices,
            "validators": validators or [],
        }

    def _store_entity_metadata(self, entity: str, dtype: str, threshold: Optional[float]):
        """Store entity configuration."""
        # Validate threshold if provided
        if threshold is not None:
            if not 0 <= threshold <= 1:
                raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        self._entity_metadata[entity] = {"dtype": dtype, "threshold": threshold}

    def _store_field_order(self, parent: str, field_order: List[str]):
        """Store field order for a structure."""
        self._field_orders[parent] = field_order

    def structure(self, name: str) -> StructureBuilder:
        """
        Start building a structured data schema for hierarchical extraction.

        This method creates a builder for defining structured data with multiple
        fields. The structure will extract repeated instances of the defined
        pattern from the text.

        Parameters
        ----------
        name : str
            The name of the structure (e.g., "product", "address", "event").
            This will be the key in the results dictionary.

        Returns
        -------
        StructureBuilder
            A builder object for adding fields to this structure.
        """
        # Auto-finish any active structure builder
        if self._active_structure_builder:
            self._active_structure_builder._auto_finish()

        self._active_structure_builder = StructureBuilder(self, name)
        return self._active_structure_builder

    def classification(
            self,
            task: str,
            labels: Union[List[str], Dict[str, str]],
            multi_label: bool = False,
            cls_threshold: float = 0.5,
            **kwargs
    ) -> 'Schema':
        """
        Add a text classification task.

        This method adds a classification task that assigns one or more labels
        to the entire input text (document-level classification).

        Parameters
        ----------
        task : str
            The name of the classification task (e.g., "sentiment", "category").
            This will be the key in the results dictionary.
        labels : List[str] or Dict[str, str]
            Either:
            - List[str]: Simple list of label names
            - Dict[str, str]: Mapping of label names to descriptions
              (descriptions help improve classification accuracy)
        multi_label : bool, default=False
            If True, multiple labels can be assigned to the text.
            If False, only the single best label is selected.
        cls_threshold : float, default=0.5
            Confidence threshold for label assignment (0-1).
            For multi_label, labels above this threshold are selected.
        **kwargs : dict
            Additional configuration options:
            - class_act : str, optional
                Activation function: "sigmoid", "softmax", or "auto"

        Returns
        -------
        Schema
            Returns self for method chaining.
        """
        # Auto-finish any active structure builder
        if self._active_structure_builder:
            self._active_structure_builder._auto_finish()
            self._active_structure_builder = None

        # Parse labels
        if isinstance(labels, dict):
            label_names = list(labels.keys())
            label_descriptions = labels
        else:
            label_names = labels
            label_descriptions = None

        # Create classification config
        config = {
            "task": task,
            "labels": label_names,
            "multi_label": multi_label,
            "cls_threshold": cls_threshold,
            "true_label": ["N/A"],
            **kwargs
        }

        if label_descriptions:
            config["label_descriptions"] = label_descriptions

        self.schema["classifications"].append(config)
        return self

    def entities(
            self,
            entity_types: Union[str, List[str], Dict[str, Union[str, Dict]]],
            dtype: Literal["str", "list"] = "list",
            threshold: Optional[float] = None
    ) -> 'Schema':
        """
        Add entity extraction task for named entity recognition.

        This method configures extraction of specific entity types from the text,
        such as people, organizations, locations, or any custom entity types.

        Parameters
        ----------
        entity_types : str, List[str], or Dict[str, Union[str, Dict]]
            Entity types to extract. Can be:
            - str: Single entity type
            - List[str]: Multiple entity types
            - Dict[str, str]: Entity types with descriptions
            - Dict[str, Dict]: Entity types with full configuration
              (including dtype, threshold, description)
        dtype : {"str", "list"}, default="list"
            Default data type for entities:
            - "str": Extract only one entity per type (the best match)
            - "list": Extract all matching entities per type
        threshold : float, optional
            Default confidence threshold for entity extraction (0-1).
            Individual entities can override this.

        Returns
        -------
        Schema
            Returns self for method chaining.
        """
        # Auto-finish any active structure builder
        if self._active_structure_builder:
            self._active_structure_builder._auto_finish()
            self._active_structure_builder = None

        entities = self._parse_entity_input(entity_types)

        for entity_name, config in entities.items():
            self.schema["entities"][entity_name] = ""

            # Only add to order if not already present
            if entity_name not in self._entity_order:
                self._entity_order.append(entity_name)

            # Get configuration
            entity_dtype = config.get("dtype", dtype)
            entity_threshold = config.get("threshold", threshold)
            description = config.get("description")

            # Store metadata
            self._store_entity_metadata(entity_name, entity_dtype, entity_threshold)

            # Store description
            if description:
                self.schema["entity_descriptions"][entity_name] = description

        return self

    def _parse_entity_input(self, entity_types) -> Dict[str, Dict]:
        """Parse different entity input formats."""
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
        else:
            raise ValueError("Invalid entity_types format")

    def relations(
            self,
            relation_types: Union[str, List[str], Dict[str, Union[str, Dict]]],
            threshold: Optional[float] = None
    ) -> 'Schema':
        """
        Add relation extraction task.

        This method configures extraction of relations between entities in the text.
        Relations automatically use "head" and "tail" fields which are handled
        by the backend processor.

        Parameters
        ----------
        relation_types : str, List[str], or Dict[str, Union[str, Dict]]
            Relation types to extract. Can be:
            - str: Single relation type
            - List[str]: Multiple relation types
            - Dict[str, str]: Relation types with descriptions
            - Dict[str, Dict]: Relation types with full configuration
              (including threshold, description)
        threshold : float, optional
            Default confidence threshold for relation extraction (0-1).
            Individual relations can override this.

        Returns
        -------
        Schema
            Returns self for method chaining.

        Examples
        --------
        >>> schema = extractor.create_schema()
        >>> schema.relations(["works_for", "located_in"])
        >>> schema.relations({"works_for": {"threshold": 0.6}, "located_in": "A location relation"})
        """
        # Auto-finish any active structure builder
        if self._active_structure_builder:
            self._active_structure_builder._auto_finish()
            self._active_structure_builder = None

        # Parse relation types similar to entities
        if isinstance(relation_types, str):
            relations = {relation_types: {}}
        elif isinstance(relation_types, list):
            relations = {name: {} for name in relation_types}
        elif isinstance(relation_types, dict):
            relations = {}
            for name, config in relation_types.items():
                if isinstance(config, str):
                    relations[name] = {"description": config}
                elif isinstance(config, dict):
                    relations[name] = config
                else:
                    relations[name] = {}
        else:
            raise ValueError("Invalid relation_types format")

        for relation_name, config in relations.items():
            # Relations use standard "head" and "tail" fields
            relation_config = {"head": "", "tail": ""}
            self.schema["relations"].append({relation_name: relation_config})

            # Only add to order if not already present
            if relation_name not in self._relation_order:
                self._relation_order.append(relation_name)

            # Store field order for relations (head and tail)
            self._field_orders[relation_name] = ["head", "tail"]

            # Get configuration
            relation_threshold = config.get("threshold", threshold)
            description = config.get("description")

            # Store metadata
            if relation_threshold is not None:
                if not 0 <= relation_threshold <= 1:
                    raise ValueError(f"Threshold must be between 0 and 1, got {relation_threshold}")
            self._relation_metadata[relation_name] = {"threshold": relation_threshold}

            # Store description if provided (for future use)
            if description:
                if "relation_descriptions" not in self.schema:
                    self.schema["relation_descriptions"] = OrderedDict()
                self.schema["relation_descriptions"][relation_name] = description

        return self

    def build(self) -> Dict[str, Any]:
        """
        Build the final schema dictionary.

        This method finalizes the schema construction and returns the internal
        schema representation. Any pending structures are automatically
        completed.

        Returns
        -------
        Dict[str, Any]
            The complete schema dictionary ready for extraction.
        """
        # Auto-finish any active structure builder
        if self._active_structure_builder:
            self._active_structure_builder._auto_finish()
            self._active_structure_builder = None

        return self.schema


@dataclass
class BatchEncodingResult:
    """Container for batch encoding results."""
    token_embeddings: torch.Tensor
    original_lengths: List[int]
    mapped_indices: List[List[Tuple[str, int, int]]]
    transformed_data: List[Dict[str, Any]]
    batch_indices: List[int]  # Maps back to original batch position


class GLiNER2(Extractor):
    """
    Advanced information extraction model with intuitive schema-based API
    and efficient batch processing capabilities.

    GLiNER2 provides a clean, powerful interface for extracting structured
    information from text including entities, classifications, and complex
    nested data structures. It uses a schema-based approach that allows
    combining multiple extraction tasks together, with support for efficient
    batch processing.

    Key Features
    ------------
    - **Entity Extraction**: Extract named entities like persons, locations, etc.
    - **Text Classification**: Single/multi-label document classification
    - **Structured Extraction**: Extract complex nested data structures
    - **Batch Processing**: Efficient processing of multiple texts
    - **Field-level Control**: Set custom thresholds and types per field
    - **Auto-completion**: No need to explicitly finish/build schemas
    - **Order Preservation**: Maintains field and entity ordering

    Quick Start
    -----------
    >>> from gliner2 import GLiNER2
    >>>
    >>> # Load pre-trained model
    >>> extractor = GLiNER2.from_pretrained("your-model")
    >>>
    >>> # Or load from API (no local model needed)
    >>> extractor = GLiNER2.from_api()
    >>>
    >>> # Simple entity extraction
    >>> results = extractor.extract_entities(
    ...     "Apple Inc. announced new products in California.",
    ...     ["company", "product", "location"]
    ... )
    >>>
    >>> # Batch processing
    >>> texts = ["Text 1...", "Text 2...", "Text 3..."]
    >>> results = extractor.batch_extract_entities(
    ...     texts,
    ...     ["person", "organization", "location"],
    ...     batch_size=8
    ... )
    >>>
    >>> # Complex multi-task extraction
    >>> schema = (extractor.create_schema()
    ...     .entities(["person", "date"])
    ...     .classification("sentiment", ["positive", "negative"])
    ...     .structure("event")
    ...         .field("name", dtype="str")
    ...         .field("location")
    ...         .field("type", choices=["conference", "launch", "meeting"])
    ... )
    >>> results = extractor.extract(text, schema)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache for processed schemas
        self._schema_cache = {}
        # Cache for entity encodings
        self._encoding_cache = {}
    
    @classmethod
    def from_api(
        cls,
        api_key: str = None,
        api_base_url: str = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> 'GLiNER2API':
        """
        Load GLiNER2 from API endpoint instead of local model.
        
        This factory method returns an API-based client that provides the same
        interface as the local model. All inference methods (extract_entities,
        extract_json, classify_text, etc.) work identically.
        
        Parameters
        ----------
        api_key : str, optional
            API authentication key. If not provided, reads from PIONEER_API_KEY
            environment variable.
        api_base_url : str, optional
            Override the default API base URL (https://api.fastino.ai).
            Can also be set via GLINER2_API_BASE_URL environment variable.
        timeout : float, default=30.0
            Request timeout in seconds.
        max_retries : int, default=3
            Maximum number of retries for failed requests.
        
        Returns
        -------
        GLiNER2API
            API-based client with the same interface as GLiNER2.
        
        Raises
        ------
        ValueError
            If no API key is provided and PIONEER_API_KEY is not set.
        
        Examples
        --------
        >>> # Using environment variable for API key
        >>> import os
        >>> os.environ['PIONEER_API_KEY'] = 'your-api-key'
        >>> extractor = GLiNER2.from_api()
        >>>
        >>> # Using explicit API key
        >>> extractor = GLiNER2.from_api(api_key="your-api-key")
        >>>
        >>> # Use exactly like local model
        >>> results = extractor.extract_entities(
        ...     "Apple released iPhone 15 in September 2023.",
        ...     ["company", "product", "date"]
        ... )
        
        Notes
        -----
        The API client supports all the same methods as the local model:
        - extract_entities / batch_extract_entities
        - extract_json / batch_extract_json
        - classify_text / batch_classify_text
        - extract / batch_extract
        - create_schema
        
        The main differences are:
        - No local GPU/model required
        - Requires network connectivity and valid API key
        - Some advanced features (validators, confidence scores) may be limited
        """
        from gliner2.api_client import GLiNER2API
        
        return GLiNER2API(
            api_key=api_key,
            api_base_url=api_base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def create_schema(self) -> Schema:
        """
        Create a new schema for defining extraction tasks.

        This is the starting point for building extraction schemas using the
        fluent API. The schema defines what information to extract from the text.

        Returns
        -------
        Schema
            A new schema instance for chaining extraction tasks.
        """
        return Schema()

    def _get_schema_hash(self, schema_dict: Dict[str, Any]) -> str:
        """Generate a hash for schema caching."""
        return hashlib.md5(json.dumps(schema_dict, sort_keys=True).encode()).hexdigest()

    def _prepare_batch_inputs(
            self,
            input_list: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Prepare batch inputs with proper padding and attention masks.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, List[int]]
            - Padded input_ids tensor
            - Attention mask tensor
            - Original sequence lengths
        """
        # Find max length
        max_length = max(inp["inputs"]["input_ids"].shape[1] for inp in input_list)

        # Prepare tensors
        batch_size = len(input_list)
        device = next(self.encoder.parameters()).device

        padded_input_ids = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
            device=device
        )
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
            device=device
        )
        original_lengths = []

        # Fill tensors
        for i, inp in enumerate(input_list):
            seq_len = inp["inputs"]["input_ids"].shape[1]
            padded_input_ids[i, :seq_len] = inp["inputs"]["input_ids"][0]
            attention_mask[i, :seq_len] = inp["inputs"]["attention_mask"][0]
            original_lengths.append(seq_len)

        return padded_input_ids, attention_mask, original_lengths

    def _batch_encode(
            self,
            prepared_records: List[Dict[str, Any]],
            batch_size: int = 8
    ) -> List[BatchEncodingResult]:
        """
        Batch encode multiple texts through the transformer encoder.

        Parameters
        ----------
        prepared_records : List[Dict[str, Any]]
            List of prepared records with inputs and metadata
        batch_size : int, default=8
            Batch size for encoder forward pass

        Returns
        -------
        List[BatchEncodingResult]
            Encoded representations for each input text
        """
        all_results = []

        with torch.no_grad():
            for batch_start in range(0, len(prepared_records), batch_size):
                batch_end = min(batch_start + batch_size, len(prepared_records))
                batch = prepared_records[batch_start:batch_end]

                # Prepare batch inputs
                input_ids, attention_mask, lengths = self._prepare_batch_inputs(batch)

                # Forward pass through encoder
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Extract embeddings for each item in batch
                for i, record in enumerate(batch):
                    seq_len = lengths[i]
                    token_embeddings = outputs.last_hidden_state[i, :seq_len, :]

                    # Extract special token embeddings
                    embs_result = self.processor.extract_special_token_embeddings_per_schema(
                        token_embeddings.unsqueeze(0),
                        input_ids[i:i + 1, :seq_len],
                        record["mapped_indices"][:seq_len],
                        num_hierarchical_schemas=len(record["schema_tokens_list"])
                    )

                    result = BatchEncodingResult(
                        token_embeddings=embs_result["token_embeddings"],
                        original_lengths=[seq_len],
                        mapped_indices=[record["mapped_indices"]],
                        transformed_data=[record["transformed"]],
                        batch_indices=[batch_start + i]
                    )

                    # Store additional needed data
                    result.embs_per_schema = embs_result["embs_per_schema"]
                    result.full_inputs = record

                    all_results.append(result)

        return all_results

    def _extract_from_encoding(
            self,
            encoding: BatchEncodingResult,
            schema_dict: Dict[str, Any],
            threshold: float,
            field_metadata: Dict[str, Dict],
            entity_metadata: Dict[str, Dict],
            relation_metadata: Dict[str, Dict],
            field_orders: Dict[str, List[str]],
            entity_order: List[str],
            relation_order: List[str],
            include_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Extract information from pre-encoded representation.

        This runs the task-specific layers (classification, span detection, etc.)
        on a single encoded text.
        """
        # Reconstruct outputs structure
        outputs = {
            "text_tokens": encoding.transformed_data[0]["text_tokens"],
            "token_embeddings": encoding.token_embeddings,
            "embs_per_schema": encoding.embs_per_schema,
            "schema_tokens_list": encoding.transformed_data[0]["schema_tokens_list"],
            "task_types": encoding.transformed_data[0]["task_types"],
            "outputs": encoding.transformed_data[0]["outputs"],
            "start_token_idx_to_text_idx": encoding.transformed_data[0].get("start_token_idx_to_text_idx", []),
            "end_token_idx_to_text_idx": encoding.transformed_data[0].get("end_token_idx_to_text_idx", [])
        }

        # Compute span representations
        span_info = self.compute_span_rep(encoding.token_embeddings)

        # Build classification field mapping
        classification_fields = self._build_classification_map(schema_dict)

        # Extract results using existing logic
        results = {}
        record = {"text": encoding.full_inputs["text"], "schema": schema_dict}

        for i, schema_tokens in enumerate(outputs["schema_tokens_list"]):
            if len(schema_tokens) < 4:
                continue

            schema_name = self._get_schema_name(schema_tokens)
            task_type = outputs["task_types"][i]

            if task_type == "classifications":
                self._extract_classification(
                    results, schema_name, schema_dict,
                    outputs["embs_per_schema"][i], schema_tokens
                )
            else:
                self._extract_spans(
                    results, schema_name, i, outputs, span_info, record,
                    threshold, field_metadata, entity_metadata, relation_metadata,
                    field_orders, entity_order, relation_order, classification_fields,
                    include_confidence=include_confidence
                )

        return results

    @torch.no_grad()
    def batch_extract(
            self,
            texts: List[str],
            schemas: Union[Schema, List[Schema], Dict[str, Any], List[Dict[str, Any]]],
            batch_size: int = 8,
            threshold: float = 0.5,
            format_results: bool = True,
            include_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extract information from multiple texts efficiently using batched encoding.

        This method processes multiple texts through the encoder in batches,
        then runs task-specific extraction individually. This provides significant
        speedup compared to processing texts one by one.

        Parameters
        ----------
        texts : List[str]
            List of input texts to process. Empty texts are handled gracefully.
        schemas : Schema, List[Schema], Dict, or List[Dict]
            Either:
            - A single schema for all texts
            - A list of schemas (one per text)
            - A raw schema dict for all texts
            - A list of schema dicts (one per text)
        batch_size : int, default=8
            Number of texts to encode together. Larger values use more memory
            but may be faster. Adjust based on your hardware.
        threshold : float, default=0.5
            Minimum confidence score (0-1) for accepting extracted spans.
        format_results : bool, default=True
            If True, returns clean, formatted results.
            If False, returns raw results with full details.
        include_confidence : bool, default=False
            If True, includes confidence scores in formatted output.

        Returns
        -------
        List[Dict[str, Any]]
            List of extraction results, one per input text, in the same order.

        Examples
        --------
        >>> # Batch entity extraction
        >>> texts = ["Apple released iPhone.", "Google announced Pixel.", "Microsoft unveiled Surface."]
        >>> results = extractor.batch_extract(
        ...     texts,
        ...     extractor.create_schema().entities(["company", "product"]),
        ...     batch_size=8
        ... )

        >>> # Different schemas per text
        >>> schemas = [
        ...     extractor.create_schema().entities(["person", "location"]),
        ...     extractor.create_schema().entities(["company", "product"]),
        ...     extractor.create_schema().classification("sentiment", ["positive", "negative"])
        ... ]
        >>> results = extractor.batch_extract(texts, schemas)

        >>> # High-throughput processing
        >>> large_texts = ["..."] * 1000  # 1000 texts
        >>> results = extractor.batch_extract(
        ...     large_texts,
        ...     schema,
        ...     batch_size=32  # Larger batch for better GPU utilization
        ... )

        Notes
        -----
        - Empty texts return empty results
        - Failed extractions return empty results (not None)
        - Results maintain the same order as input texts
        - Batch size affects memory usage and speed
        - For very long texts, consider smaller batch sizes
        """
        if not texts:
            return []

        # Ensure we're in eval mode
        self.eval()
        self.processor.change_mode(is_training=False)

        # Handle schema input variations
        if isinstance(schemas, list):
            if len(schemas) != len(texts):
                raise ValueError(f"Number of schemas ({len(schemas)}) must match number of texts ({len(texts)})")
            schema_list = schemas
        else:
            # Single schema for all texts
            schema_list = [schemas] * len(texts)

        # Process schemas and prepare records
        prepared_records = []
        schema_metadata = []

        for i, (text, schema) in enumerate(zip(texts, schema_list)):
            # Clean up text
            if not text:
                text = "."
            elif not text.endswith(('.', '!', '?')):
                text += "."

            # Handle different schema types
            if hasattr(schema, 'schema') and hasattr(schema, '_auto_finish'):
                # This is a StructureBuilder
                schema._auto_finish()
                schema = schema.schema

            # Extract schema information
            if isinstance(schema, Schema):
                schema_dict = schema.build()
                metadata = {
                    "field_metadata": schema._field_metadata,
                    "entity_metadata": schema._entity_metadata,
                    "relation_metadata": getattr(schema, '_relation_metadata', {}),
                    "field_orders": schema._field_orders,
                    "entity_order": schema._entity_order,
                    "relation_order": getattr(schema, '_relation_order', [])
                }
            elif isinstance(schema, dict):
                schema_dict = schema
                metadata = {
                    "field_metadata": {},
                    "entity_metadata": {},
                    "relation_metadata": {},
                    "field_orders": {},
                    "entity_order": [],
                    "relation_order": []
                }
            else:
                raise ValueError(f"Invalid schema type at index {i}")

            # Check cache
            schema_hash = self._get_schema_hash(schema_dict)

            # Prepare schema
            self._prepare_schema(schema_dict)

            # Create record
            record = {"text": text, "schema": schema_dict}

            # Transform record
            try:
                transformed = self.processor.transform_single_record(record)

                # Format input
                format_result = self.processor.format_input_with_mapping(
                    transformed["schema_tokens_list"],
                    transformed["text_tokens"]
                )

                prepared = {
                    "text": text,
                    "schema": schema_dict,
                    "schema_hash": schema_hash,
                    "transformed": transformed,
                    "inputs": format_result["inputs"],
                    "mapped_indices": format_result["mapped_indices"],
                    "subword_list": format_result["subword_list"],
                    "schema_tokens_list": transformed["schema_tokens_list"],
                    "batch_index": i
                }

                prepared_records.append(prepared)
                schema_metadata.append(metadata)

            except Exception as e:
                # Handle transformation errors gracefully
                print(f"Warning: Failed to process text at index {i}: {str(e)}")
                prepared_records.append(None)
                schema_metadata.append(None)

        # Batch encode all valid records
        valid_records = [r for r in prepared_records if r is not None]
        if not valid_records:
            return [{}] * len(texts)

        encoded_results = self._batch_encode(valid_records, batch_size)

        # Map encoded results back to original indices
        encoded_by_index = {}
        for enc in encoded_results:
            encoded_by_index[enc.batch_indices[0]] = enc

        # Extract from each encoding
        final_results = []

        for i, (record, metadata) in enumerate(zip(prepared_records, schema_metadata)):
            if record is None or i not in encoded_by_index:
                final_results.append({})
                continue

            try:
                encoding = encoded_by_index[i]

                # Extract using task-specific layers
                raw_results = self._extract_from_encoding(
                    encoding,
                    record["schema"],
                    threshold,
                    metadata["field_metadata"],
                    metadata["entity_metadata"],
                    metadata.get("relation_metadata", {}),
                    metadata["field_orders"],
                    metadata["entity_order"],
                    metadata.get("relation_order", []),
                    include_confidence=include_confidence
                )

                # Format if requested
                if format_results:
                    # Get all requested relation types from schema
                    requested_relations = metadata.get("relation_order", [])
                    # Also check schema dict for relations if not in metadata
                    if not requested_relations and "relations" in record["schema"]:
                        requested_relations = [
                            list(rel.keys())[0] 
                            for rel in record["schema"]["relations"]
                        ]
                    results = self.format_results(
                        raw_results, 
                        include_confidence,
                        requested_relations=requested_relations if requested_relations else None
                    )
                else:
                    results = raw_results

                final_results.append(results)

            except Exception as e:
                print(f"Warning: Extraction failed for text at index {i}: {str(e)}")
                final_results.append({})

        return final_results

    # Batch convenience methods
    def batch_extract_entities(
            self,
            texts: List[str],
            entity_types: Union[List[str], Dict[str, Union[str, Dict]]],
            batch_size: int = 8,
            threshold: float = 0.5,
            format_results: bool = True,
            include_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Batch entity extraction without explicit schema building.

        This is a convenience method for batch processing entity extraction tasks.
        Internally creates schemas with only entity extraction.

        Parameters
        ----------
        texts : List[str]
            List of texts to extract entities from.
        entity_types : List[str] or Dict
            Entity types to extract (see entities for format details).
        batch_size : int, default=8
            Number of texts to process together.
        threshold : float, default=0.5
            Minimum confidence threshold for entity extraction.
        format_results : bool, default=True
            Whether to format the results nicely.
        include_confidence : bool, default=False
            Whether to include confidence scores.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries with "entities" key containing extracted entities.

        Examples
        --------
        >>> texts = [
        ...     "Apple CEO Tim Cook announced new products.",
        ...     "Google's Sundar Pichai spoke at the conference.",
        ...     "Microsoft's Satya Nadella revealed Surface Pro."
        ... ]
        >>> results = extractor.batch_extract_entities(
        ...     texts,
        ...     ["company", "person", "product"],
        ...     batch_size=8
        ... )
        >>> # Returns: [
        >>> #     {'entities': {'company': ['Apple'], 'person': ['Tim Cook']}},
        >>> #     {'entities': {'company': ['Google'], 'person': ['Sundar Pichai']}},
        >>> #     {'entities': {'company': ['Microsoft'], 'person': ['Satya Nadella'], 'product': ['Surface Pro']}}
        >>> # ]
        """
        schema = self.create_schema().entities(entity_types)
        return self.batch_extract(texts, schema, batch_size, threshold, format_results, include_confidence)

    def batch_classify_text(
            self,
            texts: List[str],
            tasks: Dict[str, Union[List[str], Dict[str, Any]]],
            batch_size: int = 8,
            threshold: float = 0.5,
            format_results: bool = True,
            include_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Batch text classification without explicit schema building.

        This is a convenience method for batch classification tasks.

        Parameters
        ----------
        texts : List[str]
            List of texts to classify.
        tasks : Dict[str, Union[List[str], Dict[str, Any]]]
            Classification tasks (see classify_text for format details).
        batch_size : int, default=8
            Number of texts to process together.
        threshold : float, default=0.5
            Confidence threshold.
        format_results : bool, default=True
            Whether to format the results nicely.
        include_confidence : bool, default=False
            Whether to include confidence scores.

        Returns
        -------
        List[Dict[str, Any]]
            List of classification results keyed by task name.

        Examples
        --------
        >>> texts = [
        ...     "This product is amazing! Best purchase ever.",
        ...     "Terrible experience, would not recommend.",
        ...     "It's okay, nothing special."
        ... ]
        >>> results = extractor.batch_classify_text(
        ...     texts,
        ...     {"sentiment": ["positive", "negative", "neutral"]},
        ...     batch_size=8
        ... )
        >>> # Returns: [
        >>> #     {'sentiment': 'positive'},
        >>> #     {'sentiment': 'negative'},
        >>> #     {'sentiment': 'neutral'}
        >>> # ]
        """
        schema = self.create_schema()

        for task_name, task_config in tasks.items():
            if isinstance(task_config, (list, dict)) and "labels" not in task_config:
                schema.classification(task_name, task_config)
            elif isinstance(task_config, dict) and "labels" in task_config:
                config_copy = task_config.copy()
                labels = config_copy.pop("labels")
                schema.classification(task_name, labels, **config_copy)
            else:
                raise ValueError(f"Invalid task config for {task_name}")

        return self.batch_extract(texts, schema, batch_size, threshold, format_results, include_confidence)

    def batch_extract_json(
            self,
            texts: List[str],
            structures: Dict[str, List[str]],
            batch_size: int = 8,
            threshold: float = 0.5,
            format_results: bool = True,
            include_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Batch structured data extraction without explicit schema building.

        This is a convenience method for batch extracting structured data.

        Parameters
        ----------
        texts : List[str]
            List of texts to extract structured data from.
        structures : Dict[str, List[str]]
            Structure definitions (see extract_json for format details).
        batch_size : int, default=8
            Number of texts to process together.
        threshold : float, default=0.5
            Minimum confidence threshold for extraction.
        format_results : bool, default=True
            Whether to format the results nicely.
        include_confidence : bool, default=False
            Whether to include confidence scores.

        Returns
        -------
        List[Dict[str, List[Dict]]]
            List of extracted structures keyed by structure name.

        Examples
        --------
        >>> texts = [
        ...     "iPhone 15 costs $999 with 256GB storage.",
        ...     "Galaxy S24 priced at $899 with 512GB.",
        ...     "Pixel 8 available for $699 with 128GB."
        ... ]
        >>> results = extractor.batch_extract_json(
        ...     texts,
        ...     {
        ...         "product": [
        ...             "name::str",
        ...             "price::str",
        ...             "storage::str::Storage capacity"
        ...         ]
        ...     },
        ...     batch_size=8
        ... )
        >>> # Returns: [
        >>> #     {'product': [{'name': 'iPhone 15', 'price': '$999', 'storage': '256GB'}]},
        >>> #     {'product': [{'name': 'Galaxy S24', 'price': '$899', 'storage': '512GB'}]},
        >>> #     {'product': [{'name': 'Pixel 8', 'price': '$699', 'storage': '128GB'}]}
        >>> # ]
        """
        schema = self.create_schema()

        for parent, fields in structures.items():
            builder = schema.structure(parent)

            for field_spec in fields:
                name, dtype, choices, description = self._parse_field_spec(field_spec)
                builder.field(name, dtype=dtype, choices=choices, description=description)

        return self.batch_extract(texts, schema, batch_size, threshold, format_results, include_confidence)

    # Keep all existing single-text methods unchanged
    @torch.no_grad()
    def extract(
            self,
            text: str,
            schema: Union[Schema, Dict[str, Any]],
            threshold: float = 0.5,
            format_results: bool = True,
            include_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Extract information from text using a schema.

        This is the main extraction method that processes text according to the
        defined schema and returns structured results.

        Parameters
        ----------
        text : str
            The input text to extract information from. Can be a sentence,
            paragraph, or full document. A period is automatically added if
            the text doesn't end with punctuation.
        schema : Schema or Dict[str, Any]
            Either:
            - A Schema instance (recommended)
            - A raw schema dictionary (for advanced use)
        threshold : float, default=0.5
            Minimum confidence score (0-1) for accepting extracted spans.
            Higher values = more precise but may miss some extractions.
            Lower values = more recall but may include false positives.
        format_results : bool, default=True
            If True, returns clean, formatted results.
            If False, returns raw results with full details.
        include_confidence : bool, default=False
            If True, includes confidence scores in formatted output.
            Only applies when format_results=True.

        Returns
        -------
        Dict[str, Any]
            Extraction results organized by task name:
            - Entity results: {"entities": [{"person": ["John"], "location": ["NYC"]}]}
            - Classification: {"sentiment": "positive"} or with confidence
            - Structures: {"product": [{"name": "iPhone", "price": "999"}]}
        """
        # Use batch_extract with a single text
        results = self.batch_extract(
            [text],
            schema,
            batch_size=1,
            threshold=threshold,
            format_results=format_results,
            include_confidence=include_confidence
        )

        # Return the first (and only) result
        return results[0] if results else {}

    def _perform_extraction(
            self,
            record: Dict[str, Any],
            default_threshold: float,
            field_metadata: Dict[str, Dict],
            entity_metadata: Dict[str, Dict],
            field_orders: Dict[str, List[str]],
            entity_order: List[str],
            include_confidence: bool = False
    ) -> Dict[str, Any]:
        """Core extraction pipeline."""
        # Setup model
        self.eval()
        self.processor.change_mode(is_training=False)

        # Prepare schema - record["schema"] is always a dict here
        self._prepare_schema(record["schema"])

        # Build classification field mapping (without modifying schema)
        classification_fields = self._build_classification_map(record["schema"])

        # Process through model pipeline
        outputs = self.processor.process_record(self.encoder, record)

        # Compute span representations
        span_info = self.compute_span_rep(outputs["token_embeddings"])

        # Extract results
        results = {}
        for i, schema_tokens in enumerate(outputs["schema_tokens_list"]):
            if len(schema_tokens) < 4:
                continue

            schema_name = self._get_schema_name(schema_tokens)
            task_type = outputs["task_types"][i]

            if task_type == "classifications":
                self._extract_classification(
                    results, schema_name, record["schema"],
                    outputs["embs_per_schema"][i], schema_tokens
                )
            else:
                self._extract_spans(
                    results, schema_name, i, outputs, span_info, record,
                    default_threshold, field_metadata, entity_metadata, {},
                    field_orders, entity_order, [], classification_fields,
                    include_confidence=include_confidence
                )

        return results

    def _prepare_schema(self, schema: Dict[str, Any]):
        """
        Prepare schema for processing.

        Parameters
        ----------
        schema : Dict[str, Any]
            The schema dictionary (not a Schema instance)
        """
        # Ensure classifications have true_label
        for cls_config in schema.get("classifications", []):
            cls_config.setdefault("true_label", ["N/A"])

    def _build_classification_map(self, schema: Dict[str, Any]) -> Dict[str, List[str]]:
        """Build mapping of classification fields to their choices without modifying schema."""
        mapping = {}
        for struct in schema.get("json_structures", []):
            for parent, fields in struct.items():
                for fname, fval in fields.items():
                    if isinstance(fval, dict) and "choices" in fval:
                        mapping[f"{parent}.{fname}"] = fval["choices"]
        return mapping

    def _get_schema_name(self, schema_tokens: List[str]) -> str:
        """Extract schema name from tokens."""
        return schema_tokens[2].split(" [DESCRIPTION] ")[0]

    def _extract_classification(
            self,
            results: Dict,
            schema_name: str,
            schema: Dict,
            embeddings: List,
            schema_tokens: List[str]
    ):
        """Extract classification results."""
        # Find classification config
        cls_config = next(
            c for c in schema["classifications"]
            if schema_tokens[2].startswith(c["task"])
        )

        # Get model predictions
        schema_emb = torch.stack(embeddings)
        cls_embeds = schema_emb[1:]  # Skip [P] token
        logits = self.classifier(cls_embeds).squeeze(-1)

        # Apply activation function
        activation = cls_config.get("class_act", "auto")
        if activation == "sigmoid":
            probs = torch.sigmoid(logits)
        elif activation == "softmax":
            probs = torch.softmax(logits, dim=-1)
        else:
            # Auto-select based on multi_label
            is_multi = cls_config.get("multi_label", False)
            probs = torch.sigmoid(logits) if is_multi else torch.softmax(logits, dim=-1)

        # Format results
        labels = cls_config["labels"]
        threshold = cls_config.get("cls_threshold", 0.5)

        if cls_config.get("multi_label", False):
            # Multi-label classification
            chosen = [
                (labels[j], probs[j].item())
                for j in range(len(labels))
                if probs[j].item() >= threshold
            ]
            # Fallback to best prediction if none above threshold
            if not chosen:
                best_idx = int(torch.argmax(probs).item())
                chosen = [(labels[best_idx], probs[best_idx].item())]
            results[schema_name] = chosen
        else:
            # Single-label classification
            best_idx = int(torch.argmax(probs).item())
            results[schema_name] = (labels[best_idx], probs[best_idx].item())

    def _extract_spans(
            self,
            results: Dict,
            schema_name: str,
            schema_idx: int,
            outputs: Dict,
            span_info: Dict,
            record: Dict,
            default_threshold: float,
            field_metadata: Dict,
            entity_metadata: Dict,
            relation_metadata: Dict,
            field_orders: Dict[str, List[str]],
            entity_order: List[str],
            relation_order: List[str],
            classification_fields: Dict[str, List[str]],
            include_confidence: bool = False
    ):
        """Extract span-based results (entities, structures, relations)."""
        # Get basic information
        schema_tokens = outputs["schema_tokens_list"][schema_idx]
        field_names = self._get_field_names(schema_tokens)
        embeddings = outputs["embs_per_schema"][schema_idx]

        if not field_names:
            results[schema_name] = [] if schema_name == "entities" else {}
            return

        # Get span predictions
        count, span_scores = self._predict_spans(embeddings, field_names, span_info)

        if count <= 0:
            # For entities, return empty list
            if schema_name == "entities":
                results[schema_name] = []
            elif outputs["task_types"][schema_idx] == "relations":
                # For relations, add empty list so format_results knows this relation type was requested
                results[schema_name] = []
            else:
                # For structures, return empty dict
                results[schema_name] = {}
            return

        # Extract based on schema type
        if schema_name == "entities":
            extracted_results = self._extract_entity_results(
                field_names, span_scores, record, outputs,
                default_threshold, entity_metadata, entity_order,
                include_confidence=include_confidence
            )
            results[schema_name] = extracted_results
        elif outputs["task_types"][schema_idx] == "relations":
            # Relations are extracted differently - they have head and tail fields
            extracted_results = self._extract_relation_results(
                schema_name, field_names, span_scores, count, record, outputs,
                default_threshold, relation_metadata, field_orders,
                include_confidence=include_confidence
            )
            # Always add to results (empty list if no relations found)
            results[schema_name] = extracted_results if extracted_results else []
        else:
            results[schema_name] = self._extract_structure_results(
                schema_name, field_names, span_scores, count, record, outputs,
                default_threshold, field_metadata, field_orders, classification_fields,
                include_confidence=include_confidence
            )

    def _get_field_names(self, schema_tokens: List[str]) -> List[str]:
        """Extract field names from schema tokens."""
        field_names = []
        for i in range(len(schema_tokens) - 1):
            if schema_tokens[i] in ("[E]", "[C]", "[R]"):
                field_names.append(schema_tokens[i + 1])
        return field_names

    def _predict_spans(self, embeddings: List, field_names: List[str], span_info: Dict) -> Tuple[
        int, Optional[torch.Tensor]]:
        """Predict count and span scores."""
        schema_emb = torch.stack(embeddings)

        # Predict how many instances to extract
        count_logits = self.count_pred(schema_emb[0].unsqueeze(0))
        pred_count = int(count_logits.argmax(dim=1).item())

        if pred_count <= 0:
            return 0, None

        # Get span representations
        struct_proj = self.count_embed(schema_emb[1:], pred_count)
        span_scores = torch.sigmoid(
            torch.einsum("lkd,bpd->bplk", span_info["span_rep"], struct_proj)
        )

        return pred_count, span_scores

    def _extract_entity_results(
            self,
            entity_names: List[str],
            span_scores: torch.Tensor,
            record: Dict,
            outputs: Dict,
            default_threshold: float,
            entity_metadata: Dict,
            entity_order: List[str],
            include_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """Extract entity results with per-entity thresholds and preserved order."""
        if span_scores is None:
            return []

        text_len = len(self.processor.tokenize_text(record["text"]))
        # Use first instance for entities
        scores = span_scores[0, :, -text_len:]

        # Use OrderedDict to preserve entity order
        entity_results = OrderedDict()

        # Process entities in the order they were defined
        for entity_name in entity_order if entity_order else entity_names:
            if entity_name not in entity_names:
                continue

            entity_idx = entity_names.index(entity_name)

            # Get entity configuration
            metadata = entity_metadata.get(entity_name, {})
            threshold = metadata.get("threshold")
            if threshold is None:
                threshold = default_threshold
            dtype = metadata.get("dtype", "list")

            # Find spans above threshold
            spans = self._find_valid_spans(
                scores[entity_idx], threshold, record, outputs, text_len
            )

            # Format results
            if dtype == "list":
                entity_results[entity_name] = self._format_span_list(spans, include_confidence)
            else:  # "str"
                if include_confidence and spans:
                    entity_results[entity_name] = spans[0]  # Full tuple
                else:
                    entity_results[entity_name] = spans[0][0] if spans else ""

        return [entity_results] if entity_results else []

    def _extract_structure_results(
            self,
            schema_name: str,
            field_names: List[str],
            span_scores: torch.Tensor,
            count: int,
            record: Dict,
            outputs: Dict,
            default_threshold: float,
            field_metadata: Dict,
            field_orders: Dict[str, List[str]],
            classification_fields: Dict[str, List[str]],
            include_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """Extract structured data results with per-field thresholds and preserved order."""
        if span_scores is None:
            return []

        text_len = len(self.processor.tokenize_text(record["text"]))
        tokens = outputs["text_tokens"]
        instances = []

        # Get the field order for this schema
        ordered_fields = field_orders.get(schema_name, field_names)

        for instance_idx in range(count):
            scores = span_scores[instance_idx, :, -text_len:]

            # Use OrderedDict to preserve field order
            instance_data = OrderedDict()

            # Process fields in the order they were defined
            for field_name in ordered_fields:
                if field_name not in field_names:
                    continue

                field_idx = field_names.index(field_name)

                # Get field configuration
                field_key = f"{schema_name}.{field_name}"
                metadata = field_metadata.get(field_key, {})
                threshold = metadata.get("threshold")
                if threshold is None:
                    threshold = default_threshold
                dtype = metadata.get("dtype", "list")
                validators = metadata.get("validators")

                # Check if this is a classification field
                if field_key in classification_fields:
                    # Handle classification field
                    choices = classification_fields[field_key]
                    field_scores = span_scores[instance_idx, field_idx, :-text_len]

                    # Debug: Check if we have any scores at all
                    if field_scores.numel() == 0:
                        continue

                    if dtype == "list":
                        # Multi-choice selection with duplicate prevention
                        selected = []
                        selected_set = set()
                        for choice in choices:
                            if choice in selected_set:
                                continue
                            choice_indices = self._find_choice_indices(choice, tokens[:-text_len])
                            if not choice_indices:
                                # Try with [selection] prefix
                                choice_indices = self._find_choice_indices(f"[selection]{choice}", tokens[:-text_len])
                            for start_idx in choice_indices:
                                if start_idx >= 0 and start_idx < field_scores.shape[0] and field_scores[
                                    start_idx, 0].item() >= threshold:
                                    selected.append(choice)
                                    selected_set.add(choice)
                                    break
                        instance_data[field_name] = selected
                    else:  # str
                        # Single choice selection
                        best_choice = None
                        best_score = -1.0

                        # Try each choice
                        for choice in choices:
                            choice_indices = self._find_choice_indices(choice, tokens[:-text_len])
                            if not choice_indices:
                                # Try with [selection] prefix
                                choice_indices = self._find_choice_indices(f"[selection]{choice}", tokens[:-text_len])

                            for start_idx in choice_indices:
                                if start_idx >= 0 and start_idx < field_scores.shape[0]:
                                    score = field_scores[start_idx, 0].item()
                                    if score > best_score:
                                        best_score = score
                                        best_choice = choice

                        # Apply threshold check
                        if best_choice and best_score >= threshold:
                            instance_data[field_name] = best_choice
                        elif threshold == 0.0 and best_choice:
                            # Special case: threshold is 0, so include best match regardless
                            instance_data[field_name] = best_choice
                        else:
                            # Set None for str fields with no valid choice
                            instance_data[field_name] = None
                else:
                    # Regular span extraction
                    spans = self._find_valid_spans(
                        scores[field_idx], threshold, record, outputs, text_len
                    )

                    # Apply regex validators if present
                    if validators:
                        spans = [
                            (text, conf, start, end)
                            for text, conf, start, end in spans
                            if all(v.validate(text) for v in validators)
                        ]

                    # Format and store results
                    if spans:
                        if dtype == "list":
                            instance_data[field_name] = self._format_span_list(spans, include_confidence)
                        else:  # "str"
                            if include_confidence:
                                instance_data[field_name] = spans[0]  # Full tuple
                            else:
                                instance_data[field_name] = spans[0][0]
                    else:
                        # Set None for str fields, empty list for list fields
                        if dtype == "list":
                            instance_data[field_name] = []
                        else:  # "str"
                            instance_data[field_name] = None

            # Only add instance if it has at least one non-empty field
            has_content = False
            for value in instance_data.values():
                if value is not None and value != []:
                    has_content = True
                    break

            if has_content:
                instances.append(instance_data)

        return instances

    def _extract_relation_results(
            self,
            schema_name: str,
            field_names: List[str],
            span_scores: torch.Tensor,
            count: int,
            record: Dict,
            outputs: Dict,
            default_threshold: float,
            relation_metadata: Dict,
            field_orders: Dict[str, List[str]],
            include_confidence: bool = False,
    ) -> List:
        """
        Extract relation results.

        Default (include_confidence=False):
            (head_text, tail_text)
        With include_confidence=True:
            (
                (head_text, head_conf, head_start, head_end),
                (tail_text, tail_conf, tail_start, tail_end),
            )
        where start/end are token indices in the text token sequence.
        """
        if span_scores is None:
            return []

        text_len = len(self.processor.tokenize_text(record["text"]))
        instances: List = []

        # Get relation threshold from metadata
        relation_threshold = default_threshold
        if schema_name in relation_metadata:
            rel_meta = relation_metadata[schema_name]
            if rel_meta.get("threshold") is not None:
                relation_threshold = rel_meta["threshold"]

        # Get the field order for this relation (typically ["head", "tail"])
        ordered_fields = field_orders.get(schema_name, field_names)

        for instance_idx in range(count):
            scores = span_scores[instance_idx, :, -text_len:]

            # Extract head and tail spans
            extracted_values: List[Optional[str]] = []
            span_meta: List[Optional[Tuple[str, float, int, int]]] = []

            # Process fields in the order they were defined (head, tail)
            for field_name in ordered_fields:
                if field_name not in field_names:
                    continue

                field_idx = field_names.index(field_name)

                # Extract spans for this field (head or tail) using relation threshold
                spans = self._find_valid_spans(
                    scores[field_idx], relation_threshold, record, outputs, text_len
                )

                # For relations, we want the best span
                if spans:
                    # spans[0] = (text_span, conf, start, end)
                    text_span, conf, start, end = spans[0]
                    extracted_values.append(text_span)
                    span_meta.append((text_span, conf, start, end))
                else:
                    extracted_values.append(None)
                    span_meta.append(None)

            # Only add instance if both source and target are present
            # Order is maintained: first is source (head), second is target (tail)
            if (
                len(extracted_values) == 2
                and extracted_values[0]
                and extracted_values[1]
                and span_meta[0] is not None
                and span_meta[1] is not None
            ):
                if include_confidence:
                    # ((head_text, head_conf, head_start, head_end),
                    #  (tail_text, tail_conf, tail_start, tail_end))
                    instances.append(
                        (
                            span_meta[0],
                            span_meta[1],
                        )
                    )
                else:
                    instances.append((extracted_values[0], extracted_values[1]))

        return instances



    def _find_choice_indices(self, choice: str, tokens: List[str]) -> List[int]:
        """Find all starting indices where a choice appears in tokens."""
        indices = []

        # For classification choices, they often appear as single tokens in the prefix
        # First check for exact match
        choice_lower = choice.lower()
        for i, token in enumerate(tokens):
            if token.lower() == choice_lower:
                indices.append(i)

        # If no exact match found, try substring matching for the prefix area
        # This helps when choices are part of constructed tokens like "[selection]Italian"
        if not indices:
            for i, token in enumerate(tokens):
                if choice_lower in token.lower():
                    indices.append(i)

        return indices

    def _find_valid_spans(
            self,
            field_scores: torch.Tensor,
            threshold: float,
            record: Dict,
            outputs: Dict,
            text_len: int
    ) -> List[Tuple[str, float, int, int]]:
        """Find all valid spans above threshold with position information."""
        # Ensure threshold has a value
        if threshold is None:
            threshold = 0.5

        valid_positions = torch.where(field_scores >= threshold)
        starts, widths = valid_positions

        spans = []
        # Check if position mappings exist
        has_char_mapping = (
                "start_token_idx_to_text_idx" in outputs and
                "end_token_idx_to_text_idx" in outputs
        )

        for start, width in zip(starts.tolist(), widths.tolist()):
            end = start + width + 1

            # Check bounds
            if 0 <= start < text_len and end <= text_len:
                if has_char_mapping:
                    try:
                        char_start = outputs["start_token_idx_to_text_idx"][start]
                        char_end = outputs["end_token_idx_to_text_idx"][end - 1]
                        text_span = record["text"][char_start:char_end].strip()
                    except (KeyError, IndexError):
                        # Fallback to token-based
                        text_tokens = outputs["text_tokens"][-text_len:]
                        span_tokens = text_tokens[start:end]
                        text_span = " ".join(span_tokens).strip()
                else:
                    # Use token-based extraction
                    text_tokens = outputs["text_tokens"][-text_len:]
                    span_tokens = text_tokens[start:end]
                    text_span = " ".join(span_tokens).strip()

                confidence = field_scores[start, width].item()

                if text_span:  # Only add non-empty spans
                    spans.append((text_span, confidence, start, end))

        return spans

    def _format_span_list(
        self, 
        spans: List[Tuple[str, float, int, int]], 
        include_confidence: bool = False
    ) -> Union[List[str], List[Tuple[str, float, int, int]]]:
        """Format spans into a clean list with proper overlap detection and greedy selection.
        
        Parameters
        ----------
        spans : List[Tuple[str, float, int, int]]
            List of (text, confidence, start, end) tuples.
        include_confidence : bool, default=False
            If True, returns full tuples with confidence and positions.
            If False, returns only text strings.
            
        Returns
        -------
        Union[List[str], List[Tuple[str, float, int, int]]]
            List of text strings (default) or full tuples if include_confidence=True.
        """
        if not spans:
            return []

        # Apply greedy selection for non-overlapping spans
        # Sort by confidence descending
        sorted_spans = sorted(spans, key=lambda x: x[1], reverse=True)

        selected = []

        for text, confidence, start, end in sorted_spans:
            # Check for overlap with already selected spans
            overlap = False
            for sel_text, sel_conf, sel_start, sel_end in selected:
                # Check position-based overlap
                if not (end <= sel_start or start >= sel_end):
                    overlap = True
                    break

            if not overlap:
                selected.append((text, confidence, start, end))

        # Return full tuples if confidence requested, otherwise just text
        if include_confidence:
            return selected
        return [text for text, _, _, _ in selected]

    def format_results(
        self, 
        results: Dict[str, Any], 
        include_confidence: bool = False,
        requested_relations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Format raw extraction results into clean, user-friendly output.

        This method processes raw extraction results to:
        - Remove duplicates and empty values
        - Simplify confidence score presentation
        - Ensure consistent structure
        - Preserve field ordering
        - Group relations under 'relation_extraction' key

        Parameters
        ----------
        results : Dict[str, Any]
            Raw extraction results from extract or similar.
        include_confidence : bool, default=False
            If True, includes confidence scores in the output.
            If False, returns only the extracted values.
        requested_relations : List[str], optional
            List of all relation types that were requested in the schema.
            If provided, ensures all requested relations appear in output, even if empty.

        Returns
        -------
        Dict[str, Any]
            Formatted results with clean structure.
        """
        formatted = {}
        relations = {}
        found_relation_keys = set()

        for key, value in results.items():
            if isinstance(value, list):
                # Check if this is a relation result (list of tuples or empty list from relations)
                if len(value) > 0 and isinstance(value[0], tuple) and len(value[0]) == 2:
                    # This is a relation result with tuples - collect under relations
                    relations[key] = value
                    found_relation_keys.add(key)
                elif len(value) == 0 and key not in formatted:
                    # Empty list - could be empty relation or empty entity list
                    # Check if it's a relation by checking requested_relations
                    if requested_relations and key in requested_relations:
                        # This is an empty relation
                        relations[key] = []
                        found_relation_keys.add(key)
                    elif key == "entities":
                        # Empty entity list
                        formatted[key] = {}
                    else:
                        # Other empty lists
                        formatted[key] = value
                # Handle entity results (list of dicts)
                elif len(value) > 0 and isinstance(value[0], dict):
                    if key == "entities":
                        # Format entities specially
                        formatted[key] = self._format_entities(value[0], include_confidence)
                    else:
                        # Structures - format each instance
                        formatted[key] = [
                            self._format_structure_instance(instance, include_confidence)
                            for instance in value
                        ]
                elif isinstance(value[0], tuple):
                    # Multi-label classification results
                    if include_confidence:
                        formatted[key] = [{"label": label, "confidence": conf} for label, conf in value]
                    else:
                        formatted[key] = [label for label, _ in value]
                else:
                    # Simple list results
                    formatted[key] = value
            elif isinstance(value, tuple):
                # Single-label classification result
                label, conf = value
                if include_confidence:
                    formatted[key] = {"label": label, "confidence": conf}
                else:
                    formatted[key] = label
            elif isinstance(value, dict):
                # Single structure instance
                formatted[key] = self._format_structure_instance(value, include_confidence)
            else:
                # Direct value
                formatted[key] = value

        # Build relation_extraction dictionary
        relation_extraction = {}
        
        # Add found relations (both non-empty and empty)
        for rel_type, rel_values in relations.items():
            relation_extraction[rel_type] = rel_values
        
        # Add empty lists for requested relations that weren't found in results
        if requested_relations:
            for rel_type in requested_relations:
                if rel_type not in relation_extraction:
                    relation_extraction[rel_type] = []
        
        # Add relation_extraction if we have any relations (found, empty, or requested)
        if relation_extraction:
            formatted["relation_extraction"] = relation_extraction

        return formatted

    def _format_entities(self, entities: Dict[str, Any], include_confidence: bool) -> Dict[str, Any]:
        """Format entity extraction results."""
        formatted = {}
        for entity_type, spans in entities.items():
            if isinstance(spans, list):
                # Remove empty strings and duplicates while preserving order
                unique_spans = []
                seen = set()
                for span in spans:
                    # Handle both tuple format (text, conf, start, end) and plain string
                    if isinstance(span, tuple):
                        text, conf, start, end = span
                        if text and text.lower() not in seen:
                            seen.add(text.lower())
                            if include_confidence:
                                unique_spans.append({"text": text, "confidence": conf, "start": start, "end": end})
                            else:
                                unique_spans.append(text)
                    else:
                        # Plain string
                        if span and span.lower() not in seen:
                            seen.add(span.lower())
                            unique_spans.append(span)
                formatted[entity_type] = unique_spans
            elif isinstance(spans, tuple):
                # Single span with confidence (str type with include_confidence=True)
                text, conf, start, end = spans
                if include_confidence:
                    formatted[entity_type] = {"text": text, "confidence": conf, "start": start, "end": end} if text else None
                else:
                    formatted[entity_type] = text if text else None
            else:
                # Single span (str type)
                formatted[entity_type] = spans if spans else None
        return formatted

    def _format_structure_instance(self, instance: Dict[str, Any], include_confidence: bool) -> Dict[str, Any]:
        """Format a single structure instance."""
        formatted = {}
        for field, value in instance.items():
            if isinstance(value, list):
                # Remove empty strings and duplicates
                unique_values = []
                seen = set()
                for v in value:
                    # Handle both tuple format (text, conf, start, end) and plain string
                    if isinstance(v, tuple):
                        text, conf, start, end = v
                        if text and text.lower() not in seen:
                            seen.add(text.lower())
                            if include_confidence:
                                unique_values.append({"text": text, "confidence": conf, "start": start, "end": end})
                            else:
                                unique_values.append(text)
                    else:
                        if v and v.lower() not in seen:
                            seen.add(v.lower())
                            unique_values.append(v)
                formatted[field] = unique_values
            elif isinstance(value, tuple):
                # Single value with confidence (str type with include_confidence=True)
                text, conf, start, end = value
                if include_confidence:
                    formatted[field] = {"text": text, "confidence": conf, "start": start, "end": end} if text else None
                else:
                    formatted[field] = text if text else None
            elif value:  # Non-empty string
                formatted[field] = value
            else:
                formatted[field] = None
        return formatted

    # Convenience methods for common use cases
    def extract_entities(
            self,
            text: str,
            entity_types: Union[List[str], Dict[str, Union[str, Dict]]],
            threshold: float = 0.5,
            format_results: bool = True,
            include_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Quick entity extraction without explicit schema building.

        This is a convenience method for simple entity extraction tasks.
        Internally creates a schema with only entity extraction.

        Parameters
        ----------
        text : str
            The input text to extract entities from.
        entity_types : List[str] or Dict
            Entity types to extract (see entities for format details).
        threshold : float, default=0.5
            Minimum confidence threshold for entity extraction.
        format_results : bool, default=True
            Whether to format the results nicely.
        include_confidence : bool, default=False
            Whether to include confidence scores.

        Returns
        -------
        Dict[str, Any]
            Dictionary with "entities" key containing extracted entities.
        """
        schema = self.create_schema().entities(entity_types)
        return self.extract(text, schema, threshold, format_results, include_confidence)

    def classify_text(
            self,
            text: str,
            tasks: Dict[str, Union[List[str], Dict[str, Any]]],
            threshold: float = 0.5,
            format_results: bool = True,
            include_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Quick text classification without explicit schema building.

        This is a convenience method for classification tasks. Supports both
        single-label and multi-label classification with optional configurations.

        Parameters
        ----------
        text : str
            The text to classify.
        tasks : Dict[str, Union[List[str], Dict[str, Any]]]
            Classification tasks where keys are task names and values are either:
            - List[str]: Simple list of labels
            - Dict with "labels" key and optional config (multi_label, cls_threshold)
        threshold : float, default=0.5
            Confidence threshold (mainly used for other extraction types).
        format_results : bool, default=True
            Whether to format the results nicely.
        include_confidence : bool, default=False
            Whether to include confidence scores.

        Returns
        -------
        Dict[str, Any]
            Classification results keyed by task name.
        """
        results = self.batch_classify_text(
            [text],
            tasks,
            batch_size=1,
            threshold=threshold,
            format_results=format_results,
            include_confidence=include_confidence
        )

        # Return the first (and only) result
        return results[0] if results else {}

    def extract_json(
            self,
            text: str,
            structures: Dict[str, List[str]],
            threshold: float = 0.5,
            format_results: bool = True,
            include_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Quick structured data extraction without explicit schema building.

        This is a convenience method for extracting structured data that follows
        a pattern with multiple fields. Extracts all instances found.

        Parameters
        ----------
        text : str
            The text to extract structured data from.
        structures : Dict[str, List[str]]
            Dictionary where keys are structure names and values are lists of
            field specifications. Field specs support flexible formats:
            - Simple: "field_name" (extracts as list)
            - With description: "field_name::description text here"
            - Typed: "field_name::str" or "field_name::list"
            - Choices: "field_name::[option1|option2|option3]" (defaults to str)
            - Choices with type: "field_name::[opt1|opt2]::list"
            - Full spec: "field_name::[opt1|opt2]::str::description here"
        threshold : float, default=0.5
            Minimum confidence threshold for extraction.
        format_results : bool, default=True
            Whether to format the results nicely.
        include_confidence : bool, default=False
            Whether to include confidence scores.

        Returns
        -------
        Dict[str, List[Dict]]
            Extracted structures keyed by structure name. Each structure
            contains a list of instances found.
        """
        results = self.batch_extract_json(
            [text],
            structures,
            batch_size=1,
            threshold=threshold,
            format_results=format_results,
            include_confidence=include_confidence
        )

        # Return the first (and only) result
        return results[0] if results else {}

    def extract_relations(
            self,
            text: str,
            relation_types: Union[str, List[str], Dict[str, Union[str, Dict]]],
            threshold: float = 0.5,
            format_results: bool = True,
            include_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Quick relation extraction without explicit schema building.

        This is a convenience method for simple relation extraction tasks.
        Internally creates a schema with only relation extraction.

        Parameters
        ----------
        text : str
            The input text to extract relations from.
        relation_types : str, List[str], or Dict
            Relation types to extract (see relations for format details).
        threshold : float, default=0.5
            Minimum confidence threshold for relation extraction.
        format_results : bool, default=True
            Whether to format the results nicely.
        include_confidence : bool, default=False
            Whether to include confidence scores.

        Returns
        -------
        Dict[str, Any]
            Dictionary with "relation_extraction" key containing all extracted relations.
            Relations are grouped by type, with each instance as a tuple (source, target).
            Format: {"relation_extraction": {"relation_name": [(source, target), ...]}}

        Examples
        --------
        >>> results = extractor.extract_relations(
        ...     "John works for Apple Inc. Mary works for Google.",
        ...     ["works_for"]
        ... )
        >>> # Returns: {"relation_extraction": {"works_for": [("John", "Apple Inc."), ("Mary", "Google")]}}
        """
        schema = self.create_schema().relations(relation_types)
        return self.extract(text, schema, threshold, format_results, include_confidence)

    def batch_extract_relations(
            self,
            texts: List[str],
            relation_types: Union[str, List[str], Dict[str, Union[str, Dict]]],
            batch_size: int = 8,
            threshold: float = 0.5,
            format_results: bool = True,
            include_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Batch relation extraction without explicit schema building.

        This is a convenience method for batch processing relation extraction tasks.
        Internally creates schemas with only relation extraction.

        Parameters
        ----------
        texts : List[str]
            List of texts to extract relations from.
        relation_types : str, List[str], or Dict
            Relation types to extract (see relations for format details).
        batch_size : int, default=8
            Number of texts to process together.
        threshold : float, default=0.5
            Minimum confidence threshold for relation extraction.
        format_results : bool, default=True
            Whether to format the results nicely.
        include_confidence : bool, default=False
            Whether to include confidence scores.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries with "relation_extraction" key containing all extracted relations.
            Relations are grouped by type, with each instance as a tuple (source, target).
            Format: [{"relation_extraction": {"relation_name": [(source, target), ...]}}]

        Examples
        --------
        >>> texts = [
        ...     "John works for Apple Inc.",
        ...     "Mary located in New York.",
        ...     "Bob works for Microsoft."
        ... ]
        >>> results = extractor.batch_extract_relations(
        ...     texts,
        ...     ["works_for", "located_in"],
        ...     batch_size=8
        ... )
        >>> # Returns: [
        >>> #     {"relation_extraction": {"works_for": [("John", "Apple Inc.")]}},
        >>> #     {"relation_extraction": {"located_in": [("Mary", "New York")]}},
        >>> #     {"relation_extraction": {"works_for": [("Bob", "Microsoft")]}}
        >>> # ]
        """
        schema = self.create_schema().relations(relation_types)
        return self.batch_extract(texts, schema, batch_size, threshold, format_results, include_confidence)

    def _parse_field_spec(self, field_spec: str) -> Tuple[str, str, Optional[List[str]], Optional[str]]:
        """
        Parse field specification string into components.

        This internal method parses the field specification format used in
        the extract_json convenience method. It supports flexible ordering and
        all combinations of type, choices, and descriptions.

        Parameters
        ----------
        field_spec : str
            Field specification in various formats.

        Returns
        -------
        Tuple[str, str, Optional[List[str]], Optional[str]]
            A tuple of (field_name, dtype, choices, description).
        """
        # Split by :: but keep at most 3 parts (name, middle, description)
        parts = field_spec.split('::', 2)

        # Field name is always first
        field_name = parts[0]

        # Defaults
        dtype = "list"
        choices = None
        description = None

        # No additional parts - just field name
        if len(parts) == 1:
            return field_name, dtype, choices, description

        # Process remaining parts
        remaining_parts = parts[1:]

        # Helper to check if a part is a type
        is_type = lambda s: s in ['str', 'list']

        # Helper to extract choices
        def parse_choices(s):
            if s.startswith('[') and s.endswith(']'):
                return [c.strip() for c in s[1:-1].split('|')]
            return None

        # Process based on number of remaining parts
        if len(remaining_parts) == 1:
            # One part after name: type, choices, or description
            part = remaining_parts[0]

            if is_type(part):
                dtype = part
            elif choices_parsed := parse_choices(part):
                choices = choices_parsed
                dtype = "str"  # Default for choices
            else:
                description = part

        elif len(remaining_parts) == 2:
            # Two parts after name
            first, second = remaining_parts

            # Check all combinations
            if choices_parsed := parse_choices(first):
                # First is choices
                choices = choices_parsed
                dtype = "str"  # Default for choices

                if is_type(second):
                    dtype = second
                else:
                    description = second

            elif is_type(first):
                # First is type, second is description
                dtype = first
                description = second

            else:
                # First part is neither choices nor type, treat as description
                # Join both parts back as description
                description = '::'.join(remaining_parts)

        return field_name, dtype, choices, description


# Legacy aliases for backwards compatibility
BuilderExtractor = GLiNER2
SchemaBuilder = Schema
JsonStructBuilder = StructureBuilder
