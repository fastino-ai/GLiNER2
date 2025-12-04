"""
GLiNER2 API Client

This module provides an API-based wrapper for GLiNER2 that mirrors the local
model interface. It allows seamless switching between local and API-based
inference.

Usage:
    >>> from gliner2 import GLiNER2
    >>> 
    >>> # Load from API (uses environment variable for API key)
    >>> extractor = GLiNER2.from_api()
    >>> 
    >>> # Use exactly like local model
    >>> results = extractor.extract_entities(
    ...     "Apple released iPhone 15 in September 2023.",
    ...     ["company", "product", "date"]
    ... )
"""

from __future__ import annotations

import os
import logging
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union, Literal
from urllib.parse import urljoin
from urllib3.util import Retry
import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)


class GLiNER2APIError(Exception):
    """Base exception for GLiNER2 API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class AuthenticationError(GLiNER2APIError):
    """Raised when API key is invalid or expired."""
    pass


class ValidationError(GLiNER2APIError):
    """Raised when request data is invalid."""
    pass


class ServerError(GLiNER2APIError):
    """Raised when server encounters an error."""
    pass


class StructureBuilderAPI:
    """
    Builder for structured data schemas for API-based extraction.
    
    This mirrors the interface of StructureBuilder from the local model.
    """
    
    def __init__(self, schema: 'SchemaAPI', parent: str):
        self.schema = schema
        self.parent = parent
        self.fields = OrderedDict()
        self.field_order = []
        self._finished = False
    
    def field(
        self,
        name: str,
        dtype: Literal["str", "list"] = "list",
        choices: Optional[List[str]] = None,
        description: Optional[str] = None,
        threshold: Optional[float] = None,
        validators: Optional[List] = None
    ) -> 'StructureBuilderAPI':
        """Add a field to the structured data."""
        # Warn if validators are used (not supported in API mode)
        if validators:
            warnings.warn(
                f"Field '{name}': RegexValidator is not supported in API mode. "
                "Validators will be ignored. Use local model for regex-based filtering.",
                UserWarning,
                stacklevel=2
            )
        
        self.fields[name] = {
            "dtype": dtype,
            "choices": choices,
            "description": description,
            "threshold": threshold
        }
        self.field_order.append(name)
        return self
    
    def _auto_finish(self):
        """Automatically finish this structure when needed."""
        if not self._finished:
            # Convert fields to API format
            # Use dict format if any field has threshold or choices (advanced features)
            # Otherwise use simple string format for backwards compatibility
            field_specs = []
            for name in self.field_order:
                config = self.fields[name]
                
                # Check if advanced features are used
                has_threshold = config.get('threshold') is not None
                has_choices = config.get('choices') is not None
                
                if has_threshold or has_choices:
                    # Use dict format for advanced features
                    field_dict = {"name": name, "dtype": config['dtype']}
                    if config.get('description'):
                        field_dict["description"] = config['description']
                    if has_threshold:
                        field_dict["threshold"] = config['threshold']
                    if has_choices:
                        field_dict["choices"] = config['choices']
                    field_specs.append(field_dict)
                else:
                    # Use simple string format: "name::type::description"
                    spec = f"{name}::{config['dtype']}"
                    if config.get('description'):
                        spec += f"::{config['description']}"
                    field_specs.append(spec)
            
            self.schema._structures[self.parent] = field_specs
            self._finished = True
    
    def __getattr__(self, name):
        """Auto-finish when any schema method is called."""
        if hasattr(self.schema, name):
            self._auto_finish()
            return getattr(self.schema, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class SchemaAPI:
    """Schema builder for API-based extraction tasks."""
    
    def __init__(self):
        self._entities = None
        self._entity_dtype = "list"
        self._entity_threshold = None
        self._classifications = {}
        self._structures = {}
        self._relations = None
        self._relation_threshold = None
        self._active_structure_builder = None
    
    def entities(
        self,
        entity_types: Union[str, List[str], Dict[str, Union[str, Dict]]],
        dtype: Literal["str", "list"] = "list",
        threshold: Optional[float] = None
    ) -> 'SchemaAPI':
        """Add entity extraction task."""
        if self._active_structure_builder:
            self._active_structure_builder._auto_finish()
            self._active_structure_builder = None
        
        # Normalize to list or dict
        if isinstance(entity_types, str):
            self._entities = [entity_types]
        elif isinstance(entity_types, list):
            self._entities = entity_types
        elif isinstance(entity_types, dict):
            self._entities = entity_types
        
        self._entity_dtype = dtype
        self._entity_threshold = threshold
        return self
    
    def classification(
        self,
        task: str,
        labels: Union[List[str], Dict[str, str]],
        multi_label: bool = False,
        cls_threshold: float = 0.5,
        **kwargs
    ) -> 'SchemaAPI':
        """Add a text classification task."""
        if self._active_structure_builder:
            self._active_structure_builder._auto_finish()
            self._active_structure_builder = None
        
        # Parse labels
        if isinstance(labels, dict):
            label_names = list(labels.keys())
        else:
            label_names = labels
        
        self._classifications[task] = {
            "labels": label_names,
            "multi_label": multi_label,
            "cls_threshold": cls_threshold
        }
        return self
    
    def structure(self, name: str) -> StructureBuilderAPI:
        """Start building a structured data schema."""
        if self._active_structure_builder:
            self._active_structure_builder._auto_finish()
        
        self._active_structure_builder = StructureBuilderAPI(self, name)
        return self._active_structure_builder
    
    def relations(
        self,
        relation_types: Union[str, List[str], Dict[str, Union[str, Dict]]],
        threshold: Optional[float] = None
    ) -> 'SchemaAPI':
        """
        Add relation extraction task.
        
        Args:
            relation_types: Relation types to extract. Can be:
                - str: Single relation type
                - List[str]: Multiple relation types  
                - Dict[str, str]: Relation types with descriptions
                - Dict[str, Dict]: Relation types with full configuration
            threshold: Default confidence threshold for relations.
        
        Returns:
            Self for method chaining.
        """
        if self._active_structure_builder:
            self._active_structure_builder._auto_finish()
            self._active_structure_builder = None
        
        # Normalize to list or dict
        if isinstance(relation_types, str):
            self._relations = [relation_types]
        elif isinstance(relation_types, list):
            self._relations = relation_types
        elif isinstance(relation_types, dict):
            self._relations = relation_types
        
        self._relation_threshold = threshold
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the schema for API request."""
        if self._active_structure_builder:
            self._active_structure_builder._auto_finish()
            self._active_structure_builder = None
        
        schema = {}
        
        if self._entities is not None:
            schema["entities"] = self._entities
            schema["entity_dtype"] = self._entity_dtype
            if self._entity_threshold is not None:
                schema["entity_threshold"] = self._entity_threshold
        
        if self._classifications:
            schema["classifications"] = self._classifications
        
        if self._structures:
            schema["structures"] = self._structures
        
        if self._relations is not None:
            schema["relations"] = self._relations
            if self._relation_threshold is not None:
                schema["relation_threshold"] = self._relation_threshold
        
        return schema


class GLiNER2API:
    """
    API-based GLiNER2 client that mirrors the local model interface.
    
    This class provides the same methods as GLiNER2 but makes HTTP requests
    to the API endpoint instead of running local inference.
    
    Attributes:
        api_key: API authentication key
        base_url: API base URL
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries for failed requests
    """
    
    DEFAULT_BASE_URL = "https://api.fastino.ai"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize the GLiNER2 API client.
        
        Args:
            api_key: API authentication key. If not provided, reads from
                     PIONEER_API_KEY environment variable.
            api_base_url: Override the default API base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        
        Raises:
            ValueError: If no API key is provided and PIONEER_API_KEY is not set.
        """
        # Read API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("PIONEER_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key must be provided either as an argument or via "
                    "PIONEER_API_KEY environment variable"
                )
        
        self.api_key = api_key
        self.base_url = api_base_url or os.environ.get(
            "GLINER2_API_BASE_URL", self.DEFAULT_BASE_URL
        )
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Setup HTTP session with retry logic
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        })
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # 1s, 2s, 4s backoff
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        logger.debug(f"Initialized GLiNER2API for {self.base_url}")
    
    def _make_request(
        self,
        task: str,
        text: Union[str, List[str]],
        schema: Union[List[str], Dict],
        threshold: float = 0.5,
        include_confidence: bool = False,
        format_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the GLiNER-2 API.
        
        Args:
            task: Task type (extract_entities, classify_text, extract_json, schema)
            text: Text to process (string or list for batch)
            schema: Schema for extraction
            threshold: Confidence threshold
            include_confidence: Whether to include confidence scores in results
            format_results: Whether to format results (False for raw extraction data)
        
        Returns:
            API response result
        
        Raises:
            GLiNER2APIError: If request fails
        """
        # Ensure base_url ends with / for proper joining
        base = self.base_url.rstrip('/') + '/'
        url = urljoin(base, "gliner-2")
        
        payload = {
            "task": task,
            "text": text,
            "schema": schema,
            "threshold": threshold,
            "include_confidence": include_confidence,
            "format_results": format_results,
        }
        
        logger.debug(f"Making POST request to {url}")
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            
            logger.debug(f"Response status: {response.status_code}")
            
            # Handle different error codes
            if response.status_code == 401:
                error_data = response.json() if response.content else None
                error_msg = (
                    error_data.get("detail", "Invalid or expired API key")
                    if error_data else "Invalid or expired API key"
                )
                raise AuthenticationError(error_msg, response_data=error_data)
            
            elif response.status_code in (400, 422):
                error_data = response.json() if response.content else None
                error_msg = (
                    error_data.get("detail", "Request validation failed")
                    if error_data else "Request validation failed"
                )
                raise ValidationError(
                    error_msg,
                    status_code=response.status_code,
                    response_data=error_data,
                )
            
            elif response.status_code >= 500:
                error_data = response.json() if response.content else None
                error_msg = (
                    error_data.get("detail", "Server error occurred")
                    if error_data else "Server error occurred"
                )
                raise ServerError(
                    error_msg,
                    status_code=response.status_code,
                    response_data=error_data,
                )
            
            elif not response.ok:
                error_data = response.json() if response.content else None
                error_msg = (
                    error_data.get("detail", f"Request failed with status {response.status_code}")
                    if error_data else f"Request failed with status {response.status_code}"
                )
                raise GLiNER2APIError(
                    error_msg,
                    status_code=response.status_code,
                    response_data=error_data,
                )
            
            data = response.json()
            return data.get("result", data)
        
        except requests.exceptions.Timeout:
            raise GLiNER2APIError(f"Request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise GLiNER2APIError(f"Connection error: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise GLiNER2APIError(f"Request failed: {str(e)}")
    
    def create_schema(self) -> SchemaAPI:
        """Create a new schema for defining extraction tasks."""
        return SchemaAPI()
    
    # -------------------------------------------------------------------------
    # Entity Extraction Methods
    # -------------------------------------------------------------------------
    
    def extract_entities(
        self,
        text: str,
        entity_types: Union[List[str], Dict[str, Union[str, Dict]]],
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Extract entities from text.
        
        Args:
            text: Input text to extract entities from.
            entity_types: List of entity types or dict with descriptions.
            threshold: Minimum confidence threshold.
            format_results: Whether to format results. If False, returns raw extraction data.
            include_confidence: Whether to include confidence scores in results.
        
        Returns:
            Dictionary with "entities" key containing extracted entities.
            If include_confidence=True, entity values include confidence scores.
            If format_results=False, returns raw extraction data with positions.
        """
        # Normalize entity types to list
        if isinstance(entity_types, dict):
            entities = list(entity_types.keys())
        else:
            entities = entity_types
        
        result = self._make_request(
            task="extract_entities",
            text=text,
            schema=entities,
            threshold=threshold,
            include_confidence=include_confidence,
            format_results=format_results,
        )
        
        # Wrap result in expected format if needed (only for formatted results)
        if format_results and isinstance(result, dict) and "entities" not in result:
            return {"entities": result}
        return result
    
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
        Batch extract entities from multiple texts.
        
        Args:
            texts: List of input texts.
            entity_types: List of entity types or dict with descriptions.
            batch_size: Batch size (used by API for optimization).
            threshold: Minimum confidence threshold.
            format_results: Whether to format results. If False, returns raw extraction data.
            include_confidence: Whether to include confidence scores.
        
        Returns:
            List of dictionaries with "entities" key.
            If include_confidence=True, entity values include confidence scores.
            If format_results=False, returns raw extraction data with positions.
        """
        # Normalize entity types to list
        if isinstance(entity_types, dict):
            entities = list(entity_types.keys())
        else:
            entities = entity_types
        
        result = self._make_request(
            task="extract_entities",
            text=texts,
            schema=entities,
            threshold=threshold,
            include_confidence=include_confidence,
            format_results=format_results,
        )
        
        # Ensure result is a list
        if isinstance(result, dict):
            return [result]
        return result
    
    # -------------------------------------------------------------------------
    # Text Classification Methods
    # -------------------------------------------------------------------------
    
    def classify_text(
        self,
        text: str,
        tasks: Dict[str, Union[List[str], Dict[str, Any]]],
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Classify text into categories.
        
        Args:
            text: Text to classify.
            tasks: Classification tasks where keys are task names.
            threshold: Confidence threshold.
            format_results: Whether to format results. If False, returns raw extraction data.
            include_confidence: Whether to include confidence scores.
        
        Returns:
            Classification results keyed by task name.
            If include_confidence=True, results include confidence scores.
            If format_results=False, returns raw extraction data.
        """
        # Convert tasks to API format
        # For classify_text task, schema should be {"categories": [...]}
        # But for multi-task, we need to use the schema task
        if len(tasks) == 1:
            # Single task - use classify_text endpoint
            task_name = list(tasks.keys())[0]
            task_config = tasks[task_name]
            
            if isinstance(task_config, dict) and "labels" in task_config:
                categories = task_config["labels"]
            else:
                categories = task_config
            
            result = self._make_request(
                task="classify_text",
                text=text,
                schema={"categories": categories},
                threshold=threshold,
                include_confidence=include_confidence,
                format_results=format_results,
            )
            
            # Wrap result with task name (only for formatted results)
            if format_results and isinstance(result, dict) and task_name not in result:
                return {task_name: result.get("classification", result)}
            return result
        else:
            # Multiple tasks - use schema endpoint
            schema = {"classifications": tasks}
            result = self._make_request(
                task="schema",
                text=text,
                schema=schema,
                threshold=threshold,
                include_confidence=include_confidence,
                format_results=format_results,
            )
            return result
    
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
        Batch classify multiple texts.
        
        Args:
            texts: List of texts to classify.
            tasks: Classification tasks.
            batch_size: Batch size.
            threshold: Confidence threshold.
            format_results: Whether to format results. If False, returns raw extraction data.
            include_confidence: Whether to include confidence scores.
        
        Returns:
            List of classification results.
            If include_confidence=True, results include confidence scores.
            If format_results=False, returns raw extraction data.
        """
        # Use schema task for batch classification
        schema = {"classifications": tasks}
        result = self._make_request(
            task="schema",
            text=texts,
            schema=schema,
            threshold=threshold,
            include_confidence=include_confidence,
            format_results=format_results,
        )
        
        if isinstance(result, dict):
            return [result]
        return result
    
    # -------------------------------------------------------------------------
    # JSON Extraction Methods
    # -------------------------------------------------------------------------
    
    def extract_json(
        self,
        text: str,
        structures: Dict[str, List[str]],
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Extract structured data from text.
        
        Args:
            text: Text to extract data from.
            structures: Structure definitions with field specs.
            threshold: Minimum confidence threshold.
            format_results: Whether to format results. If False, returns raw extraction data.
            include_confidence: Whether to include confidence scores.
        
        Returns:
            Extracted structures keyed by structure name.
            If include_confidence=True, field values include confidence scores.
            If format_results=False, returns raw extraction data with positions.
        """
        result = self._make_request(
            task="extract_json",
            text=text,
            schema=structures,
            threshold=threshold,
            include_confidence=include_confidence,
            format_results=format_results,
        )
        return result
    
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
        Batch extract structured data from multiple texts.
        
        Args:
            texts: List of texts.
            structures: Structure definitions.
            batch_size: Batch size.
            threshold: Confidence threshold.
            format_results: Whether to format results. If False, returns raw extraction data.
            include_confidence: Whether to include confidence scores.
        
        Returns:
            List of extracted structures.
            If include_confidence=True, field values include confidence scores.
            If format_results=False, returns raw extraction data with positions.
        """
        result = self._make_request(
            task="extract_json",
            text=texts,
            schema=structures,
            threshold=threshold,
            include_confidence=include_confidence,
            format_results=format_results,
        )
        
        if isinstance(result, dict):
            return [result]
        return result
    
    # -------------------------------------------------------------------------
    # Relation Extraction Methods
    # -------------------------------------------------------------------------
    
    def extract_relations(
        self,
        text: str,
        relation_types: Union[str, List[str], Dict[str, Union[str, Dict]]],
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Extract relations between entities from text.
        
        Args:
            text: Input text to extract relations from.
            relation_types: Relation types to extract. Can be:
                - str: Single relation type
                - List[str]: Multiple relation types
                - Dict[str, str]: Relation types with descriptions
                - Dict[str, Dict]: Relation types with full configuration
            threshold: Minimum confidence threshold.
            format_results: Whether to format results. If False, returns raw extraction data.
            include_confidence: Whether to include confidence scores in results.
        
        Returns:
            Dictionary with "relation_extraction" key containing extracted relations.
            Relations are grouped by type with tuples (source, target).
            Format: {"relation_extraction": {"relation_name": [("source", "target"), ...]}}
        """
        # Build schema with relations
        schema = self.create_schema().relations(relation_types).build()
        
        result = self._make_request(
            task="schema",
            text=text,
            schema=schema,
            threshold=threshold,
            include_confidence=include_confidence,
            format_results=format_results,
        )
        
        return result
    
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
        Batch extract relations from multiple texts.
        
        Args:
            texts: List of input texts.
            relation_types: Relation types to extract.
            batch_size: Batch size (used by API for optimization).
            threshold: Minimum confidence threshold.
            format_results: Whether to format results.
            include_confidence: Whether to include confidence scores.
        
        Returns:
            List of dictionaries with "relation_extraction" key.
            Format: [{"relation_extraction": {"relation_name": [("source", "target"), ...]}}]
        """
        # Build schema with relations
        schema = self.create_schema().relations(relation_types).build()
        
        result = self._make_request(
            task="schema",
            text=texts,
            schema=schema,
            threshold=threshold,
            include_confidence=include_confidence,
            format_results=format_results,
        )
        
        # Ensure result is a list
        if isinstance(result, dict):
            return [result]
        return result
    
    # -------------------------------------------------------------------------
    # General Extraction Methods
    # -------------------------------------------------------------------------
    
    def extract(
        self,
        text: str,
        schema: Union[SchemaAPI, Dict[str, Any]],
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Extract information from text using a schema.
        
        Args:
            text: Input text to extract from.
            schema: Schema defining what to extract.
            threshold: Minimum confidence threshold.
            format_results: Whether to format results. If False, returns raw extraction data.
            include_confidence: Whether to include confidence scores.
        
        Returns:
            Extraction results organized by task name.
            If include_confidence=True, values include confidence scores.
            If format_results=False, returns raw extraction data with positions.
        """
        # Build schema dict if needed
        if isinstance(schema, SchemaAPI):
            schema_dict = schema.build()
        elif hasattr(schema, 'build'):
            schema_dict = schema.build()
        else:
            schema_dict = schema
        
        # Determine task type based on schema contents
        has_entities = "entities" in schema_dict
        has_classifications = "classifications" in schema_dict
        has_structures = "structures" in schema_dict
        has_relations = "relations" in schema_dict
        
        # Use schema task for multi-task extraction
        if sum([has_entities, has_classifications, has_structures, has_relations]) > 1:
            return self._make_request(
                task="schema",
                text=text,
                schema=schema_dict,
                threshold=threshold,
                include_confidence=include_confidence,
                format_results=format_results,
            )
        
        # Single task - use specific endpoint or schema task
        if has_entities:
            return self.extract_entities(
                text,
                schema_dict["entities"],
                threshold=threshold,
                include_confidence=include_confidence,
                format_results=format_results,
            )
        elif has_classifications:
            return self.classify_text(
                text,
                schema_dict["classifications"],
                threshold=threshold,
                include_confidence=include_confidence,
                format_results=format_results,
            )
        elif has_structures:
            return self.extract_json(
                text,
                schema_dict["structures"],
                threshold=threshold,
                include_confidence=include_confidence,
                format_results=format_results,
            )
        elif has_relations:
            # Relations use the schema task
            return self._make_request(
                task="schema",
                text=text,
                schema=schema_dict,
                threshold=threshold,
                include_confidence=include_confidence,
                format_results=format_results,
            )
        else:
            raise ValueError("Schema must contain at least one extraction task")
    
    def batch_extract(
        self,
        texts: List[str],
        schemas: Union[SchemaAPI, List[SchemaAPI], Dict[str, Any], List[Dict[str, Any]]],
        batch_size: int = 8,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extract information from multiple texts.
        
        Args:
            texts: List of input texts.
            schemas: Single schema for all texts or list of schemas.
            batch_size: Batch size.
            threshold: Confidence threshold.
            format_results: Whether to format results. If False, returns raw extraction data.
            include_confidence: Whether to include confidence scores.
        
        Returns:
            List of extraction results.
            If include_confidence=True, values include confidence scores.
            If format_results=False, returns raw extraction data with positions.
        """
        if not texts:
            return []
        
        # Handle schema variations
        if isinstance(schemas, list):
            if len(schemas) != len(texts):
                raise ValueError(
                    f"Number of schemas ({len(schemas)}) must match number of texts ({len(texts)})"
                )
            # Warn user about multi-schema batch limitation
            warnings.warn(
                "Multi-schema batch (different schemas per text) is not natively supported by the API. "
                "Each text will be processed individually, which may be slower than single-schema batch. "
                "For better performance, use the same schema for all texts.",
                UserWarning,
                stacklevel=2
            )
            # Process each text with its schema individually
            results = []
            for text, schema in zip(texts, schemas):
                results.append(self.extract(text, schema, threshold, include_confidence=include_confidence, format_results=format_results))
            return results
        
        # Single schema for all texts
        if isinstance(schemas, SchemaAPI):
            schema_dict = schemas.build()
        elif hasattr(schemas, 'build'):
            schema_dict = schemas.build()
        else:
            schema_dict = schemas
        
        return self._make_request(
            task="schema",
            text=texts,
            schema=schema_dict,
            threshold=threshold,
            include_confidence=include_confidence,
            format_results=format_results,
        )
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

