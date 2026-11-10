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

from gliner2.inference.chunking import merge_chunk_results, split_text_into_chunks

logger = logging.getLogger(__name__)


class GLiNER2APIError(Exception):
    """Base exception for GLiNER2 API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data
        # Populated when a partitioned batch fails after earlier partitions
        # succeeded. Callers can persist completed work before deciding to retry.
        self.partial_results: List[Any] = []
        self.failed_range: Optional[tuple[int, int]] = None


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
                     FASTINO_API_KEY environment variable.
            api_base_url: Override the default API base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        
        Raises:
            ValueError: If no API key is provided and FASTINO_API_KEY is not set.
        """
        # Read API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("FASTINO_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key must be provided either as an argument or via "
                    "FASTINO_API_KEY environment variable"
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

    @staticmethod
    def _safe_json(response: requests.Response) -> Optional[Dict[str, Any]]:
        """Parse response JSON safely, returning None for empty/invalid payloads."""
        if not response.content:
            return None

        try:
            data = response.json()
        except ValueError:
            logger.debug(
                "Response body was not valid JSON (status=%s)",
                response.status_code,
            )
            return None

        return data if isinstance(data, dict) else None
    
    def _make_request(
        self,
        task: str,
        text: Union[str, List[str]],
        schema: Union[List[str], Dict],
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
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
            include_spans: Whether to include character-level start/end positions
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
            "include_spans": include_spans,
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
                error_data = self._safe_json(response)
                error_msg = (
                    error_data.get("detail", "Invalid or expired API key")
                    if error_data else "Invalid or expired API key"
                )
                raise AuthenticationError(
                    error_msg,
                    status_code=response.status_code,
                    response_data=error_data,
                )
            
            elif response.status_code in (400, 422):
                error_data = self._safe_json(response)
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
                error_data = self._safe_json(response)
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
                error_data = self._safe_json(response)
                error_msg = (
                    error_data.get("detail", f"Request failed with status {response.status_code}")
                    if error_data else f"Request failed with status {response.status_code}"
                )
                raise GLiNER2APIError(
                    error_msg,
                    status_code=response.status_code,
                    response_data=error_data,
                )

            if not response.content:
                raise GLiNER2APIError(
                    "Empty response body from API",
                    status_code=response.status_code,
                )

            try:
                data = response.json()
            except ValueError as e:
                raise GLiNER2APIError(
                    "Invalid JSON response from API",
                    status_code=response.status_code,
                ) from e

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

    # Shared request plumbing -------------------------------------------------

    def _task_call(self, task, text, schema, threshold, format_results,
                   include_confidence, include_spans):
        """Dispatch a task through the shared HTTP request implementation."""
        return self._make_request(
            task=task,
            text=text,
            schema=schema,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            format_results=format_results,
        )

    @staticmethod
    def _validate_batch_size(batch_size: int) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

    @staticmethod
    def _coerce_batch_result(result: Any, expected: int) -> List[Any]:
        """Validate one server reply against its client-side partition."""
        if isinstance(result, list):
            if len(result) != expected:
                raise GLiNER2APIError(
                    "API returned "
                    f"{len(result)} results for a partition of {expected} texts"
                )
            return result
        if expected == 1:
            return [result]
        raise GLiNER2APIError(
            "API returned a non-list result for a partition of "
            f"{expected} texts"
        )

    @staticmethod
    def _mark_partial_failure(
        error: GLiNER2APIError,
        partial_results: List[Any],
        start: int,
        end: int,
    ) -> None:
        error.partial_results = list(partial_results)
        error.failed_range = (start, end)

    def _partitioned_task_call(
        self,
        task: str,
        texts: List[str],
        schema: Union[List[str], Dict],
        batch_size: int,
        threshold: float,
        format_results: bool,
        include_confidence: bool,
        include_spans: bool,
    ) -> List[Any]:
        """Send ordered client-side partitions without changing wire fields."""
        if not texts:
            return []
        self._validate_batch_size(batch_size)
        results: List[Any] = []
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            partition = texts[start:end]
            try:
                response = self._task_call(
                    task,
                    partition,
                    schema,
                    threshold,
                    format_results,
                    include_confidence,
                    include_spans,
                )
                results.extend(
                    self._coerce_batch_result(response, len(partition))
                )
            except GLiNER2APIError as error:
                self._mark_partial_failure(error, results, start, end)
                raise
        return results

    @staticmethod
    def _entity_schema(entity_types):
        """Normalize entity descriptions to the established wire schema."""
        return list(entity_types.keys()) if isinstance(entity_types, dict) else entity_types

    # Entity Extraction Methods ----------------------------------------------

    def extract_entities(
        self,
        text: str,
        entity_types: Union[List[str], Dict[str, Union[str, Dict]]],
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False
    ) -> Dict[str, Any]:
        """Extract entities from text."""
        result = self._task_call(
            "extract_entities", text, self._entity_schema(entity_types),
            threshold, format_results, include_confidence, include_spans,
        )
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
        include_confidence: bool = False,
        include_spans: bool = False
    ) -> List[Dict[str, Any]]:
        """Batch extract entities from multiple texts."""
        return self._partitioned_task_call(
            "extract_entities",
            texts,
            self._entity_schema(entity_types),
            batch_size,
            threshold,
            format_results,
            include_confidence,
            include_spans,
        )

    def extract_entities_long(
        self,
        text: str,
        entity_types: Union[List[str], Dict[str, Union[str, Dict]]],
        threshold: float = 0.5,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        batch_size: int = 8,
        num_workers: int = 0,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        overlap_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract entities from a long document using local chunk merging."""
        return self.batch_extract_entities_long(
            [text],
            entity_types,
            batch_size=batch_size,
            threshold=threshold,
            num_workers=num_workers,
            format_results=format_results,
            include_confidence=include_confidence,
            include_spans=include_spans,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            overlap_policy=overlap_policy,
        )[0]

    def batch_extract_entities_long(
        self,
        texts: List[str],
        entity_types: Union[List[str], Dict[str, Union[str, Dict]]],
        batch_size: int = 8,
        threshold: float = 0.5,
        num_workers: int = 0,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        overlap_policy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not format_results:
            raise ValueError(
                "batch_extract_entities_long currently requires format_results=True"
            )
        if not texts:
            return []
        self._validate_batch_size(batch_size)
        if num_workers:
            warnings.warn(
                "num_workers is ignored by the synchronous API client",
                UserWarning,
                stacklevel=2,
            )

        def request(chunk_texts, _):
            return self.batch_extract_entities(
                chunk_texts,
                entity_types,
                batch_size=batch_size,
                threshold=threshold,
                format_results=True,
                include_confidence=True,
                include_spans=True,
            )

        return self._run_long_batch(
            texts,
            request,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_confidence=include_confidence,
            include_spans=include_spans,
            overlap_policy=overlap_policy,
            normalize_result=self._normalize_entity_chunk_result,
        )

    # Text Classification Methods --------------------------------------------

    def classify_text(
        self,
        text: str,
        tasks: Dict[str, Union[List[str], Dict[str, Any]]],
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False
    ) -> Dict[str, Any]:
        """Classify text into categories."""
        if len(tasks) == 1:
            task_name = next(iter(tasks))
            task_config = tasks[task_name]
            categories = task_config["labels"] if (
                isinstance(task_config, dict) and "labels" in task_config
            ) else task_config
            result = self._task_call(
                "classify_text", text, {"categories": categories},
                threshold, format_results, include_confidence, include_spans,
            )
            if format_results and isinstance(result, dict) and task_name not in result:
                return {task_name: result.get("classification", result)}
            return result
        return self._task_call(
            "schema", text, {"classifications": tasks},
            threshold, format_results, include_confidence, include_spans,
        )

    def batch_classify_text(
        self,
        texts: List[str],
        tasks: Dict[str, Union[List[str], Dict[str, Any]]],
        batch_size: int = 8,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False
    ) -> List[Dict[str, Any]]:
        """Batch classify multiple texts."""
        return self._partitioned_task_call(
            "schema",
            texts,
            {"classifications": tasks},
            batch_size,
            threshold,
            format_results,
            include_confidence,
            include_spans,
        )

    def classify_text_long(
        self,
        text: str,
        tasks: Dict[str, Union[List[str], Dict[str, Any]]],
        threshold: float = 0.5,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        batch_size: int = 8,
        num_workers: int = 0,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        overlap_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.batch_classify_text_long(
            [text],
            tasks,
            batch_size=batch_size,
            threshold=threshold,
            num_workers=num_workers,
            format_results=format_results,
            include_confidence=include_confidence,
            include_spans=include_spans,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            overlap_policy=overlap_policy,
        )[0]

    def batch_classify_text_long(
        self,
        texts: List[str],
        tasks: Dict[str, Union[List[str], Dict[str, Any]]],
        batch_size: int = 8,
        threshold: float = 0.5,
        num_workers: int = 0,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        overlap_policy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not format_results:
            raise ValueError(
                "batch_classify_text_long currently requires format_results=True"
            )
        if not texts:
            return []
        self._validate_batch_size(batch_size)
        if num_workers:
            warnings.warn(
                "num_workers is ignored by the synchronous API client",
                UserWarning,
                stacklevel=2,
            )

        def request(chunk_texts, _):
            return self.batch_classify_text(
                chunk_texts,
                tasks,
                batch_size=batch_size,
                threshold=threshold,
                format_results=True,
                include_confidence=True,
                include_spans=True,
            )

        return self._run_long_batch(
            texts,
            request,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_confidence=include_confidence,
            include_spans=include_spans,
            overlap_policy=overlap_policy,
        )

    # JSON Extraction Methods -------------------------------------------------

    def extract_json(
        self,
        text: str,
        structures: Dict[str, List[str]],
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False
    ) -> Dict[str, Any]:
        """Extract structured data from text."""
        return self._task_call(
            "extract_json", text, structures,
            threshold, format_results, include_confidence, include_spans,
        )

    def batch_extract_json(
        self,
        texts: List[str],
        structures: Dict[str, List[str]],
        batch_size: int = 8,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False
    ) -> List[Dict[str, Any]]:
        """Batch extract structured data from multiple texts."""
        return self._partitioned_task_call(
            "extract_json",
            texts,
            structures,
            batch_size,
            threshold,
            format_results,
            include_confidence,
            include_spans,
        )

    def extract_json_long(
        self,
        text: str,
        structures: Dict[str, List[str]],
        threshold: float = 0.5,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        batch_size: int = 8,
        num_workers: int = 0,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        overlap_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.batch_extract_json_long(
            [text],
            structures,
            batch_size=batch_size,
            threshold=threshold,
            num_workers=num_workers,
            format_results=format_results,
            include_confidence=include_confidence,
            include_spans=include_spans,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            overlap_policy=overlap_policy,
        )[0]

    def batch_extract_json_long(
        self,
        texts: List[str],
        structures: Dict[str, List[str]],
        batch_size: int = 8,
        threshold: float = 0.5,
        num_workers: int = 0,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        overlap_policy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not format_results:
            raise ValueError(
                "batch_extract_json_long currently requires format_results=True"
            )
        if not texts:
            return []
        self._validate_batch_size(batch_size)
        if num_workers:
            warnings.warn(
                "num_workers is ignored by the synchronous API client",
                UserWarning,
                stacklevel=2,
            )

        def request(chunk_texts, _):
            return self.batch_extract_json(
                chunk_texts,
                structures,
                batch_size=batch_size,
                threshold=threshold,
                format_results=True,
                include_confidence=True,
                include_spans=True,
            )

        return self._run_long_batch(
            texts,
            request,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_confidence=include_confidence,
            include_spans=include_spans,
            overlap_policy=overlap_policy,
        )

    # Relation Extraction Methods --------------------------------------------

    def extract_relations(
        self,
        text: str,
        relation_types: Union[str, List[str], Dict[str, Union[str, Dict]]],
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False
    ) -> Dict[str, Any]:
        """Extract relations between entities from text."""
        schema = self.create_schema().relations(relation_types).build()
        return self._task_call(
            "schema", text, schema,
            threshold, format_results, include_confidence, include_spans,
        )

    def batch_extract_relations(
        self,
        texts: List[str],
        relation_types: Union[str, List[str], Dict[str, Union[str, Dict]]],
        batch_size: int = 8,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False
    ) -> List[Dict[str, Any]]:
        """Batch extract relations from multiple texts."""
        schema = self.create_schema().relations(relation_types).build()
        return self._partitioned_task_call(
            "schema",
            texts,
            schema,
            batch_size,
            threshold,
            format_results,
            include_confidence,
            include_spans,
        )

    def extract_relations_long(
        self,
        text: str,
        relation_types: Union[
            str, List[str], Dict[str, Union[str, Dict]]
        ],
        threshold: float = 0.5,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        batch_size: int = 8,
        num_workers: int = 0,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        overlap_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.batch_extract_relations_long(
            [text],
            relation_types,
            batch_size=batch_size,
            threshold=threshold,
            num_workers=num_workers,
            format_results=format_results,
            include_confidence=include_confidence,
            include_spans=include_spans,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            overlap_policy=overlap_policy,
        )[0]

    def batch_extract_relations_long(
        self,
        texts: List[str],
        relation_types: Union[
            str, List[str], Dict[str, Union[str, Dict]]
        ],
        batch_size: int = 8,
        threshold: float = 0.5,
        num_workers: int = 0,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        overlap_policy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not format_results:
            raise ValueError(
                "batch_extract_relations_long currently requires format_results=True"
            )
        if not texts:
            return []
        self._validate_batch_size(batch_size)
        if num_workers:
            warnings.warn(
                "num_workers is ignored by the synchronous API client",
                UserWarning,
                stacklevel=2,
            )

        def request(chunk_texts, _):
            return self.batch_extract_relations(
                chunk_texts,
                relation_types,
                batch_size=batch_size,
                threshold=threshold,
                format_results=True,
                include_confidence=True,
                include_spans=True,
            )

        return self._run_long_batch(
            texts,
            request,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_confidence=include_confidence,
            include_spans=include_spans,
            overlap_policy=overlap_policy,
        )

    # General Extraction Methods ---------------------------------------------

    @staticmethod
    def _schema_dict(schema):
        """Accept the established builder protocol or a plain dict."""
        return schema.build() if hasattr(schema, "build") else schema

    def extract(
        self,
        text: str,
        schema: Union[SchemaAPI, Dict[str, Any]],
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False
    ) -> Dict[str, Any]:
        """Extract information from text using a schema."""
        schema_dict = self._schema_dict(schema)
        if not any(key in schema_dict for key in [
            "entities", "classifications", "structures", "relations"
        ]):
            raise ValueError("Schema must contain at least one extraction task")
        return self._task_call(
            "schema", text, schema_dict,
            threshold, format_results, include_confidence, include_spans,
        )

    def batch_extract(
        self,
        texts: List[str],
        schemas: Union[SchemaAPI, List[SchemaAPI], Dict[str, Any], List[Dict[str, Any]]],
        batch_size: int = 8,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False
    ) -> List[Dict[str, Any]]:
        """Extract information from multiple texts."""
        if not texts:
            return []
        self._validate_batch_size(batch_size)
        if isinstance(schemas, list):
            if len(schemas) != len(texts):
                raise ValueError(
                    f"Number of schemas ({len(schemas)}) must match number of texts ({len(texts)})"
                )
            warnings.warn(
                "Multi-schema batch (different schemas per text) is not natively supported by the API. "
                "Each text will be processed individually, which may be slower than single-schema batch. "
                "For better performance, use the same schema for all texts.",
                UserWarning,
                stacklevel=2,
            )
            results: List[Dict[str, Any]] = []
            for index, (text, schema) in enumerate(zip(texts, schemas)):
                try:
                    results.append(
                        self.extract(
                            text, schema, threshold,
                            include_confidence=include_confidence,
                            include_spans=include_spans,
                            format_results=format_results,
                        )
                    )
                except GLiNER2APIError as error:
                    self._mark_partial_failure(
                        error, results, index, index + 1
                    )
                    raise
            return results
        return self._partitioned_task_call(
            "schema",
            texts,
            self._schema_dict(schemas),
            batch_size,
            threshold,
            format_results,
            include_confidence,
            include_spans,
        )

    # Long-document extraction ----------------------------------------------

    @staticmethod
    def _scalar_entity_labels(schema: Any) -> set[str]:
        schema_dict = GLiNER2API._schema_dict(schema)
        entities = schema_dict.get("entities") if isinstance(schema_dict, dict) else None
        labels: set[str] = set()
        global_scalar = (
            isinstance(schema_dict, dict)
            and schema_dict.get("entity_dtype", "list") != "list"
        )
        if isinstance(entities, str) and global_scalar:
            labels.add(entities)
        elif isinstance(entities, list) and global_scalar:
            labels.update(str(label) for label in entities)
        elif isinstance(entities, dict):
            if schema_dict.get("entity_dtype", "list") != "list":
                labels.update(str(label) for label in entities)
            labels.update(
                str(label)
                for label, config in entities.items()
                if isinstance(config, dict)
                and config.get("dtype", "list") != "list"
            )
        return labels

    @staticmethod
    def _normalize_entity_chunk_result(result: Any) -> Dict[str, Any]:
        if isinstance(result, dict) and "entities" in result:
            return result
        if isinstance(result, dict):
            return {"entities": result}
        raise GLiNER2APIError(
            "entity extraction returned a non-object chunk result"
        )

    def _merge_long_batch(
        self,
        texts: List[str],
        chunks_by_document: List[List[Any]],
        chunk_results: List[Any],
        *,
        include_confidence: bool,
        include_spans: bool,
        overlap_policy: Optional[str],
        scalar_labels_by_document: Optional[List[set[str]]] = None,
        normalize_result=None,
    ) -> List[Dict[str, Any]]:
        normalizer = normalize_result or (lambda result: result)
        normalized = [normalizer(result) for result in chunk_results]
        merged: List[Dict[str, Any]] = []
        offset = 0
        scalar_labels_by_document = scalar_labels_by_document or [
            set() for _ in texts
        ]
        for text, chunks, scalar_labels in zip(
            texts, chunks_by_document, scalar_labels_by_document
        ):
            end = offset + len(chunks)
            merged.append(
                merge_chunk_results(
                    text,
                    chunks,
                    normalized[offset:end],
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                    scalar_entity_labels=scalar_labels,
                    overlap_policy=overlap_policy,
                )
            )
            offset = end
        return merged

    def _run_long_batch(
        self,
        texts: List[str],
        request_chunks,
        *,
        chunk_size: int,
        chunk_overlap: int,
        include_confidence: bool,
        include_spans: bool,
        overlap_policy: Optional[str],
        scalar_labels_by_document: Optional[List[set[str]]] = None,
        normalize_result=None,
    ) -> List[Dict[str, Any]]:
        chunks_by_document = [
            split_text_into_chunks(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for text in texts
        ]
        chunk_texts = [
            chunk.text
            for chunks in chunks_by_document
            for chunk in chunks
        ]
        try:
            chunk_results = request_chunks(chunk_texts, chunks_by_document)
        except GLiNER2APIError as error:
            # Promote complete chunk prefixes into complete document results.
            # Keep the original chunk range for diagnostics and expose the
            # document range at the public long-batch layer.
            error.failed_chunk_range = error.failed_range
            completed_chunks = list(error.partial_results)
            completed_documents = 0
            consumed = 0
            for chunks in chunks_by_document:
                if consumed + len(chunks) > len(completed_chunks):
                    break
                consumed += len(chunks)
                completed_documents += 1
            if completed_documents:
                error.partial_results = self._merge_long_batch(
                    texts[:completed_documents],
                    chunks_by_document[:completed_documents],
                    completed_chunks[:consumed],
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                    overlap_policy=overlap_policy,
                    scalar_labels_by_document=(
                        scalar_labels_by_document[:completed_documents]
                        if scalar_labels_by_document is not None else None
                    ),
                    normalize_result=normalize_result,
                )
            else:
                error.partial_results = []
            error.failed_range = (
                completed_documents,
                min(completed_documents + 1, len(texts)),
            )
            raise
        return self._merge_long_batch(
            texts,
            chunks_by_document,
            chunk_results,
            include_confidence=include_confidence,
            include_spans=include_spans,
            overlap_policy=overlap_policy,
            scalar_labels_by_document=scalar_labels_by_document,
            normalize_result=normalize_result,
        )

    def extract_long(
        self,
        text: str,
        schema: Union[SchemaAPI, Dict[str, Any]],
        threshold: float = 0.5,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        batch_size: int = 8,
        num_workers: int = 0,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        overlap_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract a long document using local chunking and offset merging."""
        return self.batch_extract_long(
            [text],
            schema,
            batch_size=batch_size,
            threshold=threshold,
            num_workers=num_workers,
            format_results=format_results,
            include_confidence=include_confidence,
            include_spans=include_spans,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            overlap_policy=overlap_policy,
        )[0]

    def batch_extract_long(
        self,
        texts: List[str],
        schemas: Union[
            SchemaAPI,
            List[SchemaAPI],
            Dict[str, Any],
            List[Dict[str, Any]],
        ],
        batch_size: int = 8,
        threshold: float = 0.5,
        num_workers: int = 0,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        overlap_policy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Batch extract long documents without extending the server protocol."""
        if not format_results:
            raise ValueError("batch_extract_long currently requires format_results=True")
        if not texts:
            return []
        self._validate_batch_size(batch_size)
        if num_workers:
            warnings.warn(
                "num_workers is ignored by the synchronous API client",
                UserWarning,
                stacklevel=2,
            )
        if isinstance(schemas, list):
            if len(schemas) != len(texts):
                raise ValueError(
                    f"Number of schemas ({len(schemas)}) must match number of texts ({len(texts)})"
                )
            schema_list = schemas
        else:
            schema_list = [schemas] * len(texts)

        def request(chunk_texts, chunks_by_document):
            if isinstance(schemas, list):
                chunk_schemas = [
                    schema
                    for schema, chunks in zip(schema_list, chunks_by_document)
                    for _ in chunks
                ]
                return self.batch_extract(
                    chunk_texts,
                    chunk_schemas,
                    batch_size=batch_size,
                    threshold=threshold,
                    format_results=True,
                    include_confidence=True,
                    include_spans=True,
                )
            return self.batch_extract(
                chunk_texts,
                schemas,
                batch_size=batch_size,
                threshold=threshold,
                format_results=True,
                include_confidence=True,
                include_spans=True,
            )

        return self._run_long_batch(
            texts,
            request,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_confidence=include_confidence,
            include_spans=include_spans,
            overlap_policy=overlap_policy,
            scalar_labels_by_document=[
                self._scalar_entity_labels(schema) for schema in schema_list
            ],
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

