"""
Pydantic models for validating schema input from JSON/dict.

This module provides validation models for creating GLiNER2 schemas
from JSON or dictionary inputs. Every model forbids unknown keys, so a
misspelled or unsupported option raises instead of being dropped.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class _StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldInput(_StrictInput):
    """Validates a single structure field.

    Args:
        name: Field name
        dtype: Data type - 'str' for single value, 'list' for multiple values
        choices: Optional list of valid choices for classification-style fields
        description: Optional description of the field
        threshold: Optional per-field extraction threshold in [0, 1]
        cardinality: Record field cardinality (used when the structure sets a mode)
        exclusive: A mention bound here cannot bind elsewhere
    """
    name: str = Field(..., min_length=1, description="Field name")
    dtype: Literal["str", "list"] = Field(default="list", description="Data type")
    choices: Optional[List[str]] = Field(default=None, description="Valid choices")
    description: Optional[str] = Field(default=None, description="Field description")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cardinality: Optional[
        Literal["optional_one", "required_one", "zero_or_more", "one_or_more"]
    ] = Field(default=None, description="Record field cardinality")
    exclusive: bool = Field(
        default=False, description="A mention bound here cannot bind elsewhere"
    )

    @field_validator('choices')
    @classmethod
    def validate_choices(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Ensure choices list is not empty if provided."""
        if v is not None and len(v) == 0:
            raise ValueError("choices must contain at least one option")
        return v


class StructureInput(_StrictInput):
    """Validates a structure block.

    Args:
        fields: List of field definitions
        mode: Optional record formation mode (natural | latent | anchorless)
        anchor: Anchor field name (required when mode == 'natural')
        occurrence_policy: How repeated surfaces are handled during training
    """
    fields: List[FieldInput] = Field(..., min_length=1, description="List of fields")
    mode: Optional[Literal["natural", "latent", "anchorless"]] = Field(
        default=None, description="Record formation mode"
    )
    anchor: Optional[str] = Field(default=None, description="Anchor field name")
    occurrence_policy: Optional[
        Literal["all", "first", "error_on_ambiguous", "latent_all"]
    ] = Field(default=None, description="Repeated-surface handling")

    @model_validator(mode='after')
    def validate_record_mode(self) -> 'StructureInput':
        if self.mode == "natural":
            if not self.anchor:
                # Default anchor = first declared field (declaration order).
                self.anchor = self.fields[0].name
            names = {f.name for f in self.fields}
            if self.anchor not in names:
                raise ValueError(f"anchor {self.anchor!r} is not a declared field")
        elif self.mode is not None and self.anchor:
            raise ValueError(f"structure mode={self.mode!r} must not set 'anchor'")
        return self


class ClassificationInput(_StrictInput):
    """Validates a classification task.

    Args:
        task: Task name
        labels: Label names, or a mapping of label name to description
        multi_label: Whether multiple labels can be selected
        cls_threshold: Multi-label selection cutoff in [0, 1]
        top_k: Maximum number of labels returned for a multi-label task
        class_act: Activation override ('auto' picks by ``multi_label``)
        prompt: Optional instruction appended to the task name in the prompt
        examples: Few-shot ``(input, label)`` pairs rendered into the prompt
    """
    task: str = Field(..., min_length=1, description="Task name")
    labels: Union[List[str], Dict[str, str]] = Field(
        ..., min_length=2, description="Classification labels"
    )
    multi_label: bool = Field(default=False, description="Multi-label classification")
    cls_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    class_act: Optional[Literal["auto", "sigmoid", "softmax"]] = None
    prompt: Optional[str] = None
    examples: Optional[List[Tuple[str, str]]] = None

    @field_validator('labels')
    @classmethod
    def validate_labels(
        cls, v: Union[List[str], Dict[str, str]]
    ) -> Union[List[str], Dict[str, str]]:
        """Ensure labels are unique and non-empty."""
        names = list(v)
        if len(names) != len(set(names)):
            raise ValueError("labels must be unique")
        if any(not label.strip() for label in names):
            raise ValueError("labels cannot be empty strings")
        return v


class EntityInput(_StrictInput):
    """Validates one entity type's configuration.

    Args:
        description: Optional description rendered into the prompt
        dtype: 'list' for every mention, 'str' for the best one
        threshold: Optional per-entity extraction threshold in [0, 1]
    """
    description: Optional[str] = None
    dtype: Optional[Literal["str", "list"]] = None
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class RelationInput(_StrictInput):
    """Validates one relation type's configuration.

    Args:
        description: Optional description rendered into the prompt
        threshold: Optional per-relation extraction threshold in [0, 1]
    """
    description: Optional[str] = None
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class AttributeGroupInput(_StrictInput):
    """Validates one entity attribute group; mirrors ``AttributeGroup``.

    Args:
        labels: Values available in this attribute group
        multi_label: Use independent sigmoid decisions instead of forcing one value
        threshold: Selection cutoff for multi-label groups
        applies_to: Optional entity types to which this group applies
        qualify_labels: Prefix model-facing values with the group name
    """
    labels: List[str] = Field(..., min_length=1)
    multi_label: bool = False
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    applies_to: Optional[List[str]] = None
    qualify_labels: bool = False


class SchemaInput(_StrictInput):
    """Root schema validation model.

    Args:
        entities: List of entity types, or a mapping of type to description or config
        structures: Dict mapping structure names to structure definitions
        classifications: List of classification task definitions
        relations: List of relation types, or a mapping of type to description or config
        entity_attributes: Attribute groups decoded on extracted entity spans
    """
    entities: Optional[Union[List[str], Dict[str, Union[str, EntityInput]]]] = Field(
        default=None,
        description="Entity types"
    )
    structures: Optional[Dict[str, StructureInput]] = Field(
        default=None,
        description="Structure definitions"
    )
    classifications: Optional[List[ClassificationInput]] = Field(
        default=None,
        description="Classification tasks"
    )
    relations: Optional[Union[List[str], Dict[str, Union[str, RelationInput]]]] = Field(
        default=None,
        description="Relation types"
    )
    entity_attributes: Optional[Dict[str, AttributeGroupInput]] = Field(
        default=None,
        description="Entity attribute groups"
    )

    @field_validator('entities', 'relations')
    @classmethod
    def validate_named_types(
            cls,
            v: Optional[Union[List[str], Dict[str, object]]],
            info: ValidationInfo,
    ) -> Optional[Union[List[str], Dict[str, object]]]:
        """Validate entity and relation name collections."""
        if v is None:
            return v
        kind = "entity" if info.field_name == "entities" else "relation"
        names = list(v)
        if len(names) == 0:
            container = "list" if isinstance(v, list) else "dict"
            raise ValueError(f"{info.field_name} {container} cannot be empty")
        if any(not name.strip() for name in names):
            raise ValueError(f"{kind} names cannot be empty strings")
        if len(names) != len(set(names)):
            raise ValueError(f"{kind} names must be unique")
        return v

    @field_validator('structures')
    @classmethod
    def validate_structures(
            cls,
            v: Optional[Dict[str, StructureInput]]
    ) -> Optional[Dict[str, StructureInput]]:
        """Validate structures format."""
        if v is None:
            return v

        if len(v) == 0:
            raise ValueError("structures dict cannot be empty")
        if any(not key.strip() for key in v.keys()):
            raise ValueError("structure names cannot be empty strings")

        return v

    @field_validator('classifications')
    @classmethod
    def validate_classifications(
            cls,
            v: Optional[List[ClassificationInput]]
    ) -> Optional[List[ClassificationInput]]:
        """Validate classifications format."""
        if v is None:
            return v

        if len(v) == 0:
            raise ValueError("classifications list cannot be empty")

        # Check for duplicate task names
        task_names = [cls_task.task for cls_task in v]
        if len(task_names) != len(set(task_names)):
            raise ValueError("classification task names must be unique")

        return v

    @model_validator(mode='after')
    def validate_at_least_one_section(self) -> 'SchemaInput':
        """Ensure at least one section is provided."""
        if all(
                getattr(self, field) is None
                for field in ['entities', 'structures', 'classifications', 'relations']
        ):
            raise ValueError(
                "At least one of entities, structures, classifications, "
                "or relations must be provided"
            )
        if self.entity_attributes is not None and self.entities is None:
            raise ValueError("entity_attributes requires entities")
        return self
