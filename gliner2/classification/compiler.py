"""Compile a ClassificationSchema into the model-schema dict + constraint set.

This is where silent failure lives. ``_collate_batch`` swallows every exception
and substitutes ``_create_fallback_record`` (``processor.py:369-374``), so a
malformed compiled schema produces *garbage predictions, not an error*. This
module's job is to make that impossible: it self-asserts the emitted shape
before returning, and either the schema round-trips through the real processor
with exact label alignment or it fails loudly at compile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .constraints import (
    AnyOtherSelected,
    Constraint,
    DictAssignment,
    IsDefault,
    Iff,
    Not,
)
from .errors import SchemaError
from .schema import ClassificationSchema, TaskSpec, _RESERVED

_MODEL_KEYS = (
    "json_structures",
    "classifications",
    "entities",
    "relations",
    "json_descriptions",
    "entity_descriptions",
)


@dataclass(frozen=True)
class CompiledClassificationSchema:
    model_schema: dict
    task_specs: tuple  # tuple[TaskSpec, ...]
    constraints: tuple  # tuple[Constraint, ...]  (includes lowered default rules)
    task_order: tuple  # tuple[str, ...]
    fingerprint: str

    def task(self, name) -> TaskSpec:
        for spec in self.task_specs:
            if spec.name == name:
                return spec
        raise SchemaError(f"unknown task {name!r}")

    # Alias so constraints._task_spec resolves against a compiled schema too.
    def task_spec(self, name) -> TaskSpec:
        return self.task(name)

    def build(self) -> dict:
        return self.model_schema


def _classification_entry(spec: TaskSpec) -> dict:
    entry = {
        "task": spec.name,
        "labels": list(spec.label_names),
        "multi_label": not spec.is_exclusive,
        "cls_threshold": spec.threshold,
        "class_act": spec.activation,
    }
    if spec.instruction:
        entry["prompt"] = spec.instruction
    descs = {l.name: l.description for l in spec.labels if l.description}
    if descs:
        entry["label_descriptions"] = descs
    if spec.examples:
        entry["examples"] = [list(pair) for pair in spec.examples]
    return entry


def _emit_model_schema(task_specs) -> dict:
    return {
        "json_structures": [],
        "classifications": [_classification_entry(spec) for spec in task_specs],
        "entities": {},
        "relations": [],
        "json_descriptions": {},
        "entity_descriptions": {},
    }


def _has_reserved(value: str) -> bool:
    return any(token in value for token in _RESERVED)


def _assert_model_schema(model: dict) -> None:
    """Cheap self-assertion: the only thing standing between a typo and silently
    wrong predictions. Runs once per compile."""
    for key in _MODEL_KEYS:
        if key not in model:
            raise SchemaError(f"compiled model schema is missing key {key!r}")
    if not isinstance(model["classifications"], list):
        raise SchemaError("compiled 'classifications' must be a list")

    for entry in model["classifications"]:
        task = entry.get("task")
        if not isinstance(task, str) or not task:
            raise SchemaError("classification entry has an invalid 'task'")
        for required in ("labels", "multi_label", "cls_threshold", "class_act"):
            if required not in entry:
                raise SchemaError(
                    f"classification task {task!r} is missing {required!r}"
                )
        if not isinstance(entry["labels"], list) or not entry["labels"]:
            raise SchemaError(f"classification task {task!r} has invalid 'labels'")
        if not isinstance(entry["multi_label"], bool):
            raise SchemaError(
                f"classification task {task!r} has non-bool 'multi_label'"
            )

        strings = [task] + list(entry["labels"])
        if "prompt" in entry:
            strings.append(entry["prompt"])
        strings += list(entry.get("label_descriptions", {}).keys())
        strings += list(entry.get("label_descriptions", {}).values())
        for pair in entry.get("examples", []):
            strings += [str(x) for x in pair]
        for value in strings:
            if isinstance(value, str) and _has_reserved(value):
                raise SchemaError(
                    f"classification task {task!r} emitted a reserved marker token in "
                    f"{value!r}; this would corrupt logit-to-label alignment"
                )


def _check_prefix_collisions(task_order) -> None:
    """The prompt is read back with boundary-aware longest match
    (_resolve_classification_config). Reject a task name that is a prefix of
    another followed by ':' or ' ', where the failure is a silently swapped task
    result. A bare prefix without a boundary (``sent`` vs ``sentiment``) is fine.
    """
    for x in task_order:
        for y in task_order:
            if x == y:
                continue
            if y.startswith(x) and len(y) > len(x) and y[len(x)] in (":", " "):
                raise SchemaError(
                    f"task name {x!r} is a boundary-prefix of {y!r}; the prompt "
                    f"resolver could confuse them. Rename one."
                )


def _static_feasibility(schema: ClassificationSchema, task_specs) -> None:
    """Cheap propagation on declared label sets only. Catches globally
    unsatisfiable constraint sets and exclusive tasks pinned to the empty set,
    e.g. all_of(('x','a'), ('x','b')) on an exclusive task."""
    constraints = schema.constraints
    if not constraints:
        return

    undetermined = DictAssignment(schema, selected={}, decided=())
    for c in constraints:
        if c.evaluate(undetermined) is False:
            raise SchemaError(
                "constraint set is unsatisfiable on the declared label sets"
            )

    for spec in task_specs:
        if not spec.is_exclusive:
            continue
        touching = [c for c in constraints if spec.name in c.references()]
        if not touching:
            continue
        reachable = False
        for label in spec.label_names:
            a = DictAssignment(schema, {spec.name: {label}}, decided=[spec.name])
            if all(c.evaluate(a) is not False for c in touching):
                reachable = True
                break
        if not reachable:
            raise SchemaError(
                f"no label of exclusive task {spec.name!r} satisfies its constraints; "
                f"the task is pinned to the empty set"
            )


def _lower_defaults(schema: ClassificationSchema, task_specs) -> tuple:
    """Only defaults lower into the AST. Cardinality stays as TaskSpec bounds."""
    constraints = list(schema.constraints)
    for spec in task_specs:
        if spec.default is not None:
            constraints.append(
                Iff(IsDefault(spec.name), Not(AnyOtherSelected(spec.name)))
            )
    return tuple(constraints)


def _fingerprint(schema: ClassificationSchema) -> str:
    payload = json.dumps(schema.to_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compile_schema(schema) -> CompiledClassificationSchema:
    if isinstance(schema, CompiledClassificationSchema):
        return schema  # idempotent
    if not isinstance(schema, ClassificationSchema):
        raise SchemaError(
            f"compile_schema expects a ClassificationSchema, got {type(schema).__name__}"
        )

    task_specs = schema.task_specs
    if not task_specs:
        raise SchemaError("cannot compile a schema with no tasks")
    task_order = schema.task_order

    for c in schema.constraints:
        c.check_schema(schema)

    _check_prefix_collisions(task_order)
    _static_feasibility(schema, task_specs)

    model = _emit_model_schema(task_specs)
    _assert_model_schema(model)

    constraints = _lower_defaults(schema, task_specs)
    return CompiledClassificationSchema(
        model_schema=model,
        task_specs=tuple(task_specs),
        constraints=constraints,
        task_order=tuple(task_order),
        fingerprint=_fingerprint(schema),
    )
