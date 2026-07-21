"""
EntityAttributeGLiNER2: assign attribute groups to extracted entity spans.

Generalizes "entity-level sentiment" to arbitrary *attribute groups*. Each
group is a set of entity-type labels that should be *assigned to* extracted
spans rather than *extracted as* their own spans:

* single-label group -> ``softmax`` over the group's label logits at the
  span position, argmax -> exactly one value is forced on every span.
* multi-label group -> ``sigmoid`` per label, keep those >= ``threshold``
  (zero or more).

Sentiment is just one configuration::

    model = EntityAttributeGLiNER2.from_pretrained("repo")
    model.set_attribute_groups({
        "sentiment": AttributeGroup(["positive", "negative", "neutral"]),
    })
    # Note: the user does NOT add sentiment labels to the schema.
    schema = model.create_schema().entities(["person", "company", "product"])
    model.extract(text, schema, format_results=False)

Attribute labels are *not* declared in the schema by the user. The engine
injects them into the entity set automatically at inference time (so the
model scores them), then consumes them as span attributes instead of
emitting them as their own entity buckets. Injection only happens for
schemas that already contain an entity task.

The engine overrides only the ``entities`` decode path; classifications,
structures and relations fall through to :class:`GLiNER2` unchanged.

Note
----
Read results with ``extract(..., format_results=False)``. The base
``format_results`` / ``_format_entity_dict`` reshape entity dicts and will
drop the extra attribute keys otherwise.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from gliner2.inference.engine import GLiNER2
from gliner2.inference.schema import Schema


@dataclass
class AttributeGroup:
    """A group of entity-type labels to *assign* to spans, not extract as spans.

    Args:
        labels: Group labels. Mutually exclusive when ``multi_label`` is False.
        multi_label: If False, softmax over labels (forced exactly one). If
            True, sigmoid per label and keep those >= ``threshold``.
        threshold: Selection cutoff, only used when ``multi_label`` is True.
        applies_to: Entity types this group annotates. ``None`` means every
            content entity type.
    """

    labels: List[str]
    multi_label: bool = False
    threshold: float = 0.5
    applies_to: Optional[List[str]] = None


class EntityAttributeGLiNER2(GLiNER2):
    """GLiNER2 variant that assigns softmax/sigmoid attribute groups to spans."""

    #: Keys the engine writes on every span dict; a group name that collides
    #: with one of these would silently overwrite the entity's own fields.
    _RESERVED_NAMES = frozenset({"text", "confidence", "start", "end"})

    def set_attribute_groups(
        self, groups: Dict[str, AttributeGroup]
    ) -> "EntityAttributeGLiNER2":
        """Register attribute groups. Label sets must be disjoint across groups."""
        groups = groups or {}
        seen: Dict[str, str] = {}
        for gname, g in groups.items():
            if not gname or gname in self._RESERVED_NAMES:
                raise ValueError(
                    f"Invalid attribute group name {gname!r}: must be non-empty "
                    f"and not one of {sorted(self._RESERVED_NAMES)}"
                )
            if not g.labels:
                raise ValueError(f"Attribute group '{gname}' has no labels")
            if not 0.0 <= g.threshold <= 1.0:
                raise ValueError(
                    f"Attribute group '{gname}' threshold must be in [0, 1], "
                    f"got {g.threshold}"
                )
            group_seen: set = set()
            for lbl in g.labels:
                if not lbl or not lbl.strip():
                    raise ValueError(f"Attribute group '{gname}' has an empty label")
                if lbl in group_seen:
                    raise ValueError(
                        f"Label '{lbl}' is duplicated within group '{gname}'"
                    )
                group_seen.add(lbl)
                if lbl in seen:
                    raise ValueError(
                        f"Label '{lbl}' is in both '{seen[lbl]}' and '{gname}'"
                    )
                seen[lbl] = gname
        self._attr_groups: Dict[str, AttributeGroup] = groups
        self._attr_labels: set = set(seen)
        return self

    @property
    def _groups(self) -> Dict[str, AttributeGroup]:
        return getattr(self, "_attr_groups", {})

    # =========================================================================
    # Schema augmentation — inject attribute labels so the model scores them
    # =========================================================================

    def batch_extract(self, texts, schemas, *args, **kwargs):
        """Inject attribute labels into schemas, then run base extraction."""
        if self._groups and self._attr_labels:
            schemas = self._inject_attribute_labels(schemas)
        return super().batch_extract(texts, schemas, *args, **kwargs)

    def _inject_attribute_labels(
        self, schemas: Union[Schema, Dict, List]
    ) -> Union[Schema, Dict, List]:
        labels = sorted(self._attr_labels)
        if isinstance(schemas, list):
            return [self._augment_schema(s, labels) for s in schemas]
        return self._augment_schema(schemas, labels)

    @staticmethod
    def _augment_schema(schema, labels: List[str]):
        """Add missing attribute labels to a schema's entity set.

        Only augments schemas that already declare an entity task, so a
        pure-classification/structure schema is never turned into an entity
        extractor. Returns a copy; the caller's schema is left untouched.
        """
        # Schema builder object
        if isinstance(schema, Schema) or (
            hasattr(schema, "build") and hasattr(schema, "entities")
        ):
            existing = set(schema.schema.get("entities", {})) if hasattr(schema, "schema") else set()
            if not existing:
                return schema  # no entity task -> nothing to attach attributes to
            missing = [lbl for lbl in labels if lbl not in existing]
            if not missing:
                return schema
            schema = copy.deepcopy(schema)
            schema.entities(missing)
            return schema

        # Dict schema
        if isinstance(schema, dict):
            ents = schema.get("entities")
            if not ents:
                return schema
            new_schema = dict(schema)
            if isinstance(ents, list):
                new_schema["entities"] = list(ents) + [
                    lbl for lbl in labels if lbl not in ents
                ]
            elif isinstance(ents, dict):
                new_ents = dict(ents)
                for lbl in labels:
                    new_ents.setdefault(lbl, "")
                new_schema["entities"] = new_ents
            return new_schema

        return schema

    # =========================================================================
    # Entity decode override
    # =========================================================================

    def _extract_span_result(
        self,
        results,
        schema_name,
        task_type,
        embs,
        span_info,
        schema_tokens,
        text_tokens,
        text_len,
        original_text,
        start_mapping,
        end_mapping,
        threshold,
        metadata,
        cls_fields,
        include_confidence,
        include_spans,
    ):
        # Only entities get attribute assignment; everything else is unchanged.
        if schema_name != "entities" or not self._groups:
            return super()._extract_span_result(
                results,
                schema_name,
                task_type,
                embs,
                span_info,
                schema_tokens,
                text_tokens,
                text_len,
                original_text,
                start_mapping,
                end_mapping,
                threshold,
                metadata,
                cls_fields,
                include_confidence,
                include_spans,
            )

        field_names: List[str] = []
        for j in range(len(schema_tokens) - 1):
            if schema_tokens[j] in ("[E]", "[C]", "[R]"):
                field_names.append(schema_tokens[j + 1])
        if not field_names:
            results["entities"] = []
            return

        count_logits = self.count_pred(embs[0].unsqueeze(0))
        pred_count = int(count_logits.argmax(dim=1).item())
        if pred_count <= 0 or span_info is None:
            results["entities"] = []
            return

        struct_proj = self.count_embed(embs[1:], pred_count)
        # Keep the pre-sigmoid logits: needed to softmax within an attribute
        # group. Sigmoid is still used for the entity-detection threshold.
        raw_logits = torch.einsum(
            "lkd,bpd->bplk", span_info["span_rep"], struct_proj
        )
        span_scores = torch.sigmoid(raw_logits)

        results["entities"] = self._extract_entities_with_attributes(
            field_names,
            span_scores,
            raw_logits,
            text_len,
            original_text,
            start_mapping,
            end_mapping,
            threshold,
            metadata,
            include_confidence,
            include_spans,
        )

    def _extract_entities_with_attributes(
        self,
        entity_names: List[str],
        span_scores: torch.Tensor,
        raw_logits: torch.Tensor,
        text_len: int,
        text: str,
        start_map: List[int],
        end_map: List[int],
        threshold: float,
        metadata: Dict,
        include_confidence: bool,
        include_spans: bool,
    ) -> List[Dict]:
        # Precompute per-group label -> field-index tensors, dropping labels
        # that are not present in this schema's entity set.
        group_idx: Dict[str, Any] = {}
        for gname, g in self._groups.items():
            present = [
                (lbl, entity_names.index(lbl))
                for lbl in g.labels
                if lbl in entity_names
            ]
            if present:
                labels, idxs = zip(*present)
                group_idx[gname] = (
                    list(labels),
                    torch.tensor(idxs, device=raw_logits.device),
                    g,
                )

        content_names = [n for n in entity_names if n not in self._attr_labels]

        scores = span_scores[0, :, -text_len:]  # (n_fields, L, K) sigmoid — detection
        logits = raw_logits[0, :, -text_len:]  # (n_fields, L, K) raw — attributes

        entity_results: "OrderedDict[str, Any]" = OrderedDict()
        for name in metadata.get("entity_order", content_names):
            if name not in content_names:
                continue
            idx = entity_names.index(name)
            meta = metadata.get("entity_metadata", {}).get(name, {})
            # An explicit per-entity threshold of 0.0 is valid and must not fall
            # back to the global threshold, so check for None instead of falsiness.
            configured = meta.get("threshold")
            ent_threshold = threshold if configured is None else configured

            starts, widths = torch.where(scores[idx] >= ent_threshold)
            found: List[Dict[str, Any]] = []
            for start, width in zip(starts.tolist(), widths.tolist()):
                end = start + width + 1
                if not (0 <= start < text_len and end <= text_len):
                    continue
                try:
                    cs, ce = start_map[start], end_map[end - 1]
                    span_text = text[cs:ce].strip()
                except (IndexError, KeyError):
                    continue
                if not span_text:
                    continue

                attrs = self._assign_attributes(logits, start, width, group_idx, name)
                # start/end/confidence are always kept here so _dedupe can run;
                # _format_entity trims them afterwards to honour the public flags.
                found.append(
                    {
                        "text": span_text,
                        "confidence": scores[idx, start, width].item(),
                        "start": cs,
                        "end": ce,
                        **attrs,
                    }
                )

            entity_results[name] = [
                self._format_entity(d, include_confidence, include_spans)
                for d in self._dedupe(found)
            ]

        return [entity_results] if entity_results else []

    @staticmethod
    def _assign_attributes(
        logits: torch.Tensor,
        start: int,
        width: int,
        group_idx: Dict[str, Any],
        entity_name: str,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for gname, (labels, idxs, g) in group_idx.items():
            if g.applies_to is not None and entity_name not in g.applies_to:
                continue  # group not scoped to this entity type
            vec = logits[idxs, start, width]  # (n_labels_in_group,)
            if g.multi_label:
                probs = torch.sigmoid(vec)
                out[gname] = [
                    {"label": labels[i], "confidence": probs[i].item()}
                    for i in range(len(labels))
                    if probs[i].item() >= g.threshold
                ]
            else:
                probs = torch.softmax(vec, dim=-1)  # forced: always sums to 1
                b = int(probs.argmax())
                out[gname] = {"label": labels[b], "confidence": probs[b].item()}
        return out

    @staticmethod
    def _dedupe(found: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        found.sort(key=lambda d: d["confidence"], reverse=True)
        kept: List[Dict[str, Any]] = []
        for d in found:
            if not any(
                not (d["end"] <= k["start"] or d["start"] >= k["end"]) for k in kept
            ):
                kept.append(d)
        return kept

    @staticmethod
    def _format_entity(
        entity: Dict[str, Any],
        include_confidence: bool,
        include_spans: bool,
    ) -> Dict[str, Any]:
        """Trim internal span fields to honour the public output flags.

        ``text`` and any attribute-group payloads are always kept; ``confidence``
        and ``start``/``end`` are only emitted when explicitly requested, matching
        the base :class:`GLiNER2` contract.
        """
        reserved = EntityAttributeGLiNER2._RESERVED_NAMES
        out: Dict[str, Any] = {"text": entity["text"]}
        if include_confidence:
            out["confidence"] = entity["confidence"]
        if include_spans:
            out["start"] = entity["start"]
            out["end"] = entity["end"]
        for key, value in entity.items():
            if key not in reserved:
                out[key] = value
        return out
