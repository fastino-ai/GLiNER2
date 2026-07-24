"""Chunked Joint IE extraction with document-level span remapping."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, List, Optional, Tuple

from gliner2.inference.chunking import split_text_into_chunks

from .result import JointEntity, JointRelation, JointResult


def extract_long_text(engine: Any, text: str, schema: Any = None,
                      chunk_size: int = 384, chunk_overlap: int = 64,
                      config: Any = None, **kwargs: Any) -> JointResult:
    """Extract each chunk independently and merge Joint IE graph fragments.

    Exact duplicate entities are keyed by ``(type, document start, document end)``;
    relations are keyed by type and both endpoint span keys. Consequently this
    function never synthesizes cross-chunk relations.
    """
    include_confidence = True if config is None else config.include_confidence
    include_spans = True if config is None else config.include_spans
    chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
    fragments: List[Tuple[Any, JointResult]] = []
    for chunk in chunks:
        raw = _extract_chunk(engine, chunk.text, schema, config, kwargs)
        fragments.append((chunk, _coerce_result(raw, chunk.text)))

    entity_by_key: Dict[Tuple[str, int, int], JointEntity] = {}
    relation_rows: Dict[Tuple[Any, ...], Tuple[str, Tuple[str, int, int],
                                                   Tuple[str, int, int], Optional[float], bool]] = {}
    for chunk, result in fragments:
        local_keys: Dict[str, Tuple[str, int, int]] = {}
        for entity in result.entities:
            start = entity.start + chunk.start_char
            end = entity.end + chunk.start_char
            key = (entity.type, start, end)
            local_keys[entity.id] = key
            remapped = JointEntity("", entity.type, text[start:end], start, end,
                                   entity.confidence, entity.sentence_id, entity.rescued)
            previous = entity_by_key.get(key)
            if previous is None or _confidence(remapped.confidence) > _confidence(previous.confidence):
                entity_by_key[key] = remapped
        for relation in result.relations:
            # Both IDs must originate in this same chunk result. This explicit
            # lookup is what prevents accidental cross-window edge creation.
            if relation.head not in local_keys or relation.tail not in local_keys:
                continue
            head_key, tail_key = local_keys[relation.head], local_keys[relation.tail]
            key = (relation.type, head_key, tail_key)
            row = (relation.type, head_key, tail_key, relation.confidence, relation.derived)
            previous = relation_rows.get(key)
            if previous is None or _confidence(row[3]) > _confidence(previous[3]):
                relation_rows[key] = row

    ordered_keys = sorted(entity_by_key, key=lambda key: (key[1], key[2], key[0]))
    key_to_id = {key: f"e{index + 1}" for index, key in enumerate(ordered_keys)}
    entities = []
    for key in ordered_keys:
        item = entity_by_key[key]
        entities.append(JointEntity(key_to_id[key], item.type, item.text,
            item.start, item.end,
            item.confidence,
            item.sentence_id, item.rescued))

    relations = [JointRelation(label, key_to_id[head], key_to_id[tail],
                               confidence, rescued)
                 for label, head, tail, confidence, rescued in relation_rows.values()]
    relations.sort(key=lambda value: (value.type, value.head, value.tail))
    return JointResult(text, entities, relations, include_confidence, include_spans)


extract_joint_long = extract_long_text


def _extract_chunk(engine: Any, text: str, schema: Any,
                   config: Any, kwargs: Dict[str, Any]) -> Any:
    method = getattr(engine, "extract_joint", None) or getattr(engine, "extract", None)
    if method is None and callable(engine):
        method = engine
    if method is None:
        raise TypeError("engine must be callable or expose extract_joint()/extract()")
    arguments = dict(kwargs)
    parameters = inspect.signature(method).parameters
    if "config" in parameters:
        arguments["config"] = config
    if schema is not None:
        if "schema" in parameters:
            arguments["schema"] = schema
            return method(text, **arguments)
        return method(text, schema, **arguments)
    return method(text, **arguments)


def _coerce_result(value: Any, text: str) -> JointResult:
    if isinstance(value, JointResult):
        return value
    if not isinstance(value, dict):
        raise TypeError("chunk extraction must return JointResult or a result dictionary")
    entities = [JointEntity(
        str(item.get("id", f"e{index}")), str(item.get("type", item.get("label", ""))),
        str(item.get("text", "")), int(item.get("start", 0)), int(item.get("end", 0)),
        item.get("confidence"), item.get("sentence_id"), bool(item.get("rescued", False)),
    ) for index, item in enumerate(value.get("entities", []))]
    relations = [JointRelation(
        str(item.get("type", item.get("label", ""))),
        str(item["head"]), str(item["tail"]), item.get("confidence"),
        bool(item.get("derived", False)),
    ) for item in value.get("relations", [])]
    return JointResult(str(value.get("text", text)), entities, relations)


def _confidence(value: Optional[float]) -> float:
    return float("-inf") if value is None else value
