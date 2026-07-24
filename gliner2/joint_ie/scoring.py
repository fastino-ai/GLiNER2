"""Raw neural scoring for joint information extraction.

This module deliberately stops before candidate pruning or constrained decoding.  It
turns GLiNER2 outputs into dense score lattices while retaining every valid span.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch


@dataclass
class CountHypothesis:
    count: int
    logit: float
    probability: float
    role_logits: torch.Tensor
    role_probabilities: torch.Tensor


@dataclass
class TaskLattice:
    """Scores for one model-facing task.

    ``role_logits`` in each count hypothesis has shape ``[slots, roles, L, W]``.
    For a relation ``roles == 2`` (head and tail).  Entity scoring always uses one
    slot, independent of the count head.  Probabilities are sigmoid values and are
    intentionally not thresholded.
    """

    name: str
    task_type: str
    roles: Tuple[str, ...]
    count_hypotheses: List[CountHypothesis]
    schema_tokens: Tuple[str, ...] = ()


@dataclass
class ScoreLattice:
    """Dense scores and exact caller-coordinate metadata for one document."""

    text: str
    text_tokens: Tuple[str, ...]
    start_mappings: Tuple[int, ...]
    end_mappings: Tuple[int, ...]
    span_starts: torch.Tensor
    span_ends: torch.Tensor
    valid_span_mask: torch.Tensor
    tasks: List[TaskLattice]
    schema: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.start_mappings)

    def _span(self, start: int, end_inclusive: int):
        from .lattice import SpanRef
        char_start = self.start_mappings[start]
        char_end = self.end_mappings[end_inclusive]
        return SpanRef(start, end_inclusive + 1, char_start, char_end,
                       self.text[char_start:char_end])

    def _task(self, name: str, task_type: str) -> TaskLattice:
        for task in self.tasks:
            if task.name == name and task.task_type == task_type:
                return task
        raise KeyError(name)

    def top_entities(self, entity_type: str, k: Optional[int] = None):
        task = next((task for task in self.tasks if task.task_type == "entities"), None)
        if task is None or entity_type not in task.roles:
            return []
        role = task.roles.index(entity_type)
        values = task.count_hypotheses[0].role_probabilities[0, role]
        rows = [(self._span(i, i + w), float(values[i, w]))
                for i in range(values.shape[0]) for w in range(values.shape[1])
                if bool(self.valid_span_mask[i, w])]
        rows.sort(key=lambda row: (-row[1], row[0].start, row[0].end))
        return rows if k is None else rows[:max(0, k)]

    def _top_role(self, relation: str, slot: int, role: int, k: Optional[int]):
        task = self._task(relation, "relations")
        rows = []
        for hypothesis in task.count_hypotheses:
            if slot >= hypothesis.count or slot >= hypothesis.role_probabilities.shape[0]:
                continue
            values = hypothesis.role_probabilities[slot, role]
            rows.extend((self._span(i, i + w), float(values[i, w]), hypothesis.count,
                         hypothesis.probability)
                        for i in range(values.shape[0]) for w in range(values.shape[1])
                        if bool(self.valid_span_mask[i, w]))
        rows.sort(key=lambda row: (-row[1], -row[3], row[0].start, row[0].end, row[2]))
        return rows if k is None else rows[:max(0, k)]

    def top_heads(self, relation: str, slot: int = 0, k: Optional[int] = None):
        return self._top_role(relation, slot, 0, k)

    def top_tails(self, relation: str, slot: int = 0, k: Optional[int] = None):
        return self._top_role(relation, slot, 1, k)


_DTYPE_NAMES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
}


def resolve_device(model: Any, requested: Optional[Union[str, torch.device]] = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def resolve_dtype(model: Any, requested: Optional[Union[str, torch.dtype]] = None) -> torch.dtype:
    if isinstance(requested, torch.dtype):
        return requested
    if isinstance(requested, str):
        try:
            return _DTYPE_NAMES[requested.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {requested!r}") from exc
    try:
        dtype = next(model.parameters()).dtype
        return dtype if dtype.is_floating_point else torch.float32
    except (AttributeError, StopIteration):
        return torch.float32


def _schema_dict(schema: Any) -> Any:
    if hasattr(schema, "build"):
        return schema.build()
    if hasattr(schema, "schema"):
        return schema.schema
    return schema


def _field_names(schema_tokens: Sequence[str]) -> Tuple[str, ...]:
    markers = {"[E]", "[C]", "[R]"}
    return tuple(
        schema_tokens[index + 1]
        for index in range(len(schema_tokens) - 1)
        if schema_tokens[index] in markers
    )


def _task_name(schema_tokens: Sequence[str], fallback: str) -> str:
    if len(schema_tokens) > 2:
        return schema_tokens[2].split(" [DESCRIPTION] ", 1)[0]
    return fallback


class RawScorer:
    """Compose around an existing :class:`gliner2.GLiNER2` model."""

    def __init__(self, model: Any, *, device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[Union[str, torch.dtype]] = None):
        self.model = model
        self.processor = model.processor
        self.device = resolve_device(model, device)
        self.dtype = resolve_dtype(model, dtype)

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> "RawScorer":
        target_device = resolve_device(self.model, device or self.device)
        target_dtype = resolve_dtype(self.model, dtype or self.dtype)
        kwargs: Dict[str, Any] = {"device": target_device}
        if target_dtype is not None:
            kwargs["dtype"] = target_dtype
        self.model.to(**kwargs)
        self.device, self.dtype = target_device, target_dtype
        return self

    def eval(self) -> "RawScorer":
        self.model.eval()
        if hasattr(self.processor, "change_mode"):
            self.processor.change_mode(is_training=False)
        return self

    @torch.inference_mode()
    def score(self, text: str, schema: Any, *, count_top_k: int = 2,
              max_len: Optional[int] = None) -> ScoreLattice:
        return self.batch_score([text], schema, batch_size=1, max_len=max_len,
                                count_top_k=count_top_k)[0]

    @torch.inference_mode()
    def batch_score(
        self,
        texts: Sequence[str],
        schemas: Any,
        *,
        batch_size: Optional[int] = None,
        max_len: Optional[int] = None,
        count_top_k: int = 2,
    ) -> List[ScoreLattice]:
        """Score documents, using exactly one encoder pass for each model batch."""
        texts = list(texts)
        if not texts:
            return []
        if isinstance(schemas, (list, tuple)):
            if len(schemas) != len(texts):
                raise ValueError("schemas must have the same length as texts")
            schema_list = list(schemas)
        else:
            schema_list = [schemas] * len(texts)
        schema_dicts = [_schema_dict(schema) for schema in schema_list]
        size = 8 if batch_size is None else batch_size
        if size <= 0:
            raise ValueError("batch_size must be positive")
        if count_top_k <= 0:
            raise ValueError("count_top_k must be positive")
        effective_max_len = max_len

        self.eval()
        results: List[ScoreLattice] = []
        for offset in range(0, len(texts), size):
            chunk_texts = texts[offset:offset + size]
            chunk_schemas = schema_dicts[offset:offset + size]
            batch = self.processor.collate_fn_inference(
                list(zip(chunk_texts, chunk_schemas)), max_len=effective_max_len
            )
            batch = batch.to(
                self.device,
                self.dtype if self.dtype != torch.float32 else None,
            )
            encoded = self.model.encoder(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask
            ).last_hidden_state
            token_embs, schema_embs = self.processor.extract_embeddings_from_batch(
                encoded, batch.input_ids, batch
            )
            span_infos = self.model.compute_span_rep_batched(token_embs)
            for local_index, caller_text in enumerate(chunk_texts):
                results.append(self._build_lattice(
                    caller_text,
                    schema_list[offset + local_index],
                    batch,
                    local_index,
                    schema_embs[local_index],
                    span_infos[local_index],
                    count_top_k,
                ))
        return results

    def _build_lattice(
        self,
        caller_text: str,
        original_schema: Any,
        batch: Any,
        index: int,
        schema_embs: Sequence[Sequence[torch.Tensor]],
        span_info: Mapping[str, torch.Tensor],
        count_top_k: int,
    ) -> ScoreLattice:
        starts_all = list(batch.start_mappings[index])
        ends_all = list(batch.end_mappings[index])

        # collate_fn_inference may append punctuation.  Start mappings at or past
        # len(caller_text) describe that synthetic suffix, so they are excluded.
        # Keeping this rule tied to starts (rather than token text) also handles a
        # caller whose final character is itself punctuation.
        caller_token_count = len(starts_all)
        while caller_token_count and starts_all[caller_token_count - 1] >= len(caller_text):
            caller_token_count -= 1
        starts = starts_all[:caller_token_count]
        ends = ends_all[:caller_token_count]

        dense_rep = span_info["span_rep"]
        all_token_count = len(starts_all)
        document_start = max(0, dense_rep.shape[0] - all_token_count)
        dense_rep = dense_rep[document_start:document_start + caller_token_count]
        length, width = dense_rep.shape[:2]
        row = torch.arange(length, device=dense_rep.device).unsqueeze(1)
        col = torch.arange(width, device=dense_rep.device).unsqueeze(0)
        dense_ends = row + col
        valid = dense_ends < caller_token_count
        span_starts = row.expand(length, width)

        tasks: List[TaskLattice] = []
        task_types = batch.task_types[index]
        token_schemas = batch.schema_tokens_list[index]
        for task_index, (tokens, task_type, embeddings) in enumerate(
            zip(token_schemas, task_types, schema_embs)
        ):
            fields = _field_names(tokens)
            if not embeddings or not fields:
                continue
            embs = torch.stack(list(embeddings))
            if task_type == "entities":
                hypotheses = [(1, 0.0, 1.0)]
            else:
                count_logits = self.model.count_pred(embs[0].unsqueeze(0))[0]
                probabilities = torch.softmax(count_logits.float(), dim=-1)
                k = min(count_top_k, count_logits.numel())
                values, indices = torch.topk(probabilities, k=k)
                hypotheses = [
                    (int(count), float(torch.logit(prob.clamp(1e-7, 1 - 1e-7)).detach().cpu()), float(prob.detach().cpu()))
                    for prob, count in zip(values, indices)
                ]

            count_scores: List[CountHypothesis] = []
            for count, count_logit, count_probability in hypotheses:
                if count <= 0:
                    role_logits = dense_rep.new_empty((0, len(fields), length, width))
                else:
                    projected = self.model.count_embed(embs[1:], count)
                    role_logits = torch.einsum("lkd,bpd->bplk", dense_rep, projected)
                    # Preserve the dense shape but make synthetic/padded spans
                    # impossible for downstream optimizers to select.
                    role_logits = role_logits.masked_fill(~valid.unsqueeze(0).unsqueeze(0), -torch.inf)
                count_scores.append(CountHypothesis(
                    count=count,
                    logit=count_logit,
                    probability=count_probability,
                    role_logits=role_logits,
                    role_probabilities=torch.sigmoid(role_logits),
                ))
            tasks.append(TaskLattice(
                name=_task_name(tokens, f"task_{task_index}"),
                task_type=task_type,
                roles=fields,
                count_hypotheses=count_scores,
                schema_tokens=tuple(tokens),
            ))

        return ScoreLattice(
            text=caller_text,
            text_tokens=tuple(batch.text_tokens[index][:caller_token_count]),
            start_mappings=tuple(starts),
            end_mappings=tuple(ends),
            span_starts=span_starts,
            span_ends=dense_ends.expand(length, width),
            valid_span_mask=valid,
            tasks=tasks,
            schema=original_schema,
            metadata={"processed_text": batch.original_texts[index]},
        )


__all__ = [
    "CountHypothesis", "RawScorer", "ScoreLattice",
    "TaskLattice", "resolve_device", "resolve_dtype",
]
JointScoreLattice = ScoreLattice
EntityScoreBlock = TaskLattice
RelationCountHypothesis = CountHypothesis
RelationScoreBlock = TaskLattice
