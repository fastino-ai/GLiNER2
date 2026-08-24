"""Raw neural scoring for joint information extraction.

This module deliberately stops before candidate pruning or constrained
decoding. Span checkpoints produce dense :class:`ScoreLattice` objects;
boundary checkpoints produce sparse ``CandidateScoreSet`` objects.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
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
    """Compose around an existing span or boundary extractor model."""

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
              max_len: Optional[int] = None,
              top_k_roles: Optional[int] = None,
              relation_pair_cap: Optional[int] = None) -> Any:
        return self.batch_score([text], schema, batch_size=1, max_len=max_len,
                                count_top_k=count_top_k,
                                top_k_roles=top_k_roles,
                                relation_pair_cap=relation_pair_cap)[0]

    @torch.inference_mode()
    def batch_score(
        self,
        texts: Sequence[str],
        schemas: Any,
        *,
        batch_size: Optional[int] = None,
        max_len: Optional[int] = None,
        count_top_k: int = 2,
        top_k_roles: Optional[int] = None,
        relation_pair_cap: Optional[int] = None,
    ) -> List[Any]:
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
        if top_k_roles is not None and top_k_roles <= 0:
            raise ValueError("top_k_roles must be positive")
        if relation_pair_cap is not None and relation_pair_cap <= 0:
            raise ValueError("relation_pair_cap must be positive")
        effective_max_len = max_len

        self.eval()
        if getattr(self.model, "architecture", None) == "boundary":
            return self._batch_score_boundary(
                texts,
                schema_list,
                schema_dicts,
                batch_size=size,
                max_len=effective_max_len,
                top_k_roles=top_k_roles,
                relation_pair_cap=relation_pair_cap,
            )

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
            for local_index, caller_text in enumerate(chunk_texts):
                results.append(self._build_lattice(
                    caller_text,
                    schema_list[offset + local_index],
                    batch,
                    local_index,
                    token_embs[local_index],
                    schema_embs[local_index],
                    count_top_k,
                ))
        return results

    def _batch_score_boundary(
        self,
        texts: Sequence[str],
        schema_list: Sequence[Any],
        schema_dicts: Sequence[Mapping[str, Any]],
        *,
        batch_size: int,
        max_len: Optional[int],
        top_k_roles: Optional[int],
        relation_pair_cap: Optional[int],
    ) -> List[Any]:
        """Score boundary checkpoints into architecture-neutral sparse sets."""
        results: List[Any] = []
        for offset in range(0, len(texts), batch_size):
            chunk_texts = list(texts[offset:offset + batch_size])
            chunk_schemas = list(
                schema_dicts[offset:offset + batch_size]
            )
            batch = self.processor.collate_fn_inference(
                list(zip(chunk_texts, chunk_schemas)),
                max_len=max_len,
                architecture="boundary",
            )
            batch = batch.to(
                self.device,
                self.dtype if self.dtype != torch.float32 else None,
            )
            core = self.model._encode_core(batch)
            output = self.model.boundary_head(
                core["text_states"],
                core["text_mask"],
                core["query_states"],
                core["query_mask"],
                return_candidates=True,
            )
            for local_index, caller_text in enumerate(chunk_texts):
                results.append(
                    self._build_boundary_score_set(
                        caller_text,
                        schema_list[offset + local_index],
                        batch,
                        local_index,
                        core,
                        output.candidates,
                        top_k_roles,
                        relation_pair_cap,
                    )
                )
        return results

    def _build_boundary_score_set(
        self,
        caller_text: str,
        original_schema: Any,
        batch: Any,
        index: int,
        core: Mapping[str, Any],
        candidates: Any,
        top_k_roles: Optional[int],
        relation_pair_cap: Optional[int],
    ) -> Any:
        from gliner2.joint_ie.candidate_scores import (
            MentionScore,
            ScoredRelationEdge,
            boundary_candidates_to_candidate_score_set,
        )

        starts_all = list(batch.start_mappings[index])
        ends_all = list(batch.end_mappings[index])
        caller_token_count = len(starts_all)
        while (
            caller_token_count
            and starts_all[caller_token_count - 1] >= len(caller_text)
        ):
            caller_token_count -= 1
        starts = tuple(starts_all[:caller_token_count])
        ends = tuple(ends_all[:caller_token_count])
        token_offset = int(core["word_offsets"][index])
        compiled_entity_specs = getattr(original_schema, "entity_specs", {})
        entity_thresholds = {
            name: getattr(spec, "threshold", None)
            for name, spec in compiled_entity_specs.items()
        }
        entity_candidate_thresholds = {
            name: getattr(spec, "candidate_threshold", None)
            for name, spec in compiled_entity_specs.items()
        }

        extra_mentions: List[MentionScore] = []
        relation_edges: List[ScoredRelationEdge] = []
        if (
            getattr(self.model, "enable_relations", False)
            and core["rel_specs"][index]
            and candidates is not None
        ):
            extra_mentions, relation_edges = self._score_boundary_relations(
                original_schema,
                index,
                core,
                candidates,
                token_offset,
                caller_token_count,
                entity_thresholds,
                entity_candidate_thresholds,
                top_k_roles,
                relation_pair_cap,
            )

        return boundary_candidates_to_candidate_score_set(
            caller_text,
            candidates,
            core["ext_specs"][index],
            sample_index=index,
            token_offset=token_offset,
            text_length=caller_token_count,
            pair_temperature=self.model.boundary_settings.pair_temperature,
            entity_thresholds=entity_thresholds,
            entity_candidate_thresholds=entity_candidate_thresholds,
            extra_mentions=extra_mentions,
            edges=relation_edges,
            text_tokens=tuple(batch.text_tokens[index][:caller_token_count]),
            start_mappings=starts,
            end_mappings=ends,
            metadata={
                "architecture": "boundary",
                "processed_text": batch.original_texts[index],
                "schema": original_schema,
            },
        )

    def _score_boundary_relations(
        self,
        compiled_schema: Any,
        sample_index: int,
        core: Mapping[str, Any],
        candidates: Any,
        token_offset: int,
        text_length: int,
        entity_thresholds: Mapping[str, Optional[float]],
        entity_candidate_thresholds: Mapping[str, Optional[float]],
        top_k_roles: Optional[int],
        relation_pair_cap: Optional[int],
    ) -> Tuple[List[Any], List[Any]]:
        """Score sparse relation pairs and their typed entity endpoints."""
        from gliner2.joint_ie.candidate_scores import (
            MentionScore,
            ScoredRelationEdge,
        )
        from gliner2.models.base import QueryLayout
        from gliner2.models.boundary.relations import RelationTypeSpec

        relation_entries = core["rel_specs"][sample_index]
        compiled_relation_specs = getattr(
            compiled_schema, "relation_specs", {}
        )
        routed_specs = []
        for entry in relation_entries:
            route = entry["spec"]
            compiled = compiled_relation_specs.get(route.relation_type)
            routed_specs.append(
                RelationTypeSpec(
                    relation_type=route.relation_type,
                    head_query_ids=route.head_query_ids,
                    tail_query_ids=route.tail_query_ids,
                    allow_self=bool(
                        getattr(compiled, "allow_self", route.allow_self)
                    ),
                )
            )
        sample_candidates = self.model._single_sample_candidates(
            candidates, sample_index
        )
        generator = self.model.relation_pair_generator
        if top_k_roles is not None or relation_pair_cap is not None:
            from gliner2.models.boundary.relations import (
                TypedRelationPairGenerator,
            )

            settings = generator.settings
            generator = TypedRelationPairGenerator(replace(
                settings,
                heads_per_relation=(
                    min(settings.heads_per_relation, top_k_roles)
                    if top_k_roles is not None
                    else settings.heads_per_relation
                ),
                tails_per_relation=(
                    min(settings.tails_per_relation, top_k_roles)
                    if top_k_roles is not None
                    else settings.tails_per_relation
                ),
                pair_cap=(
                    min(settings.pair_cap, relation_pair_cap)
                    if relation_pair_cap is not None
                    else settings.pair_cap
                ),
            ))
        pairs = generator.generate(
            sample_candidates,
            [QueryLayout(queries=())],
            routed_specs,
        )
        if not len(pairs):
            return [], []
        relation_states = torch.stack(
            [entry["query_state"] for entry in relation_entries]
        ).unsqueeze(0)
        relation_logits = self.model.relation_scorer(
            core["text_states"][sample_index:sample_index + 1],
            relation_states,
            sample_candidates,
            pairs,
        )
        relation_logits = (
            relation_logits
            / self.model.boundary_settings.relation_temperature
        )

        entity_queries = {
            spec["field_name"]: query_id
            for query_id, spec in enumerate(core["ext_specs"][sample_index])
            if spec["task_type"] == "entities"
        }
        relation_specs = compiled_relation_specs
        endpoint_pairs: List[Tuple[int, int]] = []
        endpoint_requirements: set[Tuple[str, int, int]] = set()
        for pair_index in range(len(pairs)):
            relation_type = pairs.relation_types[pair_index]
            relation_spec = relation_specs.get(relation_type)
            if relation_spec is None:
                continue
            head = (
                int(pairs.head_start[pair_index]),
                int(pairs.head_end[pair_index]),
            )
            tail = (
                int(pairs.tail_start[pair_index]),
                int(pairs.tail_end[pair_index]),
            )
            endpoint_pairs.extend((head, tail))
            endpoint_requirements.update(
                (entity_type, *head)
                for entity_type in getattr(relation_spec, "head", ())
                if entity_type in entity_queries
            )
            endpoint_requirements.update(
                (entity_type, *tail)
                for entity_type in getattr(relation_spec, "tail", ())
                if entity_type in entity_queries
            )

        unique_pairs = list(dict.fromkeys(endpoint_pairs))
        entity_rows = list(entity_queries.items())
        if not unique_pairs or not entity_rows:
            return [], []
        device = core["query_states"].device
        query_ids = torch.tensor(
            [query_id for _, query_id in entity_rows],
            dtype=torch.long,
            device=device,
        )
        explicit_pairs = torch.tensor(
            unique_pairs, dtype=torch.long, device=device
        )
        explicit_indices = explicit_pairs.view(
            1, 1, len(unique_pairs), 2
        ).expand(1, len(entity_rows), len(unique_pairs), 2)
        entity_logits = self.model.boundary_head.score_explicit_spans(
            core["text_states"][sample_index:sample_index + 1],
            core["text_mask"][sample_index:sample_index + 1],
            core["query_states"][
                sample_index:sample_index + 1
            ].index_select(1, query_ids),
            core["query_mask"][
                sample_index:sample_index + 1
            ].index_select(1, query_ids),
            explicit_indices,
        )[0] / self.model.boundary_settings.pair_temperature
        row_by_type = {
            entity_type: row
            for row, (entity_type, _) in enumerate(entity_rows)
        }
        pair_column = {
            pair: column for column, pair in enumerate(unique_pairs)
        }
        extra_mentions: List[MentionScore] = []
        for entity_type, start, end in sorted(endpoint_requirements):
            local_start, local_end = start - token_offset, end - token_offset
            if not (0 <= local_start < local_end <= text_length):
                continue
            logit = float(
                entity_logits[
                    row_by_type[entity_type], pair_column[(start, end)]
                ].detach().float()
            )
            threshold = entity_thresholds.get(entity_type)
            extra_mentions.append(
                MentionScore(
                    query_id=entity_queries[entity_type],
                    entity_type=entity_type,
                    start=local_start,
                    end=local_end,
                    logit=logit,
                    probability=(
                        1.0 / (1.0 + math.exp(-logit))
                        if logit >= 0
                        else math.exp(logit) / (1.0 + math.exp(logit))
                    ),
                    threshold=0.5 if threshold is None else float(threshold),
                    candidate_threshold=entity_candidate_thresholds.get(
                        entity_type
                    ),
                )
            )

        mention_keys = {mention.key for mention in extra_mentions}
        edges: List[ScoredRelationEdge] = []
        for pair_index in range(len(pairs)):
            relation_type = pairs.relation_types[pair_index]
            relation_spec = relation_specs.get(relation_type)
            if relation_spec is None:
                continue
            head_span = (
                int(pairs.head_start[pair_index]) - token_offset,
                int(pairs.head_end[pair_index]) - token_offset,
            )
            tail_span = (
                int(pairs.tail_start[pair_index]) - token_offset,
                int(pairs.tail_end[pair_index]) - token_offset,
            )
            logit = float(relation_logits[pair_index].detach().float())
            probability = (
                1.0 / (1.0 + math.exp(-logit))
                if logit >= 0
                else math.exp(logit) / (1.0 + math.exp(logit))
            )
            relation_threshold = getattr(relation_spec, "threshold", None)
            for head_type in getattr(relation_spec, "head", ()):
                head_key = (head_type, *head_span)
                if head_key not in mention_keys:
                    continue
                for tail_type in getattr(relation_spec, "tail", ()):
                    tail_key = (tail_type, *tail_span)
                    if tail_key not in mention_keys:
                        continue
                    edges.append(
                        ScoredRelationEdge(
                            relation_type=relation_type,
                            head=head_key,
                            tail=tail_key,
                            logit=logit,
                            probability=probability,
                            threshold=(
                                0.5
                                if relation_threshold is None
                                else float(relation_threshold)
                            ),
                            candidate_threshold=getattr(
                                relation_spec,
                                "candidate_threshold",
                                None,
                            ),
                        )
                    )
        return extra_mentions, edges

    def _build_lattice(
        self,
        caller_text: str,
        original_schema: Any,
        batch: Any,
        index: int,
        token_emb: torch.Tensor,
        schema_embs: Sequence[Sequence[torch.Tensor]],
        count_top_k: int,
    ) -> ScoreLattice:
        starts_all = list(batch.start_mappings[index])
        ends_all = list(batch.end_mappings[index])

        # collate_fn_inference may append punctuation.  Start mappings at or past
        # len(caller_text) describe that synthetic suffix, so they are excluded.
        caller_token_count = len(starts_all)
        while caller_token_count and starts_all[caller_token_count - 1] >= len(caller_text):
            caller_token_count -= 1
        starts = starts_all[:caller_token_count]
        ends = ends_all[:caller_token_count]

        all_token_count = len(starts_all)
        document_start = max(0, token_emb.shape[0] - all_token_count)
        document_tokens = token_emb[
            document_start:document_start + caller_token_count
        ]
        length = document_tokens.shape[0]
        width = self.model.max_width
        row = torch.arange(length, device=document_tokens.device).unsqueeze(1)
        col = torch.arange(width, device=document_tokens.device).unsqueeze(0)
        dense_ends = row + col
        valid = dense_ends < caller_token_count
        span_starts = row.expand(length, width)

        # First construct every task/count hypothesis and its count-aware query
        # vectors.  All non-empty queries are concatenated so the token-side
        # factorized span computation runs once for the whole document.
        task_specs = []
        query_chunks = []
        query_offset = 0
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
                    (
                        int(count),
                        float(torch.logit(prob.clamp(1e-7, 1 - 1e-7)).detach().cpu()),
                        float(prob.detach().cpu()),
                    )
                    for prob, count in zip(values, indices)
                ]

            hypothesis_specs = []
            for count, count_logit, count_probability in hypotheses:
                if count <= 0:
                    hypothesis_specs.append((
                        count, count_logit, count_probability, None, 0, 0
                    ))
                    continue

                projected = self.model.count_embed(embs[1:], count)
                slots, roles, hidden = projected.shape
                flat = projected.reshape(slots * roles, hidden)
                start = query_offset
                stop = start + flat.shape[0]
                query_offset = stop
                query_chunks.append(flat)
                hypothesis_specs.append((
                    count, count_logit, count_probability,
                    (start, stop), slots, roles
                ))

            task_specs.append((
                task_index, tokens, task_type, fields, hypothesis_specs
            ))

        direct_logits = None
        if query_chunks and length > 0:
            queries = torch.cat(query_chunks, dim=0).unsqueeze(0)
            direct_logits = self.model.span_rep.score_queries(
                document_tokens.unsqueeze(0), queries, mask_invalid=True
            )[0]

        tasks: List[TaskLattice] = []
        for task_index, tokens, task_type, fields, hypothesis_specs in task_specs:
            count_scores: List[CountHypothesis] = []
            for (
                count, count_logit, count_probability, query_slice, slots, roles
            ) in hypothesis_specs:
                if count <= 0 or query_slice is None:
                    role_logits = document_tokens.new_empty(
                        (0, len(fields), length, width)
                    )
                else:
                    start, stop = query_slice
                    role_logits = direct_logits[start:stop].reshape(
                        slots, roles, length, width
                    )
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
