"""Score classification tasks in one encoder pass.

RawScorer cannot see ``[L]`` tasks (its ``_field_names`` matches only
``{[E],[C],[R]}``), and its lattice path computes span-rep einsums that are the
wrong head for classification. So this is a dedicated scorer.

The load-bearing decision: label order is recovered from the ``[L]`` tokens that
were *actually encoded* (``batch.schema_tokens_list``), not from the schema
dict. ``inference/engine.py`` reads ``cls_config["labels"]`` and relies on
``sampling is None`` at inference to keep order, but the processor shuffles and
drops labels whenever sampling is present. Reading the encoded tokens is correct
by construction under any prompt-side transformation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import torch

from ..joint_ie.candidates import center_logit
from ..joint_ie.scoring import resolve_device, resolve_dtype
from .compiler import CompiledClassificationSchema, compile_schema
from .errors import SchemaError

_L = "[L]"


def _label_names(schema_tokens: Sequence[str]) -> tuple:
    """Recover label order from the tokens that were actually encoded."""
    return tuple(schema_tokens[i + 1]
                 for i in range(len(schema_tokens) - 1)
                 if schema_tokens[i] == _L)


def _recover_task_name(schema_tokens: Sequence[str], known: Sequence[str]) -> str:
    """Resolve the owning task by boundary-aware longest match, mirroring
    ``GLiNER2._resolve_classification_config``."""
    prompt_str = schema_tokens[2] if len(schema_tokens) > 2 else ""
    best = None
    for name in known:
        if prompt_str.startswith(name):
            rest = prompt_str[len(name):]
            if rest == "" or rest[0] in (":", " "):
                if best is None or len(name) > len(best):
                    best = name
    if best is not None:
        return best
    return prompt_str.split(" [DESCRIPTION] ", 1)[0].split(":", 1)[0]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(values: Sequence[float]) -> list:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass(frozen=True)
class ClassificationScores:
    """Raw per-label logits plus the compiled schema's fingerprint.

    ``probability`` is presentation (temperature then activation); ``utility`` is
    the objective (temperature then ``center_logit``). Two separate paths on
    purpose.
    """
    text: str
    tasks: Mapping[str, Mapping[str, float]]
    fingerprint: str
    specs: Mapping[str, Any]  # task name -> TaskSpec

    def __post_init__(self):
        frozen = MappingProxyType({
            t: MappingProxyType(dict(v)) for t, v in self.tasks.items()
        })
        object.__setattr__(self, "tasks", frozen)

    def _spec(self, task):
        try:
            return self.specs[task]
        except KeyError:
            raise SchemaError(f"unknown task {task!r}") from None

    def logit(self, task: str, label: str) -> float:
        return self.tasks[task][label]

    def probability(self, task: str, label: str) -> float:
        spec = self._spec(task)
        temp = spec.temperature
        activation = spec.activation
        if activation == "auto":
            activation = "sigmoid" if not spec.is_exclusive else "softmax"
        if activation == "softmax":
            names = list(self.tasks[task])
            scaled = [self.tasks[task][n] / temp for n in names]
            probs = _softmax(scaled)
            return probs[names.index(label)]
        return _sigmoid(self.tasks[task][label] / temp)

    def utility(self, task: str, label: str) -> float:
        spec = self._spec(task)
        return center_logit(self.tasks[task][label] / spec.temperature, spec.threshold)

    def top(self, task: str, k: int = 5) -> tuple:
        items = [(label, self.probability(task, label)) for label in self.tasks[task]]
        items.sort(key=lambda pair: (-pair[1], pair[0]))
        return tuple(items[:k])


class ClassificationScorer:
    """Own scorer: composes around the model like RawScorer, but reads the
    classification head and aligns from encoded tokens."""

    def __init__(self, model: Any, *, device=None, dtype=None):
        self.model = model
        self.processor = model.processor
        self.device = resolve_device(model, device)
        self.dtype = resolve_dtype(model, dtype)

    def to(self, device=None, dtype=None) -> "ClassificationScorer":
        target_device = resolve_device(self.model, device or self.device)
        target_dtype = resolve_dtype(self.model, dtype or self.dtype)
        kwargs = {"device": target_device}
        if target_dtype is not None:
            kwargs["dtype"] = target_dtype
        self.model.to(**kwargs)
        self.device, self.dtype = target_device, target_dtype
        return self

    def eval(self) -> "ClassificationScorer":
        self.model.eval()
        if hasattr(self.processor, "change_mode"):
            self.processor.change_mode(is_training=False)
        return self

    @torch.inference_mode()
    def score(self, text: str, compiled, *, max_len=None) -> ClassificationScores:
        return self.batch_score([text], compiled, batch_size=1, max_len=max_len)[0]

    @torch.inference_mode()
    def batch_score(self, texts: Sequence[str], compiled, *, batch_size: int = 8,
                    max_len: Optional[int] = None,
                    hidden_states: Optional[Sequence[torch.Tensor]] = None) -> list:
        """Score each text's classification tasks.

        Args:
            texts: Input texts.
            compiled: One schema for every text, or one per text.
            batch_size: Texts per encoder pass; ignored with ``hidden_states``.
            max_len: Optional word-token cap per text.
            hidden_states: Optional precomputed encoder output, one tensor per
                text; all texts then collate as one batch.

        Returns:
            One ``ClassificationScores`` per text.

        Raises:
            ValueError: On a bad ``batch_size`` or a count mismatch.
        """
        texts = list(texts)
        if not texts:
            return []
        compiled_list = self._normalize(compiled, len(texts))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if hidden_states is not None:
            if len(hidden_states) != len(texts):
                raise ValueError("hidden_states must have the same length as texts")
            batch_size = len(texts)

        self.eval()
        results: list = []
        for offset in range(0, len(texts), batch_size):
            chunk_texts = texts[offset:offset + batch_size]
            chunk_compiled = compiled_list[offset:offset + batch_size]
            rows = list(zip(chunk_texts, [c.build() for c in chunk_compiled]))
            batch = self.processor.collate_fn_inference(
                rows,
                max_len=max_len,
                architecture=getattr(self.model, "architecture", "span"),
            )
            batch = batch.to(
                self.device,
                self.dtype if self.dtype != torch.float32 else None,
            )
            encoded = self.model.encode_tokens(batch, hidden_states)
            _, schema_embs = self.processor.extract_embeddings_from_batch(
                encoded, batch.input_ids, batch
            )
            for local_index, text in enumerate(chunk_texts):
                results.append(self._score_document(
                    text, chunk_compiled[local_index], batch, local_index,
                    schema_embs[local_index],
                ))
        return results

    def _normalize(self, compiled, n):
        if isinstance(compiled, (list, tuple)):
            if len(compiled) != n:
                raise ValueError("schemas must have the same length as texts")
            return [compile_schema(c) for c in compiled]
        return [compile_schema(compiled)] * n

    def _score_document(self, text, compiled: CompiledClassificationSchema, batch,
                        index, schema_embs) -> ClassificationScores:
        task_types = batch.task_types[index]
        schema_tokens_list = batch.schema_tokens_list[index]
        known = compiled.task_order
        tasks: dict = {}
        for t_idx, task_type in enumerate(task_types):
            if task_type != "classifications":
                continue
            tokens = schema_tokens_list[t_idx]
            names = _label_names(tokens)
            embs = schema_embs[t_idx]
            label_embs = embs[1:]  # drop the [P] prompt row
            stacked = torch.stack([torch.as_tensor(e) for e in label_embs])
            logits = self.model.classifier(stacked).squeeze(-1)
            if logits.shape[0] != len(names):
                raise SchemaError(
                    f"logit/label count mismatch: {logits.shape[0]} logits vs "
                    f"{len(names)} recovered label names"
                )
            task_name = _recover_task_name(tokens, known)
            tasks[task_name] = {
                name: float(logits[j].item()) for j, name in enumerate(names)
            }
        return ClassificationScores(
            text=text,
            tasks=tasks,
            fingerprint=compiled.fingerprint,
            specs={spec.name: spec for spec in compiled.task_specs},
        )
