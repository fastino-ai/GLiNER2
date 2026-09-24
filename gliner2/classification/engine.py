"""The ``Classifier`` facade and its per-call ``ClassificationConfig``.

One config class, mirroring ``joint_ie``'s ``JointIEConfig`` discipline:
prediction controls live in the config, not in constructor/`from_pretrained`
kwargs. ``from_pretrained`` accepts only model-loading options and rejects
anything else with a message pointing at ``ClassificationConfig``.

The compile cache is an LRU over the schema fingerprint, so re-classifying with
the same schema never recompiles, while a genuinely different schema (different
fingerprint) always does.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import torch

from ..joint_ie.calibration import Calibrator
from ..models.loading import LOAD_OPTIONS
from .compiler import CompiledClassificationSchema, compile_schema, _fingerprint
from .decoding import build_problem, decode as _decode
from .errors import SchemaError
from .result import ResultBuilder
from .schema import ClassificationSchema
from .scoring import ClassificationScorer

_DECODERS = ("auto", "independent", "exact", "beam")
_ON_INFEASIBLE = ("relax", "min_violations", "raise")
_MODEL_LOAD_OPTIONS = LOAD_OPTIONS
_CACHE_CAP = 128


@dataclass(frozen=True)
class ClassificationConfig:
    decoder: str = "auto"
    exact_node_budget: int = 200_000
    beam_size: int = 16
    candidate_threshold: float = 0.5
    max_candidates_per_task: int = 64
    batch_size: int = 8
    max_len: Optional[int] = None
    include_confidence: bool = True
    on_infeasible: str = "relax"
    calibrator: Optional[Calibrator] = None

    def __post_init__(self):
        if self.decoder not in _DECODERS:
            raise ValueError(f"decoder must be one of {_DECODERS}")
        if self.on_infeasible not in _ON_INFEASIBLE:
            raise ValueError(f"on_infeasible must be one of {_ON_INFEASIBLE}")
        if self.exact_node_budget <= 0:
            raise ValueError("exact_node_budget must be positive")
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if not 0 <= self.candidate_threshold <= 1:
            raise ValueError("candidate_threshold must be in [0, 1]")
        if self.max_candidates_per_task <= 0:
            raise ValueError("max_candidates_per_task must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_len is not None and self.max_len <= 0:
            raise ValueError("max_len must be positive or None")


class Classifier:
    """Composes around a GLiNER2 model. Components are direct-import and
    injectable for testing; there is no ``_load_component`` name-guessing."""

    def __init__(self, model: Any, *, device=None, dtype=None,
                 scorer=None, result_builder=None, decoder=None):
        self.model = model
        self.scorer = scorer or ClassificationScorer(model, device=device, dtype=dtype)
        self._result_builder = result_builder or ResultBuilder()
        self._decode = decoder or _decode
        self._compile_cache: "OrderedDict[str, CompiledClassificationSchema]" = OrderedDict()

    # ---- construction --------------------------------------------------

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, *, device=None, dtype=None,
                        **kwargs) -> "Classifier":
        from gliner2 import AutoExtractor
        unknown = sorted(set(kwargs) - _MODEL_LOAD_OPTIONS)
        if unknown:
            raise TypeError(
                f"from_pretrained does not accept {unknown}; prediction controls "
                f"belong in ClassificationConfig(...) passed as config= per call. "
                f"from_pretrained accepts only {sorted(_MODEL_LOAD_OPTIONS)}."
            )
        model = AutoExtractor.from_pretrained(repo_or_dir, **kwargs)
        return cls(model, device=device, dtype=dtype)

    # ---- lifecycle -----------------------------------------------------

    @property
    def device(self):
        return self.scorer.device

    @property
    def dtype(self):
        return self.scorer.dtype

    def to(self, device=None, dtype=None) -> "Classifier":
        self.scorer.to(device=device, dtype=dtype)
        return self

    def eval(self) -> "Classifier":
        self.scorer.eval()
        return self

    # ---- compilation ---------------------------------------------------

    def compile_schema(self, schema) -> CompiledClassificationSchema:
        if isinstance(schema, CompiledClassificationSchema):
            return schema
        if not isinstance(schema, ClassificationSchema):
            raise SchemaError(
                f"expected a ClassificationSchema, got {type(schema).__name__}")
        key = _fingerprint(schema)
        cached = self._compile_cache.get(key)
        if cached is not None:
            self._compile_cache.move_to_end(key)
            return cached
        compiled = compile_schema(schema)
        self._compile_cache[key] = compiled
        self._compile_cache.move_to_end(key)
        while len(self._compile_cache) > _CACHE_CAP:
            self._compile_cache.popitem(last=False)
        return compiled

    # ---- scoring -------------------------------------------------------

    @torch.inference_mode()
    def score(self, text: str, schema, *, config: Optional[ClassificationConfig] = None):
        config = config or ClassificationConfig()
        compiled = self.compile_schema(schema)
        return self.scorer.score(text, compiled, max_len=config.max_len)

    def _compile_each(self, schema):
        if isinstance(schema, (list, tuple)):
            return [self.compile_schema(s) for s in schema]
        return self.compile_schema(schema)

    @torch.inference_mode()
    def batch_score(self, texts, schema, *, config: Optional[ClassificationConfig] = None,
                    hidden_states=None):
        """Score texts against one schema or one schema per text.

        Args:
            texts: Input texts.
            schema: A schema for every text, or a list with one per text.
            config: Prediction controls.
            hidden_states: Optional precomputed encoder output, one tensor per text.

        Returns:
            One ``ClassificationScores`` per text.
        """
        config = config or ClassificationConfig()
        return self.scorer.batch_score(texts, self._compile_each(schema),
                                       batch_size=config.batch_size,
                                       max_len=config.max_len,
                                       hidden_states=hidden_states)

    # ---- decoding ------------------------------------------------------

    @torch.inference_mode()
    def decode(self, scores, schema, *, active=None,
               config: Optional[ClassificationConfig] = None):
        config = config or ClassificationConfig()
        compiled = self.compile_schema(schema)
        if scores.fingerprint != compiled.fingerprint:
            raise SchemaError(
                "scores were produced for a different schema (fingerprint mismatch); "
                "re-score before decoding"
            )
        problem = build_problem(compiled, scores, config, active=active)

        def widen():
            return build_problem(compiled, scores, config, active=active,
                                 full_retention_tasks=problem.task_order)

        solution = self._decode(problem, config, widen=widen)
        return self._result_builder.build(
            compiled, scores, solution, active_order=problem.task_order,
            include_confidence=config.include_confidence)

    # ---- one-shot ------------------------------------------------------

    @torch.inference_mode()
    def classify(self, text: str, schema, *, active=None,
                 config: Optional[ClassificationConfig] = None):
        config = config or ClassificationConfig()
        scores = self.score(text, schema, config=config)
        return self.decode(scores, schema, active=active, config=config)

    @torch.inference_mode()
    def batch_classify(self, texts, schema, *, active=None,
                       config: Optional[ClassificationConfig] = None,
                       hidden_states=None):
        """Classify texts against one schema or one schema per text.

        Args:
            texts: Input texts.
            schema: A schema for every text, or a list with one per text.
            active: Optional subset of tasks to decode.
            config: Prediction controls.
            hidden_states: Optional precomputed encoder output, one tensor per text.

        Returns:
            One decoded classification result per text.
        """
        config = config or ClassificationConfig()
        texts = list(texts)
        compiled = self._compile_each(schema)
        scores_list = self.batch_score(texts, compiled, config=config,
                                       hidden_states=hidden_states)
        per_text = compiled if isinstance(compiled, list) else [compiled] * len(texts)
        return [self.decode(scores, one, active=active, config=config)
                for scores, one in zip(scores_list, per_text, strict=True)]

    @torch.inference_mode()
    def classify_long(self, text: str, schema, *, active=None,
                      config: Optional[ClassificationConfig] = None,
                      chunk_size: int = 384, chunk_overlap: int = 64,
                      aggregate: str = "max"):
        """Aggregate per-chunk logits, then decode once (see ``long_text``)."""
        from .long_text import classify_long
        return classify_long(self, text, schema, config=config, active=active,
                             chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                             aggregate=aggregate)
