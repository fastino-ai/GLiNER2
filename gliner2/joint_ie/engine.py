"""High-level constrained Joint IE engine built by composition around GLiNER2."""
from __future__ import annotations

from dataclasses import dataclass, fields
import importlib
import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from gliner2.inference.schema import Schema
from .scoring import RawScorer, ScoreLattice, resolve_device, resolve_dtype


@dataclass(frozen=True)
class JointIEConfig:
    """Call-scoped controls for scoring, candidate construction and decoding."""

    optimizer: str = "beam"
    beam_size: int = 32
    candidate_threshold: float = 0.05
    relation_role_threshold: float = 0.05
    top_k_entities: int = 32
    top_k_roles: int = 12
    count_top_k: int = 2
    batch_size: int = 8
    max_len: Optional[int] = None
    include_confidence: bool = True
    include_spans: bool = True
    entity_threshold: Optional[float] = None
    relation_pair_cap: int = 128
    max_edges_per_type: int = 256
    rescue_per_role: Optional[int] = None
    entity_weight: float = 1.0
    role_weight: float = 1.0
    count_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.optimizer.lower() not in {"beam", "greedy", "auto"}:
            raise ValueError("optimizer must be 'beam' or 'greedy'")
        for name in ("beam_size", "top_k_entities", "top_k_roles", "count_top_k",
                     "batch_size", "relation_pair_cap", "max_edges_per_type"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rescue_per_role is not None and self.rescue_per_role <= 0:
            raise ValueError("rescue_per_role must be positive")


_MODEL_LOAD_OPTIONS = frozenset({"quantize", "compile", "map_location"})
_PREDICTION_OPTIONS = frozenset(item.name for item in fields(JointIEConfig))


def _coerce_config(value: Optional[JointIEConfig]) -> JointIEConfig:
    if value is None:
        return JointIEConfig()
    if not isinstance(value, JointIEConfig):
        raise TypeError("config must be a JointIEConfig")
    return value


def _load_component(module_name: str, names: Sequence[str]) -> Any:
    try:
        module = importlib.import_module(f"gliner2.joint_ie.{module_name}")
    except ImportError as exc:
        if exc.name == f"gliner2.joint_ie.{module_name}":
            return None
        raise
    for name in names:
        component = getattr(module, name, None)
        if component is not None:
            return component
    return None


def _materialize(component: Any, **kwargs: Any) -> Any:
    if component is None or not inspect.isclass(component):
        return component
    try:
        return component(**kwargs)
    except TypeError:
        if kwargs:
            return component()
        raise


def _invoke(callable_obj: Any, **context: Any) -> Any:
    signature = inspect.signature(callable_obj)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return callable_obj(**context)
    kwargs = {name: context[name] for name, parameter in signature.parameters.items()
              if name in context and parameter.kind in (
                  inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    missing = [parameter for parameter in signature.parameters.values()
               if parameter.default is inspect.Parameter.empty
               and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
               and parameter.name not in kwargs]
    if len(missing) == 1:
        for alias in ("schema", "problem", "lattice", "solution", "candidates"):
            if alias in context:
                return callable_obj(context[alias], **kwargs)
    return callable_obj(**kwargs)


def _method(component: Any, names: Sequence[str]) -> Any:
    if component is None:
        return None
    if callable(component) and not inspect.isclass(component):
        return component
    for name in names:
        candidate = getattr(component, name, None)
        if callable(candidate):
            return candidate
    return None


def _schema_cache_key(schema: Any) -> Any:
    if hasattr(schema, "build"):
        schema = schema.build()
    elif hasattr(schema, "schema"):
        schema = schema.schema
    try:
        import json
        return json.dumps(schema, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return id(schema)


class JointIEEngine:
    """Joint extraction facade around an already-loaded GLiNER2 model."""

    def __init__(self, model: Any, *, compiler: Any = None,
                 candidate_generator: Any = None, optimizer: Any = None,
                 result_builder: Any = None,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[Union[str, torch.dtype]] = None):
        self.model = model
        self.scorer = RawScorer(model, device=device, dtype=dtype)
        self.to(device=device, dtype=dtype)
        self.eval()
        compiler = compiler or _load_component(
            "compiler", ("JointSchemaCompiler", "Compiler", "SchemaCompiler", "compile_schema"))
        self.compiler = _materialize(compiler)
        self._candidate_component = candidate_generator or _load_component(
            "candidates", ("CandidateBuilder", "ProblemBuilder", "generate_candidates"))
        self._optimizer_override = optimizer
        self._result_component = result_builder or _load_component(
            "result", ("ResultBuilder", "build_result", "decode_result"))
        self._compiled_schemas: Dict[Any, Any] = {}

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, *, device: Any = None,
                        dtype: Any = None, **kwargs: Any) -> "JointIEEngine":
        """Load a model; prediction controls belong in ``JointIEConfig`` per call."""
        from gliner2 import GLiNER2
        rejected = sorted(set(kwargs) & _PREDICTION_OPTIONS)
        if rejected:
            raise TypeError(
                f"prediction options {rejected} are not accepted by from_pretrained; "
                "pass JointIEConfig(...) as config= during prediction"
            )
        unknown = sorted(set(kwargs) - _MODEL_LOAD_OPTIONS)
        if unknown:
            raise TypeError(f"unknown from_pretrained options: {unknown}")
        model = GLiNER2.from_pretrained(repo_or_dir, **kwargs)
        return cls(model, device=device, dtype=dtype)

    @property
    def device(self) -> torch.device:
        return resolve_device(self.model, self.scorer.device)

    @property
    def dtype(self) -> torch.dtype:
        return resolve_dtype(self.model, self.scorer.dtype)

    def eval(self) -> "JointIEEngine":
        self.scorer.eval()
        return self

    def to(self, device: Any = None, dtype: Any = None) -> "JointIEEngine":
        self.scorer.to(device=device, dtype=dtype)
        return self

    def create_schema(self) -> Any:
        factory = _load_component("schema", ("JointSchema", "Schema", "create_schema"))
        if factory is None:
            return Schema()
        return factory() if callable(factory) else factory

    @staticmethod
    def _is_compiled(schema: Any) -> bool:
        try:
            from .compiler import CompiledJointSchema
            if isinstance(schema, CompiledJointSchema):
                return True
        except ImportError:
            pass
        return (hasattr(schema, "model_schema") and hasattr(schema, "entity_specs") and
                hasattr(schema, "relation_specs") and hasattr(schema, "constraints") and
                callable(getattr(schema, "build", None)))

    def compile_schema(self, schema: Any) -> Any:
        if self._is_compiled(schema):
            return schema
        key = _schema_cache_key(schema)
        if key not in self._compiled_schemas:
            compile_fn = _method(self.compiler, ("compile", "compile_schema"))
            self._compiled_schemas[key] = _invoke(compile_fn, schema=schema) if compile_fn else schema
        return self._compiled_schemas[key]

    def score(self, text: str, schema: Any, *, config: Optional[JointIEConfig] = None) -> ScoreLattice:
        config = _coerce_config(config)
        compiled = schema if self._is_compiled(schema) else self.compile_schema(schema)
        return self.scorer.score(text, compiled, count_top_k=config.count_top_k,
                                 max_len=config.max_len)

    def batch_score(self, texts: Sequence[str], schemas: Any, *,
                    config: Optional[JointIEConfig] = None) -> List[ScoreLattice]:
        config = _coerce_config(config)
        texts = list(texts)
        if isinstance(schemas, (list, tuple)):
            if len(schemas) != len(texts):
                raise ValueError("schemas must have the same length as texts")
            compiled = [schema if self._is_compiled(schema) else self.compile_schema(schema)
                        for schema in schemas]
        else:
            compiled = schemas if self._is_compiled(schemas) else self.compile_schema(schemas)
        return self.scorer.batch_score(texts, compiled, batch_size=config.batch_size,
                                       max_len=config.max_len,
                                       count_top_k=config.count_top_k)

    def _make_candidates(self, config: JointIEConfig) -> Any:
        component = self._candidate_component
        if component is None or not inspect.isclass(component):
            return component
        return component(
            candidate_threshold=config.candidate_threshold,
            relation_role_threshold=config.relation_role_threshold,
            top_k_entities=config.top_k_entities, top_k_roles=config.top_k_roles,
            count_top_k=config.count_top_k, entity_threshold=config.entity_threshold,
            relation_pair_cap=config.relation_pair_cap,
            max_edges_per_type=config.max_edges_per_type,
            rescue_per_role=config.rescue_per_role,
            entity_weight=config.entity_weight, role_weight=config.role_weight,
            count_weight=config.count_weight,
        )

    def _make_optimizer(self, config: JointIEConfig) -> Any:
        if self._optimizer_override is not None:
            return _materialize(self._optimizer_override)
        module = importlib.import_module("gliner2.joint_ie.optimizers")
        if config.optimizer.lower() in {"auto", "greedy"}:
            return module.GreedyOptimizer()
        return module.BeamOptimizer(beam_width=config.beam_size)

    def _make_result_builder(self, config: JointIEConfig) -> Any:
        component = self._result_component
        if component is None or not inspect.isclass(component):
            return component
        return component(include_confidence=config.include_confidence,
                         include_spans=config.include_spans)

    def _problem_from_lattice(self, lattice: ScoreLattice, compiled_schema: Any,
                              candidate_generator: Any) -> Any:
        from .candidates import RelationHypothesis
        entity_task = next((task for task in lattice.tasks if task.task_type == "entities"), None)
        if entity_task is None or not entity_task.count_hypotheses:
            length, width = lattice.valid_span_mask.shape
            entity_logits = torch.empty((0, length, width), device=lattice.valid_span_mask.device)
            entity_types: Tuple[str, ...] = ()
        else:
            entity_logits = entity_task.count_hypotheses[0].role_logits[0]
            entity_types = entity_task.roles
        relation_specs = getattr(compiled_schema, "relation_specs", {})
        hypotheses = []
        for task in lattice.tasks:
            if task.task_type != "relations":
                continue
            spec = relation_specs.get(task.name)
            head_types = getattr(spec, "head", entity_types)
            tail_types = getattr(spec, "tail", entity_types)
            for alternative, count in enumerate(task.count_hypotheses):
                if count.count <= 0:
                    continue
                hypotheses.append(RelationHypothesis(
                    relation_type=task.name, role_logits=count.role_logits,
                    head_types=head_types, tail_types=tail_types,
                    threshold=getattr(spec, "threshold", None),
                    candidate_threshold=getattr(spec, "candidate_threshold", None),
                    count_probability=count.probability, count_utility=count.logit,
                    count_alternative=alternative, hypothesis_id=task.name))
        build = _method(candidate_generator, ("build", "generate", "generate_candidates"))
        if build is None:
            raise RuntimeError("Joint IE decoding requires a CandidateBuilder")
        specs = getattr(compiled_schema, "entity_specs", {})
        return _invoke(
            build, entity_logits=entity_logits, entity_types=entity_types,
            relation_hypotheses=hypotheses,
            constraints=getattr(compiled_schema, "constraints", ()),
            entity_thresholds={n: s.threshold for n, s in specs.items()},
            entity_candidate_thresholds={n: s.candidate_threshold for n, s in specs.items()},
            entity_max_candidates={n: s.max_candidates for n, s in specs.items()
                                   if s.max_candidates is not None},
            lattice=lattice, schema=compiled_schema)

    def _decode(self, lattice: ScoreLattice, compiled_schema: Any,
                config: JointIEConfig) -> Any:
        candidates = self._make_candidates(config)
        optimizer = self._make_optimizer(config)
        result_builder = self._make_result_builder(config)
        problem = self._problem_from_lattice(lattice, compiled_schema, candidates)
        solve = _method(optimizer, ("optimize", "solve", "decode"))
        build = _method(result_builder, ("build", "decode", "build_result"))
        if solve is None or build is None:
            raise RuntimeError("Joint IE decoding requires optimizer and result components")
        solution = _invoke(solve, problem=problem, candidates=problem, candidate_set=problem)
        return _invoke(build, solution=solution, optimized=solution, problem=problem,
                       candidates=problem, text=lattice.text, lattice=lattice, config=config,
                       include_confidence=config.include_confidence,
                       include_spans=config.include_spans)

    @torch.inference_mode()
    def extract(self, text: str, schema: Any, *,
                config: Optional[JointIEConfig] = None,
                return_lattice: bool = False) -> Any:
        config = _coerce_config(config)
        compiled = self.compile_schema(schema)
        lattice = self.score(text, compiled, config=config)
        result = self._decode(lattice, compiled, config)
        return (result, lattice) if return_lattice else result

    @torch.inference_mode()
    def batch_extract(self, texts: Sequence[str], schemas: Any, *,
                      config: Optional[JointIEConfig] = None) -> List[Any]:
        config = _coerce_config(config)
        texts = list(texts)
        if isinstance(schemas, (list, tuple)):
            if len(schemas) != len(texts):
                raise ValueError("schemas must have the same length as texts")
            compiled = [self.compile_schema(schema) for schema in schemas]
            scorer_schemas: Any = compiled
        else:
            one = self.compile_schema(schemas)
            compiled = [one] * len(texts)
            scorer_schemas = one
        lattices = self.batch_score(texts, scorer_schemas, config=config)
        return [self._decode(lattice, schema, config)
                for lattice, schema in zip(lattices, compiled)]

    @torch.inference_mode()
    def extract_long(self, text: str, schema: Any, *,
                     config: Optional[JointIEConfig] = None,
                     chunk_size: int = 384, chunk_overlap: int = 64) -> Any:
        from .long_text import extract_long_text
        config = _coerce_config(config)
        return extract_long_text(self, text, schema=schema, config=config,
                                 chunk_size=chunk_size, chunk_overlap=chunk_overlap)


JointIE = JointIEEngine
__all__ = ["JointIEEngine"]
