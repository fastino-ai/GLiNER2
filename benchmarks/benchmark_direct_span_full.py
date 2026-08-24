#!/usr/bin/env python3
"""GLiNER2 exact-legacy vs current direct-span benchmark.

One script, two fresh subprocesses:
  * legacy: exact git commit a91fd1d2...
  * current: the working tree on disk (including uncommitted patches)

Primary measurement:
  * controlled span-scaling grid over token length L and query count Q
  * default grid: L={64,256}, Q={1,8,32}, B={1,8}

Secondary measurements:
  * entities / structures / relations / mixed: full end-to-end inference
  * Joint-IE: batch_score() for speed, small full batch_extract() for parity
  * training: real forward+backward on CUDA only

CPU policy (automatic): 75% logical CPUs for PyTorch compute, 25% for
DataLoader preprocessing. CUDA policy: 1 host compute/orchestration thread and
all remaining logical CPUs as preprocessing workers. No MPS support.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

LEGACY_COMMIT = "a91fd1d2c72debe43907296c8e375a036c6a4faf"
VERSION = "exact-legacy-lq-scaling-unfold-baddbmm-v2"
CURRENT_DIRECT_KERNEL = "unfold_baddbmm"

ENTITY_TYPES = ["person", "organization", "location", "date", "product", "event"]
RELATION_TYPES = ["works_for", "founded", "located_in", "acquired"]
STRUCTURES = {
    "employment": ["person::str", "organization::str", "location::str", "date::str"],
    "event_record": ["name::str", "organization::str", "location::str", "date::str"],
}
TASKS = ("entities", "structures", "relations", "mixed", "joint")

SEEDS = [
    "Apple CEO Tim Cook announced a new iPhone in Cupertino on September 12, 2023.",
    "Google CEO Sundar Pichai presented Gemini features in Mountain View.",
    "Microsoft CEO Satya Nadella discussed Azure infrastructure in Seattle.",
    "NVIDIA CEO Jensen Huang presented CUDA systems at GTC in San Jose.",
    "AMD CEO Lisa Su introduced EPYC processors during an event in Austin.",
    "Sarah Chen founded Nova Robotics in Boston in 2021 and opened an office in Berlin.",
    "Daniel Ortiz works for Horizon Systems in Madrid.",
    "Atlas Analytics acquired BrightData Labs in London in March 2024.",
    "Maria Ivanova joined CloudWorks in Sofia and presented Orion in Vienna.",
    "Researchers from ETH Zurich and the University of Toronto presented work at NeurIPS.",
]


def csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def csv_int(value: str) -> List[int]:
    return [int(x) for x in csv(value)]


def csv_float(value: str) -> List[float]:
    return [float(x) for x in csv(value)]


def seed_everything(seed: int) -> None:
    """Reset all RNGs used by data preparation/model dropout outside timed regions."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**20


def clear(device: Optional[torch.device] = None) -> None:
    gc.collect()
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()


def cpu_plan(device: str) -> Tuple[int, int, int]:
    """Return (logical_cpus, torch_compute_threads, preprocessing_workers)."""
    total = os.cpu_count() or 1
    if device == "cpu":
        if total == 1:
            return 1, 1, 0
        workers = max(1, round(total * 0.25))
        workers = min(workers, total - 1)
        return total, total - workers, workers
    if device == "cuda":
        return total, 1, max(0, total - 1)
    raise ValueError(f"unsupported device: {device}")


def set_torch_threads(n: int) -> None:
    torch.set_num_threads(max(1, n))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine() or "CPU"


def available_devices(requested: Sequence[str]) -> List[str]:
    if requested == ["auto"]:
        return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    bad = set(requested) - {"cpu", "cuda"}
    if bad:
        raise ValueError(f"only cpu,cuda are supported; got {sorted(bad)}")
    return [d for d in requested if d != "cuda" or torch.cuda.is_available()]


def dtypes_for(device: str, requested: Sequence[str]) -> List[str]:
    if requested != ["auto"]:
        return list(requested)
    if device == "cpu":
        return ["fp32", "bf16", "fp16"]
    out = ["fp32", "fp16"]
    if torch.cuda.is_bf16_supported():
        out.append("bf16")
    return out


def jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return float(obj.detach().float().cpu())
        return obj.detach().cpu().tolist()
    if hasattr(obj, "to_dict"):
        try:
            return jsonable(obj.to_dict(include_confidence=True, include_spans=True))
        except TypeError:
            try:
                return jsonable(obj.to_dict())
            except Exception:
                pass
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


CONF_KEYS = {"confidence", "score", "probability"}


def semantic(obj: Any) -> Any:
    obj = jsonable(obj)
    if isinstance(obj, dict):
        return {k: semantic(v) for k, v in sorted(obj.items()) if k not in CONF_KEYS}
    if isinstance(obj, list):
        vals = [semantic(v) for v in obj]
        try:
            return sorted(vals, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
        except Exception:
            return vals
    return obj


def confidences(obj: Any, out: Optional[List[float]] = None) -> List[float]:
    out = [] if out is None else out
    obj = jsonable(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in CONF_KEYS and isinstance(v, (int, float)):
                out.append(float(v))
            else:
                confidences(v, out)
    elif isinstance(obj, list):
        for v in obj:
            confidences(v, out)
    return sorted(out)


def signature(outputs: Sequence[Any]) -> Dict[str, Any]:
    hashes, confs = [], []
    for output in outputs:
        payload = json.dumps(semantic(output), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        hashes.append(hashlib.sha256(payload.encode()).hexdigest())
        confs.append(confidences(output))
    return {"hashes": hashes, "confidences": confs}


def compare_signature(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    n = max(len(a["hashes"]), len(b["hashes"]))
    exact = sum(x == y for x, y in zip(a["hashes"], b["hashes"]))
    diffs: List[float] = []
    shape_mismatch = 0
    for x, y in zip(a["confidences"], b["confidences"]):
        if len(x) != len(y):
            shape_mismatch += 1
        else:
            diffs.extend(abs(float(i) - float(j)) for i, j in zip(x, y))
    return {
        "documents": n,
        "exact_documents": exact,
        "mismatched_documents": n - exact,
        "match_rate": exact / max(1, n),
        "confidence_shape_mismatches": shape_mismatch,
        "confidence_mean_abs": statistics.mean(diffs) if diffs else 0.0,
        "confidence_max_abs": max(diffs) if diffs else 0.0,
    }


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def load_corpus(path: Path, limit: int) -> List[str]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            rows.append(value["text"] if isinstance(value, dict) else str(value))
            if limit and len(rows) >= limit:
                break
        return rows
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("texts", value.get("documents", value.get("data", [])))
        rows = [x["text"] if isinstance(x, dict) else str(x) for x in value]
        return rows[:limit] if limit else rows
    rows = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows[:limit] if limit else rows


def generate_corpus(model_id: str, docs: int, target_lengths: Sequence[int], seed: int) -> List[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    rng = random.Random(seed)
    out = []
    for i in range(docs):
        target = target_lengths[i % len(target_lengths)]
        parts = []
        while True:
            parts.append(rng.choice(SEEDS))
            text = " ".join(parts)
            if len(tokenizer.encode(text, add_special_tokens=False)) >= target:
                break
        words = text.split()
        lo, hi = 1, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = " ".join(words[:mid])
            if len(tokenizer.encode(candidate, add_special_tokens=False)) <= target:
                lo = mid
            else:
                hi = mid - 1
        out.append(" ".join(words[:lo]))
    return out



def generate_exact_token_corpus(
    model_id: str,
    docs: int,
    target_tokens: int,
    seed: int,
) -> List[str]:
    """Generate deterministic texts that re-tokenize to exactly target_tokens."""
    from transformers import AutoTokenizer

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    rng = random.Random(seed)
    out: List[str] = []

    for _ in range(docs):
        parts: List[str] = []
        ids: List[int] = []
        while len(ids) < target_tokens + 32:
            parts.append(rng.choice(SEEDS))
            ids = tokenizer.encode(" ".join(parts), add_special_tokens=False)

        # Decode an exact token prefix. For the tokenizer used by the benchmark,
        # encode(decode(prefix)) should normally round-trip exactly.
        candidate = tokenizer.decode(
            ids[:target_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        actual = tokenizer.encode(candidate, add_special_tokens=False)

        # A normalization-sensitive tokenizer can occasionally change the count
        # after decode/re-encode. Search nearby prefixes before giving up.
        if len(actual) != target_tokens:
            found = None
            lo = max(1, target_tokens - 16)
            hi = min(len(ids), target_tokens + 16)
            for n in range(lo, hi + 1):
                c = tokenizer.decode(
                    ids[:n],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if len(tokenizer.encode(c, add_special_tokens=False)) == target_tokens:
                    found = c
                    break
            if found is None:
                raise RuntimeError(
                    f"could not construct text with exactly {target_tokens} tokens"
                )
            candidate = found

        out.append(candidate)

    return out


def corpus_hash(texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode())
        h.update(b"\0")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Inference tasks
# ---------------------------------------------------------------------------


def structure_schema(model):
    schema = model.create_schema()
    for name, fields in STRUCTURES.items():
        builder = schema.structure(name)
        for spec in fields:
            field, dtype = spec.split("::", 1)
            builder.field(field, dtype=dtype)
    return schema


def mixed_schema(model):
    schema = model.create_schema()
    schema.entities(["person", "organization", "location", "date", "product"])
    schema.classification("document_type", ["news", "technical", "business", "other"])
    schema.structure("employment").field("person").field("organization").field("location")
    schema.structure("event_record").field("name").field("date").field("location")
    schema.relations(["works_for", "located_in", "founded"])
    return schema


def standard_schema(model, task: str):
    if task == "entities":
        return model.create_schema().entities(ENTITY_TYPES)
    if task == "structures":
        return structure_schema(model)
    if task == "relations":
        return model.create_schema().relations(RELATION_TYPES)
    if task == "mixed":
        return mixed_schema(model)
    raise ValueError(task)


def run_standard(model, task: str, texts: Sequence[str], bs: int, threshold: float, max_len: int, workers: int):
    return model.batch_extract(
        list(texts), standard_schema(model, task),
        batch_size=bs,
        threshold=threshold,
        num_workers=workers,
        format_results=True,
        include_confidence=True,
        include_spans=True,
        max_len=max_len,
    )



def scaling_entity_types(query_count: int) -> List[str]:
    """Deterministic synthetic entity labels used only for controlled Q scaling."""
    if query_count <= 0:
        raise ValueError("query_count must be positive")
    return [f"benchmark_entity_{i:02d}" for i in range(query_count)]


def run_scaling(
    model,
    texts: Sequence[str],
    query_count: int,
    bs: int,
    max_len: int,
    workers: int,
):
    """Controlled E2E entity extraction for a fixed token length and query count.

    Synthetic labels make Q explicit while keeping the model/public inference path
    identical between exact legacy and current. Formatting/confidence/span output is
    disabled so the table is dominated by model/scoring work rather than result
    serialization.
    """
    cache = getattr(model, "_benchmark_scaling_schemas", None)
    if cache is None:
        cache = {}
        setattr(model, "_benchmark_scaling_schemas", cache)
    schema = cache.get(query_count)
    if schema is None:
        schema = model.create_schema().entities(scaling_entity_types(query_count))
        cache[query_count] = schema

    return model.batch_extract(
        list(texts),
        schema,
        batch_size=bs,
        threshold=0.5,
        num_workers=workers,
        format_results=False,
        include_confidence=False,
        include_spans=False,
        max_len=max_len,
    )


def joint_components(model):
    from gliner2.joint_ie import JointIE, JointIEConfig
    from gliner2.joint_ie.schema import JointSchema

    schema = JointSchema().entities(["person", "organization", "location", "product"])
    schema.relation("works_for", head=["person"], tail=["organization"])
    schema.relation("located_in", head=["organization"], tail=["location"])
    return JointIE(model), schema, JointIEConfig


def joint_config(JointIEConfig, bs: int, max_len: int):
    return JointIEConfig(
        optimizer="greedy",
        batch_size=bs,
        count_top_k=2,
        max_len=max_len,
        include_confidence=True,
        include_spans=True,
    )


def run_joint_score(model, texts: Sequence[str], bs: int, max_len: int) -> int:
    engine, schema, JointIEConfig = joint_components(model)
    previous = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    try:
        compiled = engine.compile_schema(schema)
        lattices = engine.batch_score(
            list(texts), compiled, config=joint_config(JointIEConfig, bs, max_len)
        )
        return len(lattices)
    finally:
        if previous is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = previous


def run_joint_full(model, texts: Sequence[str], bs: int, max_len: int):
    engine, schema, JointIEConfig = joint_components(model)
    previous = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    try:
        results = engine.batch_extract(
            list(texts), schema, config=joint_config(JointIEConfig, bs, max_len)
        )
        return [jsonable(x) for x in results]
    finally:
        if previous is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = previous


def timed(fn, device: torch.device, repeats: int) -> Tuple[float, List[float], Optional[float]]:
    times, peaks = [], []
    for _ in range(repeats):
        reset_peak(device)
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
        p = peak_mb(device)
        if p is not None:
            peaks.append(p)
    return statistics.median(times), times, max(peaks) if peaks else None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def training_examples():
    from gliner2.training.data import Classification, InputExample, Relation, Structure

    return [
        InputExample(
            text="Tim Cook works for Apple in Cupertino and presented the iPhone.",
            entities={"person": ["Tim Cook"], "organization": ["Apple"], "location": ["Cupertino"], "product": ["iPhone"]},
            classifications=[Classification("document_type", ["news", "technical", "business"], "news")],
            structures=[Structure("employment", person="Tim Cook", organization="Apple", location="Cupertino")],
            relations=[Relation("works_for", head="Tim Cook", tail="Apple")],
        ),
        InputExample(
            text="Sundar Pichai works for Google in Mountain View and discussed Gemini.",
            entities={"person": ["Sundar Pichai"], "organization": ["Google"], "location": ["Mountain View"], "product": ["Gemini"]},
            classifications=[Classification("document_type", ["news", "technical", "business"], "technical")],
            structures=[Structure("employment", person="Sundar Pichai", organization="Google", location="Mountain View")],
            relations=[Relation("works_for", head="Sundar Pichai", tail="Google")],
        ),
        InputExample(
            text="Sarah Chen founded Nova Robotics in Boston in 2021.",
            entities={"person": ["Sarah Chen"], "organization": ["Nova Robotics"], "location": ["Boston"], "date": ["2021"]},
            classifications=[Classification("document_type", ["news", "technical", "business"], "business")],
            structures=[Structure("event_record", name="Nova Robotics", date="2021", location="Boston")],
            relations=[Relation("founded", head="Sarah Chen", tail="Nova Robotics")],
        ),
        InputExample(
            text="Daniel Ortiz works for Horizon Systems in Madrid.",
            entities={"person": ["Daniel Ortiz"], "organization": ["Horizon Systems"], "location": ["Madrid"]},
            classifications=[Classification("document_type", ["news", "technical", "business"], "business")],
            structures=[Structure("employment", person="Daniel Ortiz", organization="Horizon Systems", location="Madrid")],
            relations=[Relation("works_for", head="Daniel Ortiz", tail="Horizon Systems")],
        ),
    ]


def training_batch(model, bs: int, max_len: int, device: torch.device, dtype: torch.dtype):
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

    model.processor.change_mode(is_training=True)
    examples = training_examples()
    rows = [examples[i % len(examples)] for i in range(bs)]
    dataset = ExtractorDataset.from_examples(rows, shuffle=False, validate=True)
    collator = ExtractorCollator(model.processor, is_training=True, max_len=max_len, architecture=model.architecture)
    return collator([dataset[i] for i in range(len(dataset))]).to(device, dtype=dtype)


def loss_dict(out: Dict[str, Any]) -> Dict[str, float]:
    result = {}
    for key in ("total_loss", "classification_loss", "structure_loss", "count_loss"):
        if key in out:
            value = out[key]
            result[key] = float(value.detach().float().cpu()) if torch.is_tensor(value) else float(value)
    return result


def benchmark_training(
    model,
    device: torch.device,
    dtype: torch.dtype,
    sizes: Sequence[int],
    repeats: int,
    warmup: int,
    max_len: int,
    base_seed: int,
):
    """Benchmark real forward+backward with identical RNG state across variants.

    Batch construction and every model invocation are seeded before entering the
    timed region. This keeps schema/data randomization and dropout comparable
    between the exact-legacy and current subprocesses without timing RNG setup.
    """
    rows = []
    for bs in sizes:
        try:
            batch_seed = base_seed + 100_000 + bs
            seed_everything(batch_seed)
            batch = training_batch(model, bs, max_len, device, dtype)

            # Eval-mode loss is the cross-process numerical parity reference.
            # Re-seed even in eval mode because this makes the benchmark robust
            # to any stochastic preprocessing/model component added later.
            model.eval()
            eval_seed = base_seed + 200_000 + bs
            seed_everything(eval_seed)
            deterministic = loss_dict(model(batch))

            model.train()
            for i in range(warmup):
                model.zero_grad(set_to_none=True)
                seed_everything(base_seed + 300_000 + bs * 1_000 + i)
                model(batch)["total_loss"].backward()
            sync(device)

            times, peaks = [], []
            for i in range(repeats):
                model.zero_grad(set_to_none=True)
                # Seed before reset/sync/timer so RNG setup is not benchmarked.
                seed_everything(base_seed + 400_000 + bs * 1_000 + i)
                reset_peak(device)
                sync(device)
                t0 = time.perf_counter()
                model(batch)["total_loss"].backward()
                sync(device)
                times.append(time.perf_counter() - t0)
                p = peak_mb(device)
                if p is not None:
                    peaks.append(p)
            med = statistics.median(times)
            rows.append({
                "batch_size": bs,
                "training_batch_seed": batch_seed,
                "eval_seed": eval_seed,
                "samples_per_second": bs / med,
                "median_seconds": med,
                "times": times,
                "peak_allocated_mb": max(peaks) if peaks else None,
                "deterministic_eval_loss": deterministic,
            })
            del batch
        except (RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
            rows.append({"batch_size": bs, "error": f"{type(exc).__name__}: {exc}"})
            clear(device)
    model.eval()
    model.processor.change_mode(is_training=False)
    return rows


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def load_model(model_id: str, device: torch.device, dtype: torch.dtype, variant: str):
    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained(model_id, map_location=str(device)).to(device=device, dtype=dtype)
    model.eval()
    if variant == "current":
        needed = [
            (model, "compute_span_scores_batched"),
            (model, "_compute_training_span_scores_batched"),
            (model.span_rep, "score_queries"),
        ]
        missing = [name for obj, name in needed if not hasattr(obj, name)]
        if missing:
            raise RuntimeError("current checkout is missing direct-span patch: " + ", ".join(missing))
    return model


def worker(repo: Path, variant: str, config_path: Path, corpus_path: Path, output_path: Path) -> int:
    repo = repo.resolve()
    sys.path.insert(0, str(repo))
    import gliner2

    if repo not in Path(gliner2.__file__).resolve().parents:
        raise RuntimeError(f"wrong gliner2 imported: {gliner2.__file__}")

    cfg = json.loads(config_path.read_text())
    corpus_payload = json.loads(corpus_path.read_text())
    if isinstance(corpus_payload, list):
        # Backward-compatible worker input.
        texts = corpus_payload
        scaling_corpora = {}
    else:
        texts = corpus_payload["main"]
        scaling_corpora = {
            int(k): v for k, v in corpus_payload.get("scaling", {}).items()
        }

    result = {
        "variant": variant,
        "direct_kernel": CURRENT_DIRECT_KERNEL if variant == "current" else "legacy_span_materialization",
        "benchmark_seed": cfg["seed"],
        "corpus_sha256": corpus_hash(texts),
        "scaling_corpus_sha256": {
            str(k): corpus_hash(v) for k, v in sorted(scaling_corpora.items())
        },
        "runs": [],
    }

    for dev_name in available_devices(cfg["devices"]):
        total, compute_threads, workers = cpu_plan(dev_name)
        set_torch_threads(compute_threads)
        os.environ["TOKENIZERS_PARALLELISM"] = "false" if workers else "true"
        print(f"\n[{variant}] {dev_name}: logical={total} compute_threads={compute_threads} preprocess_workers={workers}", flush=True)

        for dtype_name in dtypes_for(dev_name, cfg["dtypes"]):
            if dev_name == "cuda" and dtype_name == "bf16" and not torch.cuda.is_bf16_supported():
                continue
            device, dtype = torch.device(dev_name), dtype_from_name(dtype_name)
            run = {"device": dev_name, "device_name": device_label(device), "dtype": dtype_name, "tasks": {}}
            print(f"  dtype={dtype_name}", flush=True)
            try:
                model = load_model(cfg["model"], device, dtype, variant)

                # Primary benchmark: controlled L x Q scaling. Batch size is kept
                # deliberately small as a secondary axis (default B={1,8}).
                scaling_rows = []
                if cfg.get("scaling", True):
                    print("    controlled L x Q scaling", flush=True)
                    for token_length in cfg["scaling_token_lengths"]:
                        scale_texts = scaling_corpora.get(int(token_length), [])
                        if not scale_texts:
                            continue
                        for query_count in cfg["scaling_query_counts"]:
                            for bs in cfg["batch_sizes"]:
                                warm = scale_texts[:min(
                                    len(scale_texts),
                                    max(cfg["warmup_docs"], bs * 2),
                                )]
                                try:
                                    run_scaling(
                                        model, warm, query_count, bs,
                                        cfg["max_len"], workers,
                                    )
                                    fn = lambda tl=token_length, q=query_count, bs=bs: run_scaling(
                                        model,
                                        scaling_corpora[int(tl)],
                                        q,
                                        bs,
                                        cfg["max_len"],
                                        workers,
                                    )
                                    med, times, peak = timed(fn, device, cfg["repeats"])
                                    dps = len(scale_texts) / med
                                    scaling_rows.append({
                                        "tokens": int(token_length),
                                        "queries": int(query_count),
                                        "batch_size": int(bs),
                                        "docs_per_second": dps,
                                        "median_seconds": med,
                                        "times": times,
                                        "peak_allocated_mb": peak,
                                    })
                                    print(
                                        f"      L={token_length:<3} Q={query_count:<2} "
                                        f"B={bs:<2} {dps:9.2f} docs/s",
                                        flush=True,
                                    )
                                except (RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
                                    scaling_rows.append({
                                        "tokens": int(token_length),
                                        "queries": int(query_count),
                                        "batch_size": int(bs),
                                        "error": f"{type(exc).__name__}: {exc}",
                                    })
                                    print(
                                        f"      L={token_length:<3} Q={query_count:<2} "
                                        f"B={bs:<2} SKIP {exc}",
                                        flush=True,
                                    )
                                    clear(device)
                run["scaling"] = scaling_rows

                for task in cfg["tasks"]:
                    print(f"    task={task}", flush=True)
                    speed_rows = []
                    for bs in cfg["batch_sizes"]:
                        warm = texts[:min(len(texts), max(cfg["warmup_docs"], bs * 2))]
                        try:
                            if task == "joint":
                                run_joint_score(model, warm, bs, cfg["max_len"])
                                fn = lambda: run_joint_score(model, texts, bs, cfg["max_len"])
                                scope = "joint_batch_score"
                            else:
                                run_standard(model, task, warm, bs, cfg["speed_threshold"], cfg["max_len"], workers)
                                fn = lambda task=task, bs=bs: run_standard(
                                    model, task, texts, bs, cfg["speed_threshold"], cfg["max_len"], workers
                                )
                                scope = "end_to_end"
                            med, times, peak = timed(fn, device, cfg["repeats"])
                            speed_rows.append({
                                "batch_size": bs,
                                "docs_per_second": len(texts) / med,
                                "median_seconds": med,
                                "times": times,
                                "peak_allocated_mb": peak,
                            })
                            print(f"      bs={bs:<3} {len(texts)/med:9.2f} docs/s", flush=True)
                        except (RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
                            speed_rows.append({"batch_size": bs, "error": f"{type(exc).__name__}: {exc}"})
                            print(f"      bs={bs:<3} SKIP {exc}", flush=True)
                            clear(device)

                    good = [r["batch_size"] for r in speed_rows if "error" not in r]
                    qbs = cfg["quality_batch_size"] or (max(good) if good else 1)
                    qn = min(len(texts), cfg["joint_quality_docs"] if task == "joint" else cfg["quality_docs"])
                    qtexts = texts[:qn]
                    quality = []
                    if task == "joint":
                        try:
                            quality.append({"threshold": None, **signature(run_joint_full(model, qtexts, qbs, cfg["max_len"]))})
                        except Exception as exc:
                            quality.append({"threshold": None, "error": f"{type(exc).__name__}: {exc}"})
                    else:
                        for threshold in cfg["thresholds"]:
                            try:
                                out = run_standard(model, task, qtexts, qbs, threshold, cfg["max_len"], workers)
                                quality.append({"threshold": threshold, **signature(out)})
                            except Exception as exc:
                                quality.append({"threshold": threshold, "error": f"{type(exc).__name__}: {exc}"})
                    run["tasks"][task] = {"speed_scope": scope, "speed": speed_rows, "quality": quality}

                if cfg["training"] and dev_name == "cuda":
                    print("    training forward+backward", flush=True)
                    run["training"] = benchmark_training(
                        model, device, dtype, cfg["training_batch_sizes"],
                        cfg["training_repeats"], cfg["training_warmup"], cfg["max_len"],
                        cfg["seed"],
                    )
                del model
                clear(device)
            except (RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
                run["error"] = f"{type(exc).__name__}: {exc}"
                print(f"    CONFIG FAIL: {run['error']}", flush=True)
                clear(device)
            result["runs"].append(run)
            output_path.write_text(json.dumps(result, indent=2))

    output_path.write_text(json.dumps(result, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Parent: export exact commit, run both, compare
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    here = Path(__file__).resolve().parent.parent
    if (here / ".git").exists():
        return here
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def export_commit(repo: Path, commit: str, dest: Path) -> None:
    subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo, check=True)
    tar_path = dest.parent / "legacy.tar"
    subprocess.run(["git", "archive", "--format=tar", "-o", str(tar_path), commit], cwd=repo, check=True)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest)
    tar_path.unlink()


def launch(script: Path, repo: Path, variant: str, cfg: Path, corpus: Path, out: Path) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    # Legacy and current run in separate Python processes. Python's hash seed
    # affects set iteration order, and GLiNER2 schema preprocessing contains
    # set -> list conversions for structure fields. Fix it explicitly so both
    # workers construct identical schemas/batches before model-path comparison.
    benchmark_seed = int(json.loads(cfg.read_text())["seed"])
    env["PYTHONHASHSEED"] = str(benchmark_seed)
    # Avoid inherited one-thread caps. Device-specific torch.set_num_threads() is
    # applied inside the worker; these are only safe upper bounds for native libs.
    total = str(os.cpu_count() or 1)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        env[name] = total
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["TOKENIZERS_PARALLELISM"] = "false"
    cmd = [
        sys.executable, str(script), "--_worker", "--_repo", str(repo),
        "--_variant", variant, "--_config", str(cfg), "--_corpus", str(corpus),
        "--_worker-output", str(out),
    ]
    print("\n" + "=" * 88)
    print(f"RUNNING {variant.upper()}: {repo}")
    print("=" * 88, flush=True)
    subprocess.run(cmd, cwd=repo, env=env, check=True)


def indexed_runs(data: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {(r["device"], r["dtype"]): r for r in data["runs"]}


def rows_by_bs(rows: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(r["batch_size"]): r for r in rows}


def scaling_index(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    return {
        (int(r["tokens"]), int(r["queries"]), int(r["batch_size"])): r
        for r in rows
    }


def compare(legacy: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    out = {"configs": []}
    lr, cr = indexed_runs(legacy), indexed_runs(current)
    for key in sorted(set(lr) & set(cr)):
        a, b = lr[key], cr[key]
        cfg = {"device": key[0], "dtype": key[1], "tasks": {}}
        if "error" in a or "error" in b:
            cfg["error"] = {"legacy": a.get("error"), "current": b.get("error")}
            out["configs"].append(cfg)
            continue

        scaling = []
        sa, sb = scaling_index(a.get("scaling", [])), scaling_index(b.get("scaling", []))
        for skey in sorted(set(sa) & set(sb)):
            x, y = sa[skey], sb[skey]
            row = {
                "tokens": skey[0],
                "queries": skey[1],
                "batch_size": skey[2],
            }
            if "error" in x or "error" in y:
                row.update({
                    "legacy_error": x.get("error"),
                    "current_error": y.get("error"),
                })
            else:
                row.update({
                    "legacy_docs_per_second": x["docs_per_second"],
                    "current_docs_per_second": y["docs_per_second"],
                    "speedup": x["median_seconds"] / y["median_seconds"],
                    "legacy_peak_allocated_mb": x.get("peak_allocated_mb"),
                    "current_peak_allocated_mb": y.get("peak_allocated_mb"),
                })
            scaling.append(row)
        cfg["scaling"] = scaling

        for task in sorted(set(a["tasks"]) & set(b["tasks"])):
            at, bt = a["tasks"][task], b["tasks"][task]
            speed = []
            aa, bb = rows_by_bs(at["speed"]), rows_by_bs(bt["speed"])
            for bs in sorted(set(aa) & set(bb)):
                x, y = aa[bs], bb[bs]
                if "error" in x or "error" in y:
                    speed.append({"batch_size": bs, "legacy_error": x.get("error"), "current_error": y.get("error")})
                else:
                    speed.append({
                        "batch_size": bs,
                        "legacy_docs_per_second": x["docs_per_second"],
                        "current_docs_per_second": y["docs_per_second"],
                        "speedup": x["median_seconds"] / y["median_seconds"],
                        "legacy_peak_allocated_mb": x.get("peak_allocated_mb"),
                        "current_peak_allocated_mb": y.get("peak_allocated_mb"),
                    })
            quality = []
            aq = {str(x.get("threshold")): x for x in at["quality"]}
            bq = {str(x.get("threshold")): x for x in bt["quality"]}
            for k in aq.keys() & bq.keys():
                x, y = aq[k], bq[k]
                if "error" in x or "error" in y:
                    quality.append({"threshold": x.get("threshold"), "legacy_error": x.get("error"), "current_error": y.get("error")})
                else:
                    quality.append({"threshold": x.get("threshold"), **compare_signature(x, y)})
            quality.sort(key=lambda x: -1 if x.get("threshold") is None else float(x["threshold"]))
            cfg["tasks"][task] = {"speed_scope": at["speed_scope"], "speed": speed, "quality": quality}

        if "training" in a and "training" in b:
            train = []
            aa, bb = rows_by_bs(a["training"]), rows_by_bs(b["training"])
            for bs in sorted(set(aa) & set(bb)):
                x, y = aa[bs], bb[bs]
                if "error" in x or "error" in y:
                    train.append({"batch_size": bs, "legacy_error": x.get("error"), "current_error": y.get("error")})
                else:
                    keys = set(x["deterministic_eval_loss"]) | set(y["deterministic_eval_loss"])
                    legacy_peak = x.get("peak_allocated_mb")
                    current_peak = y.get("peak_allocated_mb")
                    train.append({
                        "batch_size": bs,
                        "legacy_samples_per_second": x["samples_per_second"],
                        "current_samples_per_second": y["samples_per_second"],
                        "speedup": x["median_seconds"] / y["median_seconds"],
                        "legacy_peak_allocated_mb": legacy_peak,
                        "current_peak_allocated_mb": current_peak,
                        "memory_ratio_legacy_over_current": (
                            legacy_peak / current_peak
                            if legacy_peak is not None and current_peak not in (None, 0)
                            else None
                        ),
                        "training_batch_seed": x.get("training_batch_seed"),
                        "loss_abs_diff": {k: abs(x["deterministic_eval_loss"].get(k, 0.0) - y["deterministic_eval_loss"].get(k, 0.0)) for k in keys},
                    })
            cfg["training"] = train
        out["configs"].append(cfg)
    return out


def print_comparison(data: Dict[str, Any]) -> None:
    for cfg in data["configs"]:
        print("\n" + "#" * 92)
        print(f"device={cfg['device']} dtype={cfg['dtype']}")
        print("#" * 92)
        if "error" in cfg:
            print(cfg["error"])
            continue

        if cfg.get("scaling"):
            print("\nCONTROLLED TOKEN x QUERY SCALING")
            print(" Tokens | Queries | B | legacy docs/s | current docs/s | speedup")
            print("-" * 72)
            for r in cfg["scaling"]:
                if "legacy_error" in r or "current_error" in r:
                    print(
                        f"{r['tokens']:7d} | {r['queries']:7d} | {r['batch_size']:1d} | "
                        f"ERROR legacy={r.get('legacy_error')} current={r.get('current_error')}"
                    )
                else:
                    print(
                        f"{r['tokens']:7d} | {r['queries']:7d} | {r['batch_size']:1d} | "
                        f"{r['legacy_docs_per_second']:13.2f} | "
                        f"{r['current_docs_per_second']:14.2f} | "
                        f"{r['speedup']:7.3f}x"
                    )

        for task, result in cfg["tasks"].items():
            suffix = " (batch_score)" if result["speed_scope"] == "joint_batch_score" else ""
            print(f"\n{task.upper()} SPEED{suffix}")
            print(" BS | legacy docs/s | current docs/s | speedup | legacy/current peak MB")
            for r in result["speed"]:
                if "legacy_error" in r or "current_error" in r:
                    print(f"{r['batch_size']:3d} | ERROR legacy={r.get('legacy_error')} current={r.get('current_error')}")
                    continue
                lm, cm = r.get("legacy_peak_allocated_mb"), r.get("current_peak_allocated_mb")
                mem = "-" if lm is None or cm is None else f"{lm:.0f}/{cm:.0f}"
                print(f"{r['batch_size']:3d} | {r['legacy_docs_per_second']:13.2f} | {r['current_docs_per_second']:14.2f} | {r['speedup']:7.3f}x | {mem}")
            print(f"{task.upper()} PARITY")
            for q in result["quality"]:
                if "legacy_error" in q or "current_error" in q:
                    print(f"  {q.get('threshold')}: ERROR")
                else:
                    label = "joint" if q.get("threshold") is None else f"th={q['threshold']:.1f}"
                    print(f"  {label}: match={q['match_rate']:.6f} conf_max={q['confidence_max_abs']:.3g}")
        if "training" in cfg:
            print("\nTRAINING FORWARD+BACKWARD")
            print(" BS | legacy samp/s | current samp/s | speedup | legacy MB | current MB | mem ratio | max eval-loss diff")
            for r in cfg["training"]:
                if "legacy_error" in r or "current_error" in r:
                    print(f"{r['batch_size']:3d} | ERROR")
                else:
                    md = max(r["loss_abs_diff"].values(), default=0.0)
                    lm = r.get("legacy_peak_allocated_mb")
                    cm = r.get("current_peak_allocated_mb")
                    mr = r.get("memory_ratio_legacy_over_current")
                    lm_s = f"{lm:.1f}" if lm is not None else "n/a"
                    cm_s = f"{cm:.1f}" if cm is not None else "n/a"
                    mr_s = f"{mr:.3f}x" if mr is not None else "n/a"
                    print(
                        f"{r['batch_size']:3d} | {r['legacy_samples_per_second']:13.2f} | "
                        f"{r['current_samples_per_second']:14.2f} | {r['speedup']:7.3f}x | "
                        f"{lm_s:>9} | {cm_s:>10} | {mr_s:>9} | {md:.3g}"
                    )


def parent(args: argparse.Namespace) -> int:
    repo = repo_root()
    tasks = csv(args.tasks)
    bad = set(tasks) - set(TASKS)
    if bad:
        raise ValueError(f"unknown tasks: {sorted(bad)}")

    texts = load_corpus(args.corpus_file, args.docs) if args.corpus_file else generate_corpus(
        args.model, args.docs, csv_int(args.generated_lengths), args.seed
    )
    if not texts:
        raise ValueError("empty corpus")

    scaling_lengths = csv_int(args.scaling_token_lengths)
    scaling_queries = csv_int(args.scaling_query_counts)
    scaling_corpora = {}
    if not args.no_scaling:
        for length in scaling_lengths:
            if length <= 0:
                raise ValueError("scaling token lengths must be positive")
            scaling_corpora[length] = generate_exact_token_corpus(
                args.model,
                args.scaling_docs,
                length,
                args.seed + 10000 + length,
            )
    if any(q <= 0 for q in scaling_queries):
        raise ValueError("scaling query counts must be positive")

    cfg = {
        "model": args.model,
        "seed": args.seed,
        "devices": csv(args.devices),
        "dtypes": csv(args.dtypes),
        "tasks": tasks,
        "batch_sizes": csv_int(args.batch_sizes),
        "scaling": not args.no_scaling,
        "scaling_token_lengths": scaling_lengths,
        "scaling_query_counts": scaling_queries,
        "scaling_docs": args.scaling_docs,
        "repeats": args.repeats,
        "warmup_docs": args.warmup_docs,
        "speed_threshold": args.speed_threshold,
        "thresholds": csv_float(args.thresholds),
        "quality_docs": args.quality_docs,
        "joint_quality_docs": args.joint_quality_docs,
        "quality_batch_size": args.quality_batch_size,
        "max_len": args.max_len,
        "training": not args.no_training,
        "training_batch_sizes": csv_int(args.training_batch_sizes),
        "training_repeats": args.training_repeats,
        "training_warmup": args.training_warmup,
    }

    print("=" * 88)
    print("GLiNER2 EXACT LEGACY vs CURRENT DIRECT-SPAN BENCHMARK")
    print(f"version:       {VERSION}")
    print(f"repo:          {repo}")
    print(f"legacy commit: {args.legacy_commit}")
    print(f"current kernel:{CURRENT_DIRECT_KERNEL:>16}")
    print(f"documents:     {len(texts)}")
    print(f"tasks:         {tasks}")
    print(f"batch sizes:   {cfg['batch_sizes']} (secondary axis)")
    if cfg["scaling"]:
        print(
            f"PRIMARY GRID:  tokens={cfg['scaling_token_lengths']} "
            f"queries={cfg['scaling_query_counts']} "
            f"docs/length={cfg['scaling_docs']}"
        )
    print(f"logical CPUs:  {os.cpu_count() or 1}")
    ctot, ccompute, cworkers = cpu_plan("cpu")
    gtot, gcompute, gworkers = cpu_plan("cuda")
    print(f"CPU policy:    {ccompute} compute + {cworkers} preprocess workers (75/25)")
    print(f"CUDA policy:   {gcompute} host compute + {gworkers} preprocess workers")
    print("Joint speed:   batch_score only; full decode is parity-only")
    print("training:      CUDA only")
    print("=" * 88)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="gliner2_legacy_") as td:
        td = Path(td)
        legacy_repo = td / "legacy"
        export_commit(repo, args.legacy_commit, legacy_repo)
        cfg_file, corpus_file = td / "config.json", td / "corpus.json"
        legacy_file, current_file = td / "legacy.json", td / "current.json"
        cfg_file.write_text(json.dumps(cfg))
        corpus_file.write_text(json.dumps({
            "main": texts,
            "scaling": {str(k): v for k, v in scaling_corpora.items()},
        }))
        launch(script, legacy_repo, "legacy", cfg_file, corpus_file, legacy_file)
        launch(script, repo, "current", cfg_file, corpus_file, current_file)
        legacy = json.loads(legacy_file.read_text())
        current = json.loads(current_file.read_text())
        if legacy["corpus_sha256"] != current["corpus_sha256"]:
            raise RuntimeError("legacy/current corpus mismatch")
        if legacy.get("scaling_corpus_sha256") != current.get("scaling_corpus_sha256"):
            raise RuntimeError("legacy/current scaling corpus mismatch")
        comparison = compare(legacy, current)
        final = {
            "metadata": {
                "version": VERSION,
                "legacy_commit": args.legacy_commit,
                "current_direct_kernel": CURRENT_DIRECT_KERNEL,
                "training_rng_policy": "seed Python/NumPy/PyTorch/CUDA before batch construction and each forward+backward, outside timed region",
                "subprocess_pythonhashseed": cfg["seed"],
                "current_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "current_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()),
                "corpus_sha256": corpus_hash(texts),
                "scaling_corpus_sha256": {
                    str(k): corpus_hash(v) for k, v in sorted(scaling_corpora.items())
                },
                "config": cfg,
            },
            "comparison": comparison,
            "legacy": legacy,
            "current": current,
        }
        args.output.write_text(json.dumps(final, indent=2))

    print_comparison(comparison)
    print(f"\nFull JSON: {args.output}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="fastino/gliner2-base-v1")
    p.add_argument("--legacy-commit", default=LEGACY_COMMIT)
    p.add_argument("--docs", type=int, default=128)
    p.add_argument("--corpus-file", type=Path)
    p.add_argument("--generated-lengths", default="64,128,256,384")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--batch-sizes", default="1,8", help="secondary batch-size axis; L x Q is the primary scaling experiment")
    p.add_argument("--devices", default="auto", help="auto or cpu,cuda")
    p.add_argument("--dtypes", default="auto", help="auto or fp32,fp16,bf16")
    p.add_argument("--scaling-token-lengths", default="64,256", help="primary L axis")
    p.add_argument("--scaling-query-counts", default="1,8,32", help="primary Q axis")
    p.add_argument("--scaling-docs", type=int, default=128, help="documents per token-length point")
    p.add_argument("--no-scaling", action="store_true", help="disable the controlled L x Q benchmark")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup-docs", type=int, default=16)
    p.add_argument("--speed-threshold", type=float, default=0.5)
    p.add_argument("--thresholds", default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--quality-docs", type=int, default=64)
    p.add_argument("--joint-quality-docs", type=int, default=16)
    p.add_argument("--quality-batch-size", type=int, default=0, help="0 = largest successful speed batch")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--no-training", action="store_true")
    p.add_argument("--training-batch-sizes", default="1,8")
    p.add_argument("--training-repeats", type=int, default=3)
    p.add_argument("--training-warmup", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("benchmarks/direct_span_exact_legacy_results.json"))
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_repo", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_variant", choices=["legacy", "current"], help=argparse.SUPPRESS)
    p.add_argument("--_config", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_corpus", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_worker-output", type=Path, help=argparse.SUPPRESS)
    return p


def main() -> int:
    args = parser().parse_args()
    if args._worker:
        return worker(args._repo, args._variant, args._config, args._corpus, args._worker_output)
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
