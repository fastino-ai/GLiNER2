#!/usr/bin/env python3
"""Benchmark decoder synchronization across GLiNER2 span and boundary models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import torch


DEFAULT_MODELS = [
    "fastino/gliner2-base-v1",
    "fastino/gliner2.5-base-v1",
]

TEXTS = [
    "Apple CEO Tim Cook introduced a new iPhone in Cupertino for $999.",
    "Sundar Pichai presented Gemini features at Google I/O in Mountain View.",
    "Microsoft appointed Sarah Chen to lead Azure research in Seattle.",
    "NVIDIA CEO Jensen Huang announced new Blackwell systems at GTC.",
    "AMD acquired Silo AI for approximately $665 million in August 2024.",
    "Maria Ivanova joined CloudWorks in Sofia and later relocated to Berlin.",
    "Atlas Analytics acquired BrightData Labs from Horizon Capital in London.",
    "Daniel Ortiz founded Nova Robotics in Madrid with Alice Morgan in 2021.",
    "The Aurora Phone costs $749 and ships in blue, black, and silver.",
    "Vertex Audio released Vertex Buds in Paris on 4 May 2026 for 249 euros.",
    "Acme renewed its supply agreement with Globex through December 2028.",
    "Dr. Maya Patel works at Stanford University and specializes in cardiology.",
    "The quarterly report showed revenue of $8.4 billion, up 17 percent.",
    "Contact Elena at elena@example.com or call the Berlin office tomorrow.",
    "Northwind opened offices in Tokyo, Toronto, and Amsterdam this year.",
    "The camera is excellent, although battery life remains disappointing.",
    "OpenAI partnered with Example Corp to deploy assistants for customer support.",
    "Robert Lewandowski scored twice as Barcelona defeated Valencia 3-1.",
    "Researchers at ETH Zurich presented a new robotics system at NeurIPS.",
    "The FDA approved Medica's therapy after a successful phase-three trial.",
    "Tesla delivered 443,956 vehicles during the second quarter of 2024.",
    "Jane Smith became CFO of Contoso after working at Fabrikam for six years.",
    "A severe storm delayed flights from Heathrow to New York and Boston.",
    "The premium subscription includes analytics and API access for $99 monthly.",
    "Amazon plans to invest $10 billion in cloud infrastructure in Ohio.",
    "Carlos Mendes reports to Priya Shah at Alpine Systems in Lisbon.",
    "The conference runs from Monday through Thursday at the Hilton Chicago.",
    "Blue River Bank approved a mortgage for the property at 14 King Street.",
    "Samsung unveiled a foldable tablet during its annual event in Seoul.",
    "The board rejected the proposal because its projected costs were too high.",
    "Lena bought three monitors from TechMarket and requested delivery by Friday.",
    "After reviewing the evidence, the court scheduled the hearing for October.",
]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(jsonable(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def result_hash(value: Any) -> str:
    payload = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def documents(count: int) -> list[str]:
    return [TEXTS[index % len(TEXTS)] for index in range(count)]


def entity_schema(model, wide: bool = False):
    labels = ["person", "organization", "location", "date"]
    if wide:
        labels += [
            "product", "event", "money", "percentage", "job title", "facility",
            "medical condition", "law", "sports team", "contact detail", "address",
            "quantity",
        ]
    return model.create_schema().entities(labels)


def relation_schema(model):
    return model.create_schema().relations([
        "works for", "founded", "acquired", "located in", "reports to",
        "partnered with", "released", "costs",
    ])


def classification_schema(model):
    schema = model.create_schema()
    schema.classification(
        "topic",
        ["technology", "business", "health", "sports", "legal", "travel"],
    )
    schema.classification(
        "tone",
        ["positive", "negative", "neutral", "urgent"],
        multi_label=True,
    )
    return schema


def record_schema(model):
    schema = model.create_schema()
    schema.structure("event").field("organization").field("person").field(
        "location"
    ).field("date").field("amount").field(
        "status", choices=["planned", "active", "completed", "cancelled"]
    )
    return schema


def mixed_schema(model):
    schema = entity_schema(model, wide=True)
    schema.classification("sentiment", ["positive", "negative", "neutral"])
    schema.relations(["works for", "founded", "acquired", "located in"])
    schema.structure("announcement").field("organization").field("subject").field(
        "location"
    ).field("date").field("amount")
    return schema


def case_schemas(model) -> dict[str, Any]:
    return {
        "entities": entity_schema(model),
        "entities_wide": entity_schema(model, wide=True),
        "relations": relation_schema(model),
        "classification": classification_schema(model),
        "records": record_schema(model),
        "mixed": mixed_schema(model),
    }


def case_functions(model, texts: list[str], batch_size: int) -> dict[str, Callable[[], Any]]:
    return {
        name: (
            lambda schema=schema: model.batch_extract(
                texts,
                schema,
                batch_size=batch_size,
                threshold=0.3,
                include_confidence=True,
                include_spans=True,
            )
        )
        for name, schema in case_schemas(model).items()
    }


def measure(
    function: Callable[[], Any],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], Any]:
    with torch.inference_mode():
        output = None
        for _ in range(warmup):
            output = function()
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            output = function()
            sync(device)
            samples.append((time.perf_counter() - start) * 1000)

    median_ms = statistics.median(samples)
    stats = {
        "median_ms": median_ms,
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None
        ),
    }
    return stats, output


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def worker(config_path: Path, output_path: Path) -> None:
    from gliner2 import AutoExtractor

    config = json.loads(config_path.read_text())
    device = torch.device(config["device"])
    dtype = dtype_from_name(config["dtype"])
    rows = []

    for model_id in config["models"]:
        print(f"\nmodel={model_id}", flush=True)
        try:
            model = AutoExtractor.from_pretrained(model_id, map_location=str(device)).eval()
            model = model.to(device=device, dtype=dtype)
            architecture = model.config.architecture
        except Exception as exc:
            rows.append({"model": model_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  LOAD ERROR: {exc}", flush=True)
            continue

        for batch_size in config["batch_sizes"]:
            texts = documents(batch_size)
            for case, function in case_functions(model, texts, batch_size).items():
                row = {
                    "model": model_id,
                    "architecture": architecture,
                    "case": case,
                    "batch_size": batch_size,
                }
                try:
                    stats, output = measure(
                        function, device, config["warmup"], config["iterations"]
                    )
                    row.update(stats)
                    row["docs_per_second"] = batch_size * 1000 / stats["median_ms"]
                    row["result_hash"] = result_hash(output)
                    print(
                        f"  B={batch_size:<2} {case:<15} "
                        f"{stats['median_ms']:8.2f} ms "
                        f"{row['docs_per_second']:8.2f} docs/s",
                        flush=True,
                    )
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"  B={batch_size:<2} {case:<15} ERROR {exc}", flush=True)
                rows.append(row)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path.write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device),
        "dtype": config["dtype"],
        "iterations": config["iterations"],
        "rows": rows,
    }, indent=2))


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def run_variant(
    script: Path,
    repo: Path,
    config: Path,
    output: Path,
    name: str,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--config",
        str(config),
        "--worker-output",
        str(output),
    ]
    print(f"\n{'=' * 80}\n{name.upper()}: {repo}\n{'=' * 80}", flush=True)
    subprocess.run(command, cwd=repo, env=env, check=True)


def index_rows(data: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (row["model"], row.get("case", ""), row.get("batch_size", 0)): row
        for row in data["rows"]
    }


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    before = index_rows(baseline)
    after = index_rows(candidate)
    rows = []
    for key in sorted(set(before) | set(after)):
        left = before.get(key, {})
        right = after.get(key, {})
        row = {
            "model": key[0],
            "case": key[1],
            "batch_size": key[2],
            "baseline_error": left.get("error"),
            "candidate_error": right.get("error"),
        }
        if not row["baseline_error"] and not row["candidate_error"]:
            row.update({
                "baseline_median_ms": left["median_ms"],
                "candidate_median_ms": right["median_ms"],
                "speedup": left["median_ms"] / right["median_ms"],
                "parity": left["result_hash"] == right["result_hash"],
                "baseline_peak_memory_mb": left["peak_memory_mb"],
                "candidate_peak_memory_mb": right["peak_memory_mb"],
            })
        rows.append(row)
    return rows


def print_comparison(rows: list[dict[str, Any]]) -> None:
    print("\nMODEL / CASE                    B   BASE MS    OPT MS  SPEEDUP  PARITY")
    print("-" * 82)
    for row in rows:
        label = f"{row['model'].split('/')[-1]} / {row['case']}"
        if row["baseline_error"] or row["candidate_error"]:
            print(f"{label:<31} {row['batch_size']:>2}  ERROR")
            continue
        print(
            f"{label:<31} {row['batch_size']:>2} "
            f"{row['baseline_median_ms']:>9.2f} "
            f"{row['candidate_median_ms']:>9.2f} "
            f"{row['speedup']:>7.3f}x  {str(row['parity']):>6}"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline-repo", type=Path)
    result.add_argument("--candidate-repo", type=Path)
    result.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    result.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--iterations", type=int, default=20)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default=None)
    result.add_argument("--output", type=Path, default=Path("decoder_sync_results.json"))
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.worker:
        worker(args.config, args.worker_output)
        return
    if args.baseline_repo is None or args.candidate_repo is None:
        raise SystemExit("--baseline-repo and --candidate-repo are required")
    for repo in (args.baseline_repo, args.candidate_repo):
        if not (repo / ".git").exists():
            raise SystemExit(f"not a git checkout: {repo}")
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("--warmup must be non-negative and --iterations must be positive")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise SystemExit("batch sizes must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    dtype = args.dtype or ("fp16" if args.device.startswith("cuda") else "fp32")
    config = {
        "models": args.models,
        "batch_sizes": args.batch_sizes,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "device": args.device,
        "dtype": dtype,
    }
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="gliner2_decoder_sync_") as temp:
        temp_path = Path(temp)
        config_path = temp_path / "config.json"
        baseline_path = temp_path / "baseline.json"
        candidate_path = temp_path / "candidate.json"
        config_path.write_text(json.dumps(config))
        run_variant(
            script, args.baseline_repo.resolve(), config_path, baseline_path, "baseline"
        )
        run_variant(
            script, args.candidate_repo.resolve(), config_path, candidate_path, "candidate"
        )
        baseline = json.loads(baseline_path.read_text())
        candidate = json.loads(candidate_path.read_text())

    comparison = compare(baseline, candidate)
    result = {
        "config": config,
        "baseline": {
            "repo": str(args.baseline_repo.resolve()),
            "commit": git_head(args.baseline_repo),
            **baseline,
        },
        "candidate": {
            "repo": str(args.candidate_repo.resolve()),
            "commit": git_head(args.candidate_repo),
            **candidate,
        },
        "comparison": comparison,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print_comparison(comparison)
    parity_failures = [row for row in comparison if row.get("parity") is False]
    errors = [
        row for row in comparison if row["baseline_error"] or row["candidate_error"]
    ]
    print(f"\noutput={args.output} errors={len(errors)} parity_failures={len(parity_failures)}")
    if errors or parity_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
