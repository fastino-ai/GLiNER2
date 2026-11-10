"""Verify that ``import gliner2`` works without torch installed.

This test spawns a subprocess so it can detect eager imports that would
otherwise be masked by torch already being in ``sys.modules`` in the
parent process.  It must be run in a venv built from
``pip install -e .`` (no extras).
"""

import subprocess
import sys
import textwrap

TORCH_FREE_SCRIPT = textwrap.dedent("""\
    import sys
    import gliner2

    banned = {"torch", "transformers", "safetensors", "gliner", "numpy"}
    hit = banned & set(sys.modules)
    assert not hit, f"torch-tainted modules imported eagerly: {hit}"

    # Schema builder round-trip
    schema = gliner2.Schema().entities(["a", "b"])
    d = schema.to_dict()
    assert d == {"entities": ["a", "b"]}, f"unexpected to_dict: {d}"

    # RegexValidator
    v = gliner2.RegexValidator(r"^\\w+$")
    assert v.validate("hello")

    # SchemaInput pydantic validation
    si = gliner2.SchemaInput(entities=["x"])
    assert si.entities == ["x"]

    # GLiNER2API class reference (no instantiation, no network)
    assert hasattr(gliner2.GLiNER2API, "__init__")

    # Accessing GLiNER2 must not make package import torch-free behavior fail.
    # It may succeed when this interpreter already provides torch, and otherwise
    # must fail with the missing optional dependency.
    try:
        gliner2.GLiNER2
    except (ModuleNotFoundError, ImportError):
        assert "torch" not in sys.modules

    print("PASS")
""")


def test_torch_free_import():
    result = subprocess.run(
        [sys.executable, "-c", TORCH_FREE_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Torch-free import test failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_managed_planning_without_any_ml_stack():
    script = """
import sys
from gliner2.training import InputExample, TrainingConfig, ManagedTrainingPlan
cfg = TrainingConfig(num_workers=0, batch_size=2, num_epochs=2, fp16=False, bf16=False)
rows = [InputExample(text='john works', entities={'person': ['john']})] * 5
for architecture in ('span', 'boundary'):
    plan = ManagedTrainingPlan(rows, cfg, architecture=architecture)
    manifest = plan.to_manifest(eval_examples=rows[:1])
    assert len(manifest['batches']) == 6
    assert manifest == plan.to_manifest(eval_examples=rows[:1])
assert not {'torch', 'transformers', 'safetensors', 'gliner', 'numpy'} & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
