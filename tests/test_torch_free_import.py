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

    # Accessing GLiNER2 must fail without torch
    try:
        gliner2.GLiNER2
    except (ModuleNotFoundError, ImportError):
        pass
    else:
        raise AssertionError("gliner2.GLiNER2 should not be importable without torch")

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
