from __future__ import annotations

import subprocess
import sys
import textwrap


SERVING_MODULES = (
    "gliner2",
    "gliner2.processor",
    "gliner2.inference.runtime",
    "gliner2.inference.engine",
    "gliner2.models.boundary.engine",
)


SCRIPT = textwrap.dedent(
    """\
    import importlib
    import sys
    sys.modules["peft"] = None
    for name in {modules!r}:
        importlib.import_module(name)
        assert "gliner2.training.trainer" not in sys.modules, name
    from gliner2.models.boundary.engine import BoundaryExtractor
    assert BoundaryExtractor.architecture == "boundary"
    print("PASS")
    """
).format(modules=list(SERVING_MODULES))


def test_serving_modules_import_without_peft() -> None:
    """The serving import graph must not pull peft, a training-only extra."""
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (
        f"A serving import pulled peft/trainer.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PASS" in result.stdout
