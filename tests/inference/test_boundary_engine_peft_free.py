"""Boundary serving imports must not require peft (training extra)."""

from __future__ import annotations

import subprocess
import sys
import textwrap


SCRIPT = textwrap.dedent(
    """\
    import sys
    sys.modules["peft"] = None
    from gliner2.models.boundary.engine import BoundaryExtractor
    assert BoundaryExtractor.architecture == "boundary"
    assert "gliner2.training.trainer" not in sys.modules
    print("PASS")
    """
)


def test_boundary_engine_imports_without_peft() -> None:
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "Boundary engine import pulled peft/trainer.\\n"
        f"stdout: {result.stdout}\\nstderr: {result.stderr}"
    )
    assert "PASS" in result.stdout
