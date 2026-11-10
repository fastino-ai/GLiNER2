"""``peft`` is an optional extra: only LoRA code paths may require it.

Each check runs in a subprocess with ``sys.modules["peft"] = None`` so it sees
the import graph of an install without peft, which a parent process that has
already imported peft would mask.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PRELUDE = textwrap.dedent(
    """\
    import sys
    sys.modules["peft"] = None
    """
)

TRAINING_IMPORTS_WITHOUT_PEFT = PRELUDE + textwrap.dedent(
    """\
    from gliner2.training import ExtractorTrainer, GLiNER2Trainer, TrainingConfig
    import gliner2.training.trainer
    assert ExtractorTrainer is GLiNER2Trainer
    assert "gliner2.training.lora" not in sys.modules
    print("PASS")
    """
)

NON_LORA_TRAINING_WITHOUT_PEFT = PRELUDE + textwrap.dedent(
    """\
    import tempfile
    from pathlib import Path
    from gliner2.training import ExtractorTrainer, InputExample, TrainingConfig
    from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model

    with tempfile.TemporaryDirectory() as tmp:
        config = TrainingConfig(
            output_dir=tmp, batch_size=1, num_epochs=1, eval_strategy="no",
            fp16=False, bf16=False, num_workers=0, logging_steps=10_000,
            warmup_ratio=0.0, scheduler_type="constant", local_rank=-1,
        )
        examples = [
            InputExample(text="apple released iphone .", entities={"company": ["apple"]})
        ] * 2
        result = ExtractorTrainer(build_tiny_span_model(), config).train(train_data=examples)
        assert result["total_steps"] == 2, result
        assert (Path(tmp) / "final").is_dir()
    assert "gliner2.training.lora" not in sys.modules
    print("PASS")
    """
)

LORA_WITHOUT_PEFT_NAMES_THE_EXTRA = PRELUDE + textwrap.dedent(
    """\
    import tempfile
    import gliner2
    from gliner2.training import ExtractorTrainer, TrainingConfig
    from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model

    def expect_install_hint(action):
        try:
            action()
        except ImportError as exc:
            message = str(exc)
            assert 'pip install "gliner2[train]"' in message, message
            assert "peft" in message, message
        else:
            raise AssertionError("expected ImportError")

    def import_lora_module():
        import gliner2.training.lora

    expect_install_hint(import_lora_module)
    expect_install_hint(lambda: gliner2.LoRAConfig)
    expect_install_hint(lambda: build_tiny_span_model().apply_lora(r=2))
    with tempfile.TemporaryDirectory() as tmp:
        expect_install_hint(
            lambda: ExtractorTrainer(
                build_tiny_span_model(), TrainingConfig(output_dir=tmp, use_lora=True)
            )
        )
    print("PASS")
    """
)


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )


def _assert_passed(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0 and "PASS" in result.stdout, (
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_training_package_imports_without_peft() -> None:
    _assert_passed(_run(TRAINING_IMPORTS_WITHOUT_PEFT))


def test_non_lora_training_runs_and_saves_without_peft() -> None:
    _assert_passed(_run(NON_LORA_TRAINING_WITHOUT_PEFT))


def test_lora_without_peft_raises_import_error_naming_the_extra() -> None:
    _assert_passed(_run(LORA_WITHOUT_PEFT_NAMES_THE_EXTRA))
