import json
from pathlib import Path

from datasets import load_dataset
from loguru import logger
import typer


def _parse_output_fields(value):
    """
    The `output` column in
    `dylanhogg/gnaf-2022-structured-training-100000-v0-instruct`
    (see https://huggingface.co/datasets/dylanhogg/gnaf-2022-structured-training-100000-v0-instruct)
    is stored as a JSON string. This helper parses it into a Python dict,
    while being robust to already-parsed dicts.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning("Could not JSON-decode output string: {}", value)
            return None

    # Fallback: keep as-is (e.g. list or other type)
    return value


def main(
    out_path: Path = typer.Option(
        Path(__file__).resolve().parent / "data" / "gnaf_train.jsonl",
        "--out",
        help="Path to the output JSONL file.",
    ),
    dataset_name: str = typer.Option(
        "dylanhogg/gnaf-2022-structured-training-100000-v0-instruct",
        "--dataset-name",
        help="Hugging Face dataset identifier to read from.",
    ),
    split: str = typer.Option(
        "train",
        "--split",
        help="Dataset split to load.",
    ),
    max_rows: int = typer.Option(
        0,
        "--max-rows",
        help="Optional limit on number of rows to export (0 means all).",
    ),
):
    """
    Convert the Hugging Face dataset into JSONL with schema:

        {"input": <input>, "output": {"entities": <output>}}
    """
    logger.info(f"Loading dataset {dataset_name!r} split={split!r}")
    ds = load_dataset(dataset_name, split=split)
    logger.info(f"Loaded {len(ds)} rows from dataset.")

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing JSONL to {out_path}")

    num_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            if max_rows and num_written >= max_rows:
                break

            input_text = row.get("input")
            output_value = row.get("output")

            if not input_text or output_value is None:
                logger.debug(
                    "Skipping row {} because of missing input/output",
                    idx,
                )
                continue

            entities = _parse_output_fields(output_value)
            if entities is None:
                logger.debug("Skipping row {} because output could not be parsed", idx)
                continue

            record = {
                "input": input_text,
                "output": {
                    "entities": entities,
                },
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

    logger.info(f"Finished. Wrote {num_written} JSONL lines to {out_path}")


if __name__ == "__main__":
    typer.run(main)
