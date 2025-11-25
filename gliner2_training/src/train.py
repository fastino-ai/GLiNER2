from datetime import datetime
import typer

from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

import os

from transformers.training_args import TrainingArguments
from gliner2 import GLiNER2

from gliner2.trainer import ExtractorDataset, ExtractorDataCollator, ExtractorTrainer

load_dotenv()
app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(base_model_id: str, num_train_epochs: int = 3, report_to: str = "none"):
    print("------------------ gliner2_ft_train_app ------------------")

    datestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    logger.info("args:")
    logger.info(f"{base_model_id=}")
    logger.info(f"{num_train_epochs=}")
    logger.info(f"{report_to=}")

    # 1. Paths to your JSONL data
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    logger.info(f"{project_root=}")
    logger.info(f"{data_dir=}")

    train_files = [
        str(data_dir / "train.jsonl"),
        # you can add more shards like:
        # str(data_dir / "train_2.jsonl"),
        # str(data_dir / "train_3.jsonl"),
    ]
    eval_files = [
        str(data_dir / "eval.jsonl"),
    ]
    logger.info(f"{train_files=}")
    logger.info(f"{eval_files=}")

    # 2. Datasets
    #    Each JSONL line must have: {"input": <text>, "output": <schema/gold>}
    train_dataset = ExtractorDataset(train_files)
    eval_dataset = ExtractorDataset(eval_files)
    logger.info(f"{train_dataset=}")
    logger.info(f"{eval_dataset=}")

    # 3. Model
    #    Pick base/large depending on your latency / quality trade-off.
    model = GLiNER2.from_pretrained(base_model_id)

    # 4. Data collator
    data_collator = ExtractorDataCollator()

    # 5. HF TrainingArguments
    #    `learning_rate` here is mostly ignored because we set per-group lrs
    #    in ExtractorTrainer.create_optimizer, but it still needs *a* value.
    run_name = f"gliner2-v1-{datestamp}-tr{len(train_dataset)}-ev{len(eval_dataset)}-ep{num_train_epochs}"
    output_dir = project_root / "checkpoints" / run_name
    logger.info(f"{run_name=}")
    logger.info(f"{output_dir=}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,  # global LR (group-specific LRs override)
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        # fp16=True,  # True if you have a GPU with fp16
        report_to=report_to,  # none/wandb/etc
        run_name=run_name,
    )
    logger.info(f"{training_args=}")

    # 6. Trainer
    #   encoder_lr: LR for encoder params
    #   custom_lr:  LR for non-encoder params (classifier etc.)
    #   weight_decay: same across groups
    #   finetune_classifier: True -> only train classifier; False -> full finetune
    trainer = ExtractorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        encoder_lr=1e-5,
        custom_lr=5e-5,
        weight_decay=0.01,
        finetune_classifier=False,  # set True if you only want to tune classifier
    )

    # 7. Train + save
    logger.info("Fine-tuning starting...")

    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    trainer.save_state()
    logger.info("Fine-tuning complete.")

    # 8. Evaluate
    eval_metrics = trainer.evaluate()
    logger.info(f"{eval_metrics=}")

    logger.info("Finished.")


if __name__ == "__main__":
    app()
