"""
GLiNER2 Trainer with Optimized DataLoader-based Preprocessing

This module provides training utilities that leverage parallel preprocessing
via DataLoader workers for maximum GPU utilization.
"""

import json
import random
from typing import Union, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments

from gliner2.processor import SchemaTransformer, PreprocessedBatch, SamplingConfig


# =============================================================================
# Dataset
# =============================================================================

class ExtractorDataset(Dataset):
    """
    Dataset for GLiNER2 training.

    Returns (text, schema) tuples that are processed by the collate function.

    Args:
        data_paths: Path or list of paths to JSONL training files
        shuffle: Whether to shuffle data on load (default: True)

    JSONL Format:
        {"input": "text here", "output": {"entities": {...}, ...}}
    """

    def __init__(self, data_paths: Union[str, List[str]], shuffle: bool = True):
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        print(f"Loading {len(data_paths)} file(s) for training...")

        self.data = []
        for path in data_paths:
            with open(path, "r", encoding="utf-8") as f:
                self.data.extend([json.loads(line) for line in f])

        if shuffle:
            random.shuffle(self.data)

        print(f"Loaded {len(self.data)} records from {len(data_paths)} file(s).")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """Return (text, schema) tuple."""
        record = self.data[idx]
        return record["input"], record["output"]


# =============================================================================
# Data Collator
# =============================================================================

class ExtractorDataCollator:
    """
    Data collator that uses processor's collate function.

    This enables parallel preprocessing via DataLoader workers.

    Args:
        processor: SchemaTransformer instance
        is_training: Whether in training mode (enables augmentation)
    """

    def __init__(self, processor: SchemaTransformer, is_training: bool = True):
        self.processor = processor
        self.is_training = is_training

    def __call__(self, batch: List[tuple]) -> PreprocessedBatch:
        """
        Collate batch of (text, schema) tuples into PreprocessedBatch.

        Args:
            batch: List of (text, schema) tuples from dataset

        Returns:
            PreprocessedBatch ready for model.forward()
        """
        if self.is_training:
            return self.processor.collate_fn_train(batch)
        else:
            return self.processor.collate_fn_inference(batch)


# =============================================================================
# Trainer
# =============================================================================

class ExtractorTrainer(Trainer):
    """
    Trainer for GLiNER2 with optimized preprocessing.

    Key features:
    - Parallel preprocessing via DataLoader workers
    - Separate learning rates for encoder and other layers
    - Optional classifier-only fine-tuning
    - FP16 support
    - Gradient accumulation

    Example:
        >>> processor = SchemaTransformer(model_name, sampling_config=config)
        >>> collator = ExtractorDataCollator(processor, is_training=True)
        >>> 
        >>> trainer = ExtractorTrainer(
        ...     model=model,
        ...     args=TrainingArguments(
        ...         output_dir="./output",
        ...         per_device_train_batch_size=32,
        ...         dataloader_num_workers=8,  # Parallel preprocessing!
        ...         dataloader_pin_memory=True,
        ...     ),
        ...     train_dataset=dataset,
        ...     data_collator=collator,
        ...     encoder_lr=1e-5,
        ...     custom_lr=5e-4,
        ...     weight_decay=0.01,
        ... )
        >>> trainer.train()
    """

    def __init__(
            self,
            encoder_lr: float = 1e-5,
            custom_lr: float = 5e-4,
            weight_decay: float = 0.01,
            finetune_classifier: bool = False,
            **kwargs
    ):
        """
        Initialize trainer.

        Args:
            encoder_lr: Learning rate for encoder parameters
            custom_lr: Learning rate for non-encoder parameters
            weight_decay: Weight decay for all parameters
            finetune_classifier: If True, freeze all except classifier
            **kwargs: Arguments passed to Trainer
        """
        self.encoder_lr = encoder_lr
        self.custom_lr = custom_lr
        self.custom_weight_decay = weight_decay
        self.finetune_classifier = finetune_classifier

        super().__init__(**kwargs)

        if self.finetune_classifier:
            self._freeze_non_classifier()

    def _freeze_non_classifier(self):
        """Freeze all parameters except classifier."""
        print("Finetuning classifier only: freezing all other parameters.")
        for name, param in self.model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    def create_optimizer(self):
        """Create optimizer with separate parameter groups."""
        if self.finetune_classifier:
            # Only classifier parameters
            classifier_params = [
                p for n, p in self.model.named_parameters()
                if n.startswith("classifier") and p.requires_grad
            ]
            if not classifier_params:
                raise ValueError("No trainable parameters in classifier.")

            groups = [{
                "params": classifier_params,
                "lr": self.custom_lr,
                "weight_decay": self.custom_weight_decay,
            }]
        else:
            # Separate encoder and other parameters
            encoder_params = list(self.model.encoder.parameters())
            other_params = [
                p for n, p in self.model.named_parameters()
                if "encoder" not in n and p.requires_grad
            ]

            groups = [
                {
                    "params": encoder_params,
                    "lr": self.encoder_lr,
                    "weight_decay": self.custom_weight_decay
                },
                {
                    "params": other_params,
                    "lr": self.custom_lr,
                    "weight_decay": self.custom_weight_decay
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)

    def compute_loss(self, model, inputs: PreprocessedBatch, return_outputs: bool = False, **kwargs):
        """
        Compute loss on preprocessed batch.

        Args:
            model: The model
            inputs: PreprocessedBatch from collator
            return_outputs: Whether to return outputs dict

        Returns:
            Loss tensor, optionally with outputs dict
        """
        # Forward pass - inputs is already PreprocessedBatch
        outputs = model(inputs, return_individual_losses=False)

        # Handle empty batch
        if outputs["batch_size"] == 0:
            device = next(model.parameters()).device
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss = outputs["total_loss"]

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# Training Utilities
# =============================================================================

def create_training_dataloader(
        dataset: ExtractorDataset,
        processor: SchemaTransformer,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        shuffle: bool = True,
        prefetch_factor: int = 2,
) -> DataLoader:
    """
    Create an optimized DataLoader for training.

    This function creates a DataLoader configured for maximum preprocessing
    efficiency using parallel workers.

    Args:
        dataset: ExtractorDataset instance
        processor: SchemaTransformer for preprocessing
        batch_size: Batch size
        num_workers: Number of parallel workers for preprocessing
        pin_memory: Pin memory for faster GPU transfer
        shuffle: Shuffle data each epoch
        prefetch_factor: Batches to prefetch per worker

    Returns:
        Configured DataLoader

    Example:
        >>> loader = create_training_dataloader(
        ...     dataset=train_dataset,
        ...     processor=processor,
        ...     batch_size=32,
        ...     num_workers=8,
        ... )
        >>> for batch in loader:
        ...     batch = batch.to(device)
        ...     loss = model(batch)["total_loss"]
    """
    collator = ExtractorDataCollator(processor, is_training=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collator,
        persistent_workers=num_workers > 0,
    )


def create_inference_dataloader(
        texts: List[str],
        schemas: List[dict],
        processor: SchemaTransformer,
        batch_size: int = 32,
        num_workers: int = 4,
) -> DataLoader:
    """
    Create a DataLoader for inference.

    Args:
        texts: List of input texts
        schemas: List of schemas (same length as texts or single schema)
        processor: SchemaTransformer for preprocessing
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        DataLoader yielding PreprocessedBatch
    """
    # Handle single schema for all texts
    if len(schemas) == 1:
        schemas = schemas * len(texts)

    dataset = list(zip(texts, schemas))
    collator = ExtractorDataCollator(processor, is_training=False)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )