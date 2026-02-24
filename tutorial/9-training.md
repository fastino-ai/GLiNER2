# GLiNER2 Training Tutorial

Complete guide to training GLiNER2 models for entity extraction, classification, structured data extraction, and relation extraction.

## Table of Contents

1. [Quick Start](#quick-start)
2. [End-to-End Training Examples](#end-to-end-training-examples)
3. [Data Preparation](#data-preparation)
4. [Training Configuration](#training-configuration)
5. [LoRA Training](#lora-training)
6. [Advanced Topics](#advanced-topics)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Example

```python
from gliner2 import GLiNER2
from gliner2.training.data import InputExample
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# 1. Create training examples
examples = [
    InputExample(
        text="John works at Google in California.",
        entities={"person": ["John"], "company": ["Google"], "location": ["California"]}
    ),
    InputExample(
        text="Apple released iPhone 15.",
        entities={"company": ["Apple"], "product": ["iPhone 15"]}
    ),
]

# 2. Initialize model and config
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(
    output_dir="./output",
    num_epochs=10,
    batch_size=8,
    encoder_lr=1e-5,
    task_lr=5e-4
)

# 3. Train
trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=examples)
```

### Quick Start with JSONL

```python
# Create train.jsonl file with format:
# {"input": "text here", "output": {"entities": {"type": ["mention1", "mention2"]}}}

from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(output_dir="./output", num_epochs=10)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data="train.jsonl")
```

---

## End-to-End Training Examples

### Example 1: Complete NER Training Pipeline

```python
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# Step 1: Prepare training data
train_examples = [
    InputExample(
        text="Tim Cook is the CEO of Apple Inc., based in Cupertino, California.",
        entities={
            "person": ["Tim Cook"],
            "company": ["Apple Inc."],
            "location": ["Cupertino", "California"]
        },
        entity_descriptions={
            "person": "Full name of a person",
            "company": "Business organization name",
            "location": "Geographic location or place"
        }
    ),
    InputExample(
        text="OpenAI released GPT-4 in March 2023. The model was developed in San Francisco.",
        entities={
            "company": ["OpenAI"],
            "model": ["GPT-4"],
            "date": ["March 2023"],
            "location": ["San Francisco"]
        },
        entity_descriptions={
            "model": "Machine learning model or AI system",
            "date": "Date or time reference"
        }
    ),
    # Add more examples...
]

# Step 2: Create and validate dataset
train_dataset = TrainingDataset(train_examples)
train_dataset.validate(strict=True, raise_on_error=True)
train_dataset.print_stats()

# Step 3: Split into train/validation
train_data, val_data, _ = train_dataset.split(
    train_ratio=0.8,
    val_ratio=0.2,
    test_ratio=0.0,
    shuffle=True,
    seed=42
)

# Step 4: Save datasets
train_data.save("train.jsonl")
val_data.save("val.jsonl")

# Step 5: Configure training
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(
    output_dir="./ner_model",
    experiment_name="ner_training",
    num_epochs=15,
    batch_size=16,
    encoder_lr=1e-5,
    task_lr=5e-4,
    warmup_ratio=0.1,
    scheduler_type="cosine",
    fp16=True,
    eval_strategy="epoch",  # Evaluates and saves at end of each epoch
    save_best=True,
    early_stopping=True,
    early_stopping_patience=3,
    report_to_wandb=True,
    wandb_project="ner_training"
)

# Step 6: Train
trainer = GLiNER2Trainer(model, config)
results = trainer.train(
    train_data=train_data,
    eval_data=val_data
)

print(f"Training completed!")
print(f"Best validation loss: {results['best_metric']:.4f}")
print(f"Total steps: {results['total_steps']}")
print(f"Training time: {results['total_time_seconds']/60:.1f} minutes")

# Step 7: Load best model for inference
best_model = GLiNER2.from_pretrained("./ner_model/best")
```

### Example 2: Multi-Task Training (NER + Classification + Relations)

```python
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, Classification, Relation
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# Create multi-task examples
examples = [
    InputExample(
        text="John Smith works at Google in California. The company is thriving and expanding rapidly.",
        entities={
            "person": ["John Smith"],
            "company": ["Google"],
            "location": ["California"]
        },
        classifications=[
            Classification(
                task="sentiment",
                labels=["positive", "negative", "neutral"],
                true_label="positive"
            )
        ],
        relations=[
            Relation("works_at", head="John Smith", tail="Google"),
            Relation("located_in", head="Google", tail="California")
        ]
    ),
    # More examples...
]

# Train multi-task model
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(
    output_dir="./multitask_model",
    num_epochs=20,
    batch_size=16,
    encoder_lr=1e-5,
    task_lr=5e-4
)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=examples)
```

### Example 3: Domain-Specific Fine-tuning (Medical NER)

```python
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# Medical domain examples
medical_examples = [
    InputExample(
        text="Patient presented with hypertension and type 2 diabetes mellitus.",
        entities={
            "condition": ["hypertension", "type 2 diabetes mellitus"]
        },
        entity_descriptions={
            "condition": "Medical condition, disease, or symptom"
        }
    ),
    InputExample(
        text="Prescribed metformin 500mg twice daily. Patient to follow up in 2 weeks.",
        entities={
            "medication": ["metformin"],
            "dosage": ["500mg"],
            "frequency": ["twice daily"],
            "duration": ["2 weeks"]
        },
        entity_descriptions={
            "medication": "Prescribed drug or medication name",
            "dosage": "Amount or strength of medication",
            "frequency": "How often medication is taken",
            "duration": "Time period for treatment"
        }
    ),
    # More medical examples...
]

# Fine-tune on medical domain
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(
    output_dir="./medical_ner",
    num_epochs=20,
    batch_size=16,
    encoder_lr=5e-6,  # Lower LR for fine-tuning
    task_lr=1e-4,
    warmup_ratio=0.05
)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=medical_examples)
```

---

## Data Preparation

### Supported Data Formats

GLiNER2 supports multiple input formats:

1. **JSONL Files** (recommended for large datasets)
2. **InputExample List** (recommended for programmatic creation)
3. **TrainingDataset Object**
4. **Raw Dict List**

```python
# Format 1: JSONL file(s)
trainer.train(train_data="train.jsonl")
trainer.train(train_data=["train1.jsonl", "train2.jsonl"])

# Format 2: InputExample list
examples = [InputExample(...), ...]
trainer.train(train_data=examples)

# Format 3: TrainingDataset
dataset = TrainingDataset.load("train.jsonl")
trainer.train(train_data=dataset)

# Format 4: Raw dicts
raw_data = [{"input": "...", "output": {...}}, ...]
trainer.train(train_data=raw_data)
```

### Creating Training Examples

#### Entity Extraction

```python
from gliner2.training.data import InputExample

# Simple entity extraction
example = InputExample(
    text="John Smith works at Google in San Francisco.",
    entities={
        "person": ["John Smith"],
        "company": ["Google"],
        "location": ["San Francisco"]
    }
)

# With entity descriptions (improves model understanding)
example = InputExample(
    text="BERT was developed by Google AI.",
    entities={
        "model": ["BERT"],
        "organization": ["Google AI"]
    },
    entity_descriptions={
        "model": "Machine learning model or architecture",
        "organization": "Company, research lab, or institution"
    }
)
```

#### Classification

```python
from gliner2.training.data import InputExample, Classification

# Single-label classification
example = InputExample(
    text="This movie is amazing! Best film of the year.",
    classifications=[
        Classification(
            task="sentiment",
            labels=["positive", "negative", "neutral"],
            true_label="positive"
        )
    ]
)

# Multi-label classification
example = InputExample(
    text="Python tutorial covers machine learning and web development.",
    classifications=[
        Classification(
            task="topic",
            labels=["programming", "machine_learning", "web_dev", "data_science"],
            true_label=["programming", "machine_learning", "web_dev"],
            multi_label=True
        )
    ]
)
```

#### Structured Data Extraction

```python
from gliner2.training.data import InputExample, Structure, ChoiceField

# Simple structure
example = InputExample(
    text="iPhone 15 Pro costs $999 and comes in titanium color.",
    structures=[
        Structure(
            "product",
            name="iPhone 15 Pro",
            price="$999",
            color="titanium"
        )
    ]
)

# With choice fields
example = InputExample(
    text="Order #12345 for laptop shipped on 2024-01-15.",
    structures=[
        Structure(
            "order",
            order_id="12345",
            product="laptop",
            date="2024-01-15",
            status=ChoiceField(
                value="shipped",
                choices=["pending", "processing", "shipped", "delivered"]
            )
        )
    ]
)
```

#### Relation Extraction

```python
from gliner2.training.data import InputExample, Relation

# Binary relations
example = InputExample(
    text="Elon Musk founded SpaceX in 2002.",
    relations=[
        Relation("founded", head="Elon Musk", tail="SpaceX"),
        Relation("founded_in", head="SpaceX", tail="2002")
    ]
)

# Custom relation fields
example = InputExample(
    text="Exercise improves mental health.",
    relations=[
        Relation(
            "causal_relation",
            cause="exercise",
            effect="mental health",
            direction="positive"
        )
    ]
)
```

### Data Validation

```python
from gliner2.training.data import TrainingDataset

# Load and validate dataset
dataset = TrainingDataset.load("train.jsonl")

# Strict validation (checks entity spans exist in text)
try:
    dataset.validate(strict=True, raise_on_error=True)
except ValidationError as e:
    print(f"Validation failed: {e}")

# Get validation report
report = dataset.validate(raise_on_error=False)
print(f"Valid: {report['valid']}, Invalid: {report['invalid']}")

# Print statistics
dataset.print_stats()
```

### Data Splitting and Management

```python
from gliner2.training.data import TrainingDataset

# Load full dataset
dataset = TrainingDataset.load("full_data.jsonl")

# Split into train/val/test
train_data, val_data, test_data = dataset.split(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    shuffle=True,
    seed=42
)

# Save splits
train_data.save("train.jsonl")
val_data.save("val.jsonl")
test_data.save("test.jsonl")

# Filter and sample
entity_only = dataset.filter(lambda ex: len(ex.entities) > 0)
small_sample = dataset.sample(n=100, seed=42)

# Combine multiple datasets
dataset1 = TrainingDataset.load("dataset1.jsonl")
dataset2 = TrainingDataset.load("dataset2.jsonl")
combined = TrainingDataset()
combined.add_many(dataset1.examples)
combined.add_many(dataset2.examples)
```

---

## Training Configuration

### Checkpoint Saving Behavior

**Important**: Checkpoints are automatically saved when evaluation runs. The `eval_strategy` parameter controls both when to evaluate and when to save checkpoints:

- `eval_strategy="steps"` (default): Evaluate and save every `eval_steps` steps
- `eval_strategy="epoch"`: Evaluate and save at the end of each epoch
- `eval_strategy="no"`: No evaluation or checkpoint saving (except final checkpoint)

The best checkpoint (based on `metric_for_best`) is always saved separately when `save_best=True`.

### Basic Configuration

```python
from gliner2.training.trainer import TrainingConfig

config = TrainingConfig(
    # Output
    output_dir="./output",
    experiment_name="my_experiment",

    # Training
    num_epochs=10,
    batch_size=32,
    gradient_accumulation_steps=1,

    # Learning rates
    encoder_lr=1e-5,
    task_lr=5e-4,

    # Optimization
    weight_decay=0.01,
    max_grad_norm=1.0,
    scheduler_type="linear",
    warmup_ratio=0.1,

    # Mixed precision
    fp16=True,

    # Checkpointing & Evaluation (saves when evaluating)
    eval_strategy="epoch",  # "epoch", "steps", or "no"
    eval_steps=500,  # Used when eval_strategy="steps"
    save_best=True,

    # Logging
    logging_steps=50,
    report_to_wandb=False,
    wandb_project=None
)
```

### Common Configurations

**Fast Prototyping:**
```python
config = TrainingConfig(
    output_dir="./quick_test",
    num_epochs=3,
    batch_size=16,
    encoder_lr=1e-5,
    task_lr=5e-4,
    max_train_samples=100,
    eval_strategy="no"
)
```

**Production Training:**
```python
config = TrainingConfig(
    output_dir="./production_model",
    num_epochs=20,
    batch_size=32,
    gradient_accumulation_steps=2,
    encoder_lr=5e-6,
    task_lr=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    scheduler_type="cosine",
    fp16=True,
    eval_strategy="steps",  # Evaluates and saves every N steps
    eval_steps=500,
    save_total_limit=5,
    save_best=True,
    early_stopping=True,
    early_stopping_patience=5,
    report_to_wandb=True,
    wandb_project="gliner2-production"
)
```

**Memory-Optimized:**
```python
config = TrainingConfig(
    output_dir="./large_model",
    num_epochs=10,
    batch_size=8,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    fp16=True,
    encoder_lr=1e-6,
    task_lr=5e-5,
    max_grad_norm=0.5,
    num_workers=2
)
```

### Complete Configuration Reference

```python
config = TrainingConfig(
    # Output
    output_dir="./output",
    experiment_name="gliner2",

    # Training steps
    num_epochs=10,
    max_steps=-1,

    # Batch size
    batch_size=32,
    eval_batch_size=64,
    gradient_accumulation_steps=1,

    # Learning rates
    encoder_lr=1e-5,
    task_lr=5e-4,

    # Optimizer
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,

    # Learning rate schedule
    scheduler_type="linear",  # "linear", "cosine", "cosine_restarts", "constant"
    warmup_ratio=0.1,
    warmup_steps=0,
    num_cycles=0.5,

    # Mixed precision
    fp16=True,
    bf16=False,

    # Checkpointing & Evaluation (saves when evaluating)
    eval_strategy="steps",  # "epoch", "steps", or "no" (default: "steps")
    eval_steps=500,  # Evaluate and save every N steps (when eval_strategy="steps")
    save_total_limit=3,
    save_best=True,
    metric_for_best="eval_loss",
    greater_is_better=False,

    # Logging
    logging_steps=50,  # Update progress bar metrics every N steps
    # Metrics (loss, learning rate, throughput) are shown in the progress bar
    logging_first_step=True,
    report_to_wandb=False,  # Enable W&B logging for experiment tracking
    wandb_project=None,
    wandb_entity=None,
    wandb_run_name=None,
    wandb_tags=[],
    wandb_notes=None,

    # Early stopping
    early_stopping=False,
    early_stopping_patience=3,
    early_stopping_threshold=0.0,

    # DataLoader
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,

    # Other
    seed=42,
    deterministic=False,
    gradient_checkpointing=False,
    max_train_samples=-1,
    max_eval_samples=-1,
    validate_data=True,
    strict_validation=False,

    # LoRA (see LoRA section)
    use_lora=False,
    lora_r=16,
    lora_alpha=32.0,
    lora_dropout=0.0,
    lora_target_modules=["encoder", "span_rep", "classifier", "count_embed", "count_pred"],
    save_adapter_only=True,
)
```

---

## LoRA Training

LoRA (Low-Rank Adaptation) enables parameter-efficient fine-tuning by training only a small number of additional parameters while keeping the base model frozen.

> 📚 **For a comprehensive guide on training and using multiple LoRA adapters**, see [Tutorial 10: LoRA Adapters - Multi-Domain Inference](./10-lora_adapters.md)

### Why Use LoRA?

- **Memory Efficient**: Train with 10-100x fewer parameters
- **Faster Training**: Fewer gradients to compute
- **Multiple Adapters**: Train different adapters for different tasks
- **Easy Deployment**: Checkpoints contain merged weights (ready for inference)
- **Granular Control**: Target specific layers or entire module groups

### LoRA Module Groups

GLiNER2 supports both coarse-grained (module groups) and fine-grained (specific layers) control:

**Module Groups** (apply to entire modules):
- `"encoder"` - All encoder layers (query, key, value, dense)
- `"span_rep"` - All linear layers in span representation
- `"classifier"` - All linear layers in classifier head
- `"count_embed"` - All linear layers in count embedding
- `"count_pred"` - All linear layers in count prediction

**Specific Encoder Layers** (fine-grained control):
- `"encoder.query"` - Only query projection layers
- `"encoder.key"` - Only key projection layers
- `"encoder.value"` - Only value projection layers
- `"encoder.dense"` - Only dense (FFN) layers

**Default**: All modules (`["encoder", "span_rep", "classifier", "count_embed", "count_pred"]`) for maximum adaptation.

### Basic LoRA Training

```python
from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# Load base model
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# Configure LoRA training
config = TrainingConfig(
    output_dir="./output_lora",
    num_epochs=10,
    batch_size=16,

    # Enable LoRA
    use_lora=True,
    lora_r=16,              # Rank (higher = more params, better approximation)
    lora_alpha=32,          # Scaling factor (typically 2*r)
    lora_dropout=0.1,       # Dropout for regularization
    lora_target_modules=["encoder"],  # All encoder layers (query, key, value, dense)
    save_adapter_only=True, # Save only adapter weights (not full model)

    # Learning rate (task_lr used for LoRA + task heads when LoRA enabled)
    task_lr=5e-4,
    # encoder_lr is ignored when LoRA is enabled

    # Other settings
    fp16=True,
    eval_strategy="epoch",  # Evaluates and saves at end of each epoch
    save_best=True
)

# Train with LoRA
trainer = GLiNER2Trainer(model, config)
results = trainer.train(train_data="train.jsonl", eval_data="val.jsonl")

# Checkpoints contain merged weights (ready for inference)
best_model = GLiNER2.from_pretrained("./output_lora/best")
```

### LoRA Configuration Parameters

```python
config = TrainingConfig(
    # Enable LoRA
    use_lora=True,

    # LoRA rank (r): Controls the rank of low-rank decomposition
    # Higher r = more parameters but better approximation
    # Typical values: 4, 8, 16, 32, 64
    # Start with 8 or 16 for most tasks
    lora_r=16,

    # LoRA alpha: Scaling factor for LoRA updates
    # Final scaling is alpha/r
    # Typical values: 8, 16, 32 (often 2*r)
    # Common practice: alpha = 2 * r
    lora_alpha=32,

    # LoRA dropout: Dropout probability applied to LoRA path
    # Helps prevent overfitting
    # Typical values: 0.0, 0.05, 0.1
    lora_dropout=0.1,

    # Target modules: Which module groups to apply LoRA to
    # Module groups:
    #   - "encoder": All encoder layers (query, key, value, dense)
    #   - "encoder.query": Only query projection layers in encoder
    #   - "encoder.key": Only key projection layers in encoder
    #   - "encoder.value": Only value projection layers in encoder
    #   - "encoder.dense": Only dense (FFN) layers in encoder
    #   - "span_rep": All linear layers in span representation module
    #   - "classifier": All linear layers in classifier head
    #   - "count_embed": All linear layers in count embedding
    #   - "count_pred": All linear layers in count prediction
    #
    # Common configurations:
    #   - ["encoder"]: Encoder only (query, key, value, dense) - good starting point
    #   - ["encoder.query", "encoder.key", "encoder.value"]: Attention only - memory efficient
    #   - ["encoder.dense"]: FFN only - alternative approach
    #   - ["encoder", "span_rep", "classifier"]: Encoder + task heads - better performance
    #   - ["encoder", "span_rep", "classifier", "count_embed", "count_pred"]: All modules (default) - maximum adaptation
    #
    # Default: All modules for maximum adaptation capacity
    lora_target_modules=["encoder", "span_rep", "classifier", "count_embed", "count_pred"],

    # Save adapter only (recommended)
    # When True: saves only LoRA adapter weights (~2-10 MB)
    # When False: saves full model with merged weights (~100-500 MB)
    save_adapter_only=True,

    # Learning rate for LoRA parameters
    # When LoRA is enabled, task_lr is used for both LoRA and task-specific heads
    task_lr=5e-4,  # Typical: 1e-4 to 1e-3
)
```

### LoRA Training Examples

**Example 1: Memory-Constrained Training**

```python
# Train on GPU with limited memory
config = TrainingConfig(
    output_dir="./lora_small_memory",
    use_lora=True,
    lora_r=8,              # Smaller rank for less memory
    lora_alpha=16,
    batch_size=32,         # Can use larger batch with LoRA
    gradient_accumulation_steps=1,
    task_lr=5e-4,
    fp16=True
)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data="train.jsonl")
```

**Example 2: High-Performance LoRA**

```python
# Use higher rank and include task heads for better performance
config = TrainingConfig(
    output_dir="./lora_high_perf",
    use_lora=True,
    lora_r=32,             # Higher rank
    lora_alpha=64,
    lora_dropout=0.05,
    lora_target_modules=["encoder", "span_rep", "classifier"],  # Encoder + task heads
    batch_size=16,
    task_lr=1e-3,          # Slightly higher LR
    num_epochs=15
)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data="train.jsonl")
```

**Example 3: Domain Adaptation with LoRA**

```python
# Fine-tune for specific domain with LoRA
config = TrainingConfig(
    output_dir="./lora_medical",
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    batch_size=16,
    task_lr=5e-4,
    num_epochs=20,
    warmup_ratio=0.05
)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=medical_examples)
```

### LoRA vs Full Fine-tuning

| Aspect | LoRA | Full Fine-tuning |
|--------|------|------------------|
| Trainable Parameters | ~0.1-1% of model | 100% of model |
| Memory Usage | Low | High |
| Training Speed | Fast | Slower |
| Checkpoint Size | Small (merged) | Large |
| Performance | Good (often comparable) | Best |
| Use Case | Limited data, multiple tasks | Large datasets, single task |

### LoRA Best Practices

1. **Start with default settings**: `r=16`, `alpha=32`, `dropout=0.1`
2. **Increase rank if needed**: If performance is insufficient, try `r=32` or `r=64`
3. **Use dropout for regularization**: Set `lora_dropout=0.1` to prevent overfitting
4. **Target attention layers first**: Start with `["query", "key", "value"]`, add `"dense"` if needed
5. **Higher learning rate**: LoRA typically works well with `task_lr=5e-4` to `1e-3`
6. **Checkpoint merging**: Checkpoints automatically contain merged weights (ready for inference)

### Loading Checkpoints

```python
# Load model from checkpoint (for inference or continued training)
trainer = GLiNER2Trainer(model, config)

# Load checkpoint (weights are merged in checkpoint)
trainer.load_checkpoint("./output_lora/checkpoint-1000")

# Continue training (LoRA will be re-applied if use_lora=True)
trainer.train(train_data="train.jsonl")
```

**Note**: Checkpoints do not save optimizer/scheduler state. Training always starts fresh, but model weights are loaded.

---

## Advanced Topics

### Custom Metrics

```python
def compute_metrics(model, eval_dataset):
    """Custom metric computation function."""
    # Your custom evaluation logic
    # For example, compute F1 score on entities

    metrics = {}
    # ... compute metrics ...
    metrics["f1"] = 0.85
    metrics["precision"] = 0.87
    metrics["recall"] = 0.83

    return metrics

trainer = GLiNER2Trainer(
    model=model,
    config=config,
    compute_metrics=compute_metrics
)

trainer.train(train_data=examples, eval_data=eval_examples)
```

### Loading Checkpoints

```python
trainer = GLiNER2Trainer(model, config)

# Load checkpoint (model weights only, no optimizer state)
trainer.load_checkpoint("./output/checkpoint-1000")

# Continue training (starts fresh with loaded weights)
trainer.train(train_data=examples)
```

**Note**: Checkpoints contain model weights only. Training state (optimizer, scheduler) is not saved, so training always starts fresh.

### Distributed Training

```python
# Launch with torchrun
# torchrun --nproc_per_node=4 train_script.py

import os

config = TrainingConfig(
    output_dir="./output",
    num_epochs=10,
    local_rank=int(os.environ.get("LOCAL_RANK", -1))  # Auto-detect DDP
)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=examples)
```

### Learning Rate Schedules

```python
# Linear warmup + linear decay (default)
config = TrainingConfig(
    scheduler_type="linear",
    warmup_ratio=0.1
)

# Cosine annealing
config = TrainingConfig(
    scheduler_type="cosine",
    warmup_ratio=0.05
)

# Cosine with restarts
config = TrainingConfig(
    scheduler_type="cosine_restarts",
    warmup_ratio=0.05,
    num_cycles=3
)

# Constant LR after warmup
config = TrainingConfig(
    scheduler_type="constant",
    warmup_steps=500
)
```

### Weights & Biases Integration

```python
config = TrainingConfig(
    output_dir="./output",
    report_to_wandb=True,
    wandb_project="my-gliner-project",
    wandb_entity="my-team",
    wandb_run_name="experiment-1",
    wandb_tags=["ner", "entity-extraction"],
    wandb_notes="Testing new architecture"
)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=examples)
# Metrics automatically logged to W&B
```

### Data Augmentation

```python
from gliner2.training.data import InputExample, TrainingDataset

def augment_example(example: InputExample) -> List[InputExample]:
    """Create augmented versions of an example."""
    augmented = [example]  # Original

    # Shuffle entity order
    if len(example.entities) > 1:
        shuffled_entities = dict(sorted(example.entities.items(), reverse=True))
        augmented.append(InputExample(
            text=example.text,
            entities=shuffled_entities
        ))

    return augmented

# Apply augmentation
dataset = TrainingDataset.load("train.jsonl")
augmented_examples = []
for ex in dataset:
    augmented_examples.extend(augment_example(ex))

augmented_dataset = TrainingDataset(augmented_examples)
trainer.train(train_data=augmented_dataset)
```

---

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
```python
# 1. Reduce batch size
config = TrainingConfig(batch_size=4)

# 2. Use gradient accumulation
config = TrainingConfig(
    batch_size=4,
    gradient_accumulation_steps=8  # Effective batch = 32
)

# 3. Enable gradient checkpointing
config = TrainingConfig(gradient_checkpointing=True)

# 4. Use mixed precision
config = TrainingConfig(fp16=True)

# 5. Use LoRA (most memory efficient)
config = TrainingConfig(
    use_lora=True,
    lora_r=8,
    batch_size=32  # Can use larger batch with LoRA
)

# 6. Reduce workers
config = TrainingConfig(num_workers=2)
```

### Training is Slow

**Solutions:**
```python
# 1. Increase batch size (if memory allows)
config = TrainingConfig(batch_size=64)

# 2. Increase workers
config = TrainingConfig(num_workers=8)

# 3. Use mixed precision
config = TrainingConfig(fp16=True)

# 4. Reduce evaluation frequency (also reduces checkpoint saves)
config = TrainingConfig(
    eval_strategy="steps",
    eval_steps=1000  # Evaluate and save every 1000 steps
)

# 5. Use LoRA (faster training)
config = TrainingConfig(use_lora=True)
```

### Validation Errors

```python
# Check specific errors
dataset = TrainingDataset(examples)
report = dataset.validate(raise_on_error=False)

print(f"Invalid examples: {report['invalid_indices']}")
for error in report['errors'][:10]:
    print(error)

# Fix common issues:
# 1. Entity not in text
example = InputExample(
    text="John works here",
    entities={"person": ["John Smith"]}  # ERROR: "John Smith" not in text
)
# Fix: Use exact match
example = InputExample(
    text="John works here",
    entities={"person": ["John"]}  # OK
)

# 2. Empty entities
example = InputExample(
    text="Some text",
    entities={"person": []}  # ERROR: empty list
)
# Fix: Remove empty entity types
example = InputExample(
    text="Some text",
    entities={}  # OK if other tasks present
)

# 3. Use loose validation during development
dataset.validate(strict=False, raise_on_error=False)
```

### Model Not Learning

**Solutions:**
```python
# 1. Check learning rates
config = TrainingConfig(
    encoder_lr=1e-5,  # Try: 5e-6, 1e-5, 5e-5
    task_lr=5e-4      # Try: 1e-4, 5e-4, 1e-3
)

# 2. Increase training epochs
config = TrainingConfig(num_epochs=20)

# 3. Check warmup
config = TrainingConfig(warmup_ratio=0.1)

# 4. Reduce weight decay
config = TrainingConfig(weight_decay=0.001)

# 5. Try different scheduler
config = TrainingConfig(scheduler_type="cosine")

# 6. Check data quality
dataset.print_stats()
dataset.validate()
```

### LoRA-Specific Issues

**Issue: LoRA not reducing memory**
```python
# Ensure LoRA is enabled
config = TrainingConfig(use_lora=True)

# Use smaller rank
config = TrainingConfig(use_lora=True, lora_r=8)

# Target only encoder (fewer modules = less memory)
config = TrainingConfig(
    use_lora=True,
    lora_target_modules=["encoder"]
)

# Target only attention layers (even less memory)
config = TrainingConfig(
    use_lora=True,
    lora_target_modules=["encoder.query", "encoder.key", "encoder.value"]
)
```

**Issue: LoRA performance worse than full fine-tuning**
```python
# Increase rank
config = TrainingConfig(use_lora=True, lora_r=32)

# Add task heads to target modules
config = TrainingConfig(
    use_lora=True,
    lora_target_modules=["encoder", "span_rep", "classifier"]
)

# Target all modules for maximum adaptation
config = TrainingConfig(
    use_lora=True,
    lora_target_modules=["encoder", "span_rep", "classifier", "count_embed", "count_pred"]
)

# Increase learning rate
config = TrainingConfig(use_lora=True, task_lr=1e-3)

# Train longer
config = TrainingConfig(use_lora=True, num_epochs=20)
```

---

## Best Practices

1. **Always validate data before training:**
   ```python
   dataset.validate()
   dataset.print_stats()
   ```

2. **Start with small subset for testing:**
   ```python
   config = TrainingConfig(max_train_samples=100)
   ```

3. **Use early stopping for long training:**
   ```python
   config = TrainingConfig(
       early_stopping=True,
       early_stopping_patience=5
   )
   ```

4. **Save intermediate checkpoints:**
   ```python
   config = TrainingConfig(
       eval_strategy="steps",  # Evaluates and saves every N steps
       eval_steps=500,
       save_best=True
   )
   ```

5. **Monitor training with W&B:**
   ```python
   config = TrainingConfig(
       report_to_wandb=True,
       wandb_project="my-project"
   )
   ```

6. **Use descriptive entity types and add descriptions:**
   ```python
   example = InputExample(
       text="...",
       entities={...},
       entity_descriptions={
           "person": "Full name of a person",
           "company": "Business organization name"
       }
   )
   ```

7. **Split your data properly:**
   ```python
   train, val, test = dataset.split(0.8, 0.1, 0.1)
   ```

8. **Use appropriate learning rates:**
   - Full fine-tuning: Encoder LR `1e-6` to `5e-5` (typically `1e-5`), Task LR `1e-4` to `1e-3` (typically `5e-4`)
   - LoRA: Task LR `1e-4` to `1e-3` (typically `5e-4`)

9. **Consider LoRA for memory-constrained scenarios:**
   ```python
   config = TrainingConfig(
       use_lora=True,
       lora_r=16,
       lora_alpha=32
   )
   ```

10. **Document your experiments:**
    ```python
    config = TrainingConfig(
        experiment_name="v1_medical_ner",
        wandb_notes="Testing with LoRA and augmented data"
    )
    ```

---

## Summary

GLiNER2 provides a flexible and powerful framework for training information extraction models:

- **Multiple data formats**: JSONL, InputExample, TrainingDataset, raw dicts
- **Four task types**: Entities, Classifications, Structures, Relations
- **Comprehensive validation**: Automatic data validation and statistics
- **Production-ready training**: FP16, gradient accumulation, distributed training
- **LoRA support**: Parameter-efficient fine-tuning with minimal memory usage
- **Extensive configuration**: 40+ config options for fine-grained control
- **Easy to use**: Quick start in 10 lines of code

Start with the Quick Start examples and gradually explore advanced features as needed!
