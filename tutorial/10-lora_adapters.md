# Tutorial 10: LoRA Adapters - Multi-Domain Inference

## Table of Contents
1. [Introduction](#introduction)
2. [Why Use LoRA Adapters?](#why-use-lora-adapters)
3. [Training Your First Adapter](#training-your-first-adapter)
4. [Training Multiple Domain Adapters](#training-multiple-domain-adapters)
5. [Loading and Swapping Adapters](#loading-and-swapping-adapters)
6. [Real-World Use Cases](#real-world-use-cases)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Introduction

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that allows you to train specialized adapters for different domains without modifying the base model. This enables:

- **Fast domain switching**: Swap between domains in milliseconds
- **Minimal storage**: Adapters are ~2-10 MB vs ~100-500 MB for full models
- **Domain specialization**: Train separate adapters for legal, medical, financial, etc.
- **Easy deployment**: Keep one base model + multiple lightweight adapters

## Why Use LoRA Adapters?

### Memory Efficiency

```
Full Model Fine-tuning:
- Legal model:     450 MB
- Medical model:   450 MB
- Financial model: 450 MB
Total:            1.35 GB

LoRA Adapters:
- Base model:      450 MB
- Legal adapter:     5 MB
- Medical adapter:   5 MB
- Financial adapter: 5 MB
Total:             465 MB (65% less!)
```

### Fast Training

LoRA adapters train **2-3x faster** than full fine-tuning because:
- Only ~1-5% of parameters are trainable
- Smaller gradient computations
- Less GPU memory required

### Easy Multi-Domain Inference

```python
# One base model, multiple domains
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# Legal domain
model.load_adapter("./legal_adapter")
legal_results = model.extract_entities(legal_text, ["company", "law"])

# Medical domain (swap in <1 second)
model.load_adapter("./medical_adapter")
medical_results = model.extract_entities(medical_text, ["disease", "drug"])
```

## Training Your First Adapter

### Step 1: Prepare Domain-Specific Data

```python
from gliner2.training.data import InputExample

# Legal domain examples
legal_examples = [
    InputExample(
        text="Apple Inc. filed a lawsuit against Samsung Electronics.",
        entities={"company": ["Apple Inc.", "Samsung Electronics"]}
    ),
    InputExample(
        text="The plaintiff Google LLC accused Microsoft Corporation of patent infringement.",
        entities={"company": ["Google LLC", "Microsoft Corporation"]}
    ),
    InputExample(
        text="Tesla Motors settled the case with the Securities and Exchange Commission.",
        entities={
            "company": ["Tesla Motors"],
            "organization": ["Securities and Exchange Commission"]
        }
    ),
    # Add 100-1000+ examples for best results
]
```

### Step 2: Configure LoRA Training

```python
from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# LoRA configuration
config = TrainingConfig(
    output_dir="./legal_adapter",
    experiment_name="legal_domain",

    # Training parameters
    num_epochs=10,
    batch_size=8,
    gradient_accumulation_steps=2,
    encoder_lr=1e-5,
    task_lr=5e-4,

    # LoRA settings
    use_lora=True,                              # Enable LoRA
    lora_r=8,                                   # Rank (4, 8, 16, 32)
    lora_alpha=16.0,                           # Scaling factor (usually 2*r)
    lora_dropout=0.0,                          # Dropout for LoRA layers
    lora_target_modules=["encoder"],           # Apply to all encoder layers (query, key, value, dense)
    save_adapter_only=True,                    # Save only adapter (not full model)

    # Optimization
    eval_strategy="epoch",  # Evaluates and saves at end of each epoch
    eval_steps=500,  # Used when eval_strategy="steps"
    logging_steps=50,
    fp16=True,  # Use mixed precision if GPU available
)
```

### Step 3: Train the Adapter

```python
# Load base model
base_model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# Create trainer
trainer = GLiNER2Trainer(model=base_model, config=config)

# Train adapter
trainer.train(train_data=legal_examples)

# Adapter automatically saved to ./legal_adapter/final/
```

**Training output:**
```
🔧 LoRA Configuration
======================================================================
Enabled            : True
Rank (r)           : 8
Alpha              : 16.0
Scaling (α/r)      : 2.0000
Dropout            : 0.0
Target modules     : query, key, value, dense
LoRA layers        : 144
----------------------------------------------------------------------
Trainable params   : 1,327,104 / 124,442,368 (1.07%)
Memory savings     : ~98.9% fewer gradients
======================================================================

***** Running Training *****
  Num examples = 1000
  Num epochs = 10
  Batch size = 8
  Effective batch size = 16
  Total optimization steps = 625
  LoRA enabled: 1,327,104 trainable / 124,442,368 total (1.07%)
```

## Training Multiple Domain Adapters

Let's train adapters for three different domains: **Legal**, **Medical**, and **Customer Support**.

### Complete Multi-Domain Training Script

```python
from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from gliner2.training.data import InputExample

# ============================================================================
# Define Domain Data
# ============================================================================

# Legal domain
legal_examples = [
    InputExample(
        text="Apple Inc. filed a lawsuit against Samsung Electronics.",
        entities={"company": ["Apple Inc.", "Samsung Electronics"]}
    ),
    InputExample(
        text="The plaintiff Google LLC accused Microsoft Corporation of patent infringement.",
        entities={"company": ["Google LLC", "Microsoft Corporation"]}
    ),
    # Add more examples...
]

# Medical domain
medical_examples = [
    InputExample(
        text="Patient diagnosed with Type 2 Diabetes and Hypertension.",
        entities={"disease": ["Type 2 Diabetes", "Hypertension"]}
    ),
    InputExample(
        text="Prescribed Metformin 500mg twice daily and Lisinopril 10mg once daily.",
        entities={
            "drug": ["Metformin", "Lisinopril"],
            "dosage": ["500mg", "10mg"]
        }
    ),
    # Add more examples...
]

# Customer support domain
support_examples = [
    InputExample(
        text="Customer John Smith reported issue with Order #12345.",
        entities={
            "customer": ["John Smith"],
            "order_id": ["Order #12345"]
        }
    ),
    InputExample(
        text="Refund of $99.99 processed for Order #98765 on 2024-01-15.",
        entities={
            "order_id": ["Order #98765"],
            "amount": ["$99.99"],
            "date": ["2024-01-15"]
        }
    ),
    # Add more examples...
]

# ============================================================================
# Training Function
# ============================================================================

def train_domain_adapter(
    base_model_name: str,
    examples: list,
    domain_name: str,
    output_dir: str = "./adapters"
):
    """Train a LoRA adapter for a specific domain."""

    adapter_path = f"{output_dir}/{domain_name}_adapter"

    config = TrainingConfig(
        output_dir=adapter_path,
        experiment_name=f"{domain_name}_domain",

        # Training
        num_epochs=10,
        batch_size=8,
        gradient_accumulation_steps=2,
        encoder_lr=1e-5,
        task_lr=5e-4,

        # LoRA
        use_lora=True,
        lora_r=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_target_modules=["encoder"],  # All encoder layers
        save_adapter_only=True,

        # Logging & Checkpointing
        eval_strategy="no",  # Set to "epoch" or "steps" if you have validation set
        eval_steps=500,  # Used when eval_strategy="steps"
        logging_steps=50,
        fp16=True,
    )

    # Load base model
    print(f"\n{'='*60}")
    print(f"Training {domain_name.upper()} adapter")
    print(f"{'='*60}")

    model = GLiNER2.from_pretrained(base_model_name)
    trainer = GLiNER2Trainer(model=model, config=config)

    # Train
    results = trainer.train(train_data=examples)

    print(f"\n✅ {domain_name.capitalize()} adapter trained!")
    print(f"📁 Saved to: {adapter_path}/final/")
    print(f"⏱️  Training time: {results['total_time_seconds']:.2f}s")

    return f"{adapter_path}/final"

# ============================================================================
# Train All Adapters
# ============================================================================

if __name__ == "__main__":
    BASE_MODEL = "fastino/gliner2-base-v1"

    # Train adapters for each domain
    legal_adapter_path = train_domain_adapter(
        BASE_MODEL, legal_examples, "legal"
    )

    medical_adapter_path = train_domain_adapter(
        BASE_MODEL, medical_examples, "medical"
    )

    support_adapter_path = train_domain_adapter(
        BASE_MODEL, support_examples, "support"
    )

    print("\n" + "="*60)
    print("🎉 All adapters trained successfully!")
    print("="*60)
    print(f"Legal adapter:    {legal_adapter_path}")
    print(f"Medical adapter:  {medical_adapter_path}")
    print(f"Support adapter:  {support_adapter_path}")
```

## Loading and Swapping Adapters

### Basic Usage

```python
from gliner2 import GLiNER2

# Load base model once
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# Load legal adapter
model.load_adapter("./adapters/legal_adapter/final")

# Use the model
result = model.extract_entities(
    "Apple Inc. sued Samsung over patent rights.",
    ["company", "legal_action"]
)
print(result)
```

### Swapping Between Adapters

```python
# Load base model
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# Legal domain
print("📋 Legal Analysis:")
model.load_adapter("./adapters/legal_adapter/final")
legal_text = "Google LLC filed a complaint against Oracle Corporation."
legal_result = model.extract_entities(legal_text, ["company", "legal_action"])
print(f"  {legal_result}")

# Swap to medical domain
print("\n🏥 Medical Analysis:")
model.load_adapter("./adapters/medical_adapter/final")
medical_text = "Patient presents with Pneumonia and was prescribed Amoxicillin."
medical_result = model.extract_entities(medical_text, ["disease", "drug"])
print(f"  {medical_result}")

# Swap to support domain
print("\n💬 Support Analysis:")
model.load_adapter("./adapters/support_adapter/final")
support_text = "Customer reported Order #12345 not delivered on time."
support_result = model.extract_entities(support_text, ["order_id", "issue"])
print(f"  {support_result}")

# Use base model without adapter
print("\n🔧 Base Model (no adapter):")
model.unload_adapter()
base_result = model.extract_entities("Some generic text", ["entity"])
print(f"  {base_result}")
```

**Output:**
```
📋 Legal Analysis:
  {'entities': [{'text': 'Google LLC', 'label': 'company', ...},
                {'text': 'Oracle Corporation', 'label': 'company', ...}]}

🏥 Medical Analysis:
  {'entities': [{'text': 'Pneumonia', 'label': 'disease', ...},
                {'text': 'Amoxicillin', 'label': 'drug', ...}]}

💬 Support Analysis:
  {'entities': [{'text': 'Order #12345', 'label': 'order_id', ...}]}

🔧 Base Model (no adapter):
  {'entities': [{'text': 'text', 'label': 'entity', ...}]}
```

### Batch Processing with Adapter Swapping

```python
def process_documents_by_domain(model, documents_by_domain, adapters):
    """
    Process multiple documents across different domains efficiently.

    Args:
        model: Base GLiNER2 model
        documents_by_domain: Dict[domain_name, List[document_text]]
        adapters: Dict[domain_name, adapter_path]

    Returns:
        Dict[domain_name, List[results]]
    """
    results = {}

    for domain, documents in documents_by_domain.items():
        print(f"Processing {domain} domain ({len(documents)} documents)...")

        # Load domain-specific adapter
        model.load_adapter(adapters[domain])

        # Process all documents for this domain
        domain_results = []
        for doc in documents:
            result = model.extract_entities(doc, get_entity_types(domain))
            domain_results.append(result)

        results[domain] = domain_results

    return results

def get_entity_types(domain):
    """Get entity types for each domain."""
    types = {
        "legal": ["company", "person", "law", "legal_action"],
        "medical": ["disease", "drug", "symptom", "procedure"],
        "support": ["customer", "order_id", "product", "issue"]
    }
    return types.get(domain, ["entity"])

# Example usage
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

documents_by_domain = {
    "legal": [
        "Apple Inc. filed suit against Samsung.",
        "Microsoft acquired LinkedIn for $26B.",
    ],
    "medical": [
        "Patient has Type 2 Diabetes.",
        "Prescribed Metformin 500mg daily.",
    ],
    "support": [
        "Issue with Order #12345 reported.",
        "Refund processed for Order #98765.",
    ]
}

adapters = {
    "legal": "./adapters/legal_adapter/final",
    "medical": "./adapters/medical_adapter/final",
    "support": "./adapters/support_adapter/final",
}

results = process_documents_by_domain(model, documents_by_domain, adapters)

# Results organized by domain
for domain, domain_results in results.items():
    print(f"\n{domain.upper()} Results:")
    for i, result in enumerate(domain_results, 1):
        print(f"  Document {i}: {len(result['entities'])} entities found")
```

## Real-World Use Cases

### Use Case 1: Multi-Tenant SaaS Platform

```python
class MultiTenantEntityExtractor:
    """Entity extraction service for multi-tenant platform."""

    def __init__(self, base_model_name: str, tenant_adapters: dict):
        """
        Args:
            base_model_name: Path to base model
            tenant_adapters: Dict mapping tenant_id to adapter_path
        """
        self.model = GLiNER2.from_pretrained(base_model_name)
        self.tenant_adapters = tenant_adapters
        self.current_tenant = None

    def extract_for_tenant(self, tenant_id: str, text: str, entity_types: list):
        """Extract entities for specific tenant."""
        # Load tenant-specific adapter if needed
        if self.current_tenant != tenant_id:
            adapter_path = self.tenant_adapters.get(tenant_id)
            if adapter_path:
                self.model.load_adapter(adapter_path)
            else:
                self.model.unload_adapter()  # Use base model
            self.current_tenant = tenant_id

        return self.model.extract_entities(text, entity_types)

# Setup
extractor = MultiTenantEntityExtractor(
    base_model_name="fastino/gliner2-base-v1",
    tenant_adapters={
        "legal_firm_123": "./adapters/legal_adapter/final",
        "hospital_456": "./adapters/medical_adapter/final",
        "ecommerce_789": "./adapters/support_adapter/final",
    }
)

# Usage
legal_result = extractor.extract_for_tenant(
    "legal_firm_123",
    "Apple sued Samsung",
    ["company"]
)

medical_result = extractor.extract_for_tenant(
    "hospital_456",
    "Patient has diabetes",
    ["disease"]
)
```

### Use Case 2: Document Classification Pipeline

```python
def classify_and_extract(document: str, model: GLiNER2, adapters: dict):
    """
    Classify document type and extract relevant entities.

    1. Classify document type using base model
    2. Load appropriate domain adapter
    3. Extract domain-specific entities
    """
    # Step 1: Classify document type
    doc_type_result = model.extract_entities(
        document,
        ["legal_document", "medical_record", "support_ticket", "financial_report"]
    )

    # Determine document type
    if doc_type_result['entities']:
        doc_type = doc_type_result['entities'][0]['label']
        doc_type = doc_type.replace("_document", "").replace("_record", "").replace("_ticket", "").replace("_report", "")
    else:
        doc_type = "general"

    # Step 2: Load appropriate adapter
    adapter_mapping = {
        "legal": adapters.get("legal"),
        "medical": adapters.get("medical"),
        "support": adapters.get("support"),
        "financial": adapters.get("financial"),
    }

    if doc_type in adapter_mapping and adapter_mapping[doc_type]:
        model.load_adapter(adapter_mapping[doc_type])

    # Step 3: Extract domain-specific entities
    entity_types = {
        "legal": ["company", "person", "law", "legal_action"],
        "medical": ["disease", "drug", "symptom", "procedure", "dosage"],
        "support": ["customer", "order_id", "product", "issue", "status"],
        "financial": ["company", "amount", "date", "stock_symbol"],
    }

    entities = model.extract_entities(
        document,
        entity_types.get(doc_type, ["entity"])
    )

    return {
        "document_type": doc_type,
        "entities": entities['entities']
    }

# Usage
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

adapters = {
    "legal": "./adapters/legal_adapter/final",
    "medical": "./adapters/medical_adapter/final",
    "support": "./adapters/support_adapter/final",
}

document = "Patient John Smith diagnosed with Type 2 Diabetes on 2024-01-15."
result = classify_and_extract(document, model, adapters)

print(f"Document Type: {result['document_type']}")
print(f"Entities: {result['entities']}")
```

### Use Case 3: A/B Testing Adapters

```python
import random

class AdapterABTester:
    """A/B test different adapter versions."""

    def __init__(self, base_model_name: str, adapter_variants: dict):
        """
        Args:
            adapter_variants: {"v1": path1, "v2": path2, ...}
        """
        self.model = GLiNER2.from_pretrained(base_model_name)
        self.adapter_variants = adapter_variants
        self.results = {variant: [] for variant in adapter_variants}

    def test_sample(self, text: str, entity_types: list, true_entities: list):
        """Test a sample with all adapter variants."""
        sample_results = {}

        for variant, adapter_path in self.adapter_variants.items():
            # Load variant
            self.model.load_adapter(adapter_path)

            # Get predictions
            pred = self.model.extract_entities(text, entity_types)

            # Compute metrics
            f1 = self.compute_f1(pred['entities'], true_entities)

            sample_results[variant] = {
                "predictions": pred['entities'],
                "f1_score": f1
            }

            self.results[variant].append(f1)

        return sample_results

    def compute_f1(self, predicted, ground_truth):
        """Simple F1 computation (simplified for demo)."""
        pred_set = {(e['text'], e['label']) for e in predicted}
        true_set = {(e['text'], e['label']) for e in ground_truth}

        if not pred_set and not true_set:
            return 1.0
        if not pred_set or not true_set:
            return 0.0

        tp = len(pred_set & true_set)
        precision = tp / len(pred_set) if pred_set else 0
        recall = tp / len(true_set) if true_set else 0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def get_summary(self):
        """Get A/B test summary."""
        summary = {}
        for variant, scores in self.results.items():
            if scores:
                summary[variant] = {
                    "avg_f1": sum(scores) / len(scores),
                    "samples": len(scores)
                }
        return summary

# Usage
tester = AdapterABTester(
    base_model_name="fastino/gliner2-base-v1",
    adapter_variants={
        "v1_r4": "./adapters/legal_v1_r4/final",
        "v2_r8": "./adapters/legal_v2_r8/final",
        "v3_r16": "./adapters/legal_v3_r16/final",
    }
)

# Test samples
test_samples = [
    {
        "text": "Apple Inc. sued Samsung Electronics.",
        "entity_types": ["company"],
        "true_entities": [
            {"text": "Apple Inc.", "label": "company"},
            {"text": "Samsung Electronics", "label": "company"}
        ]
    },
    # More samples...
]

for sample in test_samples:
    results = tester.test_sample(
        sample["text"],
        sample["entity_types"],
        sample["true_entities"]
    )

# Get summary
summary = tester.get_summary()
for variant, metrics in summary.items():
    print(f"{variant}: Avg F1 = {metrics['avg_f1']:.3f} ({metrics['samples']} samples)")
```

## Best Practices

### 1. Choosing LoRA Hyperparameters

```python
# Small datasets (< 1K examples)
config = TrainingConfig(
    lora_r=4,           # Lower rank = fewer parameters
    lora_alpha=8.0,     # alpha = 2 * r
    num_epochs=10,
)

# Medium datasets (1K-10K examples)
config = TrainingConfig(
    lora_r=8,           # Standard rank
    lora_alpha=16.0,
    num_epochs=5,
)

# Large datasets (> 10K examples)
config = TrainingConfig(
    lora_r=16,          # Higher rank = more capacity
    lora_alpha=32.0,
    num_epochs=3,
)
```

### 2. Target Module Selection

**Understanding Module Groups:**

GLiNER2 supports fine-grained control over which layers receive LoRA adaptation:

```python
# Option 1: Encoder only - all layers (query, key, value, dense)
# Use case: General domain adaptation, good starting point
# Memory: Moderate (~1-2% of model parameters)
lora_target_modules=["encoder"]

# Option 2: Encoder - attention layers only
# Use case: Very memory-constrained scenarios
# Memory: Low (~0.5-1% of model parameters)
lora_target_modules=["encoder.query", "encoder.key", "encoder.value"]

# Option 3: Encoder - FFN layers only
# Use case: Alternative to attention-only, sometimes better for certain tasks
# Memory: Low (~0.5-1% of model parameters)
lora_target_modules=["encoder.dense"]

# Option 4: Encoder + task heads
# Use case: When you want to adapt both representation and task-specific layers
# Memory: Moderate-High (~2-4% of model parameters)
lora_target_modules=["encoder", "span_rep", "classifier"]

# Option 5: All modules (DEFAULT)
# Use case: Maximum adaptation capacity, best performance
# Memory: High (~3-5% of model parameters)
lora_target_modules=["encoder", "span_rep", "classifier", "count_embed", "count_pred"]
```

**Recommendations:**
- **Start with encoder only** (`["encoder"]`) for most tasks
- **Add task heads** if performance is insufficient
- **Use attention-only** for extreme memory constraints
- **Use all modules** (default) when you need maximum performance

### 3. Adapter Organization

```
project/
├── base_model/
│   └── gliner2-base-v1/
├── adapters/
│   ├── legal/
│   │   ├── v1_r8/
│   │   │   └── final/
│   │   └── v2_r16/
│   │       └── final/
│   ├── medical/
│   │   └── final/
│   └── support/
│       └── final/
└── scripts/
    ├── train_adapters.py
    └── evaluate_adapters.py
```

### 4. Version Control for Adapters

```python
# adapter_metadata.json
{
    "legal_v1": {
        "path": "./adapters/legal/v1_r8/final",
        "base_model": "fastino/gliner2-base-v1",
        "lora_r": 8,
        "lora_alpha": 16.0,
        "trained_on": "2024-01-15",
        "training_samples": 5000,
        "eval_f1": 0.87,
        "notes": "Initial legal domain adapter"
    },
    "legal_v2": {
        "path": "./adapters/legal/v2_r16/final",
        "base_model": "fastino/gliner2-base-v1",
        "lora_r": 16,
        "lora_alpha": 32.0,
        "trained_on": "2024-02-01",
        "training_samples": 10000,
        "eval_f1": 0.92,
        "notes": "Improved with more data and higher rank"
    }
}
```

### 5. Monitoring Adapter Performance

```python
def evaluate_adapter(model, adapter_path, test_data):
    """Evaluate adapter performance on test data."""
    model.load_adapter(adapter_path)

    results = {
        "total": 0,
        "correct": 0,
        "precision_sum": 0,
        "recall_sum": 0,
    }

    for sample in test_data:
        pred = model.extract_entities(sample["text"], sample["entity_types"])

        # Compute metrics
        metrics = compute_metrics(pred['entities'], sample["true_entities"])
        results["total"] += 1
        results["precision_sum"] += metrics["precision"]
        results["recall_sum"] += metrics["recall"]

    avg_precision = results["precision_sum"] / results["total"]
    avg_recall = results["recall_sum"] / results["total"]
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": f1,
        "samples": results["total"]
    }
```

## Troubleshooting

### Issue 1: Adapter Not Affecting Predictions

**Symptom**: Predictions are the same with and without adapter.

**Solution**:
```python
# Check if adapter is actually loaded
print(f"Has adapter: {model.has_adapter}")

# Check LoRA layers
from gliner2.training.lora import LoRALayer
lora_count = sum(1 for m in model.modules() if isinstance(m, LoRALayer))
print(f"LoRA layers: {lora_count}")

# Should be > 0 if adapter is loaded
assert lora_count > 0, "No LoRA layers found!"
```

### Issue 2: Out of Memory During Training

**Solution**:
```python
config = TrainingConfig(
    # Reduce batch size
    batch_size=4,  # Instead of 8
    gradient_accumulation_steps=4,  # Maintain effective batch size

    # Use smaller LoRA rank
    lora_r=4,  # Instead of 8

    # Enable mixed precision
    fp16=True,

    # Target only attention layers (fewer parameters)
    lora_target_modules=["encoder.query", "encoder.key", "encoder.value"],
)
```

### Issue 3: Adapter File Not Found

**Solution**:
```python
import os
from gliner2.training.lora import LoRAAdapterConfig

adapter_path = "./adapters/legal_adapter/final"

# Check if path exists
if not os.path.exists(adapter_path):
    print(f"Path does not exist: {adapter_path}")
    # List available checkpoints
    checkpoint_dir = "./adapters/legal_adapter"
    if os.path.exists(checkpoint_dir):
        checkpoints = os.listdir(checkpoint_dir)
        print(f"Available checkpoints: {checkpoints}")

# Check if it's a valid adapter
if LoRAAdapterConfig.is_adapter_path(adapter_path):
    print("Valid adapter path!")
    config = LoRAAdapterConfig.load(adapter_path)
    print(f"Adapter config: {config}")
else:
    print("Not a valid adapter path!")
```

### Issue 4: Slow Adapter Switching

**Problem**: Switching between adapters takes too long.

**Solution**:
```python
# Pre-load adapters in memory (if you have enough RAM)
adapters = {}
for domain, path in adapter_paths.items():
    # Load adapter weights into memory
    adapters[domain] = load_adapter_to_memory(path)

# Fast switching from memory (not implemented in base API,
# but possible with custom caching layer)
```

## Summary

### Key Takeaways

✅ **LoRA adapters** enable efficient multi-domain inference
✅ **Training** is 2-3x faster than full fine-tuning
✅ **Storage** savings of 65-95% compared to multiple full models
✅ **Swapping** adapters takes < 1 second
✅ **Domain specialization** improves accuracy on specific tasks

### Quick Reference

```python
# Training
config = TrainingConfig(
    use_lora=True,
    lora_r=8,
    lora_alpha=16.0,
    save_adapter_only=True,
)
trainer.train(train_data=examples)

# Loading
model = GLiNER2.from_pretrained("base-model")
model.load_adapter("./adapter/final")

# Swapping
model.load_adapter("./other_adapter/final")

# Unloading
model.unload_adapter()

# Checking
print(model.has_adapter)
print(model.adapter_config)
```

### Next Steps

1. **Train your first adapter** with domain-specific data
2. **Evaluate performance** on test set
3. **Experiment with hyperparameters** (rank, alpha, target modules)
4. **Deploy multiple adapters** for different use cases
5. **Monitor and iterate** based on real-world performance

For more information:
- LoRA Paper: https://arxiv.org/abs/2106.09685
- Implementation: `gliner2/training/lora.py`
- Tests: `tests/test_lora_adapters.py`
- Verification Guide: `LORA_VERIFICATION_TESTS.md`
