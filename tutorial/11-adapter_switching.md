# Tutorial 11: LoRA Adapter Switching/Routing

## Quick Start

Switch between domain-specific adapters during inference without reloading the base model.

```python
from gliner2 import GLiNER2

# Load base model once
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# Load legal adapter
model.load_adapter("./legal_adapter")
legal_result = model.extract_entities("Apple sued Google", ["company"])

# Switch to medical adapter
model.load_adapter("./medical_adapter")
medical_result = model.extract_entities("Patient has diabetes", ["disease"])

# Use base model (no adapter)
model.unload_adapter()
base_result = model.extract_entities("Some text", ["entity"])
```

## Basic Usage

### Loading an Adapter

```python
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
model.load_adapter("./path/to/adapter")
```

The adapter path should point to a directory containing:
- `adapter_config.json`
- `adapter_weights.safetensors`

### Checking Adapter Status

```python
# Check if adapter is loaded
if model.has_adapter:
    print("Adapter is loaded")

# Get adapter configuration
config = model.adapter_config
print(f"LoRA rank: {config.lora_r}")
```

### Unloading an Adapter

```python
# Remove adapter, use base model
model.unload_adapter()
```

## Switching Between Adapters

Adapters automatically swap when you call `load_adapter()`:

```python
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# Legal domain
model.load_adapter("./legal_adapter")
result1 = model.extract_entities("Apple Inc. filed suit", ["company"])

# Medical domain (previous adapter auto-unloaded)
model.load_adapter("./medical_adapter")
result2 = model.extract_entities("Patient has diabetes", ["disease"])

# Support domain
model.load_adapter("./support_adapter")
result3 = model.extract_entities("Order #12345 issue", ["order_id"])
```

## Routing by Document Type

Route documents to appropriate adapters:

```python
def extract_with_routing(model, text, doc_type, adapters):
    """Route document to domain-specific adapter."""
    adapter_path = adapters.get(doc_type)

    if adapter_path:
        model.load_adapter(adapter_path)
    else:
        model.unload_adapter()  # Use base model

    # Define entity types per domain
    entity_types = {
        "legal": ["company", "person", "law"],
        "medical": ["disease", "drug", "symptom"],
        "support": ["order_id", "customer", "issue"]
    }

    return model.extract_entities(
        text,
        entity_types.get(doc_type, ["entity"])
    )

# Setup
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
adapters = {
    "legal": "./legal_adapter",
    "medical": "./medical_adapter",
    "support": "./support_adapter"
}

# Use
result = extract_with_routing(
    model,
    "Apple sued Google",
    "legal",
    adapters
)
```

## Batch Processing by Domain

Process multiple documents efficiently:

```python
def process_by_domain(model, documents, adapters):
    """Process documents grouped by domain."""
    results = {}

    for domain, docs in documents.items():
        # Load domain adapter
        model.load_adapter(adapters[domain])

        # Process all documents for this domain
        results[domain] = [
            model.extract_entities(doc, get_entity_types(domain))
            for doc in docs
        ]

    return results

# Example
documents = {
    "legal": ["Apple sued Samsung", "Microsoft acquired LinkedIn"],
    "medical": ["Patient has diabetes", "Prescribed Metformin"]
}

adapters = {
    "legal": "./legal_adapter",
    "medical": "./medical_adapter"
}

results = process_by_domain(model, documents, adapters)
```

## Simple Router Class

```python
class AdapterRouter:
    """Simple adapter router for multi-domain inference."""

    def __init__(self, base_model_name, adapters):
        self.model = GLiNER2.from_pretrained(base_model_name)
        self.adapters = adapters
        self.current_domain = None

    def extract(self, text, domain, entity_types):
        """Extract entities using domain-specific adapter."""
        # Load adapter if domain changed
        if self.current_domain != domain:
            adapter_path = self.adapters.get(domain)
            if adapter_path:
                self.model.load_adapter(adapter_path)
            else:
                self.model.unload_adapter()
            self.current_domain = domain

        return self.model.extract_entities(text, entity_types)

# Usage
router = AdapterRouter(
    "fastino/gliner2-base-v1",
    {
        "legal": "./legal_adapter",
        "medical": "./medical_adapter"
    }
)

result = router.extract("Apple sued Google", "legal", ["company"])
```

## Summary

- **Load adapter**: `model.load_adapter(path)`
- **Unload adapter**: `model.unload_adapter()`
- **Check status**: `model.has_adapter`
- **Get config**: `model.adapter_config`
- **Auto-swap**: Loading a new adapter automatically unloads the previous one

For training adapters, see [Tutorial 10: LoRA Adapters](10-lora_adapters.md).
