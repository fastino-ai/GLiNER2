# GLiNER2 API Extractor

Use GLiNER2 through a cloud API without loading models locally. Perfect for production deployments, low-memory environments, or when you need instant access without GPU setup.

## Table of Contents
- [Getting Started](#getting-started)
- [Basic Usage](#basic-usage)
- [Entity Extraction](#entity-extraction)
- [Text Classification](#text-classification)
- [Structured Extraction](#structured-extraction)
- [Relation Extraction](#relation-extraction)
- [Combined Schemas](#combined-schemas)
- [Batch Processing](#batch-processing)
- [Confidence Scores](#confidence-scores)
- [Error Handling](#error-handling)
- [API vs Local](#api-vs-local)

## Getting Started

### Get Your API Key

1. Visit [gliner.pioneer.ai](https://gliner.pioneer.ai)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Generate a new API key

### Installation

```bash
pip install gliner2
```

### Set Your API Key

**Option 1: Environment Variable (Recommended)**
```bash
export PIONEER_API_KEY="your-api-key-here"
```

**Option 2: Pass Directly**
```python
extractor = GLiNER2.from_api(api_key="your-api-key-here")
```

## Basic Usage

```python
from gliner2 import GLiNER2

# Load from API (uses PIONEER_API_KEY environment variable)
extractor = GLiNER2.from_api()

# Use exactly like the local model!
results = extractor.extract_entities(
    "Apple CEO Tim Cook announced the iPhone 15 in Cupertino.",
    ["company", "person", "product", "location"]
)
print(results)
# Output: {
#     'entities': {
#         'company': ['Apple'],
#         'person': ['Tim Cook'],
#         'product': ['iPhone 15'],
#         'location': ['Cupertino']
#     }
# }
```

## Entity Extraction

### Simple Extraction

```python
extractor = GLiNER2.from_api()

text = "Elon Musk founded SpaceX in 2002 and Tesla in 2003."
results = extractor.extract_entities(
    text,
    ["person", "company", "date"]
)
# Output: {
#     'entities': {
#         'person': ['Elon Musk'],
#         'company': ['SpaceX', 'Tesla'],
#         'date': ['2002', '2003']
#     }
# }
```

### With Confidence Scores

```python
results = extractor.extract_entities(
    "Microsoft acquired LinkedIn for $26.2 billion.",
    ["company", "price"],
    include_confidence=True
)
# Output: {
#     'entities': {
#         'company': [
#             {'text': 'Microsoft', 'confidence': 0.98, 'start': 0, 'end': 1},
#             {'text': 'LinkedIn', 'confidence': 0.97, 'start': 2, 'end': 3}
#         ],
#         'price': [
#             {'text': '$26.2 billion', 'confidence': 0.95, 'start': 5, 'end': 7}
#         ]
#     }
# }
```

### Custom Threshold

```python
# Only return high-confidence extractions
results = extractor.extract_entities(
    text,
    ["person", "company"],
    threshold=0.8  # Minimum 80% confidence
)
```

## Text Classification

### Single-Label Classification

```python
extractor = GLiNER2.from_api()

text = "I absolutely love this product! It exceeded all my expectations."
results = extractor.classify_text(
    text,
    {"sentiment": ["positive", "negative", "neutral"]}
)
# Output: {'sentiment': {'category': 'positive'}}
```

### Multi-Task Classification

```python
text = "Breaking: Major earthquake hits coastal city. Rescue teams deployed."
results = extractor.classify_text(
    text,
    {
        "category": ["politics", "sports", "technology", "disaster", "business"],
        "urgency": ["low", "medium", "high"]
    }
)
# Output: {'category': 'disaster', 'urgency': 'high'}
```

## Structured Extraction

### Contact Information

```python
extractor = GLiNER2.from_api()

text = """
Contact John Smith at john.smith@email.com or call +1-555-123-4567.
He works as a Senior Engineer at TechCorp Inc.
"""

results = extractor.extract_json(
    text,
    {
        "contact": [
            "name::str::Full name of the person",
            "email::str::Email address",
            "phone::str::Phone number",
            "job_title::str::Professional title",
            "company::str::Company name"
        ]
    }
)
# Output: {
#     'contact': [{
#         'name': 'John Smith',
#         'email': 'john.smith@email.com',
#         'phone': '+1-555-123-4567',
#         'job_title': 'Senior Engineer',
#         'company': 'TechCorp Inc.'
#     }]
# }
```

### Product Information

```python
text = "iPhone 15 Pro Max - $1199, 256GB storage, Natural Titanium color"

results = extractor.extract_json(
    text,
    {
        "product": [
            "name::str",
            "price::str",
            "storage::str",
            "color::str"
        ]
    }
)
# Output: {
#     'product': [{
#         'name': 'iPhone 15 Pro Max',
#         'price': '$1199',
#         'storage': '256GB',
#         'color': 'Natural Titanium'
#     }]
# }
```

## Relation Extraction

Extract relationships between entities as directional tuples (source, target).

### Basic Relation Extraction

```python
extractor = GLiNER2.from_api()

text = "John works for Apple Inc. and lives in San Francisco. Apple Inc. is located in Cupertino."
results = extractor.extract_relations(
    text,
    ["works_for", "lives_in", "located_in"]
)
# Output: {
#     'relation_extraction': {
#         'works_for': [('John', 'Apple Inc.')],
#         'lives_in': [('John', 'San Francisco')],
#         'located_in': [('Apple Inc.', 'Cupertino')]
#     }
# }
```

### With Descriptions

```python
text = "Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne, California."

schema = extractor.create_schema().relations({
    "founded": "Founding relationship where person created organization",
    "located_in": "Geographic relationship where entity is in a location"
})

results = extractor.extract(text, schema)
# Output: {
#     'relation_extraction': {
#         'founded': [('Elon Musk', 'SpaceX')],
#         'located_in': [('SpaceX', 'Hawthorne, California')]
#     }
# }
```

### Batch Relation Extraction

```python
texts = [
    "John works for Microsoft and lives in Seattle.",
    "Sarah founded TechStartup in 2020.",
    "Bob reports to Alice at Google."
]

results = extractor.batch_extract_relations(
    texts,
    ["works_for", "founded", "reports_to", "lives_in"]
)
# Returns list of relation extraction results for each text
```

## Combined Schemas

Combine entities, classification, relations, and structured extraction in a single call.

```python
extractor = GLiNER2.from_api()

text = """
Tech Review: The new MacBook Pro M3 is absolutely fantastic! Apple has outdone themselves.
I tested it in San Francisco last week. Tim Cook works for Apple, which is located in Cupertino.
Highly recommended for developers. Rating: 5 out of 5 stars.
"""

schema = (extractor.create_schema()
    .entities(["company", "product", "location", "person"])
    .classification("sentiment", ["positive", "negative", "neutral"])
    .relations(["works_for", "located_in"])
    .structure("review")
        .field("product_name", dtype="str")
        .field("rating", dtype="str")
        .field("recommendation", dtype="str")
)

results = extractor.extract(text, schema)
# Output: {
#     'entities': {
#         'company': ['Apple'],
#         'product': ['MacBook Pro M3'],
#         'location': ['San Francisco', 'Cupertino'],
#         'person': ['Tim Cook']
#     },
#     'sentiment': 'positive',
#     'relation_extraction': {
#         'works_for': [('Tim Cook', 'Apple')],
#         'located_in': [('Apple', 'Cupertino')]
#     },
#     'review': [{
#         'product_name': 'MacBook Pro M3',
#         'rating': '5 out of 5 stars',
#         'recommendation': 'Highly recommended for developers'
#     }]
# }
```

## Batch Processing

Process multiple texts efficiently in a single API call.

```python
extractor = GLiNER2.from_api()

texts = [
    "Google's Sundar Pichai unveiled Gemini AI in Mountain View.",
    "Microsoft CEO Satya Nadella announced Copilot at Build 2023.",
    "Amazon's Andy Jassy revealed new AWS services in Seattle."
]

results = extractor.batch_extract_entities(
    texts,
    ["company", "person", "product", "location"]
)

for i, result in enumerate(results):
    print(f"Text {i+1}: {result}")
```

## Confidence Scores

### Entity Extraction with Confidence

```python
results = extractor.extract_entities(
    "Apple released the iPhone 15 in September 2023.",
    ["company", "product", "date"],
    include_confidence=True
)
# Each entity includes: text, confidence, start position, end position
```

### Raw Results (Advanced)

For full control over the extraction data:

```python
results = extractor.extract_entities(
    "Apple CEO Tim Cook announced new products.",
    ["company", "person"],
    format_results=False,  # Get raw extraction data
    include_confidence=True
)
# Returns tuples: (text, confidence, start_token, end_token)
```

## Error Handling

```python
from gliner2 import GLiNER2, GLiNER2APIError, AuthenticationError, ValidationError

try:
    extractor = GLiNER2.from_api()
    results = extractor.extract_entities(text, entity_types)
    
except AuthenticationError:
    print("Invalid API key. Check your PIONEER_API_KEY.")
    
except ValidationError as e:
    print(f"Invalid request: {e}")
    
except GLiNER2APIError as e:
    print(f"API error: {e}")
```

### Connection Settings

```python
extractor = GLiNER2.from_api(
    api_key="your-key",
    timeout=60.0,      # Request timeout (seconds)
    max_retries=5      # Retry failed requests
)
```

## API vs Local

| Feature | API (`from_api()`) | Local (`from_pretrained()`) |
|---------|-------------------|----------------------------|
| Setup | Just API key | GPU/CPU + model download |
| Memory | ~0 MB | 2-8 GB+ |
| Latency | Network dependent | Faster for single texts |
| Batch | Optimized | Optimized |
| Cost | Per request | Free after setup |
| Offline | ❌ | ✅ |
| RegexValidator | ❌ | ✅ |

### When to Use API

- Production deployments without GPU
- Serverless functions (AWS Lambda, etc.)
- Quick prototyping
- Low-memory environments
- Mobile/edge applications

### When to Use Local

- High-volume processing
- Offline requirements
- Sensitive data (no network transfer)
- Need for RegexValidator
- Cost optimization at scale

## Seamless Switching

The API mirrors the local interface exactly, making switching trivial:

```python
# Development: Use API for quick iteration
extractor = GLiNER2.from_api()

# Production: Switch to local if needed
# extractor = GLiNER2.from_pretrained("your-model")

# Same code works with both!
results = extractor.extract_entities(text, entity_types)
```

## Limitations

The API currently does not support:

1. **RegexValidator** - Use local model for regex-based filtering
2. **Multi-schema batch** - Different schemas per text in batch (works but slower)
3. **Custom models** - API uses the default GLiNER2 model

## Best Practices

1. **Store API key securely** - Use environment variables, not hardcoded strings
2. **Handle errors gracefully** - Network issues can occur
3. **Use batch processing** - More efficient than individual calls
4. **Set appropriate timeouts** - Increase for large texts
5. **Cache results** - Avoid redundant API calls for same content

