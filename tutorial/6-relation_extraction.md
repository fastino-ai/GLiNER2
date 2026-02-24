# GLiNER2 Relation Extraction Tutorial

Learn how to extract relations between entities from text using GLiNER2's relation extraction capabilities.

## Table of Contents
- [Basic Relation Extraction](#basic-relation-extraction)
- [Multiple Relation Types](#multiple-relation-types)
- [Relation Extraction with Descriptions](#relation-extraction-with-descriptions)
- [Custom Thresholds](#custom-thresholds)
- [Batch Processing](#batch-processing)
- [Combining with Other Tasks](#combining-with-other-tasks)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)

## Basic Relation Extraction

### Simple Example

```python
from gliner2 import GLiNER2

# Load model
extractor = GLiNER2.from_pretrained("your-model-name")

# Extract relations
text = "John works for Apple Inc. and lives in San Francisco."
results = extractor.extract_relations(
    text,
    ["works_for", "lives_in"]
)
print(results)
# Output: {
#     'relation_extraction': {
#         'works_for': [('John', 'Apple Inc.')],
#         'lives_in': [('John', 'San Francisco')]
#     }
# }
```

### Using Schema Builder

```python
# Same extraction using schema
schema = extractor.create_schema().relations([
    "works_for", "lives_in"
])
results = extractor.extract(text, schema)
```

### Understanding the Output Format

Relations are returned as tuples `(source, target)` grouped under the `relation_extraction` key. **All requested relation types are included in the output, even if no relations are found** (they appear as empty lists `[]`):

```python
text = "Alice manages the Engineering team. Bob reports to Alice."
results = extractor.extract_relations(
    text,
    ["manages", "reports_to", "founded"]  # Note: "founded" not found in text
)
# Output: {
#     'relation_extraction': {
#         'manages': [('Alice', 'Engineering team')],
#         'reports_to': [('Bob', 'Alice')],
#         'founded': []  # Empty list - relation type requested but not found
#     }
# }
```

This ensures consistent output structure - all requested relation types will always be present in the results, making it easier to process the output programmatically.

## Multiple Relation Types

You can extract multiple relation types in a single call:

```python
text = """
Sarah founded TechCorp in 2020. She is married to Mike,
who works at Google. TechCorp is located in Seattle.
"""

results = extractor.extract_relations(
    text,
    ["founded", "married_to", "works_at", "located_in"]
)
# Output: {
#     'relation_extraction': {
#         'founded': [('Sarah', 'TechCorp')],
#         'married_to': [('Sarah', 'Mike')],
#         'works_at': [('Mike', 'Google')],
#         'located_in': [('TechCorp', 'Seattle')]
#     }
# }
```

### Multiple Instances per Relation Type

GLiNER2 automatically extracts all relation instances found in the text:

```python
text = """
John works for Microsoft. Mary works for Google.
Bob works for Apple. All three live in California.
"""

results = extractor.extract_relations(
    text,
    ["works_for", "lives_in"]
)
# Output: {
#     'relation_extraction': {
#         'works_for': [
#             ('John', 'Microsoft'),
#             ('Mary', 'Google'),
#             ('Bob', 'Apple')
#         ],
#         'lives_in': [
#             ('John', 'California'),
#             ('Mary', 'California'),
#             ('Bob', 'California')
#         ]
#     }
# }
```

## Relation Extraction with Descriptions

Providing descriptions helps improve extraction accuracy by clarifying the relation semantics:

```python
schema = extractor.create_schema().relations({
    "works_for": "Employment relationship where person works at organization",
    "founded": "Founding relationship where person created organization",
    "acquired": "Acquisition relationship where company bought another company",
    "located_in": "Geographic relationship where entity is in a location"
})

text = """
Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne, California.
Tesla acquired SolarCity in 2016. Many engineers work for SpaceX.
"""

results = extractor.extract(text, schema)
```

### Advanced Configuration

```python
schema = extractor.create_schema().relations({
    "works_for": {
        "description": "Employment or professional relationship",
        "threshold": 0.7  # Higher precision for employment relations
    },
    "located_in": {
        "description": "Geographic containment relationship",
        "threshold": 0.6  # Moderate threshold
    },
    "reports_to": {
        "description": "Organizational hierarchy relationship",
        "threshold": 0.8  # Very high precision
    }
})
```

## Custom Thresholds

### Global Threshold

```python
# High-precision relation extraction
results = extractor.extract_relations(
    text,
    ["acquired", "merged_with"],
    threshold=0.8  # High confidence required
)
```

### Per-Relation Thresholds

```python
schema = extractor.create_schema().relations({
    "acquired": {
        "description": "Company acquisition relationship",
        "threshold": 0.9  # Very high precision
    },
    "partnered_with": {
        "description": "Partnership or collaboration relationship",
        "threshold": 0.6  # Moderate threshold
    },
    "competes_with": {
        "description": "Competitive relationship",
        "threshold": 0.5  # Lower threshold for implicit relations
    }
})
```

### With Confidence Scores and Character Positions

You can include confidence scores and character-level start/end positions for relation extractions:

```python
# Extract relations with confidence scores
text = "John works for Apple Inc. and lives in San Francisco."
results = extractor.extract_relations(
    text,
    ["works_for", "lives_in"],
    include_confidence=True
)
print(results)
# Output: {
#     'relation_extraction': {
#         'works_for': [{
#             'head': {'text': 'John', 'confidence': 0.95},
#             'tail': {'text': 'Apple Inc.', 'confidence': 0.92}
#         }],
#         'lives_in': [{
#             'head': {'text': 'John', 'confidence': 0.94},
#             'tail': {'text': 'San Francisco', 'confidence': 0.91}
#         }]
#     }
# }

# Extract with character positions (spans)
results = extractor.extract_relations(
    text,
    ["works_for", "lives_in"],
    include_spans=True
)
print(results)
# Output: {
#     'relation_extraction': {
#         'works_for': [{
#             'head': {'text': 'John', 'start': 0, 'end': 4},
#             'tail': {'text': 'Apple Inc.', 'start': 15, 'end': 25}
#         }],
#         'lives_in': [{
#             'head': {'text': 'John', 'start': 0, 'end': 4},
#             'tail': {'text': 'San Francisco', 'start': 33, 'end': 46}
#         }]
#     }
# }

# Extract with both confidence and spans
results = extractor.extract_relations(
    text,
    ["works_for", "lives_in"],
    include_confidence=True,
    include_spans=True
)
print(results)
# Output: {
#     'relation_extraction': {
#         'works_for': [{
#             'head': {'text': 'John', 'confidence': 0.95, 'start': 0, 'end': 4},
#             'tail': {'text': 'Apple Inc.', 'confidence': 0.92, 'start': 15, 'end': 25}
#         }],
#         'lives_in': [{
#             'head': {'text': 'John', 'confidence': 0.94, 'start': 0, 'end': 4},
#             'tail': {'text': 'San Francisco', 'confidence': 0.91, 'start': 33, 'end': 46}
#         }]
#     }
# }
```

**Note**: When `include_spans` or `include_confidence` is True, relations are returned as dictionaries with `head` and `tail` keys, each containing the extracted text along with optional confidence scores and character positions. When both are False (default), relations are returned as simple tuples `(head, tail)`.

## Batch Processing

Process multiple texts efficiently:

```python
texts = [
    "John works for Microsoft and lives in Seattle.",
    "Sarah founded TechStartup in 2020.",
    "Bob reports to Alice at Google."
]

results = extractor.batch_extract_relations(
    texts,
    ["works_for", "founded", "reports_to", "lives_in"],
    batch_size=8
)
# Output: [
#     {
#         'relation_extraction': {
#             'works_for': [('John', 'Microsoft')],
#             'lives_in': [('John', 'Seattle')],
#             'founded': [],      # Not found in first text
#             'reports_to': []   # Not found in first text
#         }
#     },
#     {
#         'relation_extraction': {
#             'works_for': [],    # Not found in second text
#             'founded': [('Sarah', 'TechStartup')],
#             'reports_to': [],   # Not found in second text
#             'lives_in': []      # Not found in second text
#         }
#     },
#     {
#         'relation_extraction': {
#             'works_for': [('Alice', 'Google')],
#             'reports_to': [('Bob', 'Alice')],
#             'founded': [],      # Not found in third text
#             'lives_in': []      # Not found in third text
#         }
#     }
# ]
```

**Note**: All requested relation types appear in each result, even if empty. This ensures consistent structure across all batch results, making it easier to process programmatically.

## Combining with Other Tasks

Relation extraction can be combined with entity extraction, classification, and structured extraction:

### Relations + Entities

```python
schema = (extractor.create_schema()
    .entities(["person", "organization", "location"])
    .relations(["works_for", "located_in"])
)

text = "Tim Cook works for Apple Inc., which is located in Cupertino, California."
results = extractor.extract(text, schema)
# Output: {
#     'entities': {
#         'person': ['Tim Cook'],
#         'organization': ['Apple Inc.'],
#         'location': ['Cupertino', 'California']
#     },
#     'relation_extraction': {
#         'works_for': [('Tim Cook', 'Apple Inc.')],
#         'located_in': [('Apple Inc.', 'Cupertino')]
#     }
# }
```

### Relations + Classification + Structures

```python
schema = (extractor.create_schema()
    .classification("document_type", ["news", "report", "announcement"])
    .entities(["person", "company"])
    .relations(["works_for", "acquired"])
    .structure("event")
        .field("date", dtype="str")
        .field("description", dtype="str")
)

text = """
BREAKING: Microsoft announced today that it acquired GitHub.
Satya Nadella, CEO of Microsoft, confirmed the deal.
The acquisition was finalized on October 26, 2018.
"""

results = extractor.extract(text, schema)
```

## Real-World Examples

### Organizational Relationships

```python
org_schema = extractor.create_schema().relations({
    "reports_to": "Direct reporting relationship in organizational hierarchy",
    "manages": "Management relationship where person manages team/department",
    "works_for": "Employment relationship",
    "founded": "Founding relationship",
    "acquired": "Company acquisition relationship"
})

text = """
Sundar Pichai is the CEO of Google. He reports to the board of directors.
Google acquired YouTube in 2006. Many engineers work for Google.
"""

results = extractor.extract(text, org_schema)
# Output: {
#     'relation_extraction': {
#         'reports_to': [('Sundar Pichai', 'board of directors')],
#         'works_for': [('engineers', 'Google')],
#         'acquired': [('Google', 'YouTube')]
#     }
# }
```

### Medical Relationships

```python
medical_schema = extractor.create_schema().relations({
    "treats": "Medical treatment relationship between doctor and patient",
    "prescribed_for": "Prescription relationship between medication and condition",
    "causes": "Causal relationship between condition and symptom",
    "located_in": "Anatomical location relationship"
})

text = """
Dr. Smith treats patients with diabetes. Metformin is prescribed for Type 2 Diabetes.
High blood sugar causes frequent urination. The pancreas is located in the abdomen.
"""

results = extractor.extract(text, medical_schema)
```

### Financial Relationships

```python
finance_schema = extractor.create_schema().relations({
    "invested_in": "Investment relationship between investor and company",
    "acquired": "Company acquisition relationship",
    "merged_with": "Merger relationship between companies",
    "owns": "Ownership relationship"
})

text = """
SoftBank invested in Uber in 2018. Microsoft acquired LinkedIn in 2016.
Disney merged with 21st Century Fox. Berkshire Hathaway owns Geico.
"""

results = extractor.extract(text, finance_schema)
```

### Geographic Relationships

```python
geo_schema = extractor.create_schema().relations({
    "located_in": "Geographic containment (city in country, etc.)",
    "borders": "Geographic adjacency relationship",
    "capital_of": "Capital city relationship",
    "flows_through": "River or waterway relationship"
})

text = """
Paris is the capital of France. France borders Germany and Spain.
The Seine flows through Paris. Paris is located in France.
"""

results = extractor.extract(text, geo_schema)
```

### Family Relationships

```python
family_schema = extractor.create_schema().relations({
    "married_to": "Marriage relationship",
    "parent_of": "Parent-child relationship",
    "sibling_of": "Sibling relationship",
    "related_to": "General family relationship"
})

text = """
John is married to Mary. They are parents of two children: Alice and Bob.
Alice and Bob are siblings. Mary is related to her sister Sarah.
"""

results = extractor.extract(text, family_schema)
```

### Academic Relationships

```python
academic_schema = extractor.create_schema().relations({
    "authored": "Publication relationship between author and paper",
    "cited": "Citation relationship between papers",
    "supervised": "Academic supervision relationship",
    "affiliated_with": "Institutional affiliation relationship"
})

text = """
Dr. Johnson authored the paper on machine learning. The paper cited
previous work by Dr. Smith. Dr. Johnson supervises graduate students
at MIT, where she is affiliated with the Computer Science department.
"""

results = extractor.extract(text, academic_schema)
```

## Best Practices

### 1. Use Clear, Specific Relation Names

```python
# Good - Clear and specific
schema.relations(["works_for", "reports_to", "manages"])

# Less ideal - Too generic
schema.relations(["related", "connected", "linked"])
```

### 2. Provide Descriptions for Ambiguous Relations

```python
# Good - Clear descriptions
schema.relations({
    "works_for": "Employment relationship where person works at organization",
    "consulted_for": "Consulting relationship where person provides services to organization"
})

# Less ideal - No context
schema.relations(["works_for", "consulted_for"])
```

### 3. Set Appropriate Thresholds

```python
# High precision for critical relations
schema.relations({
    "acquired": {
        "description": "Company acquisition",
        "threshold": 0.9  # Very high precision
    },
    "partnered_with": {
        "description": "Partnership relationship",
        "threshold": 0.6  # Moderate threshold
    }
})
```

### 4. Combine with Entity Extraction

```python
# Extract both entities and relations for better context
schema = (extractor.create_schema()
    .entities(["person", "organization"])
    .relations(["works_for", "founded"])
)
```

### 5. Use Batch Processing for Multiple Texts

```python
# Efficient batch processing
results = extractor.batch_extract_relations(
    texts,
    relation_types,
    batch_size=8  # Adjust based on your hardware
)
```

### 6. Handle Multiple Instances

```python
# GLiNER2 automatically extracts all instances
text = "John works for Apple. Mary works for Google. Bob works for Microsoft."
results = extractor.extract_relations(text, ["works_for"])
# Returns all three work relationships
```

### 7. Handle Empty Relations

All requested relation types are always included in the output, even if empty:

```python
results = extractor.extract_relations(
    "John works for Microsoft.",
    ["works_for", "founded", "acquired"]
)
# Output: {
#     'relation_extraction': {
#         'works_for': [('John', 'Microsoft')],
#         'founded': [],      # Empty - not found in text
#         'acquired': []      # Empty - not found in text
#     }
# }

# This makes it easy to check for relations programmatically:
for rel_type, rels in results['relation_extraction'].items():
    if rels:  # Non-empty
        print(f"Found {len(rels)} {rel_type} relations")
    else:  # Empty
        print(f"No {rel_type} relations found")
```

### 7. Validate Relation Direction

Relations are directional tuples `(source, target)`:
- `works_for`: (person, organization)
- `located_in`: (entity, location)
- `reports_to`: (subordinate, manager)
- `manages`: (manager, team)

Make sure your relation names match the expected direction.

## Common Use Cases

### Knowledge Graph Construction

```python
# Extract entities and relations for knowledge graph
schema = (extractor.create_schema()
    .entities(["person", "organization", "location", "product"])
    .relations([
        "works_for", "founded", "located_in", "created",
        "acquired", "partnered_with"
    ])
)

# Process documents to build knowledge graph
documents = [...]  # Your documents
all_relations = []
all_entities = []

for doc in documents:
    results = extractor.extract(doc, schema)
    all_relations.append(results.get("relation_extraction", {}))
    all_entities.append(results.get("entities", {}))
```

### Relationship Analysis

```python
# Analyze organizational structures
org_texts = [...]  # Organizational documents
results = extractor.batch_extract_relations(
    org_texts,
    ["reports_to", "manages", "works_for", "collaborates_with"],
    batch_size=8
)

# Analyze relationship patterns
for result in results:
    relations = result.get("relation_extraction", {})
    # Process relations for analysis
```

### Document Understanding

```python
# Comprehensive document understanding
schema = (extractor.create_schema()
    .classification("document_type", ["contract", "report", "email"])
    .entities(["person", "organization", "date", "amount"])
    .relations(["signed_by", "involves", "dated", "worth"])
    .structure("contract_term")
        .field("term", dtype="str")
        .field("value", dtype="str")
)

# Extract all information types in one pass
results = extractor.extract(document_text, schema)
```
