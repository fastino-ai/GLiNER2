# GLiNER2 Entity Extraction Tutorial

Learn how to extract named entities from text using GLiNER2's flexible entity recognition capabilities.

## Table of Contents
- [Basic Entity Extraction](#basic-entity-extraction)
- [Entity Extraction with Descriptions](#entity-extraction-with-descriptions)
- [Single vs Multiple Entities](#single-vs-multiple-entities)
- [Custom Thresholds](#custom-thresholds)
- [Advanced Configuration](#advanced-configuration)
- [Domain-Specific Entities](#domain-specific-entities)
- [Best Practices](#best-practices)

## Basic Entity Extraction

### Simple Example

```python
from gliner2 import GLiNER2

# Load model
extractor = GLiNER2.from_pretrained("your-model-name")

# Extract common entities
text = "Apple Inc. CEO Tim Cook announced the new iPhone 15 in Cupertino, California on September 12, 2023."
results = extractor.extract_entities(
    text,
    ["company", "person", "product", "location", "date"]
)
print(results)
# Output: {
#     'entities': {
#         'company': ['Apple Inc.'],
#         'person': ['Tim Cook'],
#         'product': ['iPhone 15'],
#         'location': ['Cupertino', 'California'],
#         'date': ['September 12, 2023']
#     }
# }
```

### Using Schema Builder

```python
# Same extraction using schema
schema = extractor.create_schema().entities([
    "company", "person", "product", "location", "date"
])
results = extractor.extract(text, schema)
```

## Entity Extraction with Descriptions

Descriptions significantly improve extraction accuracy by providing context.

```python
# Medical entity extraction
schema = extractor.create_schema().entities({
    "drug": "Pharmaceutical drugs, medications, or treatment names",
    "disease": "Medical conditions, illnesses, or disorders",
    "symptom": "Clinical symptoms or patient-reported symptoms",
    "dosage": "Medication amounts like '50mg' or '2 tablets daily'",
    "organ": "Body parts or organs mentioned in medical context"
})

medical_text = """
Patient was prescribed Metformin 500mg twice daily for Type 2 Diabetes.
She reported fatigue and occasional dizziness. Liver function tests ordered.
"""

results = extractor.extract(medical_text, schema)
print(results)
# Output: {
#     'entities': {
#         'drug': ['Metformin'],
#         'disease': ['Type 2 Diabetes'],
#         'symptom': ['fatigue', 'dizziness'],
#         'dosage': ['500mg twice daily'],
#         'organ': ['Liver']
#     }
# }
```

## Single vs Multiple Entities

Control whether to extract one or multiple entities per type.

### Multiple Entities (Default)

```python
# Default behavior - extracts all matching entities
schema = extractor.create_schema().entities(
    ["person", "organization"],
    dtype="list"  # Default
)

text = "Bill Gates and Steve Jobs founded Microsoft and Apple respectively."
results = extractor.extract(text, schema)
# Output: {
#     'entities': {
#         'person': ['Bill Gates', 'Steve Jobs'],
#         'organization': ['Microsoft', 'Apple']
#     }
# }
```

### Single Entity per Type

```python
# Extract only the best match per entity type
schema = extractor.create_schema().entities(
    ["company", "ceo"],
    dtype="str"  # Single entity mode
)

text = "Apple CEO Tim Cook met with Microsoft CEO Satya Nadella."
results = extractor.extract(text, schema)
# Output: {
#     'entities': {
#         'company': 'Apple',  # Just one, despite multiple in text
#         'ceo': 'Tim Cook'    # Just one
#     }
# }
```

## Custom Thresholds

Set confidence thresholds for precise control.

### Global Threshold

```python
# High-precision extraction
results = extractor.extract_entities(
    text,
    ["email", "phone", "address"],
    threshold=0.8  # High confidence required
)
```

### With Confidence Scores and Character Positions

You can include confidence scores and character-level start/end positions using `include_confidence` and `include_spans` parameters:

```python
# Extract entities with confidence scores
text = "Apple Inc. CEO Tim Cook announced iPhone 15 in Cupertino."
results = extractor.extract_entities(
    text,
    ["company", "person", "product"],
    include_confidence=True
)
print(results)
# Output: {
#     'entities': {
#         'company': [
#             {'text': 'Apple Inc.', 'confidence': 0.95},
#             {'text': 'Tim Cook', 'confidence': 0.92}
#         ],
#         'product': [
#             {'text': 'iPhone 15', 'confidence': 0.88}
#         ]
#     }
# }

# Extract with character positions (spans)
results = extractor.extract_entities(
    text,
    ["company", "person"],
    include_spans=True
)
print(results)
# Output: {
#     'entities': {
#         'company': [
#             {'text': 'Apple Inc.', 'start': 0, 'end': 9}
#         ],
#         'person': [
#             {'text': 'Tim Cook', 'start': 15, 'end': 23}
#         ]
#     }
# }

# Extract with both confidence and spans
results = extractor.extract_entities(
    text,
    ["company", "product"],
    include_confidence=True,
    include_spans=True
)
print(results)
# Output: {
#     'entities': {
#         'company': [
#             {'text': 'Apple Inc.', 'confidence': 0.95, 'start': 0, 'end': 9}
#         ],
#         'product': [
#             {'text': 'confidence': 0.88, 'start': 15, 'end': 24}
#         ]
#     }
# }
```

**Note**: When `include_spans` is True, the output format changes:
- **Default** (both False): Returns simple text strings: `['Apple Inc.', 'Tim Cook']`
- **include_confidence=True**: Returns dicts with `{'text': '...', 'confidence': 0.95}`
- **include_spans=True**: Returns dicts with `{'text': '...', 'start': 0, 'end': 9}
- **Both True**: Returns dicts with `{'text': '...', 'confidence': 0.95, 'start': 0, 'end': 9}

### Per-Entity Thresholds

```python
# Different thresholds for different entities
schema = extractor.create_schema().entities({
    "email": {
        "description": "Email addresses",
        "dtype": "list",
        "threshold": 0.9  # Very high precision for emails
    },
    "phone": {
        "description": "Phone numbers including mobile and landline",
        "dtype": "list",
        "threshold": 0.7  # Moderate threshold
    },
    "name": {
        "description": "Person names",
        "dtype": "list",
        "threshold": 0.5  # Lower threshold for names
    }
})

contact_text = "Contact John Doe at john.doe@email.com or call 555-1234."
results = extractor.extract(contact_text, schema, threshold=0.6)  # Default threshold
```

## Advanced Configuration

### Mixed Configuration

```python
# Combine different entity configurations
schema = extractor.create_schema()

# Add simple entities
schema.entities(["date", "time", "currency"])

# Add entities with descriptions
schema.entities({
    "technical_term": "Technical jargon or specialized terminology",
    "metric": "Measurements, KPIs, or quantitative values"
})

# Add entities with full configuration
schema.entities({
    "competitor": {
        "description": "Competing companies or products",
        "dtype": "list",
        "threshold": 0.7
    },
    "revenue": {
        "description": "Revenue figures or financial amounts",
        "dtype": "str",  # Only extract one
        "threshold": 0.8
    }
})
```

### Incremental Entity Addition

```python
# Build schema incrementally
schema = extractor.create_schema()

# Add entities in stages
schema.entities(["person", "location"])  # Basic entities
schema.entities({"company": "Company or organization names"})  # With description
schema.entities({  # With full config
    "financial_term": {
        "description": "Financial instruments, metrics, or terminology",
        "threshold": 0.75
    }
})
```

## Domain-Specific Entities

### Legal Entities

```python
legal_schema = extractor.create_schema().entities({
    "party": "Parties involved in legal proceedings (plaintiff, defendant, etc.)",
    "law_firm": "Law firm or legal practice names",
    "court": "Court names or judicial bodies",
    "statute": "Legal statutes, laws, or regulations cited",
    "case": "Legal case names or citations",
    "judge": "Names of judges or magistrates",
    "legal_term": "Legal terminology or concepts"
})

legal_text = """
In the case of Smith v. Jones, Judge Sarah Williams of the Superior Court
ruled that the defendant violated Section 15.2 of the Consumer Protection Act.
The plaintiff was represented by Miller & Associates.
"""
results = extractor.extract(legal_text, legal_schema)
```

### Financial Entities

```python
finance_schema = extractor.create_schema().entities({
    "ticker": "Stock ticker symbols (e.g., AAPL, GOOGL)",
    "financial_metric": "Financial metrics like P/E ratio, market cap",
    "currency_amount": "Monetary values with currency symbols",
    "percentage": "Percentage values (e.g., 5.2%, -3%)",
    "financial_org": "Banks, investment firms, financial institutions",
    "market_index": "Stock market indices (S&P 500, NASDAQ, etc.)"
})

finance_text = """
AAPL rose 3.5% to $185.50 after beating earnings expectations.
The company's P/E ratio of 28.5 attracted Goldman Sachs analysts.
The NASDAQ composite gained 1.2% for the day.
"""
results = extractor.extract(finance_text, finance_schema)
```

### Scientific Entities

```python
science_schema = extractor.create_schema().entities({
    "chemical": "Chemical compounds or elements",
    "organism": "Biological organisms, species names",
    "gene": "Gene names or identifiers",
    "measurement": "Scientific measurements with units",
    "research_method": "Research techniques or methodologies",
    "institution": "Universities or research institutions"
})

science_text = """
Researchers at MIT discovered that the BRCA1 gene mutation increases
cancer risk by 70%. The study used CRISPR-Cas9 to modify DNA sequences
in Mus musculus specimens, measuring tumor growth in millimeters.
"""
results = extractor.extract(science_text, science_schema)
```

## Best Practices

### 1. Use Descriptive Entity Names

```python
# Good - Clear, specific entity types
schema.entities(["drug_name", "medical_device", "procedure_name"])

# Less ideal - Too generic
schema.entities(["thing", "item", "stuff"])
```

### 2. Provide Context with Descriptions

```python
# Good - Clear descriptions
schema.entities({
    "acquisition_company": "Company that is acquiring another company",
    "target_company": "Company being acquired",
    "acquisition_price": "Purchase price or valuation of acquisition"
})

# Less ideal - No context
schema.entities(["company1", "company2", "price"])
```
