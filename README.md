# GLiNER2: Unified Schema-Based Information Extraction

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/gliner2.svg)](https://badge.fury.io/py/gliner2)
[![Downloads](https://pepy.tech/badge/gliner2)](https://pepy.tech/project/gliner2)

> *Extract entities, classify text, parse structured data, and extract relations—all in one efficient model.*

GLiNER2 unifies **Named Entity Recognition**, **Text Classification**, **Structured Data Extraction**, and **Relation Extraction** into a single 205M parameter model. It provides efficient CPU-based inference without requiring complex pipelines or external API dependencies.

## ✨ Why GLiNER2?

- **🎯 One Model, Four Tasks**: Entities, classification, structured data, and relations in a single forward pass
- **💻 CPU First**: Lightning-fast inference on standard hardware—no GPU required
- **🛡️ Privacy**: 100% local processing, zero external dependencies

## 🚀 Installation & Quick Start

```bash
pip install gliner2
```

```python
from gliner2 import GLiNER2

# Load model once, use everywhere
extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# Extract entities in one line
text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
result = extractor.extract_entities(text, ["company", "person", "product", "location"])

print(result)
# {'entities': {'company': ['Apple'], 'person': ['Tim Cook'], 'product': ['iPhone 15'], 'location': ['Cupertino']}}
```

### 🌐 API Access: GLiNER XL 1B

Our biggest and most powerful model—**GLiNER XL 1B**—is available exclusively via API. No GPU required, no model downloads, just instant access to state-of-the-art extraction. Get your API key at [gliner.pioneer.ai](https://gliner.pioneer.ai).

```python
from gliner2 import GLiNER2

# Access GLiNER XL 1B via API
extractor = GLiNER2.from_api()  # Uses PIONEER_API_KEY env variable

result = extractor.extract_entities(
    "OpenAI CEO Sam Altman announced GPT-5 at their San Francisco headquarters.",
    ["company", "person", "product", "location"]
)
# {'entities': {'company': ['OpenAI'], 'person': ['Sam Altman'], 'product': ['GPT-5'], 'location': ['San Francisco']}}
```

## 📦 Available Models

| Model | Parameters | Description | Use Case                                         |
|-------|------------|-------------|--------------------------------------------------|
| `fastino/gliner2-base-v1` | 205M | base size   | Extraction / classification |
| `fastino/gliner2-large-v1` | 340M | large size  | Extraction / classification                      |

The models are available on [Hugging Face](https://huggingface.co/collections/fastino/gliner2-family).

## 📚 Documentation & Tutorials

Comprehensive guides for all GLiNER2 features:

### Core Features
- **[Text Classification](tutorial/1-classification.md)** - Single and multi-label classification with confidence scores
- **[Entity Extraction](tutorial/2-ner.md)** - Named entity recognition with descriptions and spans
- **[Structured Data Extraction](tutorial/3-json_extraction.md)** - Parse complex JSON structures from text
- **[Combined Schemas](tutorial/4-combined.md)** - Multi-task extraction in a single pass
- **[Regex Validators](tutorial/5-validator.md)** - Filter and validate extracted spans
- **[Relation Extraction](tutorial/6-relation_extraction.md)** - Extract relationships between entities
- **[API Access](tutorial/7-api.md)** - Use GLiNER2 via cloud API

### Training & Customization
- **[Training Data Format](tutorial/8-train_data.md)** - Complete guide to preparing training data
- **[Model Training](tutorial/9-training.md)** - Train custom models for your domain
- **[LoRA Adapters](tutorial/10-lora_adapters.md)** - Parameter-efficient fine-tuning
- **[Adapter Switching](tutorial/11-adapter_switching.md)** - Switch between domain adapters

## 🎯 Core Capabilities

### 1. Entity Extraction
Extract named entities with optional descriptions for precision:

```python
# Basic entity extraction
entities = extractor.extract_entities(
    "Patient received 400mg ibuprofen for severe headache at 2 PM.",
    ["medication", "dosage", "symptom", "time"]
)
# Output: {'entities': {'medication': ['ibuprofen'], 'dosage': ['400mg'], 'symptom': ['severe headache'], 'time': ['2 PM']}}

# Enhanced with descriptions for medical accuracy
entities = extractor.extract_entities(
    "Patient received 400mg ibuprofen for severe headache at 2 PM.",
    {
        "medication": "Names of drugs, medications, or pharmaceutical substances",
        "dosage": "Specific amounts like '400mg', '2 tablets', or '5ml'",
        "symptom": "Medical symptoms, conditions, or patient complaints",
        "time": "Time references like '2 PM', 'morning', or 'after lunch'"
    }
)
# Same output but with higher accuracy due to context descriptions

# With confidence scores
entities = extractor.extract_entities(
    "Apple Inc. CEO Tim Cook announced iPhone 15 in Cupertino.",
    ["company", "person", "product", "location"],
    include_confidence=True
)
# Output: {
#     'entities': {
#         'company': [{'text': 'Apple Inc.', 'confidence': 0.95}],
#         'person': [{'text': 'Tim Cook', 'confidence': 0.92}],
#         'product': [{'text': 'iPhone 15', 'confidence': 0.88}],
#         'location': [{'text': 'Cupertino', 'confidence': 0.90}]
#     }
# }

# With character positions (spans)
entities = extractor.extract_entities(
    "Apple Inc. CEO Tim Cook announced iPhone 15 in Cupertino.",
    ["company", "person", "product"],
    include_spans=True
)
# Output: {
#     'entities': {
#         'company': [{'text': 'Apple Inc.', 'start': 0, 'end': 9}],
#         'person': [{'text': 'Tim Cook', 'start': 15, 'end': 23}],
#         'product': [{'text': 'iPhone 15', 'start': 35, 'end': 44}]
#     }
# }

# With both confidence and spans
entities = extractor.extract_entities(
    "Apple Inc. CEO Tim Cook announced iPhone 15 in Cupertino.",
    ["company", "person", "product"],
    include_confidence=True,
    include_spans=True
)
# Output: {
#     'entities': {
#         'company': [{'text': 'Apple Inc.', 'confidence': 0.95, 'start': 0, 'end': 9}],
#         'person': [{'text': 'Tim Cook', 'confidence': 0.92, 'start': 15, 'end': 23}],
#         'product': [{'text': 'iPhone 15', 'confidence': 0.88, 'start': 35, 'end': 44}]
#     }
# }
```

### 2. Text Classification
Single or multi-label classification with configurable confidence:

```python
# Sentiment analysis
result = extractor.classify_text(
    "This laptop has amazing performance but terrible battery life!",
    {"sentiment": ["positive", "negative", "neutral"]}
)
# Output: {'sentiment': 'negative'}

# Multi-aspect classification
result = extractor.classify_text(
    "Great camera quality, decent performance, but poor battery life.",
    {
        "aspects": {
            "labels": ["camera", "performance", "battery", "display", "price"],
            "multi_label": True,
            "cls_threshold": 0.4
        }
    }
)
# Output: {'aspects': ['camera', 'performance', 'battery']}

# With confidence scores
result = extractor.classify_text(
    "This laptop has amazing performance but terrible battery life!",
    {"sentiment": ["positive", "negative", "neutral"]},
    include_confidence=True
)
# Output: {'sentiment': {'label': 'negative', 'confidence': 0.82}}

# Multi-label with confidence
schema = extractor.create_schema().classification(
    "topics",
    ["technology", "business", "health", "politics", "sports"],
    multi_label=True,
    cls_threshold=0.3
)
text = "Apple announced new health monitoring features in their latest smartwatch, boosting their stock price."
results = extractor.extract(text, schema, include_confidence=True)
# Output: {
#     'topics': [
#         {'label': 'technology', 'confidence': 0.92},
#         {'label': 'business', 'confidence': 0.78},
#         {'label': 'health', 'confidence': 0.65}
#     ]
# }
```

### 3. Structured Data Extraction
Parse complex structured information with field-level control:

```python
# Product information extraction
text = "iPhone 15 Pro Max with 256GB storage, A17 Pro chip, priced at $1199. Available in titanium and black colors."

result = extractor.extract_json(
    text,
    {
        "product": [
            "name::str::Full product name and model",
            "storage::str::Storage capacity like 256GB or 1TB", 
            "processor::str::Chip or processor information",
            "price::str::Product price with currency",
            "colors::list::Available color options"
        ]
    }
)
# Output: {
#     'product': [{
#         'name': 'iPhone 15 Pro Max',
#         'storage': '256GB', 
#         'processor': 'A17 Pro chip',
#         'price': '$1199',
#         'colors': ['titanium', 'black']
#     }]
# }

# Multiple structured entities
text = "Apple Inc. headquarters in Cupertino launched iPhone 15 for $999 and MacBook Air for $1299."

result = extractor.extract_json(
    text,
    {
        "company": [
            "name::str::Company name",
            "location::str::Company headquarters or office location"
        ],
        "products": [
            "name::str::Product name and model",
            "price::str::Product retail price"
        ]
    }
)
# Output: {
#     'company': [{'name': 'Apple Inc.', 'location': 'Cupertino'}],
#     'products': [
#         {'name': 'iPhone 15', 'price': '$999'},
#         {'name': 'MacBook Air', 'price': '$1299'}
#     ]
# }

# With confidence scores
result = extractor.extract_json(
    "The MacBook Pro costs $1999 and features M3 chip, 16GB RAM, and 512GB storage.",
    {
        "product": [
            "name::str",
            "price",
            "features"
        ]
    },
    include_confidence=True
)
# Output: {
#     'product': [{
#         'name': {'text': 'MacBook Pro', 'confidence': 0.95},
#         'price': [{'text': '$1999', 'confidence': 0.92}],
#         'features': [
#             {'text': 'M3 chip', 'confidence': 0.88},
#             {'text': '16GB RAM', 'confidence': 0.90},
#             {'text': '512GB storage', 'confidence': 0.87}
#         ]
#     }]
# }

# With character positions (spans)
result = extractor.extract_json(
    "The MacBook Pro costs $1999 and features M3 chip.",
    {
        "product": [
            "name::str",
            "price"
        ]
    },
    include_spans=True
)
# Output: {
#     'product': [{
#         'name': {'text': 'MacBook Pro', 'start': 4, 'end': 15},
#         'price': [{'text': '$1999', 'start': 22, 'end': 27}]
#     }]
# }

# With both confidence and spans
result = extractor.extract_json(
    "The MacBook Pro costs $1999 and features M3 chip, 16GB RAM, and 512GB storage.",
    {
        "product": [
            "name::str",
            "price",
            "features"
        ]
    },
    include_confidence=True,
    include_spans=True
)
# Output: {
#     'product': [{
#         'name': {'text': 'MacBook Pro', 'confidence': 0.95, 'start': 4, 'end': 15},
#         'price': [{'text': '$1999', 'confidence': 0.92, 'start': 22, 'end': 27}],
#         'features': [
#             {'text': 'M3 chip', 'confidence': 0.88, 'start': 32, 'end': 39},
#             {'text': '16GB RAM', 'confidence': 0.90, 'start': 41, 'end': 49},
#             {'text': '512GB storage', 'confidence': 0.87, 'start': 55, 'end': 68}
#         ]
#     }]
# }
```

### 4. Relation Extraction
Extract relationships between entities as directional tuples:

```python
# Basic relation extraction
text = "John works for Apple Inc. and lives in San Francisco. Apple Inc. is located in Cupertino."

result = extractor.extract_relations(
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

# With descriptions for better accuracy
schema = extractor.create_schema().relations({
    "works_for": "Employment relationship where person works at organization",
    "founded": "Founding relationship where person created organization",
    "acquired": "Acquisition relationship where company bought another company",
    "located_in": "Geographic relationship where entity is in a location"
})

text = "Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne, California."
results = extractor.extract(text, schema)
# Output: {
#     'relation_extraction': {
#         'founded': [('Elon Musk', 'SpaceX')],
#         'located_in': [('SpaceX', 'Hawthorne, California')]
#     }
# }

# With confidence scores
results = extractor.extract_relations(
    "John works for Apple Inc. and lives in San Francisco.",
    ["works_for", "lives_in"],
    include_confidence=True
)
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

# With character positions (spans)
results = extractor.extract_relations(
    "John works for Apple Inc. and lives in San Francisco.",
    ["works_for", "lives_in"],
    include_spans=True
)
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

# With both confidence and spans
results = extractor.extract_relations(
    "John works for Apple Inc. and lives in San Francisco.",
    ["works_for", "lives_in"],
    include_confidence=True,
    include_spans=True
)
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

### 5. Multi-Task Schema Composition
Combine all extraction types when you need comprehensive analysis:

```python
# Use create_schema() for multi-task scenarios
schema = (extractor.create_schema()
    # Extract key entities
    .entities({
        "person": "Names of people, executives, or individuals",
        "company": "Organization, corporation, or business names", 
        "product": "Products, services, or offerings mentioned"
    })
    
    # Classify the content
    .classification("sentiment", ["positive", "negative", "neutral"])
    .classification("category", ["technology", "business", "finance", "healthcare"])
    
    # Extract relationships
    .relations(["works_for", "founded", "located_in"])
    
    # Extract structured product details
    .structure("product_info")
        .field("name", dtype="str")
        .field("price", dtype="str")
        .field("features", dtype="list")
        .field("availability", dtype="str", choices=["in_stock", "pre_order", "sold_out"])
)

# Comprehensive extraction in one pass
text = "Apple CEO Tim Cook unveiled the revolutionary iPhone 15 Pro for $999. The device features an A17 Pro chip and titanium design. Tim Cook works for Apple, which is located in Cupertino."

results = extractor.extract(text, schema)
# Output: {
#     'entities': {
#         'person': ['Tim Cook'], 
#         'company': ['Apple'], 
#         'product': ['iPhone 15 Pro']
#     },
#     'sentiment': 'positive',
#     'category': 'technology',
#     'relation_extraction': {
#         'works_for': [('Tim Cook', 'Apple')],
#         'located_in': [('Apple', 'Cupertino')]
#     },
#     'product_info': [{
#         'name': 'iPhone 15 Pro',
#         'price': '$999',
#         'features': ['A17 Pro chip', 'titanium design'],
#         'availability': 'in_stock'
#     }]
# }
```

## 🏭 Example Usage Scenarios

### Financial Document Processing

```python
financial_text = """
Transaction Report: Goldman Sachs processed a $2.5M equity trade for Tesla Inc. 
on March 15, 2024. Commission: $1,250. Status: Completed.
"""

# Extract structured financial data
result = extractor.extract_json(
    financial_text,
    {
        "transaction": [
            "broker::str::Financial institution or brokerage firm",
            "amount::str::Transaction amount with currency",
            "security::str::Stock, bond, or financial instrument",
            "date::str::Transaction date",
            "commission::str::Fees or commission charged", 
            "status::str::Transaction status",
            "type::[equity|bond|option|future|forex]::str::Type of financial instrument"
        ]
    }
)
# Output: {
#     'transaction': [{
#         'broker': 'Goldman Sachs',
#         'amount': '$2.5M', 
#         'security': 'Tesla Inc.',
#         'date': 'March 15, 2024',
#         'commission': '$1,250',
#         'status': 'Completed',
#         'type': 'equity'
#     }]
# }
```

### Healthcare Information Extraction

```python
medical_record = """
Patient: Sarah Johnson, 34, presented with acute chest pain and shortness of breath.
Prescribed: Lisinopril 10mg daily, Metoprolol 25mg twice daily.
Follow-up scheduled for next Tuesday.
"""

result = extractor.extract_json(
    medical_record,
    {
        "patient_info": [
            "name::str::Patient full name",
            "age::str::Patient age",
            "symptoms::list::Reported symptoms or complaints"
        ],
        "prescriptions": [
            "medication::str::Drug or medication name",
            "dosage::str::Dosage amount and frequency",
            "frequency::str::How often to take the medication"
        ]
    }
)
# Output: {
#     'patient_info': [{
#         'name': 'Sarah Johnson',
#         'age': '34',
#         'symptoms': ['acute chest pain', 'shortness of breath']
#     }],
#     'prescriptions': [
#         {'medication': 'Lisinopril', 'dosage': '10mg', 'frequency': 'daily'},
#         {'medication': 'Metoprolol', 'dosage': '25mg', 'frequency': 'twice daily'}
#     ]
# }
```

### Legal Contract Analysis

```python
contract_text = """
Service Agreement between TechCorp LLC and DataSystems Inc., effective January 1, 2024.
Monthly fee: $15,000. Contract term: 24 months with automatic renewal.
Termination clause: 30-day written notice required.
"""

# Multi-task extraction for comprehensive analysis
schema = (extractor.create_schema()
    .entities(["company", "date", "duration", "fee"])
    .classification("contract_type", ["service", "employment", "nda", "partnership"])
    .relations(["signed_by", "involves", "dated"])
    .structure("contract_terms")
        .field("parties", dtype="list")
        .field("effective_date", dtype="str")
        .field("monthly_fee", dtype="str")
        .field("term_length", dtype="str")
        .field("renewal", dtype="str", choices=["automatic", "manual", "none"])
        .field("termination_notice", dtype="str")
)

results = extractor.extract(contract_text, schema)
# Output: {
#     'entities': {
#         'company': ['TechCorp LLC', 'DataSystems Inc.'],
#         'date': ['January 1, 2024'],
#         'duration': ['24 months'],
#         'fee': ['$15,000']
#     },
#     'contract_type': 'service',
#     'relation_extraction': {
#         'involves': [('TechCorp LLC', 'DataSystems Inc.')],
#         'dated': [('Service Agreement', 'January 1, 2024')]
#     },
#     'contract_terms': [{
#         'parties': ['TechCorp LLC', 'DataSystems Inc.'],
#         'effective_date': 'January 1, 2024',
#         'monthly_fee': '$15,000',
#         'term_length': '24 months', 
#         'renewal': 'automatic',
#         'termination_notice': '30-day written notice'
#     }]
# }
```

### Knowledge Graph Construction

```python
# Extract entities and relations for knowledge graph building
text = """
Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne, California.
SpaceX acquired Swarm Technologies in 2021. Many engineers work for SpaceX.
"""

schema = (extractor.create_schema()
    .entities(["person", "organization", "location", "date"])
    .relations({
        "founded": "Founding relationship where person created organization",
        "acquired": "Acquisition relationship where company bought another company",
        "located_in": "Geographic relationship where entity is in a location",
        "works_for": "Employment relationship where person works at organization"
    })
)

results = extractor.extract(text, schema)
# Output: {
#     'entities': {
#         'person': ['Elon Musk', 'engineers'],
#         'organization': ['SpaceX', 'Swarm Technologies'],
#         'location': ['Hawthorne, California'],
#         'date': ['2002', '2021']
#     },
#     'relation_extraction': {
#         'founded': [('Elon Musk', 'SpaceX')],
#         'acquired': [('SpaceX', 'Swarm Technologies')],
#         'located_in': [('SpaceX', 'Hawthorne, California')],
#         'works_for': [('engineers', 'SpaceX')]
#     }
# }
```

## ⚙️ Advanced Configuration

### Custom Confidence Thresholds

```python
# High-precision extraction for critical fields
result = extractor.extract_json(
    text,
    {
        "financial_data": [
            "account_number::str::Bank account number",  # default threshold
            "amount::str::Transaction amount",           # default threshold  
            "routing_number::str::Bank routing number"   # default threshold
        ]
    },
    threshold=0.9  # High confidence for all fields
)

# Per-field thresholds using schema builder (for multi-task scenarios)
schema = (extractor.create_schema()
    .structure("sensitive_data")
        .field("ssn", dtype="str", threshold=0.95)         # Highest precision
        .field("email", dtype="str", threshold=0.8)        # Medium precision  
        .field("phone", dtype="str", threshold=0.7)        # Lower precision
)
```

### Field Types and Constraints

```python
# Structured extraction with choices and types
result = extractor.extract_json(
    "Premium subscription at $99/month with mobile and web access.",
    {
        "subscription": [
            "tier::[basic|premium|enterprise]::str::Subscription level",
            "price::str::Monthly or annual cost",
            "billing::[monthly|annual]::str::Billing frequency", 
            "features::[mobile|web|api|analytics]::list::Included features"
        ]
    }
)
# Output: {
#     'subscription': [{
#         'tier': 'premium',
#         'price': '$99/month', 
#         'billing': 'monthly',
#         'features': ['mobile', 'web']
#     }]
# }
```

## 🔍 Regex Validators

Filter extracted spans to ensure they match expected patterns, improving extraction quality and reducing false positives.

```python
from gliner2 import GLiNER2, RegexValidator

extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# Email validation
email_validator = RegexValidator(r"^[\w\.-]+@[\w\.-]+\.\w+$")
schema = (extractor.create_schema()
    .structure("contact")
        .field("email", dtype="str", validators=[email_validator])
)

text = "Contact: john@company.com, not-an-email, jane@domain.org"
results = extractor.extract(text, schema)
# Output: {'contact': [{'email': 'john@company.com'}]}  # Only valid emails

# Phone number validation (US format)
phone_validator = RegexValidator(r"\(\d{3}\)\s\d{3}-\d{4}", mode="partial")
schema = (extractor.create_schema()
    .structure("contact")
        .field("phone", dtype="str", validators=[phone_validator])
)

text = "Call (555) 123-4567 or 5551234567"
results = extractor.extract(text, schema)
# Output: {'contact': [{'phone': '(555) 123-4567'}]}  # Second number filtered out

# URL validation
url_validator = RegexValidator(r"^https?://", mode="partial")
schema = (extractor.create_schema()
    .structure("links")
        .field("url", dtype="list", validators=[url_validator])
)

text = "Visit https://example.com or www.site.com"
results = extractor.extract(text, schema)
# Output: {'links': [{'url': ['https://example.com']}]}  # www.site.com filtered out

# Exclude test data
import re
no_test_validator = RegexValidator(r"^(test|demo|sample)", exclude=True, flags=re.IGNORECASE)
schema = (extractor.create_schema()
    .structure("products")
        .field("name", dtype="list", validators=[no_test_validator])
)

text = "Products: iPhone, Test Phone, Samsung Galaxy"
results = extractor.extract(text, schema)
# Output: {'products': [{'name': ['iPhone', 'Samsung Galaxy']}]}  # Test Phone excluded

# Multiple validators (all must pass)
username_validators = [
    RegexValidator(r"^[a-zA-Z0-9_]+$"),  # Alphanumeric + underscore
    RegexValidator(r"^.{3,20}$"),        # 3-20 characters
    RegexValidator(r"^(?!admin)", exclude=True, flags=re.IGNORECASE)  # No "admin"
]

schema = (extractor.create_schema()
    .structure("user")
        .field("username", dtype="str", validators=username_validators)
)

text = "Users: ab, john_doe, user@domain, admin, valid_user123"
results = extractor.extract(text, schema)
# Output: {'user': [{'username': 'john_doe'}]}  # Only valid usernames
```

## 📦 Batch Processing

Process multiple texts efficiently in a single call:

```python
# Batch entity extraction
texts = [
    "Google's Sundar Pichai unveiled Gemini AI in Mountain View.",
    "Microsoft CEO Satya Nadella announced Copilot at Build 2023.",
    "Amazon's Andy Jassy revealed new AWS services in Seattle."
]

results = extractor.batch_extract_entities(
    texts,
    ["company", "person", "product", "location"],
    batch_size=8
)
# Returns list of results, one per input text

# Batch relation extraction
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
# Returns list of relation extraction results for each text
# All requested relation types appear in each result, even if empty

# Batch with confidence and spans
results = extractor.batch_extract_entities(
    texts,
    ["company", "person"],
    include_confidence=True,
    include_spans=True,
    batch_size=8
)
```

## 🎓 Training Custom Models

Train GLiNER2 on your own data to specialize for your domain or use case.

### Quick Start Training

```python
from gliner2 import GLiNER2
from gliner2.training.data import InputExample
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# 1. Prepare training data
examples = [
    InputExample(
        text="John works at Google in California.",
        entities={"person": ["John"], "company": ["Google"], "location": ["California"]}
    ),
    InputExample(
        text="Apple released iPhone 15.",
        entities={"company": ["Apple"], "product": ["iPhone 15"]}
    ),
    # Add more examples...
]

# 2. Configure training
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

### Training Data Format (JSONL)

GLiNER2 uses JSONL format where each line contains an `input` and `output` field:

```jsonl
{"input": "Tim Cook is the CEO of Apple Inc., based in Cupertino, California.", "output": {"entities": {"person": ["Tim Cook"], "company": ["Apple Inc."], "location": ["Cupertino", "California"]}, "entity_descriptions": {"person": "Full name of a person", "company": "Business organization name", "location": "Geographic location or place"}}}
{"input": "OpenAI released GPT-4 in March 2023.", "output": {"entities": {"company": ["OpenAI"], "model": ["GPT-4"], "date": ["March 2023"]}}}
```

**Classification Example:**
```jsonl
{"input": "This movie is absolutely fantastic! I loved every minute of it.", "output": {"classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": ["positive"]}]}}
{"input": "The service was terrible and the food was cold.", "output": {"classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": ["negative"]}]}}
```

**Structured Extraction Example:**
```jsonl
{"input": "iPhone 15 Pro Max with 256GB storage, priced at $1199.", "output": {"json_structures": [{"product": {"name": "iPhone 15 Pro Max", "storage": "256GB", "price": "$1199"}}]}}
```

**Relation Extraction Example:**
```jsonl
{"input": "John works for Apple Inc. and lives in San Francisco.", "output": {"relations": [{"works_for": {"head": "John", "tail": "Apple Inc."}}, {"lives_in": {"head": "John", "tail": "San Francisco"}}]}}
```

### Training from JSONL File

```python
from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# Load model and train from JSONL file
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(output_dir="./output", num_epochs=10)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data="train.jsonl")  # Path to your JSONL file
```

### LoRA Training (Parameter-Efficient Fine-Tuning)

Train lightweight adapters for domain-specific tasks:

```python
from gliner2 import GLiNER2
from gliner2.training.data import InputExample
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# Prepare domain-specific data
legal_examples = [
    InputExample(
        text="Apple Inc. filed a lawsuit against Samsung Electronics.",
        entities={"company": ["Apple Inc.", "Samsung Electronics"]}
    ),
    # Add more examples...
]

# Configure LoRA training
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(
    output_dir="./legal_adapter",
    num_epochs=10,
    batch_size=8,
    encoder_lr=1e-5,
    task_lr=5e-4,
    
    # LoRA settings
    use_lora=True,                    # Enable LoRA
    lora_r=8,                         # Rank (4, 8, 16, 32)
    lora_alpha=16.0,                  # Scaling factor (usually 2*r)
    lora_dropout=0.0,                 # Dropout for LoRA layers
    save_adapter_only=True            # Save only adapter (~5MB vs ~450MB)
)

# Train adapter
trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=legal_examples)

# Use the adapter
model.load_adapter("./legal_adapter/final")
results = model.extract_entities(legal_text, ["company", "law"])
```

**Benefits of LoRA:**
- **Smaller size**: Adapters are ~2-10 MB vs ~450 MB for full models
- **Faster training**: 2-3x faster than full fine-tuning
- **Easy switching**: Swap adapters in milliseconds for different domains

### Complete Training Example

```python
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# Prepare training data
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
    # Add more examples...
]

# Create and validate dataset
train_dataset = TrainingDataset(train_examples)
train_dataset.validate(strict=True, raise_on_error=True)
train_dataset.print_stats()

# Split into train/validation
train_data, val_data, _ = train_dataset.split(
    train_ratio=0.8,
    val_ratio=0.2,
    test_ratio=0.0,
    shuffle=True,
    seed=42
)

# Configure training
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
    eval_strategy="epoch",
    save_best=True,
    early_stopping=True,
    early_stopping_patience=3
)

# Train
trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=train_data, val_data=val_data)

# Load best model
model = GLiNER2.from_pretrained("./ner_model/best")
```

For more details, see the [Training Tutorial](tutorial/9-training.md) and [Data Format Guide](tutorial/8-train_data.md).

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use GLiNER2 in your research, please cite:

```bibtex
@inproceedings{zaratiana-etal-2025-gliner2,
    title = "{GL}i{NER}2: Schema-Driven Multi-Task Learning for Structured Information Extraction",
    author = "Zaratiana, Urchade  and
      Pasternak, Gil  and
      Boyd, Oliver  and
      Hurn-Maloney, George  and
      Lewis, Ash",
    editor = {Habernal, Ivan  and
      Schulam, Peter  and
      Tiedemann, J{\"o}rg},
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-demos.10/",
    pages = "130--140",
    ISBN = "979-8-89176-334-0",
    abstract = "Information extraction (IE) is fundamental to numerous NLP applications, yet existing solutions often require specialized models for different tasks or rely on computationally expensive large language models. We present GLiNER2, a unified framework that enhances the original GLiNER architecture to support named entity recognition, text classification, and hierarchical structured data extraction within a single efficient model. Built on a fine-tuned encoder architecture, GLiNER2 maintains CPU efficiency and compact size while introducing multi-task composition through an intuitive schema-based interface. Our experiments demonstrate competitive performance across diverse IE tasks with substantial improvements in deployment accessibility compared to LLM-based alternatives. We release GLiNER2 as an open-source library available through pip, complete with pre-trained models and comprehensive documentation."
}
```

## 🙏 Acknowledgments

Built upon the original [GLiNER](https://github.com/urchade/GLiNER) architecture by the team at [Fastino AI](https://fastino.ai).

---

<div align="center">
    <strong>Ready to extract insights from your data?</strong><br>
    <code>pip install gliner2</code>
</div>
