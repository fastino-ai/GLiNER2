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

The models are available on [Hugging Face](https://huggingface.co/collections/fastino/gliner2).

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