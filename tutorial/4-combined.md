# GLiNER2 Combining Schemas Tutorial

## Table of Contents
- [Why Combine Schemas](#why-combine-schemas)
- [Basic Combinations](#basic-combinations)
- [Advanced Multi-Task Schemas](#advanced-multi-task-schemas)
- [Real-World Applications](#real-world-applications)

## Why Combine Schemas

Combining schemas allows you to:
- Extract multiple types of information in one pass
- Maintain context between different extraction tasks
- Improve efficiency by avoiding multiple model calls
- Build comprehensive information extraction pipelines

## Basic Combinations

### Entities + Classification

```python
from gliner2 import GLiNER2

extractor = GLiNER2.from_pretrained("your-model-name")

# Sentiment analysis with entity extraction
schema = (extractor.create_schema()
    .entities(["person", "product", "company"])
    .classification("sentiment", ["positive", "negative", "neutral"])
    .classification("category", ["review", "news", "opinion"])
)

text = "Tim Cook announced that Apple's new iPhone is exceeding sales expectations."
results = extractor.extract(text, schema)
# Output: {
#     'entities': {
#         'person': ['Tim Cook'],
#         'product': ['iPhone'],
#         'company': ['Apple']
#     },
#     'sentiment': 'positive',
#     'category': 'news'
# }
```

### Entities + Structures

```python
schema = (extractor.create_schema()
    .entities({
        "person": "Names of people mentioned",
        "date": "Dates and time references"
    })
    .structure("appointment")
        .field("patient", dtype="str")
        .field("doctor", dtype="str")
        .field("date")
        .field("time")
        .field("type", dtype="str", choices=["checkup", "followup", "consultation"])
)

text = """
Dr. Sarah Johnson confirmed the appointment with John Smith for
March 15th at 2:30 PM. This will be a follow-up consultation
regarding his previous visit on February 1st.
"""
results = extractor.extract(text, schema)
```

### Classification + Structures

```python
schema = (extractor.create_schema()
    .classification("email_type",
        ["order_confirmation", "shipping_update", "promotional", "support"])
    .classification("priority", ["urgent", "normal", "low"])
    .structure("order_info")
        .field("order_number", dtype="str")
        .field("items")
        .field("total", dtype="str")
        .field("status", dtype="str",
               choices=["pending", "processing", "shipped", "delivered"])
)
```

## Advanced Multi-Task Schemas

### Complete Document Analysis

```python
# Comprehensive invoice extraction
invoice_schema = (extractor.create_schema()
    # Document classification
    .classification("document_type",
        ["invoice", "credit_note", "purchase_order", "receipt"])
    .classification("payment_status",
        ["paid", "unpaid", "partial", "overdue"])

    # Key entities
    .entities({
        "company": "Company names (buyer or seller)",
        "person": "Contact person names",
        "date": "Important dates",
        "amount": "Monetary amounts"
    })

    # Structured information
    .structure("invoice_header")
        .field("invoice_number", dtype="str")
        .field("issue_date", dtype="str")
        .field("due_date", dtype="str")
        .field("vendor_name", dtype="str")
        .field("customer_name", dtype="str")

    .structure("line_item")
        .field("description", dtype="str")
        .field("quantity")
        .field("unit_price")
        .field("amount")
        .field("tax_rate", dtype="str", choices=["0%", "5%", "10%", "20%"])

    .structure("payment_info")
        .field("method", dtype="str",
               choices=["bank_transfer", "credit_card", "check", "cash"])
        .field("terms", description="Payment terms like NET30")
        .field("bank_details", dtype="list")
)
```

### Customer Feedback Analysis

```python
feedback_schema = (extractor.create_schema()
    # Overall classifications
    .classification("sentiment", ["positive", "negative", "neutral", "mixed"])
    .classification("intent", {
        "complaint": "Customer expressing dissatisfaction",
        "compliment": "Customer expressing satisfaction",
        "suggestion": "Customer providing improvement ideas",
        "question": "Customer asking for information"
    }, multi_label=True)

    # Extract mentioned entities
    .entities({
        "product": "Products or services mentioned",
        "feature": "Specific features discussed",
        "competitor": "Competing products mentioned",
        "price_mention": "Price points or cost references"
    })

    # Structured feedback components
    .structure("issue")
        .field("problem", dtype="str")
        .field("severity", dtype="str", choices=["critical", "major", "minor"])
        .field("affected_area", dtype="list")

    .structure("suggestion")
        .field("improvement", dtype="str")
        .field("benefit", description="Expected benefit of the suggestion")
)
```

### News Article Analysis

```python
news_schema = (extractor.create_schema()
    # Article metadata
    .classification("category",
        ["politics", "business", "technology", "sports", "entertainment"])
    .classification("bias", ["left", "center", "right", "neutral"])
    .classification("factuality", ["fact", "opinion", "analysis", "speculation"])

    # Key entities
    .entities({
        "person": "People mentioned in the article",
        "organization": "Companies, agencies, or groups",
        "location": "Places, cities, or countries",
        "event": "Named events or incidents"
    })

    # Structured content
    .structure("quote")
        .field("speaker", dtype="str")
        .field("statement", dtype="str")
        .field("context", description="Context of the quote")

    .structure("claim")
        .field("statement", dtype="str")
        .field("source", dtype="str")
        .field("evidence", dtype="list")
)
```

## Real-World Applications

### E-commerce Product Listing

```python
product_schema = (extractor.create_schema()
    # Listing classification
    .classification("condition", ["new", "used", "refurbished", "for_parts"])
    .classification("listing_type", ["buy_now", "auction", "best_offer"])

    # Extract key entities
    .entities({
        "brand": "Product brand or manufacturer",
        "model": "Specific model name or number",
        "color": "Product colors mentioned",
        "size": "Size specifications"
    })

    # Product details
    .structure("product")
        .field("title", dtype="str")
        .field("price", dtype="str")
        .field("features", dtype="list")
        .field("category", dtype="str")

    # Shipping information
    .structure("shipping")
        .field("method", dtype="list",
               choices=["standard", "express", "overnight", "international"])
        .field("cost", dtype="str")
        .field("delivery_time", description="Estimated delivery timeframe")

    # Seller information
    .structure("seller")
        .field("name", dtype="str")
        .field("rating", dtype="str")
        .field("location", dtype="str")
)
```

### Healthcare Clinical Note

```python
clinical_schema = (extractor.create_schema()
    # Note classification
    .classification("visit_type",
        ["initial_consultation", "follow_up", "emergency", "routine_checkup"])
    .classification("urgency", ["urgent", "routine", "elective"])

    # Medical entities
    .entities({
        "symptom": "Patient reported symptoms",
        "diagnosis": "Medical diagnoses or conditions",
        "medication": "Prescribed or mentioned medications",
        "procedure": "Medical procedures or tests",
        "body_part": "Anatomical references"
    })

    # Patient information
    .structure("patient_info")
        .field("name", dtype="str")
        .field("age", dtype="str")
        .field("gender", dtype="str", choices=["male", "female", "other"])
        .field("chief_complaint", dtype="str")

    # Clinical findings
    .structure("vital_signs")
        .field("blood_pressure", dtype="str")
        .field("heart_rate", dtype="str")
        .field("temperature", dtype="str")
        .field("respiratory_rate", dtype="str")

    # Treatment plan
    .structure("prescription")
        .field("medication", dtype="str")
        .field("dosage", dtype="str")
        .field("frequency")
        .field("duration")
        .field("route", dtype="str", choices=["oral", "IV", "topical", "injection"])
)
```

### Legal Document Analysis

```python
legal_schema = (extractor.create_schema()
    # Document classification
    .classification("document_type",
        ["contract", "memorandum", "brief", "motion", "order"])
    .classification("jurisdiction",
        ["federal", "state", "local", "international"])

    # Legal entities
    .entities({
        "party": "Parties involved (plaintiff, defendant, etc.)",
        "attorney": "Legal representatives",
        "judge": "Judicial officers",
        "statute": "Laws or regulations cited",
        "case_citation": "Referenced legal cases"
    })

    # Contract terms
    .structure("contract_term")
        .field("clause_type", dtype="str",
               choices=["payment", "delivery", "warranty", "liability", "termination"])
        .field("obligation", dtype="str")
        .field("party_responsible", dtype="str")
        .field("deadline")

    # Legal claims
    .structure("claim")
        .field("type", dtype="str")
        .field("plaintiff", dtype="str")
        .field("defendant", dtype="str")
        .field("amount", dtype="str")
        .field("basis", description="Legal basis for the claim")
)
```

## Using Confidence Scores and Character Positions with Combined Schemas

When using combined schemas, `include_confidence` and `include_spans` parameters apply to all extraction types:

```python
schema = (extractor.create_schema()
    .entities(["person", "company"])
    .classification("sentiment", ["positive", "negative", "neutral"])
    .relations(["works_for"])
    .structure("product")
        .field("name", dtype="str")
        .field("price", dtype="str")
)

text = "Tim Cook works for Apple. The iPhone 15 costs $999. This is exciting!"
results = extractor.extract(
    text,
    schema,
    include_confidence=True,
    include_spans=True
)
# Output: {
#     'entities': {
#         'person': [
#             {'text': 'Tim Cook', 'confidence': 0.95, 'start': 0, 'end': 8}
#         ],
#         'company': [
#             {'text': 'Apple', 'confidence': 0.92, 'start': 20, 'end': 25}
#         ]
#     },
#     'sentiment': {'label': 'positive', 'confidence': 0.88},
#     'relation_extraction': {
#         'works_for': [{
#             'head': {'text': 'Tim Cook', 'confidence': 0.95, 'start': 0, 'end': 8},
#             'tail': {'text': 'Apple', 'confidence': 0.92, 'start': 20, 'end': 25}
#         }]
#     },
#     'product': [{
#         'name': {'text': 'iPhone 15', 'confidence': 0.90, 'start': 30, 'end': 39},
#         'price': {'text': '$999', 'confidence': 0.88, 'start': 46, 'end': 51}
#     }]
# }
```

**Note**: The `include_confidence` and `include_spans` parameters work consistently across all extraction types (entities, classifications, relations, and structures) when using combined schemas.
