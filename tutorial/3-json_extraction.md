# GLiNER2 JSON Structure Extraction Tutorial

Learn how to extract complex structured data from text using GLiNER2's hierarchical extraction capabilities.

## Table of Contents
- [Quick API with extract_json](#quick-api-with-extract_json)
- [Field Types and Specifications](#field-types-and-specifications)
- [Multiple Instances](#multiple-instances)
- [Schema Builder (Multi-Task)](#schema-builder-multi-task)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)

## Quick API with extract_json

For structure-only extraction, use the `extract_json()` method with the simple dictionary format:

### Basic Structure Extraction

```python
from gliner2 import GLiNER2

# Load model
extractor = GLiNER2.from_pretrained("your-model-name")

# Simple product extraction
text = "The MacBook Pro costs $1999 and features M3 chip, 16GB RAM, and 512GB storage."
results = extractor.extract_json(
    text,
    {
        "product": [
            "name::str",
            "price",
            "features"
        ]
    }
)
print(results)
# Output: {
#     'product': [{
#         'name': 'MacBook Pro',
#         'price': ['$1999'],
#         'features': ['M3 chip', '16GB RAM', '512GB storage']
#     }]
# }
```

### Contact Information

```python
text = """
Contact: John Smith
Email: john@example.com
Phones: 555-1234, 555-5678
Address: 123 Main St, NYC
"""

results = extractor.extract_json(
    text,
    {
        "contact": [
            "name::str",
            "email::str",
            "phone::list",
            "address"
        ]
    }
)
# Output: {
#     'contact': [{
#         'name': 'John Smith',
#         'email': 'john@example.com',
#         'phone': ['555-1234', '555-5678'],
#         'address': ['123 Main St, NYC']
#     }]
# }
```

## Field Types and Specifications

### Field Specification Format

Fields support flexible specifications using `::` separators:

```
"field_name::type::description"
"field_name::[choice1|choice2|choice3]::type::description"
"field_name::description"  # defaults to list type
"field_name"               # simple field, defaults to list
```

### String vs List Fields

```python
text = """
Tech Conference 2024 on June 15th in San Francisco.
Topics include AI, Machine Learning, and Cloud Computing.
Registration fee: $299 for early bird tickets.
"""

results = extractor.extract_json(
    text,
    {
        "event": [
            "name::str::Event or conference name",
            "date::str::Event date",
            "location::str",
            "topics::list::Conference topics",
            "registration_fee::str"
        ]
    }
)
# Output: {
#     'event': [{
#         'name': 'Tech Conference 2024',
#         'date': 'June 15th',
#         'location': 'San Francisco',
#         'topics': ['AI', 'Machine Learning', 'Cloud Computing'],
#         'registration_fee': '$299'
#     }]
# }
```

### Choice Fields (Classification within Structure)

```python
text = """
Reservation at Le Bernardin for 4 people on March 15th at 7:30 PM.
We'd prefer outdoor seating. Two guests are vegetarian and one is gluten-free.
"""

results = extractor.extract_json(
    text,
    {
        "reservation": [
            "restaurant::str::Restaurant name",
            "date::str",
            "time::str",
            "party_size::[1|2|3|4|5|6+]::str::Number of guests",
            "seating::[indoor|outdoor|bar]::str::Seating preference",
            "dietary::[vegetarian|vegan|gluten-free|none]::list::Dietary restrictions"
        ]
    }
)
# Output: {
#     'reservation': [{
#         'restaurant': 'Le Bernardin',
#         'date': 'March 15th',
#         'time': '7:30 PM',
#         'party_size': '4',
#         'seating': 'outdoor',
#         'dietary': ['vegetarian', 'gluten-free']
#     }]
# }
```

## Multiple Instances

GLiNER2 automatically extracts ALL instances of a structure found in text:

### Multiple Transactions

```python
text = """
Recent transactions:
- Jan 5: Starbucks $5.50 (food)
- Jan 5: Uber $23.00 (transport)
- Jan 6: Amazon $156.99 (shopping)
"""

results = extractor.extract_json(
    text,
    {
        "transaction": [
            "date::str",
            "merchant::str",
            "amount::str",
            "category::[food|transport|shopping|utilities]::str"
        ]
    }
)
# Output: {
#     'transaction': [
#         {'date': 'Jan 5', 'merchant': 'Starbucks', 'amount': '$5.50', 'category': 'food'},
#         {'date': 'Jan 5', 'merchant': 'Uber', 'amount': '$23.00', 'category': 'transport'},
#         {'date': 'Jan 6', 'merchant': 'Amazon', 'amount': '$156.99', 'category': 'shopping'}
#     ]
# }
```

### Multiple Hotel Bookings

```python
text = """
Alice Brown booked the Hilton Downtown from March 10 to March 12. She selected a double room
for $340 total with breakfast and parking included.

Robert Taylor reserved The Grand Hotel, April 1 to April 5, suite at $1,200 total.
Amenities include breakfast, wifi, gym, and spa access.
"""

results = extractor.extract_json(
    text,
    {
        "booking": [
            "guest::str::Guest name",
            "hotel::str::Hotel name",
            "check_in::str",
            "check_out::str",
            "room_type::[single|double|suite|deluxe]::str",
            "total_price::str",
            "amenities::[breakfast|wifi|parking|gym|spa]::list"
        ]
    }
)
# Output: {
#     'booking': [
#         {
#             'guest': 'Alice Brown',
#             'hotel': 'Hilton Downtown',
#             'check_in': 'March 10',
#             'check_out': 'March 12',
#             'room_type': 'double',
#             'total_price': '$340',
#             'amenities': ['breakfast', 'parking']
#         },
#         {
#             'guest': 'Robert Taylor',
#             'hotel': 'The Grand Hotel',
#             'check_in': 'April 1',
#             'check_out': 'April 5',
#             'room_type': 'suite',
#             'total_price': '$1,200',
#             'amenities': ['breakfast', 'wifi', 'gym', 'spa']
#         }
#     ]
# }
```

## Schema Builder (Multi-Task)

Use `create_schema()` only when combining structured extraction with other tasks (entities, classification):

### Multi-Task Extraction

```python
# Use schema builder for multi-task scenarios
schema = (extractor.create_schema()
    # Extract entities
    .entities(["person", "company", "location"])

    # Classify sentiment
    .classification("sentiment", ["positive", "negative", "neutral"])

    # Extract structured product info
    .structure("product")
        .field("name", dtype="str")
        .field("price", dtype="str")
        .field("features", dtype="list")
        .field("category", dtype="str", choices=["electronics", "software", "service"])
)

text = "Apple CEO Tim Cook announced iPhone 15 for $999 with amazing new features. This is exciting!"
results = extractor.extract(text, schema)
# Output: {
#     'entities': {'person': ['Tim Cook'], 'company': ['Apple'], 'location': []},
#     'sentiment': 'positive',
#     'product': [{
#         'name': 'iPhone 15',
#         'price': '$999',
#         'features': ['amazing new features'],
#         'category': 'electronics'
#     }]
# }
```

### Advanced Configuration

```python
schema = (extractor.create_schema()
    .classification("urgency", ["low", "medium", "high"])

    .structure("support_ticket")
        .field("ticket_id", dtype="str", threshold=0.9)      # High precision
        .field("customer", dtype="str", description="Customer name")
        .field("issue", dtype="str", description="Problem description")
        .field("priority", dtype="str", choices=["low", "medium", "high", "urgent"])
        .field("tags", dtype="list", choices=["bug", "feature", "support", "billing"])
)
```

## Examples

### Financial Transaction Processing

```python
text = """
Goldman Sachs processed a $2.5M equity trade for Tesla Inc. on March 15, 2024.
Commission: $1,250. Status: Completed.
"""

results = extractor.extract_json(
    text,
    {
        "transaction": [
            "broker::str::Financial institution",
            "amount::str::Transaction amount",
            "security::str::Stock or financial instrument",
            "date::str::Transaction date",
            "commission::str::Fees charged",
            "status::[pending|completed|failed]::str",
            "type::[equity|bond|option|future]::str"
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
#         'status': 'completed',
#         'type': 'equity'
#     }]
# }
```

### Medical Prescription Extraction

```python
text = """
Patient: Sarah Johnson, 34, presented with chest pain.
Prescribed: Lisinopril 10mg daily, Metoprolol 25mg twice daily.
Follow-up scheduled for next Tuesday.
"""

results = extractor.extract_json(
    text,
    {
        "patient": [
            "name::str::Patient full name",
            "age::str::Patient age",
            "symptoms::list::Reported symptoms"
        ],
        "prescription": [
            "medication::str::Drug name",
            "dosage::str::Dosage amount",
            "frequency::str::How often to take"
        ]
    }
)
# Output: {
#     'patient': [{
#         'name': 'Sarah Johnson',
#         'age': '34',
#         'symptoms': ['chest pain']
#     }],
#     'prescription': [
#         {'medication': 'Lisinopril', 'dosage': '10mg', 'frequency': 'daily'},
#         {'medication': 'Metoprolol', 'dosage': '25mg', 'frequency': 'twice daily'}
#     ]
# }
```

### E-commerce Order Processing

```python
text = """
Order #ORD-2024-001 for Alexandra Thompson
Items: Laptop Stand (2x $45.99), Wireless Mouse (1x $29.99), USB Hub (3x $35.50)
Subtotal: $228.46, Tax: $18.28, Total: $246.74
Status: Processing
"""

results = extractor.extract_json(
    text,
    {
        "order": [
            "order_id::str::Order number",
            "customer::str::Customer name",
            "items::list::Product names",
            "quantities::list::Item quantities",
            "unit_prices::list::Individual prices",
            "subtotal::str",
            "tax::str",
            "total::str",
            "status::[pending|processing|shipped|delivered]::str"
        ]
    }
)
# Output: {
#     'order': [{
#         'order_id': 'ORD-2024-001',
#         'customer': 'Alexandra Thompson',
#         'items': ['Laptop Stand', 'Wireless Mouse', 'USB Hub'],
#         'quantities': ['2', '1', '3'],
#         'unit_prices': ['$45.99', '$29.99', '$35.50'],
#         'subtotal': '$228.46',
#         'tax': '$18.28',
#         'total': '$246.74',
#         'status': 'processing'
#     }]
# }
```

## Confidence Scores and Character Positions

You can include confidence scores and character-level start/end positions for structured extraction:

```python
# Extract with confidence scores
text = "The MacBook Pro costs $1999 and features M3 chip, 16GB RAM, and 512GB storage."
results = extractor.extract_json(
    text,
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

# Extract with character positions (spans)
results = extractor.extract_json(
    text,
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

# Extract with both confidence and spans
results = extractor.extract_json(
    text,
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

**Note**: When `include_spans` or `include_confidence` is True:
- **String fields** (`dtype="str"`): Return dicts with `{'text': '...', 'confidence': 0.95, 'start': 0, 'end': 5}` (or subset)
- **List fields** (`dtype="list"`): Return lists of dicts, each with text, confidence, and positions
- **Default** (both False): Returns simple strings or lists of strings

## Best Practices

### Data Types

- Use `::str` for single values (IDs, names, amounts)
- Use `::list` or default for multiple values (features, items, tags)
- Use choices `[opt1|opt2|opt3]` for standardized values
- Add descriptions for complex or domain-specific fields

### Quick Decision Guide

**Use `extract_json()`** for:
- Structure-only extraction
- Quick data parsing
- Single extraction task

**Use `create_schema().extract()`** for:
- Multi-task scenarios (entities + structures + classification)
- When you need entities or classification alongside structures
- Complex extraction pipelines
