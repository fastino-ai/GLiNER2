# GLiNER2 Training Dataset Formats

GLiNER2 uses JSONL format where each line contains an `input` and `output` field (or alternatively `text` and `schema`). The `input`/`text` is the text to process, and the `output`/`schema` is the schema with labels/annotations.

## Quick Format Reference

### General Structure

**Primary Format**:
```jsonl
{"input": "text to process", "output": {"schema_definition": "with_annotations"}}
```

**Alternative Format** (also supported):
```jsonl
{"text": "text to process", "schema": {"schema_definition": "with_annotations"}}
```

Both formats are equivalent - use whichever is more convenient for your workflow.

### Valid Output Schema Keys

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `entities` | `dict[str, list[str]]` | No | Entity type → list of entity mentions |
| `entity_descriptions` | `dict[str, str]` | No | Entity type → description |
| `classifications` | `list[dict]` | No | List of classification tasks |
| `json_structures` | `list[dict]` | No | List of structured data extractions |
| `json_descriptions` | `dict[str, dict[str, str]]` | No | Parent → field → description |
| `relations` | `list[dict]` | No | List of relation extractions |

### Classification Task Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task` | `str` | Yes | Task identifier |
| `labels` | `list[str]` | Yes | Available label options |
| `true_label` | `list[str]` or `str` | Yes | Correct label(s) |
| `multi_label` | `bool` | No | Enable multi-label classification |
| `prompt` | `str` | No | Custom prompt for the task |
| `examples` | `list[list[str]]` or `list[tuple[str, str]]` | No | Few-shot examples as [[input, output], ...] pairs. Internally converted to list of lists. |
| `label_descriptions` | `dict[str, str]` | No | Label → description mapping |

### Entity Fields Format

Entities use a simple dictionary where keys are entity types and values are lists of mentions:

| Component | Type | Required | Description |
|-----------|------|----------|-------------|
| Entity type (key) | `str` | Yes | Name of the entity type (e.g., "person", "location") |
| Entity mentions (value) | `list[str]` | Yes | List of entity text spans found in input |

**Format**: `{"entity_type": ["mention1", "mention2", ...]}`

### JSON Structure Fields Format

Each structure is a dictionary with a parent name as key and field definitions as value:

| Component | Type | Required | Description |
|-----------|------|----------|-------------|
| Parent name (key) | `str` | Yes | Name of the structure (e.g., "product", "contact") |
| Fields (value) | `dict` | Yes | Field name → field value mappings |
| Field value | `str` or `list[str]` or `dict` | Yes | String, list of strings, or choice dict |
| Choice dict | `dict` with `value` and `choices` | No | For classification-style fields |

**Format**: `[{"parent": {"field1": "value", "field2": ["list", "values"]}}]`

**Multiple Instances**: When the same parent appears multiple times, each instance is a separate dict in the list:
```jsonl
[{"hotel": {"name": "Hotel A", ...}}, {"hotel": {"name": "Hotel B", ...}}]
```

### Relation Fields Format

Relations use flexible field structures - you can use ANY field names (not just "head" and "tail"):

| Component | Type | Required | Description |
|-----------|------|----------|-------------|
| Relation name (key) | `str` | Yes | Name of the relation type (e.g., "works_for") |
| Fields (value) | `dict` | Yes | Field name → field value mappings |
| Field value | `str` or `list[str]` | Yes | String or list of strings |

**Standard Format**: `[{"relation_name": {"head": "entity1", "tail": "entity2"}}]`

**⚠️ Critical Constraint**: For a given relation type, the **first occurrence** defines the field structure:
- The first instance of "works_for" determines what fields ALL "works_for" instances must have
- All subsequent instances of the same relation type must use the same field names
- Different relation types can have different field structures
- **This consistency is enforced during validation** - inconsistent field structures will raise a `ValidationError`

**Example**: If first "works_for" has `{"head": "...", "tail": "..."}`, all other "works_for" instances must also have "head" and "tail" fields.

**Validation**: The `TrainingDataset.validate_relation_consistency()` method checks that all relation types have consistent field structures across the entire dataset.

---

## Alternative Input Formats

The training data loader supports multiple input formats:

1. **JSONL files**: `{"input": "...", "output": {...}}` or `{"text": "...", "schema": {...}}`
2. **Python API**: Use `InputExample` and `TrainingDataset` classes from `gliner2.training.data`
3. **Dict lists**: List of dictionaries in the same format as JSONL

All formats are automatically detected and converted to the internal format. See `gliner2.training.data.DataLoader_Factory` for details.

---

## 1. Classification Tasks

### Basic Single-Label Classification

```jsonl
{"input": "This movie is absolutely fantastic! I loved every minute of it.", "output": {"classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": ["positive"]}]}}
{"input": "The service at this restaurant was terrible and the food was cold.", "output": {"classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": ["negative"]}]}}
{"input": "The weather today is okay, nothing special.", "output": {"classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": ["neutral"]}]}}
```

### Multi-label Classification

```jsonl
{"input": "This smartphone has an amazing camera but the battery life is poor.", "output": {"classifications": [{"task": "product_aspects", "labels": ["camera", "battery", "screen", "performance", "design"], "true_label": ["camera", "battery"], "multi_label": true}]}}
{"input": "Great performance and beautiful design!", "output": {"classifications": [{"task": "product_aspects", "labels": ["camera", "battery", "screen", "performance", "design"], "true_label": ["performance", "design"], "multi_label": true}]}}
```

### Classification with Label Descriptions

```jsonl
{"input": "Breaking: New AI model achieves human-level performance on reasoning tasks.", "output": {"classifications": [{"task": "news_category", "labels": ["technology", "politics", "sports", "entertainment"], "true_label": ["technology"], "label_descriptions": {"technology": "Articles about computers, AI, software, and tech innovations", "politics": "Government, elections, and political news", "sports": "Athletic events, teams, and competitions", "entertainment": "Movies, music, celebrities, and entertainment news"}}]}}
```

### Classification with Custom Prompts

```jsonl
{"input": "The patient shows signs of improvement after treatment.", "output": {"classifications": [{"task": "medical_assessment", "labels": ["improving", "stable", "declining", "critical"], "true_label": ["improving"], "prompt": "Assess the patient's medical condition based on the clinical notes."}]}}
```

### Classification with Few-Shot Examples

Few-shot examples are provided as a list of `[input, output]` pairs. Each example is a list/tuple with exactly 2 elements:

```jsonl
{"input": "This service exceeded all my expectations!", "output": {"classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": ["positive"], "examples": [["Great product, highly recommend!", "positive"], ["Terrible experience, very disappointed.", "negative"], ["It's okay, nothing special.", "neutral"]]}]}}
```

**Format**: `"examples": [[input_text, output_label], [input_text, output_label], ...]`

Each example pair must have exactly 2 elements: the input text and the corresponding label.

### Classification with Both Examples and Descriptions

```jsonl
{"input": "The algorithm demonstrates linear time complexity.", "output": {"classifications": [{"task": "complexity", "labels": ["constant", "linear", "quadratic", "exponential"], "true_label": ["linear"], "examples": [["O(1) lookup time", "constant"], ["O(n) iteration", "linear"]], "label_descriptions": {"constant": "O(1) - fixed time regardless of input size", "linear": "O(n) - time scales linearly with input", "quadratic": "O(n²) - nested iterations", "exponential": "O(2ⁿ) - recursive branching"}}]}}
```

### Multiple Classification Tasks

```jsonl
{"input": "Exciting new smartphone with innovative features!", "output": {"classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": ["positive"]}, {"task": "category", "labels": ["technology", "sports", "politics", "entertainment"], "true_label": ["technology"]}]}}
```

### true_label: String vs List Format

Both formats are supported - use list for consistency or string for brevity:

```jsonl
{"input": "Sample text A", "output": {"classifications": [{"task": "label", "labels": ["a", "b"], "true_label": ["a"]}]}}
{"input": "Sample text B", "output": {"classifications": [{"task": "label", "labels": ["a", "b"], "true_label": "b"}]}}
{"input": "This is great!", "output": {"classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": "positive"}]}}
```

**Note**:
- String format (`"true_label": "positive"`) and list format (`"true_label": ["positive"]`) are both valid for single-label classification
- Internally, string values are automatically converted to lists (`["positive"]`)
- For multi-label classification, always use list format: `"true_label": ["label1", "label2"]`

---

## 2. Named Entity Recognition (NER)

### Basic NER

```jsonl
{"input": "John Smith works at OpenAI in San Francisco and will visit London next month.", "output": {"entities": {"person": ["John Smith"], "organization": ["OpenAI"], "location": ["San Francisco", "London"]}}}
{"input": "Apple Inc. CEO Tim Cook announced the iPhone 15 release date.", "output": {"entities": {"person": ["Tim Cook"], "organization": ["Apple Inc."], "product": ["iPhone 15"]}}}
{"input": "The meeting on January 15, 2024 will be held at Microsoft headquarters.", "output": {"entities": {"date": ["January 15, 2024"], "organization": ["Microsoft"]}}}
```

### NER with Entity Descriptions

```jsonl
{"input": "Dr. Sarah Johnson prescribed Metformin 500mg twice daily for diabetes treatment.", "output": {"entities": {"person": ["Dr. Sarah Johnson"], "medication": ["Metformin"], "dosage": ["500mg"], "condition": ["diabetes"]}, "entity_descriptions": {"person": "Names of people mentioned in the text", "medication": "Names of drugs or pharmaceutical products", "dosage": "Specific amounts or dosages of medications", "condition": "Medical conditions or diseases"}}}
```

### NER with Multiple Instances of Same Entity Type

```jsonl
{"input": "Alice, Bob, and Charlie attended the meeting with David.", "output": {"entities": {"person": ["Alice", "Bob", "Charlie", "David"]}}}
```

### NER with Empty Entity Types

```jsonl
{"input": "The conference will be held next week.", "output": {"entities": {"person": [], "organization": [], "location": []}}}
```

### Partial NER (Some Entity Types Present)

```jsonl
{"input": "Microsoft announced new features.", "output": {"entities": {"organization": ["Microsoft"], "person": []}}}
```

---

## 3. JSON Structure Extraction

### Basic Structure with String Fields

```jsonl
{"input": "Contact John Doe at john.doe@email.com or call (555) 123-4567.", "output": {"json_structures": [{"contact": {"name": "John Doe", "email": "john.doe@email.com", "phone": "(555) 123-4567"}}]}}
```

### Structure with List Fields

```jsonl
{"input": "Product features include: wireless charging, water resistance, and face recognition.", "output": {"json_structures": [{"product": {"features": ["wireless charging", "water resistance", "face recognition"]}}]}}
```

### Structure with Mixed String and List Fields

```jsonl
{"input": "iPhone 15 costs $999 and comes in blue, black, and white colors.", "output": {"json_structures": [{"product": {"name": "iPhone 15", "price": "$999", "colors": ["blue", "black", "white"]}}]}}
```

### Multiple Instances of Same Structure Type

When the **same structure type** (parent name) appears multiple times in the text, each instance is a **separate dictionary** in the `json_structures` list:

```jsonl
{"input": "We have two hotels available: Hotel Paradise with 4 stars, pool, and wifi for $150/night, and Budget Inn with 2 stars and parking for $80/night.", "output": {"json_structures": [{"hotel": {"name": "Hotel Paradise", "stars": "4", "amenities": ["pool", "wifi"], "price": "$150/night"}}, {"hotel": {"name": "Budget Inn", "stars": "2", "amenities": ["parking"], "price": "$80/night"}}]}}
```

**Note**: Both instances use the same parent key "hotel" but are separate objects in the list. This is how you represent multiple occurrences of the same structure type.

Another example with three products:

```jsonl
{"input": "Available products: iPhone 15 for $999, MacBook Pro for $1999, and AirPods for $199.", "output": {"json_structures": [{"product": {"name": "iPhone 15", "price": "$999"}}, {"product": {"name": "MacBook Pro", "price": "$1999"}}, {"product": {"name": "AirPods", "price": "$199"}}]}}
```

### Structure with Classification Fields (Choices)

```jsonl
{"input": "Book a single room at Grand Hotel for 2 nights with breakfast included.", "output": {"json_structures": [{"booking": {"hotel": "Grand Hotel", "room_type": {"value": "single", "choices": ["single", "double", "suite"]}, "nights": "2", "meal_plan": {"value": "breakfast", "choices": ["none", "breakfast", "half-board", "full-board"]}}}]}}
```

### Structure with Multiple Choice Fields

```jsonl
{"input": "Order a large pepperoni pizza for delivery, extra cheese.", "output": {"json_structures": [{"order": {"size": {"value": "large", "choices": ["small", "medium", "large", "xlarge"]}, "type": {"value": "pepperoni", "choices": ["cheese", "pepperoni", "veggie", "supreme"]}, "method": {"value": "delivery", "choices": ["pickup", "delivery", "dine-in"]}, "extras": ["extra cheese"]}}]}}
```

### Structure with Field Descriptions

```jsonl
{"input": "Patient: Mary Wilson, Age: 45, diagnosed with hypertension, prescribed Lisinopril 10mg daily.", "output": {"json_structures": [{"medical_record": {"patient_name": "Mary Wilson", "age": "45", "diagnosis": "hypertension", "medication": "Lisinopril", "dosage": "10mg daily"}}], "json_descriptions": {"medical_record": {"patient_name": "Full name of the patient", "age": "Patient's age in years", "diagnosis": "Medical condition diagnosed", "medication": "Prescribed medication name", "dosage": "Medication dosage and frequency"}}}}
```

### Structure with Null/Empty Field Values

```jsonl
{"input": "Product name is Widget X. Price not available.", "output": {"json_structures": [{"product": {"name": "Widget X", "price": "", "description": ""}}]}}
```

### Structure with Some Fields Missing

```jsonl
{"input": "Contact Sarah at sarah@example.com", "output": {"json_structures": [{"contact": {"name": "Sarah", "email": "sarah@example.com", "phone": ""}}]}}
```

### Multiple Different Structure Types

```jsonl
{"input": "John Doe works at TechCorp. Product ABC costs $50 with free shipping.", "output": {"json_structures": [{"employee": {"name": "John Doe", "company": "TechCorp"}}, {"product": {"name": "ABC", "price": "$50", "shipping": "free"}}]}}
```

### Structure with Only List Fields

```jsonl
{"input": "Available colors: red, blue, green. Sizes: S, M, L, XL.", "output": {"json_structures": [{"options": {"colors": ["red", "blue", "green"], "sizes": ["S", "M", "L", "XL"]}}]}}
```

---

## 4. Relation Extraction

Relations use flexible field structures. While "head" and "tail" are common, you can use ANY field names.

**⚠️ Important**: The first occurrence of each relation type defines the field structure for ALL instances of that type.

### Basic Relation (Head and Tail)

```jsonl
{"input": "Alice manages the engineering team.", "output": {"relations": [{"manages": {"head": "Alice", "tail": "engineering team"}}]}}
{"input": "John works for Microsoft.", "output": {"relations": [{"works_for": {"head": "John", "tail": "Microsoft"}}]}}
```

### Multiple Instances - Same Field Structure

All instances of the same relation type MUST have the same fields (determined by first occurrence):

```jsonl
{"input": "Alice works for Google. Bob works for Microsoft. Charlie works for Amazon.", "output": {"relations": [{"works_for": {"head": "Alice", "tail": "Google"}}, {"works_for": {"head": "Bob", "tail": "Microsoft"}}, {"works_for": {"head": "Charlie", "tail": "Amazon"}}]}}
```

**Note**: All three "works_for" instances use the same fields (head, tail) as defined by the first occurrence.

### Multiple Different Relation Types

Different relation types can have different field structures:

```jsonl
{"input": "John works for Apple Inc. and lives in San Francisco. Apple Inc. is located in Cupertino.", "output": {"relations": [{"works_for": {"head": "John", "tail": "Apple Inc."}}, {"lives_in": {"head": "John", "tail": "San Francisco"}}, {"located_in": {"head": "Apple Inc.", "tail": "Cupertino"}}]}}
```

**Note**: Each relation type ("works_for", "lives_in", "located_in") can independently define its own field structure.

### Custom Field Names (Beyond Head/Tail)

You can use custom field names - the first occurrence defines what fields to use:

```jsonl
{"input": "Alice sent $100 to Bob. Charlie sent $50 to David.", "output": {"relations": [{"transaction": {"sender": "Alice", "recipient": "Bob", "amount": "$100"}}, {"transaction": {"sender": "Charlie", "recipient": "David", "amount": "$50"}}]}}
```

**Note**: First "transaction" uses sender/recipient/amount, so all "transaction" instances must use these same fields.

### Relations with Additional Fields

```jsonl
{"input": "John Smith is the CEO of TechCorp which is headquartered in Silicon Valley.", "output": {"relations": [{"employment": {"head": "John Smith", "tail": "TechCorp", "role": "CEO"}}, {"located_in": {"head": "TechCorp", "tail": "Silicon Valley"}}]}}
```

### Relations Combined with Entities

```jsonl
{"input": "Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne.", "output": {"entities": {"person": ["Elon Musk"], "organization": ["SpaceX"], "location": ["Hawthorne"], "date": ["2002"]}, "relations": [{"founded": {"head": "Elon Musk", "tail": "SpaceX"}}, {"located_in": {"head": "SpaceX", "tail": "Hawthorne"}}]}}
```

### Empty Relations (Negative Example)

```jsonl
{"input": "The weather is nice today.", "output": {"relations": []}}
```

### Bidirectional Relations

```jsonl
{"input": "Alice and Bob are colleagues.", "output": {"relations": [{"colleague_of": {"head": "Alice", "tail": "Bob"}}, {"colleague_of": {"head": "Bob", "tail": "Alice"}}]}}
```

### Field Consistency: Relations vs JSON Structures

**Key Difference**:

- **Relations**: First occurrence defines field structure for ALL instances of that relation type
  - All "works_for" relations must have same fields
  - Enforced consistency per relation type

- **JSON Structures**: Fields can vary between instances of the same parent type
  - Uses union of all fields across instances
  - More flexible - instances can have different subsets of fields

**Example - Relations (Strict Consistency)**:
```jsonl
{"input": "Alice works for Google. Bob works for Microsoft.", "output": {"relations": [{"works_for": {"head": "Alice", "tail": "Google"}}, {"works_for": {"head": "Bob", "tail": "Microsoft"}}]}}
```
✓ Valid: Both "works_for" have same fields (head, tail)

**Example - JSON Structures (Flexible Fields)**:
```jsonl
{"input": "Product A costs $10. Product B costs $20 and weighs 5kg.", "output": {"json_structures": [{"product": {"name": "A", "price": "$10"}}, {"product": {"name": "B", "price": "$20", "weight": "5kg"}}]}}
```
✓ Valid: Second instance has extra "weight" field - this is allowed for json_structures

---

## 5. Combined Multi-Task Examples

### Entities + Classifications

```jsonl
{"input": "Apple Inc. announced record profits. This is great news for investors.", "output": {"entities": {"organization": ["Apple Inc."]}, "classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": ["positive"]}]}}
```

### Entities + JSON Structures

```jsonl
{"input": "Contact John Doe at john@example.com. He works at TechCorp.", "output": {"entities": {"person": ["John Doe"], "organization": ["TechCorp"]}, "json_structures": [{"contact": {"name": "John Doe", "email": "john@example.com", "company": "TechCorp"}}]}}
```

### Entities + Relations

```jsonl
{"input": "Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne.", "output": {"entities": {"person": ["Elon Musk"], "organization": ["SpaceX"], "location": ["Hawthorne"], "date": ["2002"]}, "relations": [{"founded": {"head": "Elon Musk", "tail": "SpaceX", "year": "2002"}}, {"located_in": {"head": "SpaceX", "tail": "Hawthorne"}}]}}
```

### Classifications + JSON Structures

```jsonl
{"input": "Premium subscription for $99/month includes unlimited access. Great value!", "output": {"classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": ["positive"]}], "json_structures": [{"subscription": {"tier": "Premium", "price": "$99/month", "features": ["unlimited access"]}}]}}
```

### Entities + Classifications + JSON Structures

```jsonl
{"input": "Apple CEO Tim Cook unveiled iPhone 15 for $999. Analysts are optimistic.", "output": {"entities": {"person": ["Tim Cook"], "organization": ["Apple"], "product": ["iPhone 15"]}, "classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": ["positive"]}], "json_structures": [{"product_announcement": {"company": "Apple", "product": "iPhone 15", "price": "$999", "presenter": "Tim Cook"}}]}}
```

### Entities + Relations + Classifications

```jsonl
{"input": "Sarah founded TechStart in 2020. The company is doing exceptionally well.", "output": {"entities": {"person": ["Sarah"], "organization": ["TechStart"], "date": ["2020"]}, "relations": [{"founded": {"head": "Sarah", "tail": "TechStart", "year": "2020"}}], "classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": ["positive"]}]}}
```

### All Four Tasks Combined

```jsonl
{"input": "Breaking: Apple announces new iPhone 15 with improved camera. Analysts are optimistic about sales projections.", "output": {"entities": {"company": ["Apple"], "product": ["iPhone 15"]}, "classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": ["positive"]}, {"task": "category", "labels": ["technology", "business", "sports", "entertainment"], "true_label": ["technology"]}], "json_structures": [{"news_article": {"company": "Apple", "product": "iPhone 15", "feature": "improved camera", "analyst_view": "optimistic"}}], "relations": [{"product_of": {"head": "iPhone 15", "tail": "Apple"}}]}}
```

### Multi-Task with Descriptions

```jsonl
{"input": "Dr. Johnson prescribed medication X for condition Y. Patient shows improvement.", "output": {"entities": {"person": ["Dr. Johnson"], "medication": ["medication X"], "condition": ["condition Y"]}, "entity_descriptions": {"person": "Healthcare provider names", "medication": "Prescribed drugs", "condition": "Medical conditions"}, "classifications": [{"task": "patient_status", "labels": ["improving", "stable", "declining"], "true_label": ["improving"], "label_descriptions": {"improving": "Patient condition getting better", "stable": "No change in condition", "declining": "Patient condition worsening"}}], "json_structures": [{"prescription": {"doctor": "Dr. Johnson", "medication": "medication X", "condition": "condition Y"}}], "json_descriptions": {"prescription": {"doctor": "Prescribing physician", "medication": "Prescribed drug name", "condition": "Diagnosed condition"}}}}
```

### Partial Multi-Task (Some Tasks Empty)

**Note**: While you can include empty dictionaries/lists for some tasks, at least one task must have content.

```jsonl
{"input": "The weather forecast predicts rain tomorrow.", "output": {"entities": {}, "classifications": [{"task": "weather", "labels": ["sunny", "rainy", "cloudy", "snowy"], "true_label": ["rainy"]}], "json_structures": []}}
```

This is valid because it has a classification task. However, if all tasks were empty, it would fail validation.

---

## 6. Format Edge Cases

### Completely Empty Output

**⚠️ Note**: Examples must have at least one task (entities, classifications, structures, or relations). Completely empty outputs are not valid training examples.

```jsonl
{"input": "Random text with no specific information.", "output": {"entities": {}, "classifications": [], "json_structures": [], "relations": []}}
```

This format will fail validation. Each example must contain at least one annotation.

### Empty Entities Dictionary

**⚠️ Note**: While an empty entities dictionary is syntactically valid, examples must have at least one task. If you only have empty entities, add at least one other task (classification, structure, or relation).

```jsonl
{"input": "The weather is nice today.", "output": {"entities": {}, "classifications": [{"task": "sentiment", "labels": ["positive", "negative"], "true_label": ["positive"]}]}}
```

### Empty Classifications List

**⚠️ Note**: While an empty classifications list is syntactically valid, examples must have at least one task. If you only have empty classifications, add at least one other task.

```jsonl
{"input": "Some generic text.", "output": {"classifications": [], "entities": {"location": ["text"]}}}
```

### Very Long Label Lists

```jsonl
{"input": "Sample text for many labels.", "output": {"classifications": [{"task": "topic", "labels": ["label1", "label2", "label3", "label4", "label5", "label6", "label7", "label8", "label9", "label10", "label11", "label12", "label13", "label14", "label15", "label16", "label17", "label18", "label19", "label20"], "true_label": ["label5"]}]}}
```

### Very Short Text

```jsonl
{"input": "Yes.", "output": {"classifications": [{"task": "response", "labels": ["yes", "no", "maybe"], "true_label": ["yes"]}]}}
{"input": "OK", "output": {"entities": {}}}
```

### Special Characters in Labels

```jsonl
{"input": "The C++ programming language.", "output": {"entities": {"programming_language": ["C++"]}}}
{"input": "Use the @ symbol for mentions.", "output": {"entities": {"symbol": ["@"]}}}
```

### Special Characters in Values

```jsonl
{"input": "Price is $1,299.99 (including tax).", "output": {"json_structures": [{"pricing": {"amount": "$1,299.99", "note": "(including tax)"}}]}}
```

### Unicode and Non-ASCII Characters

```jsonl
{"input": "Café Münchën serves crème brûlée.", "output": {"entities": {"location": ["Café Münchën"], "food": ["crème brûlée"]}}}
{"input": "东京 Tokyo is the capital.", "output": {"entities": {"location": ["东京", "Tokyo"]}}}
```

### Quotes and Escaping

```jsonl
{"input": "He said \"hello\" to me.", "output": {"entities": {"quote": ["\"hello\""]}}}
```

### Newlines in Text

```jsonl
{"input": "First line.\nSecond line.", "output": {"entities": {"text": ["First line", "Second line"]}}}
```

### Numbers as Strings vs Entity Names

```jsonl
{"input": "Room 123 on floor 4.", "output": {"json_structures": [{"location": {"room": "123", "floor": "4"}}]}}
```

### Boolean-like Values

```jsonl
{"input": "Status is active, notifications enabled.", "output": {"json_structures": [{"settings": {"status": "active", "notifications": "enabled"}}]}}
```

### Empty String Values

```jsonl
{"input": "Name: John, Age: unknown", "output": {"json_structures": [{"person": {"name": "John", "age": ""}}]}}
```

### Multiple Empty Lines in JSONL

```jsonl
{"input": "First example.", "output": {"entities": {"type": ["example"]}}}
{"input": "Second example.", "output": {"entities": {"type": ["example"]}}}
```

---

## Schema Component Reference

### entities
- **Type**: `dict[str, list[str]]`
- **Format**: `{"entity_type": ["mention1", "mention2", ...]}`
- **Example**: `{"person": ["John", "Alice"], "location": ["NYC"]}`

### entity_descriptions
- **Type**: `dict[str, str]`
- **Format**: `{"entity_type": "description text"}`
- **Example**: `{"person": "Names of people", "location": "Geographic places"}`

### classifications
- **Type**: `list[dict]`
- **Required fields**: `task`, `labels`, `true_label`
- **Optional fields**: `multi_label`, `prompt`, `examples`, `label_descriptions`
- **Example**: `[{"task": "sentiment", "labels": ["pos", "neg"], "true_label": ["pos"]}]`

### json_structures
- **Type**: `list[dict]`
- **Single instance**: `[{"parent_name": {"field1": "value1", "field2": ["list", "values"]}}]`
- **Multiple instances (same parent)**: `[{"parent": {...}}, {"parent": {...}}]` - Same parent key, separate dicts
- **Multiple types**: `[{"parent1": {...}}, {"parent2": {...}}]` - Different parent keys
- **Choice format**: `{"field": {"value": "selected", "choices": ["opt1", "opt2"]}}`
- **Example**: `[{"product": {"name": "Item", "price": "$10"}}, {"product": {"name": "Item2", "price": "$20"}}]`

### json_descriptions
- **Type**: `dict[str, dict[str, str]]`
- **Format**: `{"parent": {"field": "description"}}`
- **Example**: `{"product": {"name": "Product name", "price": "Cost in USD"}}`

### relations
- **Type**: `list[dict]`
- **Standard format**: `[{"relation_name": {"head": "entity1", "tail": "entity2"}}]`
- **With custom fields**: `[{"relation_name": {"sender": "A", "recipient": "B", "amount": "$100"}}]`
- **Example**: `[{"works_for": {"head": "John", "tail": "Company"}}, {"founded": {"head": "Alice", "tail": "StartupX"}}]`
- **⚠️ Field constraint**: First occurrence of each relation type defines field structure for ALL instances of that type
- **Note**: While "head" and "tail" are common, you can use ANY field names - just keep them consistent per relation type

---

## Tips for Dataset Creation

1. **Use diverse examples** to improve model generalization
2. **Include edge cases** - but remember each example must have at least one task
3. **Provide descriptions** when possible to improve accuracy
4. **Balance your classes** in classification tasks
5. **Use realistic text** that matches your target domain
6. **Include multiple instances** for JSON structures when applicable
7. **For negative examples**, include at least one task (e.g., empty entities but a classification, or empty classifications but entities)
8. **Mix task types** to train multi-task capabilities
9. **Use consistent formatting** for similar examples
10. **Include special characters** to ensure robust handling
11. **Validate your dataset** using `TrainingDataset.validate(strict=True)` to catch annotation errors early
12. **Check relation consistency** using `validate_relation_consistency()` to ensure all relation types have consistent field structures

## Validation Checklist

Make sure your JSONL file is valid by checking:
- [ ] Each line is valid JSON
- [ ] Required fields (`input`/`output` or `text`/`schema`) are present
- [ ] **At least one task is present** (entities, classifications, structures, or relations)
- [ ] Schema structure matches the expected format
- [ ] Entity spans exist in the input text (entities can be found in the input) - checked in strict validation mode
- [ ] Classification labels are from the defined label set
- [ ] `true_label` is a list or string (string format is converted to list internally)
- [ ] For multi-label classification, `multi_label` is set to `true` when multiple labels are provided
- [ ] JSON structure fields match between instances of the same parent (flexible - union of fields is used)
- [ ] **Relation field consistency**: All instances of the same relation type use the same field names (determined by first occurrence)
- [ ] No trailing commas in JSON objects
- [ ] Special characters are properly escaped
- [ ] File encoding is UTF-8

### Validation Modes

The implementation supports two validation modes:

- **Standard validation**: Checks format correctness, required fields, label consistency
- **Strict validation**: Additionally checks that entity mentions and relation values exist in the input text (case-insensitive substring matching)

Use strict validation during dataset creation to catch annotation errors early.
