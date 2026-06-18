"""
Test entity extraction with include_confidence and include_spans parameters.

This test demonstrates the new API features for entity extraction.
"""

import json
from gliner2 import GLiNER2


def test_entity_extraction():
    """Test entity extraction with various output formats."""
    
    print("=" * 80)
    print("ENTITY EXTRACTION TESTS")
    print("=" * 80)
    
    print("\nLoading model: fastino/gliner2-base-v1...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print("Model loaded successfully!\n")
    
    text = "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino on September 12."
    entity_types = ["company", "person", "product", "location", "date"]
    
    print(f"\nTest Text: {text}")
    print(f"Entity Types: {entity_types}")
    print("\n" + "-" * 80)
    
    # Test 1: Basic extraction (default)
    print("\n1. BASIC EXTRACTION (text only)")
    print("-" * 40)
    result = model.extract_entities(text, entity_types)
    print(json.dumps(result, indent=2))
    
    # Test 2: With confidence scores
    print("\n2. WITH CONFIDENCE SCORES")
    print("-" * 40)
    result = model.extract_entities(text, entity_types, include_confidence=True)
    print(json.dumps(result, indent=2))
    
    # Test 3: With span positions
    print("\n3. WITH SPAN POSITIONS")
    print("-" * 40)
    result = model.extract_entities(text, entity_types, include_spans=True)
    print(json.dumps(result, indent=2))
    
    # Test 4: With both confidence and spans
    print("\n4. WITH CONFIDENCE AND SPAN POSITIONS")
    print("-" * 40)
    result = model.extract_entities(text, entity_types, include_confidence=True, include_spans=True)
    print(json.dumps(result, indent=2))
    
    # Test 5: Verify character positions
    print("\n5. VERIFY CHARACTER POSITIONS")
    print("-" * 40)
    for entity_type, entities in result["entities"].items():
        for entity in entities:
            extracted_text = text[entity["start"]:entity["end"]]
            print(f"{entity_type}: '{entity['text']}' at [{entity['start']}:{entity['end']}]")
            print(f"  Verification: text[{entity['start']}:{entity['end']}] = '{extracted_text}'")
            print(f"  Match: {extracted_text == entity['text']}")
    
    print("\n" + "=" * 80)
    print("Entity extraction tests completed!")
    print("=" * 80)


def test_batch_entity_extraction():
    """Test batch entity extraction with various output formats."""
    
    print("\n" + "=" * 80)
    print("BATCH ENTITY EXTRACTION TESTS")
    print("=" * 80)
    
    print("\nLoading model: fastino/gliner2-base-v1...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print("Model loaded successfully!\n")
    
    texts = [
        "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino.",
        "Google's Sundar Pichai spoke at the conference in Mountain View.",
        "Microsoft released Windows 11 in Redmond."
    ]
    entity_types = ["company", "person", "product", "location"]
    
    print(f"\nNumber of texts: {len(texts)}")
    print(f"Entity Types: {entity_types}")
    
    # Test with full metadata
    print("\n1. BATCH WITH CONFIDENCE AND SPANS")
    print("-" * 40)
    results = model.batch_extract_entities(
        texts, entity_types, batch_size=2, 
        include_confidence=True, include_spans=True
    )
    
    for i, (text, result) in enumerate(zip(texts, results)):
        print(f"\nText {i+1}: {text}")
        print(f"Result: {json.dumps(result, indent=2)}")
    
    print("\n" + "=" * 80)
    print("Batch entity extraction tests completed!")
    print("=" * 80)


def test_duplicate_entities_at_different_positions():
    """Same entity text at different positions should both be returned."""
    # Simulate two spans: "John" at pos 11 and "John" at pos 35
    entities = {
        "name": [
            ("John", 0.99, 11, 15),
            ("John", 0.98, 35, 39),
        ]
    }
    result = GLiNER2._format_entity_dict(None, entities, include_confidence=False)
    assert len(result["name"]) == 2, "Both occurrences of 'John' should be returned"


if __name__ == "__main__":
    test_entity_extraction()
    test_batch_entity_extraction()

