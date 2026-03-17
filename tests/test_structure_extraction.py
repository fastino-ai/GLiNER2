"""
Test structure extraction with include_confidence and include_spans parameters.

This test demonstrates the new API features for structured data extraction.
"""

import json
from gliner2 import GLiNER2


def test_structure_extraction():
    """Test structure extraction with various output formats."""

    print("=" * 80)
    print("STRUCTURE EXTRACTION TESTS")
    print("=" * 80)

    print("\nLoading model: fastino/gliner2-base-v1...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print("Model loaded successfully!\n")

    text = "Apple announced a new iPhone 15 Pro Max at $1099 during their September event in Cupertino."

    print(f"\nTest Text: {text}")
    print("\n" + "-" * 80)

    # Define schema
    schema = model.create_schema()
    schema.structure("product_announcement").field("company").field("product").field(
        "price"
    ).field("date").field("location")

    print("\nSchema: product_announcement")
    print("  Fields: company, product, price, date, location")

    # Test 1: Basic extraction (default)
    print("\n1. BASIC EXTRACTION (text only)")
    print("-" * 40)
    result = model.extract(text, schema)
    print(json.dumps(result, indent=2))

    # Test 2: With confidence scores
    print("\n2. WITH CONFIDENCE SCORES")
    print("-" * 40)
    result = model.extract(text, schema, include_confidence=True)
    print(json.dumps(result, indent=2))

    # Test 3: With span positions
    print("\n3. WITH SPAN POSITIONS")
    print("-" * 40)
    result = model.extract(text, schema, include_spans=True)
    print(json.dumps(result, indent=2))

    # Test 4: With both confidence and spans
    print("\n4. WITH CONFIDENCE AND SPAN POSITIONS")
    print("-" * 40)
    result = model.extract(text, schema, include_confidence=True, include_spans=True)
    print(json.dumps(result, indent=2))

    # Test 5: Verify character positions
    print("\n5. VERIFY CHARACTER POSITIONS")
    print("-" * 40)
    for struct in result["product_announcement"]:
        for field_name, field_values in struct.items():
            print(f"\n{field_name}:")
            for value in field_values:
                extracted_text = text[value["start"] : value["end"]]
                print(f"  '{value['text']}' at [{value['start']}:{value['end']}]")
                print(
                    f"  Verification: '{extracted_text}' - Match: {extracted_text == value['text']}"
                )

    print("\n" + "=" * 80)
    print("Structure extraction tests completed!")
    print("=" * 80)


def test_structure_with_single_values():
    """Test structure extraction with single-value fields (dtype='str')."""

    print("\n" + "=" * 80)
    print("STRUCTURE EXTRACTION WITH SINGLE VALUES")
    print("=" * 80)

    print("\nLoading model: fastino/gliner2-base-v1...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print("Model loaded successfully!\n")

    text = "Apple announced iPhone 15 at $999 on September 12."

    print(f"\nTest Text: {text}")
    print("\n" + "-" * 80)

    # Define schema with dtype='str' for single values
    schema = model.create_schema()
    schema.structure("product_info").field("company", dtype="str").field(
        "product", dtype="str"
    ).field("price", dtype="str").field("date", dtype="str")

    print("\nSchema: product_info (all fields dtype='str')")
    print("  Fields: company, product, price, date")

    # Test with all flags
    print("\n1. WITH CONFIDENCE AND SPAN POSITIONS")
    print("-" * 40)
    result = model.extract(text, schema, include_confidence=True, include_spans=True)
    print(json.dumps(result, indent=2))

    # Test 2: Only spans (no confidence)
    print("\n2. WITH SPAN POSITIONS ONLY")
    print("-" * 40)
    result = model.extract(text, schema, include_spans=True)
    print(json.dumps(result, indent=2))

    # Test 3: Basic (no metadata)
    print("\n3. BASIC (text only)")
    print("-" * 40)
    result = model.extract(text, schema)
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 80)
    print("Single-value structure tests completed!")
    print("=" * 80)


def test_batch_structure_extraction():
    """Test batch structure extraction."""

    print("\n" + "=" * 80)
    print("BATCH STRUCTURE EXTRACTION TESTS")
    print("=" * 80)

    print("\nLoading model: fastino/gliner2-base-v1...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print("Model loaded successfully!\n")

    texts = [
        "Apple announced iPhone 15 at $999 on September 12.",
        "Google released Pixel 8 for $699 in October.",
        "Microsoft launched Surface Pro 9 at $1299.",
    ]

    print(f"\nNumber of texts: {len(texts)}")

    schema = model.create_schema()
    schema.structure("product_launch").field("company").field("product").field(
        "price"
    ).field("date")

    print("\n1. BATCH WITH CONFIDENCE AND SPANS")
    print("-" * 40)
    results = model.batch_extract(
        texts, schema, batch_size=2, include_confidence=True, include_spans=True
    )

    for i, (text, result) in enumerate(zip(texts, results)):
        print(f"\nText {i+1}: {text}")
        print(f"Result: {json.dumps(result, indent=2)}")

    print("\n" + "=" * 80)
    print("Batch structure extraction tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_structure_extraction()
    test_structure_with_single_values()
    test_batch_structure_extraction()
