"""
Test: sanitization must not orphan a structure's record-metadata anchor.

`InputExample.sanitize()` drops structure fields whose values cannot be found
in the text. `record_metadata` names one of those fields as the record anchor,
and nothing in sanitize consulted it, so a validated example could emerge whose
declaration pointed at a field that no longer existed. Compiling record specs
for that example then raises:

    record 'record' declares anchor 'type' but no matching field query was
    found in the layout

which aborts training inside a DataLoader worker.

The second case is quieter and matters as much: a natural-mode structure with
no explicit anchor defaults to its FIRST declared field, so dropping that field
silently re-points the anchor at a different one.

Run:
    pytest tests/test_sanitize_record_anchor.py
"""

from gliner2.training.data import InputExample, Structure

TEXT = "Ada Lovelace wrote the first algorithm, published in 1843."


def _example(field_values, **kwargs):
    # Field values go through `_field_values`; a bare `fields=` keyword would be
    # captured by `**fields` and create a single field named "fields".
    return InputExample(
        text=TEXT,
        structures=[
            Structure(struct_name="record", _field_values=field_values, **kwargs)
        ],
    )


def test_structure_dropped_when_explicit_anchor_is_invalid():
    # "Wikispecies" does not occur in the text, so the anchor field is dropped.
    example = _example(
        {"type": "Wikispecies", "author": "Ada Lovelace"},
        mode="natural",
        anchor="type",
    )
    warnings, _ = example.sanitize()

    assert example.structures == []
    assert any("lost its anchor field" in w for w in warnings)


def test_structure_kept_when_anchor_survives():
    example = _example(
        {"author": "Ada Lovelace", "type": "Wikispecies"},
        mode="natural",
        anchor="author",
    )
    example.sanitize()

    assert len(example.structures) == 1
    assert "author" in example.structures[0]._fields
    assert "type" not in example.structures[0]._fields


def test_defaulted_anchor_is_not_silently_repointed():
    # No explicit anchor: get_record_metadata defaults to the first field.
    example = _example(
        {"type": "Wikispecies", "author": "Ada Lovelace"}, mode="natural"
    )
    assert example.structures[0].get_record_metadata()["record"]["anchor"] == "type"

    warnings, _ = example.sanitize()

    assert example.structures == []
    assert any("lost its anchor field" in w for w in warnings)


def test_metadata_and_fields_agree_after_sanitize():
    example = _example(
        {"type": "Wikispecies", "author": "Ada Lovelace"},
        mode="natural",
        anchor="type",
    )
    example.sanitize()

    for struct in example.structures:
        metadata = struct.get_record_metadata()
        if not metadata:
            continue
        anchor = metadata[struct.struct_name].get("anchor")
        assert anchor is None or anchor in struct._fields


def test_structure_without_a_mode_is_unaffected():
    # Structures with no mode compile no record spec, so the anchor rule must
    # not remove them.
    example = _example({"type": "Wikispecies", "author": "Ada Lovelace"})
    example.structures[0].mode = None
    example.sanitize()

    assert len(example.structures) == 1
    assert "author" in example.structures[0]._fields
