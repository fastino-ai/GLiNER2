import json
from loguru import logger
import typer

from gliner2 import GLiNER2

_result_n = 0


def print_result(text: str, result: dict):
    global _result_n
    _result_n += 1
    print(f"----- Result {_result_n} -----")
    assert isinstance(result, dict), f"Result must be a dict, got {type(result)}"
    print(f"text={text}")
    print(f"result={json.dumps(result, indent=2)}")
    print("")
    # logger.info(f"{type(result)=}")


def run_gliner2(extractor: GLiNER2, text: str):
    threshold = 0.3
    format_results = False
    include_confidence = True

    # schema = extractor.create_schema().entities(entity_types, dtype="list")
    # results = extractor.extract(text, schema, threshold=threshold, format_results=format_results, include_confidence=include_confidence)
    # result = extractor.extract_entities(text=text, entity_types=entity_types, threshold=threshold, format_results=format_results, include_confidence=include_confidence)
    # print_result(text, result)

    # Align the inference schema with the training JSONL (`output.entities` keys)
    structures = {
        "address": [
            # Core street number information
            "number_first::str::Primary street number (e.g. 12, 101)",
            "number_last::str::Last street number in a range if present (e.g. 14 in 12-14)",
            "number_first_suffix::str::Suffix for the primary street number (e.g. A, B)",
            # Unit / flat information
            "flat_type::str::Type of unit/flat (e.g. unit, apt)",
            "flat_number::str::Unit or flat number (e.g. 5, 22)",
            "flat_number_suffix::str::Suffix for the unit/flat number if present",
            # Street details
            'street_name::str::Street name (e.g. "warringah", "henley")',
            "street_type_code::str::Street type (e.g. street, road, avenue, lane, highway, drive, court, close, circuit, boulevard, gardens, crescent, place, way, turn, rise)",
            # Building / level information
            'building_name::str::Building or complex name (e.g. "northgate plaza", "figtree gardens")',
            "level_type::str::Type of level (e.g. level, floor)",
            "level_number::str::Level number when present (e.g. 10, 2)",
            # Locality and region
            'locality_name::str::Suburb or locality name (e.g. "maroubra", "camira")',
            "state_abbreviation::str::Australian state abbreviation (e.g. nsw, vic, qld, sa, tas, act)",
            "postcode::str::Australian 4-digit postcode",
        ],
    }
    result = extractor.extract_json(
        text=text,
        structures=structures,
        threshold=threshold,
        format_results=format_results,
        include_confidence=include_confidence,
    )
    print_result(text, result)


def main(model_id: str):
    print("------------------ run_gliner2 ------------------")

    logger.info(f"{model_id=}")
    extractor = GLiNER2.from_pretrained(model_id)

    texts = [
        "48a Pirrama Rd Pyrmont NSW 2009",
        "6/341 George Str, Sydney NSW 2000",
        "108-110 Kippax Lane, Surry Hills NSW 2010",
        "L41, Tower Two, 200 Barangaroo Ave, Sydney NSW 2000",
        "Unit 18/14-18 Flood St, Bondi, NSW 2026",
        "Aptt 16, 400 Bondi Rd, Bondi NSW 2026",
        "Lvl 15/333 George St Sydney NSW 2000",
        "Shop 4/11-31 York St, Sydney NSW 2000",
        "Floor 3, 152-156 Clarence St, Sydney NSW 2000",
    ]

    for text in texts:
        run_gliner2(extractor, text)

    logger.info(f"{model_id=}")


if __name__ == "__main__":
    typer.run(main)
