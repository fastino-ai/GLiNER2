"""
Hard overfitting tests for NER, hierarchical structure, and classification.

All use deliberately non-obvious label names with batch_size > 1.
Runs inference before and after 30 training steps.
"""

from gliner2 import GLiNER2
from gliner2.training.data import InputExample, Structure, Classification
from gliner2.training.trainer import TrainingConfig, GLiNER2Trainer

# ── NER ──────────────────────────────────────────────────────────────────

NER_TYPES = ["zx9_alpha", "vibrance", "morph_class"]

def make_ner_examples():
    return [
        InputExample(
            text="The reactor core temperature reached 450 degrees at midnight near the coastal facility.",
            entities={"zx9_alpha": ["reactor core"], "vibrance": ["450 degrees"], "morph_class": ["coastal facility"]},
        ),
        InputExample(
            text="Heavy rainfall disrupted the satellite uplink during the summit conference.",
            entities={"zx9_alpha": ["satellite uplink"], "vibrance": ["Heavy rainfall"], "morph_class": ["summit conference"]},
        ),
        InputExample(
            text="Protocol seven was activated after the seismic anomaly below the northern ridge.",
            entities={"zx9_alpha": ["Protocol seven"], "vibrance": ["seismic anomaly"], "morph_class": ["northern ridge"]},
        ),
        InputExample(
            text="The cargo manifest listed twelve crates of synthetic polymer for the offshore platform.",
            entities={"zx9_alpha": ["cargo manifest"], "vibrance": ["synthetic polymer"], "morph_class": ["offshore platform"]},
        ),
    ]

def run_ner_inference(model, examples, tag=""):
    print(f"\n{'='*60}")
    print(f"  NER INFERENCE {tag}")
    print(f"{'='*60}")
    model.eval()
    hits, total = 0, 0
    for ex in examples:
        result = model.extract_entities(ex.text, NER_TYPES, include_spans=True)
        entities = result.get("entities", {})
        print(f"\nText: {ex.text}")
        for label, expected_spans in ex.entities.items():
            raw = entities.get(label, [])
            predicted = [e["text"] if isinstance(e, dict) else e for e in raw]
            for span in expected_spans:
                total += 1
                found = span in predicted
                hits += found
                print(f"  [{'HIT' if found else 'MISS'}] {label}: '{span}'  (pred: {predicted})")
    print(f"\nRecall: {hits}/{total} = {hits/total:.0%}" if total else "\nNo entities expected")
    return hits, total

# ── STRUCTURE ────────────────────────────────────────────────────────────

def make_structure_examples():
    return [
        InputExample(
            text="The reactor core temperature reached 450 degrees at midnight near the coastal facility.",
            structures=[
                Structure("qx_report", component="reactor core", reading="450 degrees", site="coastal facility"),
            ],
        ),
        InputExample(
            text="Heavy rainfall disrupted the satellite uplink during the summit conference.",
            structures=[
                Structure("qx_report", component="satellite uplink", reading="Heavy rainfall", site="summit conference"),
            ],
        ),
        InputExample(
            text="Protocol seven was activated after the seismic anomaly below the northern ridge.",
            structures=[
                Structure("qx_report", component="Protocol seven", reading="seismic anomaly", site="northern ridge"),
            ],
        ),
        InputExample(
            text="The cargo manifest listed twelve crates of synthetic polymer for the offshore platform.",
            structures=[
                Structure("qx_report", component="cargo manifest", reading="synthetic polymer", site="offshore platform"),
            ],
        ),
    ]

STRUCT_SCHEMA = {"qx_report": ["component", "reading", "site"]}

def run_structure_inference(model, examples, tag=""):
    print(f"\n{'='*60}")
    print(f"  STRUCTURE INFERENCE {tag}")
    print(f"{'='*60}")
    model.eval()
    hits, total = 0, 0
    for ex in examples:
        result = model.extract_json(ex.text, STRUCT_SCHEMA)
        print(f"\nText: {ex.text}")
        print(f"  Result: {result}")
        structs = ex.structures[0]
        for field_name, expected_val in structs._fields.items():
            total += 1
            instances = result.get("qx_report", [])
            found = False
            for inst in (instances if isinstance(instances, list) else [instances]):
                if isinstance(inst, dict):
                    pred_val = inst.get(field_name, "")
                    if isinstance(pred_val, list):
                        found = any(expected_val in v if isinstance(v, str) else expected_val == v for v in pred_val)
                    elif isinstance(pred_val, str):
                        found = expected_val in pred_val
                if found:
                    break
            hits += found
            print(f"  [{'HIT' if found else 'MISS'}] {field_name}: expected '{expected_val}'")
    print(f"\nRecall: {hits}/{total} = {hits/total:.0%}" if total else "")
    return hits, total

# ── CLASSIFICATION ───────────────────────────────────────────────────────

CLS_TASKS = {"vortex_mode": {"labels": ["flux_high", "flux_low", "flux_neutral"]}}

def make_classification_examples():
    return [
        InputExample(
            text="The reactor core temperature reached 450 degrees at midnight near the coastal facility.",
            classifications=[Classification(task="vortex_mode", labels=["flux_high", "flux_low", "flux_neutral"], true_label="flux_high")],
        ),
        InputExample(
            text="Heavy rainfall disrupted the satellite uplink during the summit conference.",
            classifications=[Classification(task="vortex_mode", labels=["flux_high", "flux_low", "flux_neutral"], true_label="flux_low")],
        ),
        InputExample(
            text="Protocol seven was activated after the seismic anomaly below the northern ridge.",
            classifications=[Classification(task="vortex_mode", labels=["flux_high", "flux_low", "flux_neutral"], true_label="flux_neutral")],
        ),
        InputExample(
            text="The cargo manifest listed twelve crates of synthetic polymer for the offshore platform.",
            classifications=[Classification(task="vortex_mode", labels=["flux_high", "flux_low", "flux_neutral"], true_label="flux_high")],
        ),
    ]

def run_cls_inference(model, examples, tag=""):
    print(f"\n{'='*60}")
    print(f"  CLASSIFICATION INFERENCE {tag}")
    print(f"{'='*60}")
    model.eval()
    hits, total = 0, 0
    for ex in examples:
        result = model.classify_text(ex.text, CLS_TASKS)
        expected = ex.classifications[0].true_label
        pred_labels = result.get("vortex_mode", [])
        if isinstance(pred_labels, dict):
            pred_labels = pred_labels.get("labels", [])
        total += 1
        found = expected in pred_labels
        hits += found
        print(f"\nText: {ex.text[:60]}...")
        print(f"  [{'HIT' if found else 'MISS'}] expected='{expected}'  pred={pred_labels}")
    print(f"\nAccuracy: {hits}/{total} = {hits/total:.0%}" if total else "")
    return hits, total

# ── MULTI-TASK COMBINATION ───────────────────────────────────────────────

COMBO_NER_TYPES = ["zx9_alpha", "morph_class"]
COMBO_CLS_TASKS = {"vortex_mode": {"labels": ["flux_high", "flux_low", "flux_neutral"]}}
COMBO_STRUCT_SCHEMA = {"qx_report": ["reading", "site"]}

def make_combo_examples():
    """Each sample has NER + classification + structure simultaneously."""
    return [
        InputExample(
            text="The reactor core temperature reached 450 degrees at midnight near the coastal facility.",
            entities={"zx9_alpha": ["reactor core"], "morph_class": ["coastal facility"]},
            classifications=[Classification(task="vortex_mode", labels=["flux_high", "flux_low", "flux_neutral"], true_label="flux_high")],
            structures=[Structure("qx_report", reading="450 degrees", site="coastal facility")],
        ),
        InputExample(
            text="Heavy rainfall disrupted the satellite uplink during the summit conference.",
            entities={"zx9_alpha": ["satellite uplink"], "morph_class": ["summit conference"]},
            classifications=[Classification(task="vortex_mode", labels=["flux_high", "flux_low", "flux_neutral"], true_label="flux_low")],
            structures=[Structure("qx_report", reading="Heavy rainfall", site="summit conference")],
        ),
        InputExample(
            text="Protocol seven was activated after the seismic anomaly below the northern ridge.",
            entities={"zx9_alpha": ["Protocol seven"], "morph_class": ["northern ridge"]},
            classifications=[Classification(task="vortex_mode", labels=["flux_high", "flux_low", "flux_neutral"], true_label="flux_neutral")],
            structures=[Structure("qx_report", reading="seismic anomaly", site="northern ridge")],
        ),
        InputExample(
            text="The cargo manifest listed twelve crates of synthetic polymer for the offshore platform.",
            entities={"zx9_alpha": ["cargo manifest"], "morph_class": ["offshore platform"]},
            classifications=[Classification(task="vortex_mode", labels=["flux_high", "flux_low", "flux_neutral"], true_label="flux_high")],
            structures=[Structure("qx_report", reading="synthetic polymer", site="offshore platform")],
        ),
    ]

def run_combo_inference(model, examples, tag=""):
    print(f"\n{'='*60}")
    print(f"  MULTI-TASK INFERENCE {tag}")
    print(f"{'='*60}")
    model.eval()
    ner_hits, ner_total = 0, 0
    cls_hits, cls_total = 0, 0
    struct_hits, struct_total = 0, 0

    for ex in examples:
        schema = model.create_schema()
        schema.entities(COMBO_NER_TYPES)
        for name, cfg in COMBO_CLS_TASKS.items():
            schema.classification(name, cfg["labels"])
        for parent, fields in COMBO_STRUCT_SCHEMA.items():
            builder = schema.structure(parent)
            for f in fields:
                builder.field(f)

        result = model.extract(ex.text, schema)
        print(f"\nText: {ex.text[:70]}...")
        print(f"  Full result: {result}")

        # Check NER
        entities = result.get("entities", {})
        for label, expected_spans in ex.entities.items():
            raw = entities.get(label, [])
            predicted = [e["text"] if isinstance(e, dict) else e for e in raw]
            for span in expected_spans:
                ner_total += 1
                found = span in predicted
                ner_hits += found
                print(f"  [{'HIT' if found else 'MISS'}] NER {label}: '{span}'  (pred: {predicted})")

        # Check classification
        expected_cls = ex.classifications[0].true_label
        pred_labels = result.get("vortex_mode", [])
        if isinstance(pred_labels, dict):
            pred_labels = pred_labels.get("labels", [])
        cls_total += 1
        cls_found = expected_cls in pred_labels
        cls_hits += cls_found
        print(f"  [{'HIT' if cls_found else 'MISS'}] CLS expected='{expected_cls}'  pred={pred_labels}")

        # Check structure
        structs = ex.structures[0]
        for field_name, expected_val in structs._fields.items():
            struct_total += 1
            instances = result.get("qx_report", [])
            found = False
            for inst in (instances if isinstance(instances, list) else [instances]):
                if isinstance(inst, dict):
                    pred_val = inst.get(field_name, "")
                    if isinstance(pred_val, list):
                        found = any(expected_val in v if isinstance(v, str) else expected_val == v for v in pred_val)
                    elif isinstance(pred_val, str):
                        found = expected_val in pred_val
                if found:
                    break
            struct_hits += found
            print(f"  [{'HIT' if found else 'MISS'}] STRUCT {field_name}: expected '{expected_val}'")

    total = ner_total + cls_total + struct_total
    hits = ner_hits + cls_hits + struct_hits
    print(f"\nNER: {ner_hits}/{ner_total}  CLS: {cls_hits}/{cls_total}  STRUCT: {struct_hits}/{struct_total}  Total: {hits}/{total}")
    return hits, total

# ── TRAIN + EVAL HARNESS ────────────────────────────────────────────────

def train_and_eval(name, model, examples, run_inference_fn, output_dir):
    print(f"\n{'#'*60}")
    print(f"#  {name}")
    print(f"{'#'*60}")

    run_inference_fn(model, examples, tag="BEFORE TRAINING")

    config = TrainingConfig(
        output_dir=output_dir,
        max_steps=30,
        batch_size=2,
        gradient_accumulation_steps=1,
        encoder_lr=2e-5,
        task_lr=5e-4,
        warmup_ratio=0.0,
        scheduler_type="constant",
        logging_steps=5,
        eval_strategy="no",
        fp16=False,
        bf16=False,
    )
    trainer = GLiNER2Trainer(model, config)
    result = trainer.train(train_data=examples)
    final_loss = result["train_metrics_history"][-1]["loss"]
    print(f"\nFinal loss: {final_loss:.6f}")

    hits, total = run_inference_fn(model, examples, tag="AFTER TRAINING (in-memory)")

    save_dir = f"{output_dir}/final"
    print(f"\nLoading saved model from {save_dir} ...")
    saved = GLiNER2.from_pretrained(save_dir)
    hits2, total2 = run_inference_fn(saved, examples, tag="AFTER TRAINING (loaded from disk)")

    return hits, total, hits2, total2, final_loss


if __name__ == "__main__":
    print("Loading base model...")
    base_model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    # 1) NER
    ner_model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    h1, t1, h1d, t1d, loss1 = train_and_eval(
        "NER OVERFIT", ner_model, make_ner_examples(),
        run_ner_inference, "/tmp/test_overfit_ner",
    )

    # 2) Structure
    struct_model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    h2, t2, h2d, t2d, loss2 = train_and_eval(
        "STRUCTURE OVERFIT", struct_model, make_structure_examples(),
        run_structure_inference, "/tmp/test_overfit_struct",
    )

    # 3) Classification
    cls_model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    h3, t3, h3d, t3d, loss3 = train_and_eval(
        "CLASSIFICATION OVERFIT", cls_model, make_classification_examples(),
        run_cls_inference, "/tmp/test_overfit_cls",
    )

    # 4) Multi-task combination
    combo_model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    h4, t4, h4d, t4d, loss4 = train_and_eval(
        "MULTI-TASK COMBO OVERFIT", combo_model, make_combo_examples(),
        run_combo_inference, "/tmp/test_overfit_combo",
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  NER:            loss={loss1:.6f}  in-mem={h1}/{t1}  disk={h1d}/{t1d}")
    print(f"  Structure:      loss={loss2:.6f}  in-mem={h2}/{t2}  disk={h2d}/{t2d}")
    print(f"  Classification: loss={loss3:.6f}  in-mem={h3}/{t3}  disk={h3d}/{t3d}")
    print(f"  Multi-task:     loss={loss4:.6f}  in-mem={h4}/{t4}  disk={h4d}/{t4d}")
    print(f"{'='*60}")
