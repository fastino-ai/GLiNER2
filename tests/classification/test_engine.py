"""Phase 8 gate: the Classifier facade wires everything correctly, caches
compiles, guards fingerprints, and enforces the one-config-class discipline.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gliner2.classification import constraints as C
from gliner2.classification.engine import ClassificationConfig, Classifier
from gliner2.classification.errors import InfeasibleError, SchemaError
from gliner2.classification.schema import ClassificationSchema


TASKS = [("sentiment", ["pos", "neu", "neg"])]
LOGITS = {"sentiment": {"pos": 2.0, "neu": 0.0, "neg": -1.0}}


def _schema():
    return ClassificationSchema().single("sentiment", ["pos", "neu", "neg"])


# ---- T-E1 : end-to-end classify ----------------------------------------

def test_classify_end_to_end(planted):
    clf = Classifier(planted(TASKS, LOGITS))
    result = clf.classify("great movie", _schema())
    assert result.value("sentiment") == "pos"
    assert result.feasible


def test_batch_classify(planted):
    clf = Classifier(planted(TASKS, LOGITS))
    results = clf.batch_classify(["a", "b", "c"], _schema())
    assert len(results) == 3
    assert all(r.value("sentiment") == "pos" for r in results)


# ---- T-E2 : compile cache ----------------------------------------------

def test_compile_cache_reuses_by_fingerprint(planted):
    clf = Classifier(planted(TASKS, LOGITS))
    a = clf.compile_schema(_schema())
    b = clf.compile_schema(_schema())  # distinct builder, same fingerprint
    assert a is b
    assert len(clf._compile_cache) == 1


def test_compile_cache_is_bounded(planted):
    clf = Classifier(planted(TASKS, LOGITS))
    from gliner2.classification.engine import _CACHE_CAP
    for i in range(_CACHE_CAP + 20):
        clf.compile_schema(ClassificationSchema().single(f"t{i}", ["a", "b"]))
    assert len(clf._compile_cache) <= _CACHE_CAP


# ---- T-E3 : score/decode split; fingerprint guard ----------------------

def test_score_then_decode(planted):
    clf = Classifier(planted(TASKS, LOGITS))
    scores = clf.score("x", _schema())
    result = clf.decode(scores, _schema())
    assert result.value("sentiment") == "pos"


def test_decode_rejects_fingerprint_mismatch(planted):
    clf = Classifier(planted(TASKS, LOGITS))
    scores = clf.score("x", _schema())
    other = ClassificationSchema().single("sentiment", ["pos", "neu", "neg", "mixed"])
    with pytest.raises(SchemaError, match="fingerprint"):
        clf.decode(scores, other)


# ---- T-E4 : active masking is score-preserving --------------------------

def test_active_masking_preserves_scores(planted):
    tasks = [("a", ["x", "y"]), ("b", ["p", "q"])]
    logits = {"a": {"x": 2.0, "y": -1.0}, "b": {"p": 1.5, "q": -1.0}}
    schema = (ClassificationSchema()
              .single("a", ["x", "y"]).single("b", ["p", "q"]))
    clf = Classifier(planted(tasks, logits))
    scores = clf.score("t", schema)
    full = clf.decode(scores, schema)
    masked = clf.decode(scores, schema, active=["a"])
    assert set(masked.tasks) == {"a"}
    # same scores drive both: the reported label for 'a' is identical.
    assert masked.value("a") == full.value("a")


# ---- T-E5 : from_pretrained rejects prediction options ------------------

def test_from_pretrained_rejects_prediction_kwargs():
    with pytest.raises(TypeError, match="ClassificationConfig"):
        Classifier.from_pretrained("repo", beam_size=4)


# ---- T-E6 : config validation ------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"decoder": "nonsense"},
    {"on_infeasible": "nope"},
    {"exact_node_budget": 0},
    {"beam_size": 0},
    {"candidate_threshold": 1.5},
    {"max_candidates_per_task": 0},
    {"batch_size": 0},
    {"max_len": 0},
])
def test_config_range_checks(kwargs):
    with pytest.raises(ValueError):
        ClassificationConfig(**kwargs)


# ---- T-E7 : lifecycle ---------------------------------------------------

def test_to_and_eval(planted):
    clf = Classifier(planted(TASKS, LOGITS))
    assert clf.eval() is clf
    assert clf.to(device="cpu") is clf
    assert clf.device is not None


# ---- T-E8 : constraint enforced through the facade ----------------------

def test_facade_enforces_cross_task_constraint(planted):
    tasks = [("intent", ["read", "delete"]),
             ("effects", ["read_only", "delete"])]
    logits = {"intent": {"read": -1.0, "delete": 3.0},
              "effects": {"read_only": 2.0, "delete": -2.0}}
    schema = (ClassificationSchema()
              .single("intent", ["read", "delete"])
              .multi("effects", ["read_only", "delete"], min_labels=1)
              .constrain(C.implies(("intent", "delete"), ("effects", "delete"))))
    clf = Classifier(planted(tasks, logits))
    result = clf.classify("rm file", schema, config=ClassificationConfig(decoder="exact"))
    assert result.feasible
    assert result.value("intent") == "delete"
    assert "delete" in result.selected("effects")


# ---- T-E9 : on_infeasible=raise propagates -----------------------------

def test_on_infeasible_raise(planted):
    tasks = [("a", ["x", "other"]), ("b", ["p", "q"])]
    logits = {"a": {"x": 1.0, "other": 0.0}, "b": {"p": 1.0, "q": 0.5}}
    schema = (ClassificationSchema()
              .single("a", ["x", "other"])
              .single("b", ["p", "q"])
              .constrain(C.label("a", "x"),
                         C.implies(("a", "x"), ("b", "p")),
                         C.implies(("a", "x"), ("b", "q"))))
    clf = Classifier(planted(tasks, logits))
    with pytest.raises(InfeasibleError):
        clf.classify("t", schema, config=ClassificationConfig(on_infeasible="raise"))


def test_on_infeasible_min_violations(planted):
    tasks = [("a", ["x", "other"]), ("b", ["p", "q"])]
    logits = {"a": {"x": 1.0, "other": 0.0}, "b": {"p": 1.0, "q": 0.5}}
    schema = (ClassificationSchema()
              .single("a", ["x", "other"])
              .single("b", ["p", "q"])
              .constrain(C.label("a", "x"),
                         C.implies(("a", "x"), ("b", "p")),
                         C.implies(("a", "x"), ("b", "q"))))
    clf = Classifier(planted(tasks, logits))
    result = clf.classify("t", schema,
                          config=ClassificationConfig(on_infeasible="min_violations"))
    assert not result.feasible
    assert result.violations


# ---- T-E10 : budget exhaustion with a beam dead end stays typed ---------

@pytest.mark.parametrize("on_infeasible", ["raise", "relax", "min_violations"])
def test_budget_exhausted_beam_dead_end_never_raises_keyerror(planted, on_infeasible):
    tasks = [("a", ["x", "y"]), ("b", ["p", "q"])]
    logits = {"a": {"x": 3.0, "y": 0.0}, "b": {"p": 2.0, "q": 1.0}}
    schema = (ClassificationSchema()
              .single("a", ["x", "y"])
              .single("b", ["p", "q"])
              .constrain(C.implies(("a", "x"), ("b", "p")),
                         C.implies(("a", "x"), ("b", "q"))))
    clf = Classifier(planted(tasks, logits))
    config = ClassificationConfig(decoder="exact", candidate_threshold=0.0,
                                  exact_node_budget=1, beam_size=1,
                                  on_infeasible=on_infeasible)
    if on_infeasible == "raise":
        with pytest.raises(InfeasibleError):
            clf.classify("t", schema, config=config)
        return
    result = clf.classify("t", schema, config=config)
    assert not result.feasible
    assert result.exact is False


# ---- T-R6 : __all__ exact ----------------------------------------------

def test_public_all_is_exact():
    import gliner2.classification as classification
    assert set(classification.__all__) == {
        "Classifier", "ClassificationSchema", "ClassificationConfig",
        "ClassificationResult", "ClassificationScores", "SchemaError",
        "InfeasibleError", "compile_schema", "constraints",
    }
    assert len(classification.__all__) == 9


# ---- T-R7 : exactly one Config class -----------------------------------

def test_only_one_config_class_in_module():
    root = Path(__file__).parents[2] / "gliner2" / "classification"
    names = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        names.extend(node.name for node in ast.walk(tree)
                     if isinstance(node, ast.ClassDef) and node.name.endswith("Config"))
    assert names == ["ClassificationConfig"]


# ---- T-E9 : ClassificationConfig.calibrator is applied ------------------

def test_calibrator_is_applied_at_decode(planted):
    from gliner2.joint_ie.calibration import Calibrator, IdentityCalibrator

    class Negate(Calibrator):
        def calibrate(self, logits):
            return -logits

    clf = Classifier(planted(TASKS, LOGITS))
    base = clf.classify("x", _schema())
    identity = clf.classify("x", _schema(),
                            config=ClassificationConfig(calibrator=IdentityCalibrator()))
    negated = clf.classify("x", _schema(),
                           config=ClassificationConfig(calibrator=Negate()))

    assert identity.to_dict() == base.to_dict()
    assert base.value("sentiment") == "pos"
    assert negated.value("sentiment") == "neg"
    assert negated.tasks["sentiment"].probabilities["neg"] > base.tasks["sentiment"].probabilities["neg"]


def test_calibrator_applies_to_batch_and_long(planted):
    from gliner2.joint_ie.calibration import TemperatureCalibrator

    clf = Classifier(planted(TASKS, LOGITS))
    cfg = ClassificationConfig(calibrator=TemperatureCalibrator(2.0))
    plain = clf.classify("x", _schema())
    single = clf.classify("x", _schema(), config=cfg)
    batched = clf.batch_classify(["x"], _schema(), config=cfg)[0]
    long = clf.classify_long("x", _schema(), config=cfg)

    p = plain.tasks["sentiment"].probabilities["pos"]
    assert single.tasks["sentiment"].probabilities["pos"] < p  # flatter after T=2
    assert batched.to_dict() == single.to_dict()
    assert long.to_dict() == single.to_dict()
