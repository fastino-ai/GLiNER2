"""Phase 5 gate: local enumeration is differentially proven optimal against
brute force, including the count-coupled case. Pure logic, no torch.
"""
from __future__ import annotations

import math
import random
from itertools import combinations

import pytest

from gliner2.classification.candidates import (
    LocalAssignment,
    enumerate_locals,
    retain,
    task_utilities,
)
from gliner2.classification.schema import LabelSpec, TaskSpec


def _spec(name, labels, **kw):
    return TaskSpec(name, tuple(LabelSpec(l) for l in labels), **kw)


def _all_subsets(items):
    items = sorted(items)
    for size in range(len(items) + 1):
        for combo in combinations(items, size):
            yield frozenset(combo)


# ---- T-U1 : centering inert for exclusive, decisive for cardinality ----

def test_centering_inert_for_exclusive_tasks():
    logits = {"a": 1.4, "b": 0.2, "c": -0.7}
    lo = _spec("t", ["a", "b", "c"], min_labels=1, max_labels=1, threshold=0.3)
    hi = _spec("t", ["a", "b", "c"], min_labels=1, max_labels=1, threshold=0.8)
    ulo = task_utilities(lo, logits)
    uhi = task_utilities(hi, logits)
    # every label re-scored by the same constant -> order preserved
    deltas = {l: ulo[l] - uhi[l] for l in logits}
    assert math.isclose(min(deltas.values()), max(deltas.values()), rel_tol=1e-9)
    assert (sorted(ulo, key=ulo.get) == sorted(uhi, key=uhi.get))


def test_centering_decisive_for_variable_cardinality():
    logits = {"a": 1.4, "b": 0.2, "c": -0.7, "d": 0.9}
    low = _spec("t", list(logits), min_labels=0, max_labels=4, threshold=0.3)
    high = _spec("t", list(logits), min_labels=0, max_labels=4, threshold=0.8)
    low_sel = enumerate_locals(low, set(logits), task_utilities(low, logits),
                               bound_labels=frozenset())[0].labels
    high_sel = enumerate_locals(high, set(logits), task_utilities(high, logits),
                                bound_labels=frozenset())[0].labels
    assert high_sel <= low_sel  # raising threshold strictly shrinks (subset)
    assert high_sel != low_sel


# ---- T-D5 : rescue ------------------------------------------------------

def test_rescued_label_retained_below_threshold():
    spec = _spec("t", ["a", "b", "c"], threshold=0.5)
    logits = {"a": 5.0, "b": -5.0, "c": -5.0}  # b, c far below
    kept = retain(spec, logits, candidate_threshold=0.1, cap=64, rescued={"b"})
    assert "b" in kept        # rescued despite tiny probability
    assert "c" not in kept


def test_cap_never_evicts_rescued():
    spec = _spec("t", ["a", "b", "c", "d"], threshold=0.5)
    logits = {"a": 5.0, "b": 4.0, "c": -5.0, "d": -6.0}
    kept = retain(spec, logits, candidate_threshold=0.1, cap=1, rescued={"c", "d"})
    assert {"c", "d"} <= kept  # cap smaller than rescued set, still kept


@pytest.mark.parametrize("rescued", [{"c"}, {"a", "c"}, {"a", "b", "c"}])
def test_rescue_only_adds_to_the_unconstrained_retention(rescued):
    # Exclusive task, nothing clears the floor: the unconstrained set falls back
    # to every label, and a rescued label must not suppress that fallback.
    spec = _spec("t", ["a", "b", "c"], min_labels=1, max_labels=1, threshold=0.5)
    logits = {"a": -0.04, "b": -1.9, "c": -3.9}
    free = retain(spec, logits, candidate_threshold=0.5, cap=64, rescued=set())
    kept = retain(spec, logits, candidate_threshold=0.5, cap=64, rescued=rescued)
    assert free == frozenset({"a", "b", "c"})
    assert kept == free


def test_cap_never_evicts_an_unconstrained_candidate_for_a_rescued_one():
    spec = _spec("t", ["a", "b", "c", "d"], threshold=0.5)
    logits = {"a": 5.0, "b": 4.0, "c": -5.0, "d": -6.0}
    free = retain(spec, logits, candidate_threshold=0.1, cap=1, rescued=set())
    kept = retain(spec, logits, candidate_threshold=0.1, cap=1, rescued={"c"})
    assert free == frozenset({"a"})
    assert kept == frozenset({"a", "c"})


# ---- T-D5b : separate thresholds; -inf skipped -------------------------

def test_infinite_logit_is_skipped():
    spec = _spec("t", ["a", "b"], threshold=0.5)
    logits = {"a": 2.0, "b": -math.inf}
    kept = retain(spec, logits, candidate_threshold=0.05, cap=64, rescued={"b"})
    assert kept == frozenset({"a"})  # -inf skipped even when rescued


def test_candidate_threshold_default_vs_task_override():
    spec = _spec("t", ["a", "b"], threshold=0.5, candidate_threshold=0.9)
    logits = {"a": 0.5, "b": -0.5}  # sigmoid ~0.62, ~0.38: below 0.9 floor
    kept = retain(spec, logits, candidate_threshold=0.05, cap=64, rescued=set())
    assert kept == frozenset()  # per-task floor wins over the config default


# ---- T-D2 : differential vs brute force --------------------------------

def _brute_best_by_bound(retained, bound, utils, minl, maxl):
    best = {}
    for s in _all_subsets(retained):
        if not (minl <= len(s) <= maxl):
            continue
        key = frozenset(s) & bound
        u = sum(utils[l] for l in s)
        if key not in best or u > best[key] + 1e-12:
            best[key] = u
    return best


def test_enumeration_matches_brute_force_per_bound_subset():
    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(2, 5)
        labels = [f"l{i}" for i in range(n)]
        retained = frozenset(labels)
        bound = frozenset(l for l in labels if rng.random() < 0.5)
        utils = {l: round(rng.uniform(-2, 2), 3) for l in labels}
        minl = rng.randint(0, n)
        maxl = rng.randint(minl, n)
        if minl == 1 and maxl == 1:
            maxl = min(n, 2)  # avoid the exclusive path for this test
        spec = _spec("t", labels, min_labels=minl, max_labels=maxl)

        locals_ = enumerate_locals(spec, retained, utils, bound_labels=bound)
        mine = {}
        for la in locals_:
            key = la.labels & bound
            mine[key] = max(mine.get(key, -math.inf), la.utility)

        brute = _brute_best_by_bound(retained, bound, utils, minl, maxl)
        assert set(mine) == set(brute)
        for key in brute:
            assert math.isclose(mine[key], brute[key], rel_tol=1e-9, abs_tol=1e-9)


# ---- T-D2b : count-coupled ---------------------------------------------

def test_count_coupled_one_local_per_count():
    labels = ["a", "b", "c", "d"]
    spec = _spec("t", labels, min_labels=0, max_labels=4)
    utils = {"a": 1.0, "b": 0.5, "c": -0.5, "d": -1.0}
    locals_ = enumerate_locals(spec, set(labels), utils,
                               bound_labels=frozenset(), count_coupled=True)
    counts = sorted(len(la.labels) for la in locals_)
    assert counts == [0, 1, 2, 3, 4]  # exactly one local per feasible count
    # a selection of size >= 2 exists, so a conditional at_least(2) is satisfiable
    assert any(len(la.labels) >= 2 for la in locals_)
    # the size-2 local picks the top-utility free labels
    two = next(la for la in locals_ if len(la.labels) == 2)
    assert two.labels == frozenset({"a", "b"})


def test_set_coupled_full_enumeration():
    labels = ["a", "b", "c"]
    spec = _spec("t", labels, min_labels=0, max_labels=3, default="c")
    utils = {"a": 1.0, "b": -1.0, "c": 0.0}
    locals_ = enumerate_locals(spec, set(labels), utils,
                               bound_labels=frozenset(), set_coupled=True)
    # min_labels forced to 1 by default, so subsets of size 1..3
    selections = {la.labels for la in locals_}
    assert frozenset({"a"}) in selections
    assert frozenset({"a", "b", "c"}) in selections
    assert frozenset() not in selections  # min_labels=1


# ---- T-N1 / T-N2 -------------------------------------------------------

def test_min_labels_top_up_picks_least_negative():
    labels = ["a", "b", "c"]
    spec = _spec("t", labels, min_labels=1, max_labels=3)
    utils = {"a": -0.1, "b": -0.9, "c": -0.5}  # all negative, must add one
    locals_ = enumerate_locals(spec, set(labels), utils, bound_labels=frozenset())
    top = locals_[0]
    assert top.labels == frozenset({"a"})  # least-negative, not first alphabetically


def test_enumeration_deterministic_under_ties():
    labels = ["a", "b", "c"]
    spec = _spec("t", labels, min_labels=1, max_labels=1)
    utils = {"a": 1.0, "b": 1.0, "c": 0.0}
    first = enumerate_locals(spec, set(labels), utils, bound_labels=frozenset())
    second = enumerate_locals(spec, set(labels), utils, bound_labels=frozenset())
    assert [la.labels for la in first] == [la.labels for la in second]
    assert first[0].labels == frozenset({"a"})  # tie broken by label name
