"""Retention, utilities, and local-assignment enumeration.

Pure functions over ``(TaskSpec, {label: logit}, constraints)``. No torch, no
model. This is where the decoder's exactness is earned: enumeration is
differentially provable against brute force.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

from ..joint_ie.candidates import center_logit, sigmoid


def task_utilities(spec, logits) -> dict:
    """Centered log-odds: positive means above the decision threshold."""
    return {l: center_logit(v / spec.temperature, spec.threshold)
            for l, v in logits.items()}


def task_probabilities(spec, logits) -> dict:
    return {l: sigmoid(v / spec.temperature) for l, v in logits.items()}


def retain(spec, logits, *, candidate_threshold, cap, rescued) -> frozenset:
    """Keep the labels the task retains unconstrained, plus every rescued label.

    Unconstrained retention is the top ``cap`` labels above the retention floor,
    or every label when fewer than ``min_labels`` clear it (an exclusive task
    whose argmax sits below the floor still needs its argmax).

    ``rescued`` = every label named in a constraint's ``label_references()`` for
    this task. Rescue only ever adds: a hard constraint can always force a
    below-threshold label, the cap can never evict a rescued label (which would
    manufacture infeasibility), and a rescued label can never displace an
    unconstrained candidate, so a constraint the unconstrained optimum already
    satisfies cannot change the decode. Direct analogue of joint IE's endpoint
    rescue.
    """
    rescued = frozenset(rescued)
    floor = spec.candidate_threshold if spec.candidate_threshold is not None else candidate_threshold
    # A non-finite (e.g. -inf) logit is impossible and is skipped entirely.
    finite = {l: v for l, v in logits.items() if math.isfinite(v)}
    probs = task_probabilities(spec, finite)
    utils = task_utilities(spec, finite)
    keep = {l for l, p in probs.items() if p >= floor}
    if len(keep) > cap:
        keep = set(sorted(keep, key=lambda l: (-utils[l], l))[:cap])
    if len(keep) < spec.min_labels:
        keep = set(finite)
    return frozenset(keep | (rescued & set(finite)))


@dataclass(frozen=True)
class LocalAssignment:
    task: str
    labels: frozenset
    utility: float

    def __post_init__(self):
        object.__setattr__(self, "labels", frozenset(self.labels))


def _utility_of(labels, utilities) -> float:
    return float(sum(utilities[l] for l in labels))


def _all_subsets(items):
    items = sorted(items)
    for size in range(len(items) + 1):
        for combo in combinations(items, size):
            yield frozenset(combo)


def _sorted_locals(locals_):
    return sorted(locals_, key=lambda la: (-la.utility, tuple(sorted(la.labels))))


def enumerate_locals(spec, retained, utilities, *, bound_labels,
                     set_coupled=False, count_coupled=False) -> list:
    """Enumerate a task's cardinality-feasible local assignments.

    - exclusive task: one local per retained label.
    - set-coupled (a set predicate reads its whole selection): every
      cardinality-feasible subset of retained labels (full enumeration, always
      exact).
    - count-coupled (a cardinality node reads its selection count): for each
      bound-subset, one local per feasible total count, completing with the
      top-utility free labels for that count.
    - otherwise: enumerate subsets of the *bound* labels and complete each with
      *free* labels greedily by descending utility (optimal, because free labels
      appear in no constraint).

    Returned sorted by descending utility; the exact decoder's early ``break``
    depends on that order.
    """
    retained = frozenset(retained)
    minl = spec.min_labels
    maxl = min(spec.effective_max_labels(), len(retained))

    if spec.is_exclusive:
        locals_ = [LocalAssignment(spec.name, frozenset({l}), utilities[l])
                   for l in retained]
        return _sorted_locals(locals_)

    # A set predicate needs the full subset lattice; it dominates count coupling.
    if set_coupled:
        locals_ = []
        for base in _all_subsets(retained):
            if minl <= len(base) <= maxl:
                locals_.append(LocalAssignment(
                    spec.name, base, _utility_of(base, utilities)))
        return _sorted_locals(locals_)

    bound = retained & frozenset(bound_labels)
    free = retained - bound
    free_ranked = sorted(free, key=lambda l: (-utilities[l], l))

    locals_ = []
    for base in _all_subsets(bound):
        if len(base) > maxl:
            continue
        if count_coupled:
            lo = max(minl, len(base))
            hi = min(maxl, len(base) + len(free))
            for count in range(lo, hi + 1):
                need = count - len(base)
                selected = frozenset(base) | frozenset(free_ranked[:need])
                if len(selected) == count:
                    locals_.append(LocalAssignment(
                        spec.name, selected, _utility_of(selected, utilities)))
            continue
        selected = set(base)
        for f in free_ranked:                       # add all positive-utility free
            if len(selected) >= maxl:
                break
            if utilities[f] > 0:
                selected.add(f)
        if len(selected) < minl:                    # top up with least-negative free
            for f in free_ranked:
                if len(selected) >= minl:
                    break
                if f not in selected:
                    selected.add(f)
        if not (minl <= len(selected) <= maxl):
            continue
        locals_.append(LocalAssignment(
            spec.name, frozenset(selected), _utility_of(selected, utilities)))
    return _sorted_locals(locals_)
