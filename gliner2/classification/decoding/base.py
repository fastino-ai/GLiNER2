"""Shared decoder machinery: problem construction, search-state assignment, and
the Solution record.

Follows ``joint_ie/optimizers/base.py``'s layering but *not* its duck-typed
``_invoke`` fallback chain: this module's constraints are one known ABC with one
known method, so a direct call is correct and a signature-guessing loop would
only hide errors.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..candidates import LocalAssignment, enumerate_locals, retain, task_utilities


@dataclass
class Solution:
    assignments: dict           # task -> LocalAssignment
    score: float
    violations: tuple = ()      # tuple[Constraint, ...] (violated)
    exact: bool = True
    decoder: str = "independent"

    @property
    def feasible(self) -> bool:
        return not self.violations


class SearchAssignment:
    """Implements the Phase-2 ``Assignment`` protocol over live search state.

    Domains for undecided tasks are precomputed once per problem: ``always[t]``
    (labels in every local) and ``possible[t]`` (labels in some local), giving
    O(1) ``holds``.
    """

    def __init__(self, problem, chosen, decided):
        self._problem = problem
        self._chosen = chosen
        self._decided = frozenset(decided)

    def is_decided(self, task):
        return task in self._decided

    def selected(self, task):
        if task in self._decided:
            return self._chosen[task].labels
        return self._problem.always[task]

    def domain(self, task):
        if task in self._decided:
            return self._chosen[task].labels
        return self._problem.possible[task]

    def holds(self, task, label):
        if label in self.selected(task):
            return True
        if label in self.domain(task):
            return False if self.is_decided(task) else None
        return False

    def levels(self, task):
        index = self._problem.index_map[task]
        return frozenset(index[l] for l in self.domain(task) if l in index)

    def index(self, task, label):
        return self._problem.index_map[task][label]

    def default(self, task):
        return self._problem.schema.task(task).default


class DecodeProblem:
    def __init__(self, schema, task_order, locals_map, constraints):
        self.schema = schema
        self.task_order = tuple(task_order)
        self.locals = locals_map
        self.constraints = tuple(constraints)

        self.possible = {}
        self.always = {}
        self.index_map = {}
        for task in self.task_order:
            local_list = self.locals[task]
            union = frozenset().union(*[la.labels for la in local_list]) if local_list else frozenset()
            inter = (frozenset.intersection(*[la.labels for la in local_list])
                     if local_list else frozenset())
            self.possible[task] = union
            self.always[task] = inter
            names = schema.task(task).label_names
            self.index_map[task] = {name: i for i, name in enumerate(names)}

        self._touch_cache = {}

    def constraints_touching(self, task):
        cached = self._touch_cache.get(task)
        if cached is None:
            cached = tuple(c for c in self.constraints if task in c.references())
            self._touch_cache[task] = cached
        return cached

    def assignment(self, chosen, decided):
        return SearchAssignment(self, chosen, decided)

    def violations_of(self, chosen):
        a = SearchAssignment(self, chosen, self.task_order)
        return tuple(c for c in self.constraints if c.evaluate(a) is False)


def _fallback_locals(spec, retained, utils):
    """Last resort when enumeration is empty: a single best-effort local that at
    least respects max_labels, so the decoder always has something to return."""
    ranked = sorted(retained, key=lambda l: (-utils[l], l))
    maxl = min(spec.effective_max_labels(), len(ranked))
    count = max(spec.min_labels, 1)
    count = min(count, maxl) if maxl else 0
    chosen = frozenset(ranked[:count])
    return [LocalAssignment(spec.name, chosen, float(sum(utils[l] for l in chosen)))]


def build_problem(compiled, scores, config, *, active=None, full_retention_tasks=()):
    """Retain candidates, enumerate locals, and assemble a DecodeProblem.

    ``active`` masks which tasks are decoded/reported (score-preserving, since
    the prompt is unchanged). Constraints referencing a masked-out task are
    dropped.
    """
    order = compiled.task_order
    if active is not None:
        active_set = set(active)
        order = tuple(t for t in order if t in active_set)
    active_set = set(order)
    full_retention = set(full_retention_tasks)

    constraints = tuple(c for c in compiled.constraints if c.references() <= active_set)

    label_refs: dict = {}
    set_tasks: set = set()
    count_tasks: set = set()
    for c in constraints:
        for task, lbl in c.label_references():
            label_refs.setdefault(task, set()).add(lbl)
        set_tasks |= set(c.set_references())
        count_tasks |= set(c.count_references())

    locals_map: dict = {}
    for task in order:
        spec = compiled.task(task)
        logits = scores.tasks[task]
        rescued = label_refs.get(task, set())
        if task in full_retention:
            retained = frozenset(spec.label_names)
        else:
            retained = retain(spec, logits,
                               candidate_threshold=config.candidate_threshold,
                               cap=config.max_candidates_per_task,
                               rescued=rescued)
        if len(retained) < spec.min_labels:          # too few finite logits
            retained = frozenset(spec.label_names)
        utils = task_utilities(spec, {l: logits[l] for l in retained})
        set_coupled = task in set_tasks
        count_coupled = (task in count_tasks) and not set_coupled
        locals_ = enumerate_locals(
            spec, retained, utils,
            bound_labels=label_refs.get(task, set()),
            set_coupled=set_coupled, count_coupled=count_coupled,
        )
        if not locals_:
            locals_ = _fallback_locals(spec, retained, utils)
        locals_map[task] = locals_

    return DecodeProblem(compiled, order, locals_map, constraints)
