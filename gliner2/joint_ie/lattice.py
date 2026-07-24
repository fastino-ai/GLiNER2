"""Dense candidate spans and calibrated Joint IE score rankings."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .calibration import Calibrator, IdentityCalibrator


def _number(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def sigmoid(logit: Any) -> float:
    """Numerically stable scalar sigmoid."""
    value = _number(logit)
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True, order=True)
class SpanRef:
    """A dense token span and its document character projection.

    Token ``start`` and ``end`` use the lattice's dense token coordinates;
    character offsets are half-open and index ``text``'s source document.
    """

    start: int
    end: int
    char_start: int
    char_end: int
    text: str
    sentence_id: Optional[int] = None

    def __post_init__(self) -> None:
        if min(self.start, self.end, self.char_start, self.char_end) < 0:
            raise ValueError("span offsets must be non-negative")
        if self.end < self.start or self.char_end < self.char_start:
            raise ValueError("span end must not precede start")

    @property
    def token_start(self) -> int:
        return self.start

    @property
    def token_end(self) -> int:
        return self.end


@dataclass
class ScoreBlock:
    """A named dense logit block.

    ``scores[label][candidate]`` is intentionally container-agnostic and may be
    backed by Python sequences, NumPy arrays, or torch tensors.
    """

    scores: Mapping[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(self.scores)

    def logits(self, label: str) -> Any:
        return self.scores[label]


@dataclass
class JointScoreLattice:
    """Scores for entity candidates and directed relation endpoint roles."""

    spans: Sequence[SpanRef]
    entity_scores: Any = field(default_factory=ScoreBlock)
    head_scores: Any = field(default_factory=ScoreBlock)
    tail_scores: Any = field(default_factory=ScoreBlock)
    calibrator: Calibrator = field(default_factory=IdentityCalibrator)

    def __post_init__(self) -> None:
        self.spans = tuple(self.spans)
        self.entity_scores = self._block(self.entity_scores)
        self.head_scores = self._block(self.head_scores)
        self.tail_scores = self._block(self.tail_scores)

    @staticmethod
    def _block(value: Any) -> ScoreBlock:
        if isinstance(value, ScoreBlock):
            return value
        return ScoreBlock(value or {})

    def probability(self, logit: Any) -> float:
        return sigmoid(self.calibrator.calibrate(logit))

    def top_entities(self, span: Optional[Any] = None, k: Optional[int] = None) -> List[Tuple[Any, ...]]:
        """Rank entity scores.

        With ``span`` (index or SpanRef), returns ``(label, probability)``.
        Without it, returns ``(span, label, probability)`` across the lattice.
        Ties are resolved by span coordinates and label.
        """
        if span is not None:
            index = self._span_index(span)
            rows = [(label, self.probability(self._at(values, index)))
                    for label, values in self.entity_scores.scores.items()]
            rows.sort(key=lambda row: (-row[1], row[0]))
        else:
            rows = [(candidate, label, self.probability(self._at(values, index)))
                    for label, values in self.entity_scores.scores.items()
                    for index, candidate in enumerate(self.spans)]
            rows.sort(key=lambda row: (-row[2], row[0].start, row[0].end,
                                       row[0].char_start, row[0].char_end, row[1]))
        return rows if k is None else rows[:max(0, k)]

    def top_heads(self, relation: str, tail: Optional[Any] = None,
                  k: Optional[int] = None) -> List[Tuple[SpanRef, float]]:
        return self._top_role(self.head_scores, relation, tail, k)

    def top_tails(self, relation: str, head: Optional[Any] = None,
                  k: Optional[int] = None) -> List[Tuple[SpanRef, float]]:
        return self._top_role(self.tail_scores, relation, head, k)

    def _top_role(self, block: ScoreBlock, relation: str, other: Optional[Any],
                  k: Optional[int]) -> List[Tuple[SpanRef, float]]:
        values = block.scores[relation]
        if other is not None:
            other_index = self._span_index(other)
            # Role matrices are [candidate, opposite_endpoint]. A vector is also
            # accepted for models whose role score is endpoint-independent.
            rows = [(span, self.probability(self._matrix_at(values, i, other_index)))
                    for i, span in enumerate(self.spans)]
        else:
            rows = [(span, self.probability(self._at(values, i)))
                    for i, span in enumerate(self.spans)]
        rows.sort(key=lambda row: (-row[1], row[0].start, row[0].end,
                                   row[0].char_start, row[0].char_end, row[0].text))
        return rows if k is None else rows[:max(0, k)]

    def _span_index(self, span: Any) -> int:
        if isinstance(span, int):
            if span < 0 or span >= len(self.spans):
                raise IndexError(span)
            return span
        return self.spans.index(span)

    @staticmethod
    def _at(values: Any, index: int) -> Any:
        return values[index]

    @staticmethod
    def _matrix_at(values: Any, row: int, column: int) -> Any:
        value = values[row]
        try:
            return value[column]
        except (IndexError, TypeError):
            return value
