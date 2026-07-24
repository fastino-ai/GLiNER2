import math

import pytest

from gliner2.joint_ie.calibration import TemperatureCalibrator
from gliner2.joint_ie.lattice import JointScoreLattice, SpanRef


def test_lattice_sigmoid_calibration_and_deterministic_ties():
    later = SpanRef(1, 2, 2, 3, "b", 0)
    earlier = SpanRef(0, 1, 0, 1, "a", 0)
    lattice = JointScoreLattice(
        [later, earlier],
        entity_scores={"Z": [0.0, 0.0], "A": [0.0, 0.0]},
        head_scores={"works_for": [[0.0, 2.0], [0.0, 0.0]]},
        tail_scores={"works_for": [[-2.0, 0.0], [2.0, 0.0]]},
        calibrator=TemperatureCalibrator(2.0),
    )
    assert lattice.top_entities(earlier) == [("A", 0.5), ("Z", 0.5)]
    assert lattice.top_entities(k=2) == [(earlier, "A", 0.5), (earlier, "Z", 0.5)]
    assert lattice.top_heads("works_for", tail=earlier)[0] == (later, pytest.approx(0.7310585))
    assert lattice.top_tails("works_for", head=later)[0] == (earlier, pytest.approx(0.7310585))


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        TemperatureCalibrator(0)
