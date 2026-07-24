"""Logit calibration used by Joint IE score lattices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class Calibrator(ABC):
    """Interface for transformations applied to logits before sigmoid."""

    @abstractmethod
    def calibrate(self, logits: Any) -> Any:
        """Return calibrated logits, preserving scalar/container shape."""

    def __call__(self, logits: Any) -> Any:
        return self.calibrate(logits)


@dataclass(frozen=True)
class IdentityCalibrator(Calibrator):
    def calibrate(self, logits: Any) -> Any:
        return logits


@dataclass(frozen=True)
class TemperatureCalibrator(Calibrator):
    """Divide logits by a positive temperature."""

    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")

    def calibrate(self, logits: Any) -> Any:
        try:
            return logits / self.temperature
        except TypeError:
            if isinstance(logits, tuple):
                return tuple(self.calibrate(value) for value in logits)
            if isinstance(logits, list):
                return [self.calibrate(value) for value in logits]
            if isinstance(logits, dict):
                return {key: self.calibrate(value) for key, value in logits.items()}
            raise
