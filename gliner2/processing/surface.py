"""Torch-free character span values shared by data validation and targets."""
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SurfaceSpan:
    """A surface pinned to a character span of the input text.

    Offsets index the original text, ``end`` exclusive.
    """

    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise ValueError(
                f"SurfaceSpan offsets must be integers, got "
                f"[{self.start!r}, {self.end!r})"
            )
        if self.start < 0:
            raise ValueError(
                f"SurfaceSpan start must be non-negative, got {self.start}"
            )
        if self.end <= self.start:
            raise ValueError(
                f"SurfaceSpan requires end > start, got [{self.start}, {self.end})"
            )

    @classmethod
    def from_mapping(cls, value: Mapping) -> "SurfaceSpan":
        missing = {"text", "start", "end"} - value.keys()
        if missing:
            raise ValueError(f"gold span is missing {sorted(missing)}")
        return cls(value["text"], value["start"], value["end"])

