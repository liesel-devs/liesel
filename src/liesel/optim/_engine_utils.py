from __future__ import annotations

from .batch import Batches, BatchManager
from .split import PositionSplit, PositionSplitManager

BatchConfig = Batches | BatchManager
SplitConfig = PositionSplit | PositionSplitManager


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
