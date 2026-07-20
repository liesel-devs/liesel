from __future__ import annotations

from math import ceil

from .batch import Batches, BatchManager
from .split import PositionSplit, PositionSplitManager

BatchConfig = Batches | BatchManager
SplitConfig = PositionSplit | PositionSplitManager


def _progress_print_rate(total: int, progress_n_updates: int) -> int:
    return max(ceil(total / progress_n_updates), 1)


def _progress_n_updates(total: int, progress_update_every: int) -> int:
    return ceil(total / progress_update_every)


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
