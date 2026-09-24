from __future__ import annotations

from collections.abc import Sequence

from .batch import Batches, BatchManager
from .optimizer import LBFGS, OptimizerLike
from .split import PositionSplit, PositionSplitManager

BatchConfig = Batches | BatchManager
SplitConfig = PositionSplit | PositionSplitManager


def _validate_optimizer_batches(
    optimizers: Sequence[OptimizerLike], batches: BatchConfig
) -> None:
    if not batches.is_full_data and any(isinstance(opt, LBFGS) for opt in optimizers):
        raise ValueError(
            "LBFGS requires full-data batches and a deterministic objective; "
            "configure full-data batches or use another optimizer."
        )


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
