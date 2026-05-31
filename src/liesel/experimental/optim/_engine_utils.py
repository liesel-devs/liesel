from __future__ import annotations

from ...model import Model
from .batch import Batches, BatchManager
from .split import PositionSplit, PositionSplitManager

BatchConfig = Batches | BatchManager
SplitConfig = PositionSplit | PositionSplitManager


def _full_data_batches_for_split(model: Model, split: SplitConfig) -> BatchConfig:
    if isinstance(split, PositionSplitManager):
        return BatchManager(
            [
                Batches(
                    position_keys=child.position_keys,
                    n=child.n_train,
                    batch_size=None,
                    axes=None,
                    default_axis=0,
                    shuffle=False,
                )
                for child in split.splits
            ]
        )

    return Batches(
        position_keys=split.position_keys or list(model.observed),
        n=split.n_train,
        batch_size=None,
        axes=None,
        default_axis=0,
        shuffle=False,
    )
