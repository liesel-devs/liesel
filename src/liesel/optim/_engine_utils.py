from __future__ import annotations

from ..model import Model
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
                    axis_size=child.train_axis_size,
                    batch_axis_size=None,
                    split_axes=None,
                    default_split_axis=0,
                    shuffle=False,
                    sample_size=child.train_sample_size,
                )
                for child in split.splits
            ]
        )

    return Batches(
        position_keys=split.position_keys or list(model.observed),
        axis_size=split.train_axis_size,
        batch_axis_size=None,
        split_axes=None,
        default_split_axis=0,
        shuffle=False,
        sample_size=split.train_sample_size,
    )
