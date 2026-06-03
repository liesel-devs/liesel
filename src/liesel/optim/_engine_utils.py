from __future__ import annotations

from .batch import Batches, BatchManager
from .split import PositionSplit, PositionSplitManager

BatchConfig = Batches | BatchManager
SplitConfig = PositionSplit | PositionSplitManager
