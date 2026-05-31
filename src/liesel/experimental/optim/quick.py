from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import optax

from ...model import Model
from ._engine_utils import BatchConfig, SplitConfig, _full_data_batches_for_split
from .batch import Batches, BatchManager
from .loss import Loss, NegLogProbLoss
from .optimizer import LBFGS, Optimizer
from .split import PositionSplit, PositionSplitManager
from .stop import Stopper
from .util import guess_n

if TYPE_CHECKING:
    from .engine import OptimEngine


class QuickOptim:
    """
    Enables quick optimizer setup by providing strong defaults.

    1. Default loss: Unnormalized negative log posterior
    2. Default batching: None
    3. Default stopping: ``Stopper(epochs=1000, patience=10, rtol=1e-6)``
    4. Default Train/Validation/Test split: None (use all data for training)
    5. Default optimizer: Adam(learning_rate=1e-3), jointly optimizing all parameter
       variables in the model.
    """

    def __init__(
        self,
        model: Model,
        loss: Loss | None = None,
        batches: BatchConfig | None = None,
        split: SplitConfig | None = None,
        optimizers: Sequence[Optimizer] | Literal["adam", "lbfgs"] = "adam",
        stopper: Stopper = Stopper(epochs=1000, patience=10, rtol=1e-6),
        seed: int | None = None,
        n: int | None = None,
    ) -> None:
        self.model = model
        self._n = n
        self._loss = loss
        self.stopper = stopper
        self._batches = batches
        self.seed = int(time.time()) if seed is None else seed
        self._split = split
        self._optimizers = optimizers

    @property
    def n(self) -> int:
        if self._n is not None:
            return self._n

        return guess_n(self.model)

    @property
    def batches(self) -> BatchConfig:
        if self._batches is not None:
            if isinstance(self.split, PositionSplitManager) and isinstance(
                self._batches, Batches
            ):
                raise ValueError(
                    "QuickOptim requires a BatchManager when used with a "
                    "PositionSplitManager. Pass batches=None for full-data batches "
                    "or provide a matching BatchManager."
                )

            if isinstance(self._batches, BatchManager):
                return self._batches

            self._batches.n = self.split.n_train
            self._batches.indices = jnp.arange(self.split.n_train)
            return self._batches

        return _full_data_batches_for_split(self.model, self.split)

    @property
    def split(self) -> SplitConfig:
        if self._split:
            return self._split

        return PositionSplit.from_model(self.model, n=self._n)

    @property
    def optimizers(self) -> Sequence[Optimizer]:
        if isinstance(self._optimizers, str):
            match self._optimizers:
                case "lbfgs":
                    return [LBFGS(list(self.model.parameters))]
                case "adam":
                    opt = Optimizer(
                        list(self.model.parameters),
                        optimizer=optax.adam(learning_rate=1e-3),
                    )
                    return [opt]

        return self._optimizers

    @property
    def loss(self) -> Loss:
        if self._loss is not None:
            return self._loss

        return NegLogProbLoss(self.model, self.split)

    def build_engine(self) -> OptimEngine:
        from .engine import OptimEngine

        engine = OptimEngine(
            loss=self.loss,
            batches=self.batches,
            optimizers=self.optimizers,
            stopper=self.stopper,
            initial_state=self.model.state,
            seed=self.seed,
        )
        return engine
