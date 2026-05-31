from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import optax

from ...model import Model
from ._engine_utils import BatchConfig, SplitConfig, _full_data_batches_for_split
from .batch import Batches
from .optimizer import LBFGS, Optimizer
from .split import PositionSplit, PositionSplitManager
from .stop import Stopper
from .util import guess_n
from .vi import Elbo

if TYPE_CHECKING:
    from .engine import OptimEngine


class LieselVI:
    """
    Enables quick optimizer setup by providing strong defaults.
    """

    def __init__(
        self,
        model: Model,
        elbo: Literal["mvn_diag", "mvn_tril", "mvn_blocked"] | Elbo = "mvn_diag",
        batches: BatchConfig | None = None,
        split: SplitConfig | None = None,
        optimizers: Sequence[Optimizer] | Literal["adam", "lbfgs"] = "adam",
        stopper: Stopper = Stopper(epochs=1000, patience=10, rtol=1e-6),
        seed: int | None = None,
        n: int | None = None,
    ) -> None:
        self.model = model
        self._n = n
        self.stopper = stopper
        self.seed = int(time.time()) if seed is None else seed
        self.split = split or PositionSplit.from_model(model)
        if isinstance(self.split, PositionSplitManager) and isinstance(
            batches, Batches
        ):
            raise ValueError(
                "LieselVI requires a BatchManager when used with a "
                "PositionSplitManager. Pass batches=None for full-data batches or "
                "provide a matching BatchManager."
            )
        self.batches = batches or _full_data_batches_for_split(model, self.split)
        self._optimizers = optimizers
        if isinstance(elbo, str):
            match elbo:
                case "mvn_diag":
                    self.elbo = Elbo.mvn_diag(model, self.split)
                case "mvn_tril":
                    self.elbo = Elbo.mvn_tril(model, self.split)
                case "mvn_blocked":
                    self.elbo = Elbo.mvn_blocked(model, self.split)
        else:
            self.elbo = elbo

    @property
    def n(self) -> int:
        if self._n is not None:
            return self._n

        return guess_n(self.model)

    @property
    def optimizers(self) -> Sequence[Optimizer]:
        if isinstance(self._optimizers, str):
            match self._optimizers:
                case "lbfgs":
                    return [LBFGS(list(self.elbo.q.parameters))]
                case "adam":
                    opt = Optimizer(
                        list(self.elbo.q.parameters),
                        optimizer=optax.adam(learning_rate=1e-3),
                    )
                    return [opt]

        return self._optimizers

    def build_engine(self) -> OptimEngine:
        from .engine import OptimEngine

        engine = OptimEngine(
            loss=self.elbo,
            batches=self.batches,
            optimizers=self.optimizers,
            stopper=self.stopper,
            initial_state=self.model.state,
            seed=self.seed,
        )
        return engine
