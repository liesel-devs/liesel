"""Opinionated optimization setup for Liesel models."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import optax

from ..model import Model
from ._engine_utils import BatchConfig, SplitConfig
from .batch import Batches, BatchManager
from .loss import Loss, NegLogProbLoss
from .optimizer import LBFGS, Optimizer
from .split import PositionSplit, PositionSplitManager
from .stop import Stopper

if TYPE_CHECKING:
    from .engine import OptimEngine
    from .state import OptimResult


class LieselOptim:
    """
    Builds an :class:`.OptimEngine` for a Liesel model using sensible defaults.

    ``LieselOptim`` is the quick-start wrapper for regular model optimization. It
    creates a negative log-posterior loss, full-data or mini-batch training data,
    and a single optimizer over all model parameters unless these pieces are supplied
    explicitly. Call :meth:`build_engine` to inspect or modify the low-level engine
    before fitting, or :meth:`fit` for the direct path.

    Parameters
    ----------
    model
        Liesel model to optimize.
    loss
        Optional custom loss. If supplied, ``loss.split`` is used as the split.
    batches
        Optional explicit batch configuration. Cannot be combined with
        ``batch_axis_size``.
    batch_axis_size
        Mini-batch size used to construct default batches. ``None`` means full-data
        batches.
    split
        Optional split. If omitted and ``loss`` is not supplied, all observed data is
        used for training. Multi-size observed data automatically uses
        :class:`.PositionSplitManager`.
    optimizers
        Either explicit optimizers or one of ``"adam"`` and ``"lbfgs"``.
    stopper
        Maximum-epoch and early-stopping configuration.
    seed
        Integer seed. If ``None``, the current Unix time is used.
    axis_size
        Optional scalar observation count for scalar default splitting.
    split_axes
        Optional mapping from observed position key to split/batch axis.
    default_split_axis
        Split/batch axis for observed keys missing from ``split_axes``.
    shuffle_batches
        Whether default mini-batches should shuffle observations.
    batch_mode
        Mode used when default batches require a :class:`.BatchManager`.
    epoch_size
        Joint epoch length used by default :class:`.BatchManager` objects in
        ``mode="resample"``.
    validation_strategy
        Validation strategy passed to :class:`.NegLogProbLoss` when ``loss`` is not
        supplied.
    scale_loss
        Whether the default :class:`.NegLogProbLoss` should divide losses by the
        training sample size. ``"auto"`` scales the internally constructed loss.
        This setting has no effect when ``loss`` is supplied.
    train_monitor
        Training-data monitor used by :class:`.OptimEngine` when no validation split
        exists. The default ``"auto"`` uses exact full-data monitoring when batches
        are full-data and ``"weighted_epoch_average"`` for mini-batch runs.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> from liesel.optim import LieselOptim
    >>> loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    >>> y = lsl.Var.new_obs(
    ...     jnp.array([0.0, 1.0]),
    ...     lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
    ...     name="y",
    ... )
    >>> model = lsl.Model([y])
    >>> engine = LieselOptim(model, seed=1).build_engine()
    >>> type(engine).__name__
    'OptimEngine'
    """

    def __init__(
        self,
        model: Model,
        *,
        loss: Loss | None = None,
        batches: BatchConfig | None = None,
        batch_axis_size: int | None = None,
        split: SplitConfig | None = None,
        optimizers: Sequence[Optimizer] | Literal["adam", "lbfgs"] = "adam",
        stopper: Stopper = Stopper(epochs=1000, patience=10, rtol=1e-6),
        seed: int | None = None,
        axis_size: int | None = None,
        split_axes: dict[str, int] | None = None,
        default_split_axis: int = 0,
        shuffle_batches: bool = True,
        batch_mode: Literal["strict", "resample"] = "resample",
        epoch_size: Literal["max", "min"] | int = "max",
        validation_strategy: Literal["log_lik", "log_prob"] = "log_lik",
        scale_loss: bool | Literal["auto"] = "auto",
        train_monitor: Literal[
            "auto", "epoch_average", "weighted_epoch_average", "full_data"
        ] = "auto",
    ) -> None:
        if batches is not None and batch_axis_size is not None:
            raise ValueError("Pass either batches or batch_axis_size, not both.")

        self.model = model
        self.seed = int(time.time()) if seed is None else seed
        self.stopper = stopper
        self.split = self._resolve_split(
            loss, split, axis_size, split_axes, default_split_axis
        )
        self.loss = self._resolve_loss(
            loss, validation_strategy=validation_strategy, scale_loss=scale_loss
        )
        self.batches = self._resolve_batches(
            batches=batches,
            batch_axis_size=batch_axis_size,
            split_axes=split_axes,
            default_split_axis=default_split_axis,
            shuffle=shuffle_batches,
            mode=batch_mode,
            epoch_size=epoch_size,
        )
        self.optimizers = self._resolve_optimizers(optimizers)
        self.train_monitor = train_monitor

    def _resolve_split(
        self,
        loss: Loss | None,
        split: SplitConfig | None,
        axis_size: int | None,
        split_axes: dict[str, int] | None,
        default_split_axis: int,
    ) -> SplitConfig:
        if loss is not None:
            if split is not None and split is not loss.split:
                raise ValueError(
                    "When both loss and split are provided, split must be loss.split."
                )

            return loss.split

        if split is not None:
            return split

        return PositionSplit.from_model(
            self.model,
            axis_size=axis_size,
            split_axes=split_axes,
            default_split_axis=default_split_axis,
            multi_size="manager",
        )

    def _resolve_loss(
        self,
        loss: Loss | None,
        validation_strategy: Literal["log_lik", "log_prob"],
        scale_loss: bool | Literal["auto"],
    ) -> Loss:
        if loss is not None:
            return loss

        if scale_loss == "auto":
            scale = True
        elif isinstance(scale_loss, bool):
            scale = scale_loss
        else:
            raise ValueError("scale_loss must be True, False, or 'auto'.")

        return NegLogProbLoss(
            self.model,
            self.split,
            validation_strategy=validation_strategy,
            scale=scale,
        )

    def _resolve_batches(
        self,
        batches: BatchConfig | None,
        batch_axis_size: int | None,
        split_axes: dict[str, int] | None,
        default_split_axis: int,
        shuffle: bool,
        mode: Literal["strict", "resample"],
        epoch_size: Literal["max", "min"] | int,
    ) -> BatchConfig:
        if batches is not None:
            return batches

        shuffle = False if batch_axis_size is None else shuffle
        if isinstance(self.split, PositionSplitManager):
            children = [
                Batches(
                    position_keys=child.position_keys,
                    axis_size=child.train_axis_size,
                    batch_axis_size=batch_axis_size,
                    shuffle=shuffle,
                    split_axes=split_axes,
                    default_split_axis=default_split_axis,
                    sample_size=child.train_sample_size,
                    sample_with_replacement=(
                        mode == "resample"
                        and batch_axis_size is not None
                        and batch_axis_size > child.train_axis_size
                    ),
                )
                for child in self.split.splits
            ]
            return BatchManager(children, mode=mode, epoch_size=epoch_size)

        position_keys = self.split.position_keys or list(self.model.observed)
        return Batches(
            position_keys=position_keys,
            axis_size=self.split.train_axis_size,
            batch_axis_size=batch_axis_size,
            shuffle=shuffle,
            split_axes=split_axes,
            default_split_axis=default_split_axis,
            sample_size=self.split.train_sample_size,
        )

    def _resolve_optimizers(
        self, optimizers: Sequence[Optimizer] | Literal["adam", "lbfgs"]
    ) -> Sequence[Optimizer]:
        if not isinstance(optimizers, str):
            return optimizers

        position_keys = list(self.model.parameters)
        match optimizers:
            case "adam":
                return [
                    Optimizer(
                        position_keys,
                        optimizer=optax.adam(learning_rate=1e-3),
                    )
                ]
            case "lbfgs":
                return [LBFGS(position_keys)]
            case _:
                raise ValueError("optimizers must be 'adam', 'lbfgs', or a sequence.")

    def build_engine(self) -> OptimEngine:
        """
        Builds the low-level optimization engine.

        Returns
        -------
        OptimEngine
            Configured engine. Users may modify engine attributes before calling
            :meth:`OptimEngine.fit`.
        """
        from .engine import OptimEngine

        return OptimEngine(
            loss=self.loss,
            batches=self.batches,
            optimizers=self.optimizers,
            stopper=self.stopper,
            initial_state=self.model.state,
            seed=self.seed,
            train_monitor=self.train_monitor,
        )

    def fit(self) -> OptimResult:
        """
        Builds an engine and runs optimization immediately.

        Returns
        -------
        OptimResult
            Result returned by :meth:`OptimEngine.fit`.
        """
        return self.build_engine().fit()
