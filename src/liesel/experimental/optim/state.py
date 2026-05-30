"""State containers and history helpers for experimental optimizers.

This module mostly supports :class:`.OptimEngine` internally. The most useful
user-facing pieces are :class:`OptimResult`, returned by optimizer runs, and
:class:`OptimHistory`, which converts recorded losses and positions into tidy
``pandas`` data frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import optax
import pandas as pd
import plotnine as p9

from liesel.goose.types import ModelState

from ...goose.pytree import register_dataclass_as_pytree
from .batch import Batches, BatchManager
from .optimizer import Optimizer
from .types import Position

Array = Any
BatchConfig = Batches | BatchManager


def position_df(
    position: Position, subset: Sequence[str] | None = None
) -> pd.DataFrame:
    """
    Converts a position history into a tidy ``pandas.DataFrame``.

    ``position`` is expected to contain arrays whose leading dimension indexes
    epochs. Remaining dimensions are flattened into numbered columns. One-dimensional
    entries keep their original name.

    Parameters
    ----------
    position
        Position history with one leading epoch dimension per entry.
    subset
        Optional sequence of position names to keep.

    Returns
    -------
    pandas.DataFrame
        Data frame with an ``"epoch"`` column and one or more columns per position
        entry.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim.state import position_df
    >>> from liesel.experimental.optim.types import Position
    >>> history = Position(
    ...     {
    ...         "theta": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    ...         "sigma": jnp.array([5.0, 7.0]),
    ...     }
    ... )
    >>> df = position_df(history)
    >>> df[["theta0", "sigma"]].to_dict("list")
    {'theta0': [1.0, 3.0], 'sigma': [5.0, 7.0]}
    >>> position_df(history, subset=["sigma"]).columns.tolist()
    ['epoch', 'sigma']
    """
    data: dict[str, Array] = dict()
    for name, value in position.items():
        if subset and name not in subset:
            continue
        pdim = int(jnp.prod(jnp.array(value.shape[1:])))
        hdim = value.shape[0]
        value = jnp.reshape(value, (hdim, pdim))
        data |= array_to_dict(value.squeeze(), names_prefix=name)

    df = pd.DataFrame(data)
    df = df.reset_index(names="epoch")

    return df.astype(float)


@register_dataclass_as_pytree
@dataclass
class OptimHistory:
    """
    Stores loss values and optional position histories for optimizer runs.

    ``OptimHistory`` is allocated before an optimization starts. Loss arrays are
    initialized with ``jnp.inf``. Position histories, when requested, are initialized
    with zeros and updated epoch by epoch by :class:`.OptimEngine`.

    Parameters
    ----------
    loss_train
        Training loss history with shape ``(epochs,)``.
    loss_validate
        Validation loss history with shape ``(epochs,)``.
    position
        Optional parameter position history. Each array has a leading epoch
        dimension.
    tracked
        Optional history for additional tracked quantities.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim.state import OptimHistory
    >>> from liesel.experimental.optim.types import Position
    >>> position = Position({"theta": jnp.array([1.0, 2.0])})
    >>> history = OptimHistory.from_epochs(epochs=3, position=position, tracked=None)
    >>> history.loss_train.shape
    (3,)
    >>> history.position["theta"].shape
    (3, 2)
    >>> history
    OptimHistory(len=3)
    """

    loss_train: jax.Array
    loss_validate: jax.Array
    position: Position | None
    tracked: Position | None

    @classmethod
    def from_epochs(
        cls, epochs: int, position: Position | None, tracked: Position | None
    ) -> OptimHistory:
        """
        Allocates an empty optimizer history for a fixed number of epochs.

        Parameters
        ----------
        epochs
            Number of epochs to allocate.
        position
            Initial position used to infer the shape of the stored parameter history.
            If ``None``, no parameter history is allocated.
        tracked
            Initial tracked position used to infer the shape of tracked history. If
            ``None``, no tracked history is allocated.

        Returns
        -------
        OptimHistory
            Initialized history object.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim.state import OptimHistory
        >>> from liesel.experimental.optim.types import Position
        >>> history = OptimHistory.from_epochs(
        ...     2, Position({"theta": jnp.array(1.0)}), tracked=None
        ... )
        >>> history.loss_validate.tolist()
        [inf, inf]
        >>> history.position["theta"].tolist()
        [0.0, 0.0]
        """
        tracked_init = (
            cls.init_position_history(tracked, epochs) if tracked is not None else None
        )

        position_init = (
            cls.init_position_history(position, epochs)
            if position is not None
            else None
        )

        inst = cls(
            loss_train=jnp.full((epochs,), fill_value=jnp.inf),
            loss_validate=jnp.full((epochs,), fill_value=jnp.inf),
            position=position_init,
            tracked=tracked_init,
        )
        return inst

    def loss_df(self) -> pd.DataFrame:
        """
        Converts training and validation losses into a ``pandas.DataFrame``.

        Returns
        -------
        pandas.DataFrame
            Data frame with ``"epoch"``, ``"loss_train"``, and
            ``"loss_validate"`` columns.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim.state import OptimHistory
        >>> history = OptimHistory.from_epochs(epochs=2, position=None, tracked=None)
        >>> history.loss_train = history.loss_train.at[0].set(1.5)
        >>> history.loss_validate = history.loss_validate.at[0].set(2.5)
        >>> history.loss_df().iloc[0].to_dict()
        {'epoch': 0.0, 'loss_train': 1.5, 'loss_validate': 2.5}
        """
        data: dict[str, Array] = dict()
        data |= array_to_dict(self.loss_train, names_prefix="loss_train")
        data |= array_to_dict(self.loss_validate, names_prefix="loss_validate")

        df = pd.DataFrame(data)
        df = df.reset_index(names="epoch")

        return df.astype(float)

    def position_df(self, subset: Sequence[str] | None = None) -> pd.DataFrame:
        """
        Converts the saved parameter position history into a data frame.

        Parameters
        ----------
        subset
            Optional sequence of parameter names to keep.

        Raises
        ------
        TypeError
            If position history was not saved.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim.state import OptimHistory
        >>> from liesel.experimental.optim.types import Position
        >>> history = OptimHistory.from_epochs(
        ...     2, Position({"theta": jnp.array([1.0, 2.0])}), tracked=None
        ... )
        >>> history.position_df().columns.tolist()
        ['epoch', 'theta0', 'theta1']
        >>> no_position = OptimHistory.from_epochs(2, position=None, tracked=None)
        >>> try:
        ...     no_position.position_df()
        ... except TypeError as error:
        ...     print("not saved" in str(error))
        True
        """
        if self.position is None:
            raise TypeError(
                "'position' is None. Probably the position history was not saved."
            )
        return position_df(self.position, subset)

    def tracked_df(self, subset: Sequence[str] | None = None) -> pd.DataFrame:
        """
        Converts tracked quantities into a data frame.

        Parameters
        ----------
        subset
            Optional sequence of tracked quantity names to keep.

        Raises
        ------
        ValueError
            If no tracked history is available.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim.state import OptimHistory
        >>> from liesel.experimental.optim.types import Position
        >>> tracked = Position({"mean": jnp.array(0.0)})
        >>> history = OptimHistory.from_epochs(epochs=2, position=None, tracked=tracked)
        >>> history.tracked = OptimHistory.update_position_history(
        ...     0, history.tracked, Position({"mean": jnp.array(1.5)})
        ... )
        >>> history.tracked_df().iloc[0].to_dict()
        {'epoch': 0.0, 'mean': 1.5}
        """
        if self.tracked is None:
            raise ValueError(f"{self.tracked=}")

        data: dict[str, Array] = dict()
        for name, value in self.tracked.items():
            if subset and name not in subset:
                continue
            data |= array_to_dict(value.squeeze(), names_prefix=name)

        df = pd.DataFrame(data)
        df = df.reset_index(names="epoch")

        return df.astype(float)

    @staticmethod
    def init_position_history(position: Position, epochs: int) -> Position:
        """
        Allocates zero-filled arrays for a position history.

        The returned arrays have one extra leading epoch dimension.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim.state import OptimHistory
        >>> from liesel.experimental.optim.types import Position
        >>> position = Position({"theta": jnp.ones((2,))})
        >>> history = OptimHistory.init_position_history(position, epochs=3)
        >>> history["theta"].shape
        (3, 2)
        >>> history["theta"].sum()
        Array(0., dtype=float32)
        """
        # initialize arrays of zeros
        pos = {
            name: jnp.zeros((epochs,) + jnp.shape(value))
            for name, value in position.items()
        }
        # fill in initial values
        # pos = jax.tree.map(lambda d, pos: d.at[0].set(pos), pos, position)
        return Position(pos)

    @staticmethod
    def update_position_history(
        i: int, position_history: Position, position: Position
    ) -> Position:
        """
        Writes a position into a position history at one epoch index.

        Parameters
        ----------
        i
            Epoch index to update.
        position_history
            History created by :meth:`init_position_history`.
        position
            Position values to store at epoch ``i``.

        Returns
        -------
        Position
            Updated position history.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim.state import OptimHistory
        >>> from liesel.experimental.optim.types import Position
        >>> position = Position({"theta": jnp.array([1.0, 2.0])})
        >>> history = OptimHistory.init_position_history(position, epochs=2)
        >>> updated = OptimHistory.update_position_history(1, history, position)
        >>> updated["theta"].tolist()
        [[0.0, 0.0], [1.0, 2.0]]
        """
        pos_history = jax.tree.map(
            lambda d, pos: d.at[i].set(pos), position_history, position
        )
        return pos_history

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name}(len={len(self.loss_train)})"


def array_to_dict(
    x: Array, names_prefix: str = "x", prefix_1d: bool = False
) -> dict[str, Array]:
    """
    Converts a one- or two-dimensional array into named columns.

    Parameters
    ----------
    x
        One- or two-dimensional array-like object.
    names_prefix
        Prefix used for generated names.
    prefix_1d
        If ``True``, one-dimensional input receives a trailing ``"0"`` in its
        column name.

    Returns
    -------
    dict
        Mapping from generated column names to arrays.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim.state import array_to_dict
    >>> array_to_dict(jnp.array([1.0, 2.0]), names_prefix="loss")
    {'loss': Array([1., 2.], dtype=float32)}
    >>> array_to_dict(jnp.array([[1.0, 2.0], [3.0, 4.0]]), names_prefix="theta")
    {'theta0': Array([1., 3.], dtype=float32), 'theta1': Array([2., 4.], dtype=float32)}
    >>> array_to_dict(jnp.array([1.0, 2.0]), names_prefix="x", prefix_1d=True)
    {'x0': Array([1., 2.], dtype=float32)}
    """

    if isinstance(x, float) or x.ndim == 1:
        if prefix_1d:
            return {f"{names_prefix}0": x}
        else:
            return {names_prefix: x}
    elif x.ndim == 2:
        return {f"{names_prefix}{i}": x[:, i] for i in range(x.shape[-1])}
    else:
        raise ValueError(f"x should have ndim <= 2, but it has x.ndim={x.ndim}")


@register_dataclass_as_pytree
@dataclass
class OptimCarry:
    """
    Mutable optimizer loop state used by :class:`.OptimEngine`.

    ``OptimCarry`` is passed through the epoch and mini-batch loops. It contains the
    current parameter position, model state, optimizer states, batch configuration,
    current losses, best-so-far values, and preallocated history.

    Most users do not need to construct this class directly. Use
    :meth:`OptimCarry.new` when testing custom engine logic.

    Parameters
    ----------
    key
        Current JAX pseudo-random key.
    position
        Current parameter position.
    tracked
        Optional tracked quantities for diagnostics.
    history
        Preallocated optimizer history.
    batches
        Batch configuration used by the optimizer.
    optimizer_states
        Optax states keyed by optimizer identifier.
    model_state
        Current model state.
    batch
        Current mini-batch position.
    fixed_position
        Non-optimized position entries.
    best_position
        Best parameter position found so far.
    best_loss
        Best validation loss found so far.
    loss_train
        Most recent training loss.
    loss_validate
        Most recent validation loss.
    epoch
        Current epoch index.
    i_batch
        Current mini-batch index within the epoch.
    """

    key: jax.Array  # random number key

    position: Position  # parameter position (estimation targets)
    tracked: Position | None  # recorded position (for diagnosis)

    history: OptimHistory
    batches: BatchConfig

    optimizer_states: dict[str, optax.OptState]
    model_state: ModelState

    batch: Position = field(default_factory=lambda: Position({}))
    fixed_position: Position = field(default_factory=lambda: Position({}))
    best_position: Position = field(default_factory=lambda: Position({}))
    best_loss: jax.Array | float = jnp.inf

    loss_train: jax.Array | float = jnp.inf
    loss_validate: jax.Array | float = jnp.inf

    epoch: int = 0  # outer while-loop index over epochs
    i_batch: int = 0  # inner for-loop index over batches

    @classmethod
    def new(
        cls,
        key: jax.Array,
        epochs: int,
        position: Position,
        tracked: Position | None,
        batches: BatchConfig,
        optimizers: Sequence[Optimizer],
        model_state: ModelState,
        save_position_history: bool,
    ) -> OptimCarry:
        """
        Creates an initialized optimizer carry.

        Optimizer states are initialized from ``position``. The history stores the
        parameter position only when ``save_position_history`` is ``True``.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import optax
        >>> from liesel.experimental.optim import Batches, Optimizer
        >>> from liesel.experimental.optim.state import OptimCarry
        >>> from liesel.experimental.optim.types import Position
        >>> position = Position({"theta": jnp.array(0.0)})
        >>> carry = OptimCarry.new(
        ...     key=jax.random.key(0),
        ...     epochs=2,
        ...     position=position,
        ...     tracked=None,
        ...     batches=Batches(["y"], n=4, batch_size=2),
        ...     optimizers=[Optimizer(["theta"], optax.sgd(0.1))],
        ...     model_state={},
        ...     save_position_history=True,
        ... )
        >>> carry.optimizer_states.keys()
        dict_keys([''])
        >>> carry.history.position["theta"].shape
        (2,)
        >>> carry.best_position == position
        True
        """
        opt_states = {opt.identifier: opt.init(position) for opt in optimizers}
        if save_position_history:
            history = OptimHistory.from_epochs(epochs, position, tracked)
        else:
            history = OptimHistory.from_epochs(epochs, None, tracked)

        inst = cls(
            key=key,
            position=position,
            tracked=tracked,
            history=history,
            batches=batches,
            optimizer_states=opt_states,
            model_state=model_state,
            best_position=position,
        )
        return inst

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name}(epoch={self.epoch}, batch={self.i_batch})"


@dataclass
class OptimResult:
    """
    Result returned by an optimizer run.

    ``OptimResult`` bundles the processed history, the best parameter position, and
    small metadata about the run. It also provides convenience plotting methods for
    losses and saved parameter histories.

    Parameters
    ----------
    history
        Processed optimizer history.
    final_epoch
        Last epoch index included in the processed history.
    best_position
        Best parameter position selected by the optimizer.
    best_epoch
        Epoch at which ``best_position`` was found.
    duration
        Wall-clock runtime in seconds.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim.state import OptimHistory, OptimResult
    >>> from liesel.experimental.optim.types import Position
    >>> history = OptimHistory.from_epochs(epochs=2, position=None, tracked=None)
    >>> result = OptimResult(
    ...     history=history,
    ...     final_epoch=1,
    ...     best_position=Position({"theta": jnp.array(1.0)}),
    ...     best_epoch=0,
    ...     duration=0.25,
    ... )
    >>> result
    OptimResult(final_epoch=1, best_epoch=0, duration=0.2s)
    """

    history: OptimHistory

    final_epoch: int
    best_position: Position
    best_epoch: int
    duration: float

    def plot_loss(
        self, legend: bool = True, title: str | None = None, window: int | None = None
    ):
        """
        Plots training and validation loss histories.

        Parameters
        ----------
        legend
            Whether to show the plot legend.
        title
            Optional plot title.
        window
            Optional number of final epochs to show. If ``None``, all epochs are
            shown.

        Returns
        -------
        plotnine.ggplot
            Plot object with training and validation loss curves. A vertical line
            marks :attr:`best_epoch` when it is inside the displayed window.
        """
        history = self.history.loss_df()
        n_iter = history.shape[0]
        if window is None:
            window = n_iter

        i = n_iter - window
        history = history.iloc[i:, :]

        plot_data = history[["loss_validate", "loss_train", "epoch"]]
        plot_data.rename(
            columns={
                "loss_validate": "Validation",
                "loss_train": "Training",
                "epoch": "Epoch",
            },
            inplace=True,
        )

        plot_data = plot_data.melt(
            id_vars="Epoch", var_name="Loss Type", value_name="Loss"
        )

        p = (
            p9.ggplot(plot_data)
            + p9.aes(x="Epoch", y="Loss", color="Loss Type", linetype="Loss Type")
            + p9.geom_line()
        )

        if self.best_epoch >= i:
            p = p + p9.geom_vline(xintercept=self.best_epoch)

        if title is not None:
            p += p9.ggtitle(title)

        if not legend:
            p += p9.theme(legend_position="none")

        return p

    def plot_params(
        self,
        position: Position | None = None,
        legend: bool = True,
        title: str | None = None,
        subset: Sequence[str] | None = None,
        window: int | None = None,
    ):
        """
        Plots saved parameter histories.

        Parameters
        ----------
        position
            Optional position history to plot. If omitted,
            ``self.history.position`` is used.
        legend
            Whether to show the plot legend.
        title
            Optional plot title.
        subset
            Optional sequence of parameter names to keep.
        window
            Optional number of final epochs to show. If ``None``, all epochs are
            shown.

        Returns
        -------
        plotnine.ggplot
            Plot object with one curve per flattened parameter column.

        Raises
        ------
        TypeError
            If no position history is available.
        """
        position = position or self.history.position
        if position is None:
            raise TypeError(
                "'position' is None and cannot be plotted. "
                "Probably the position history was not saved."
            )
        history = position_df(position, subset)
        n_iter = history.shape[0]
        if window is None:
            window = n_iter

        i = n_iter - window
        history = history.iloc[i:, :]

        plot_data = history.melt(
            id_vars="epoch", var_name="Parameter", value_name="Value"
        )
        plot_data.rename(columns={"epoch": "Epoch"}, inplace=True)

        p = (
            p9.ggplot(plot_data)
            + p9.aes(
                x="Epoch",
                y="Value",
                color="Parameter",
                group="Parameter",
            )
            + p9.geom_line()
        )
        if self.best_epoch >= i:
            p = p + p9.geom_vline(xintercept=self.best_epoch)

        if title is not None:
            p += p9.ggtitle(title)

        if not legend:
            p += p9.theme(legend_position="none")

        return p

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(final_epoch={self.final_epoch}, best_epoch={self.best_epoch}, "
            f"duration={self.duration:.1f}s)"
        )
        return out
