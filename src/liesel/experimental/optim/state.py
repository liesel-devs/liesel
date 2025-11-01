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
from .batch import Batches
from .optimizer import Optimizer
from .types import Position

Array = Any


def position_df(
    position: Position, subset: Sequence[str] | None = None
) -> pd.DataFrame:
    data: dict[str, Array] = dict()
    for name, value in position.items():
        if subset and name not in subset:
            continue
        pdim = int(jnp.prod(jnp.array(value.shape[1:])))
        hdim = value.shape[0]
        value = jnp.reshape(value, (hdim, pdim))
        data |= array_to_dict(value.squeeze(), names_prefix=name)

    df = pd.DataFrame(data)
    df = df.reset_index(names="iteration")

    return df.astype(float)


@register_dataclass_as_pytree
@dataclass
class OptimHistory:
    loss_train: jax.Array
    loss_validation: jax.Array
    position: dict[str, jax.Array]
    tracked: dict[str, jax.Array] | None

    @classmethod
    def new(
        cls, niter: int, position: Position, tracked: Position | None
    ) -> OptimHistory:
        tracked_init = (
            cls.init_position_history(tracked, niter) if tracked is not None else None
        )

        inst = cls(
            loss_train=jnp.full((niter,), fill_value=jnp.inf),
            loss_validation=jnp.full((niter,), fill_value=jnp.inf),
            position=cls.init_position_history(position, niter),
            tracked=tracked_init,
        )
        return inst

    def loss_df(self) -> pd.DataFrame:
        """
        Turns a :attr:`.OptimResult.history` dictionary into a ``pandas.DataFrame``.
        """
        data: dict[str, Array] = dict()
        data |= array_to_dict(self.loss_train, names_prefix="loss_train")
        data |= array_to_dict(self.loss_validation, names_prefix="loss_validation")

        df = pd.DataFrame(data)
        df = df.reset_index(names="iteration")

        return df.astype(float)

    def position_df(self, subset: Sequence[str] | None = None) -> pd.DataFrame:
        return position_df(self.position, subset)

    def tracked_df(self, subset: Sequence[str] | None = None) -> pd.DataFrame:
        if self.tracked is None:
            raise ValueError(f"{self.tracked=}")

        data: dict[str, Array] = dict()
        for name, value in self.tracked.items():
            if subset and name not in subset:
                continue
            data |= array_to_dict(value.squeeze(), names_prefix=name)

        df = pd.DataFrame(data)
        df = df.reset_index(names="iteration")

        return df.astype(float)

    @staticmethod
    def init_position_history(position: Position, niter: int) -> dict[str, jax.Array]:
        # initialize arrays of zeros
        pos = {
            name: jnp.zeros((niter,) + jnp.shape(value))
            for name, value in position.items()
        }
        # fill in initial values
        # pos = jax.tree.map(lambda d, pos: d.at[0].set(pos), pos, position)
        return pos

    @staticmethod
    def update_position_history(
        i: int, position_history: dict[str, jax.Array], position: Position
    ) -> dict[str, jax.Array]:
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
    """Turns a 2d-array into a dict."""

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
    key: jax.Array  # random number key

    position: Position  # parameter position (estimation targets)
    tracked: Position  # recorded position (for diagnosis)

    history: OptimHistory
    batches: Batches

    optimizer_states: list[optax.OptState]
    model_state: ModelState

    batch: Position = field(default_factory=lambda: Position({}))
    fixed_position: Position = field(default_factory=lambda: Position({}))

    loss_train: jax.Array | float = jnp.inf
    loss_validation: jax.Array | float = jnp.inf

    i_it: int = 0  # outer while loop index over iterations
    i_batch: int = 0  # inner for loop index over batches

    @classmethod
    def new(
        cls,
        key: jax.Array,
        niter: int,
        position: Position,
        tracked: Position | None,
        batches: Batches,
        optimizers: Sequence[Optimizer],
        model_state: ModelState,
    ) -> OptimCarry:
        opt_states = {opt.identifier: opt.init(position) for opt in optimizers}
        inst = cls(
            key=key,
            position=position,
            tracked=tracked,
            history=OptimHistory.new(niter, position, tracked),
            batches=batches,
            optimizer_states=opt_states,
            model_state=model_state,
        )
        return inst

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name}(it={self.i_it}, batch={self.i_batch})"


@dataclass
class OptimResult:
    history: OptimHistory

    final_it: int
    best_position: Position
    best_it: int
    duration: float

    def plot_loss(
        self, legend: bool = True, title: str | None = None, window: int | None = None
    ):
        history = self.history.loss_df()
        n_iter = history.shape[0]
        if window is None:
            window = n_iter

        i = n_iter - window
        history = history.iloc[i:, :]

        plot_data = history[["loss_validation", "loss_train", "iteration"]]

        plot_data = plot_data.melt(
            id_vars="iteration", var_name="loss_type", value_name="loss"
        )

        p = (
            p9.ggplot(plot_data)
            + p9.aes(x="iteration", y="loss", color="loss_type", linetype="loss_type")
            + p9.geom_line()
        )

        if self.best_it >= i:
            p = p + p9.geom_vline(xintercept=self.best_it)

        if title is not None:
            p += p9.ggtitle(title)

        if not legend:
            p += p9.theme(legend_position="none")

        return p

    def plot_param_history(
        self,
        position: Position | None = None,
        legend: bool = True,
        title: str | None = None,
        subset: Sequence[str] | None = None,
        window: int | None = None,
    ):
        position = position or self.history.position
        history = position_df(position, subset)
        n_iter = history.shape[0]
        if window is None:
            window = n_iter

        i = n_iter - window
        history = history.iloc[i:, :]

        plot_data = history.melt(id_vars="iteration")

        p = (
            p9.ggplot(plot_data)
            + p9.aes(
                x="iteration",
                y="value",
                color="variable",
                group="variable",
            )
            + p9.geom_line()
            + p9.geom_vline(xintercept=self.best_it)
        )

        if title is not None:
            p += p9.ggtitle(title)

        if not legend:
            p += p9.theme(legend_position="none")

        return p

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(final_it={self.final_it}, best_it={self.best_it}, "
            f"duration={self.duration:.1f}s)"
        )
        return out
