from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ...model import Model
from .types import ModelInterface, ModelState, Position


@dataclass
class Batches:
    position_keys: Sequence[str]
    n: int
    batch_size: int | None
    shuffle: bool = True
    axes: dict[str, int] | None = None
    default_axis: int = 0

    def __post_init__(self):
        if self.batch_size is None:
            self.batch_size = self.n

        if self.batch_size < 1:
            raise ValueError(f"{self.batch_size=} is < 1, which is not allowed.")

        if self.n < self.batch_size:
            raise ValueError(
                f"{self.n=} is < {self.batch_size=}, which is not allowed."
            )

        if self.axes is None:
            self.axes = {}

        self.indices = jnp.arange(self.n)

    @property
    def n_full_batches(self) -> int:
        assert self.batch_size is not None
        return int(self.n // self.batch_size)

    @property
    def likelihood_scalar(self) -> float:
        assert self.batch_size is not None
        return float(self.n / self.batch_size)

    def permute_indices(self, key: jax.Array) -> jax.Array:
        if self.shuffle:
            all_indices = jax.random.permutation(key, self.indices)
        else:
            all_indices = self.indices

        return all_indices

    @property
    def batch_indices(self) -> jax.Array:
        assert self.batch_size is not None
        idx = self.indices[: self.n_full_batches * self.batch_size]
        batch_indices = jnp.reshape(idx, (self.n_full_batches, self.batch_size))
        return batch_indices

    def get_batched_position(self, position: Position, batch_index: int) -> Position:
        idx = self.batch_indices[batch_index]
        batched_position = {}
        assert isinstance(self.axes, dict)
        for key in self.position_keys:
            axis = self.axes.get(key, self.default_axis)
            batched = jnp.take(position[key], idx, axis=axis)
            batched_position[key] = batched

        return Position(batched_position)

    def extract_batched_position(
        self,
        interface: ModelInterface | Model,
        model_state: ModelState,
        batch_number: int,
    ) -> Position:
        obs = interface.extract_position(self.position_keys, model_state)
        return self.get_batched_position(obs, batch_number)

    def _tree_flatten(self):
        children = (self.indices,)
        aux_data = {
            "position_keys": self.position_keys,
            "n": self.n,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "axes": self.axes,
            "default_axis": self.default_axis,
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        bi = cls(**aux_data)
        bi.indices = children[0]
        return bi

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(n={self.n}, "
            f"batch_size={self.batch_size}, default_axis={self.default_axis})"
        )
        return out


jax.tree_util.register_pytree_node(
    Batches, Batches._tree_flatten, Batches._tree_unflatten
)
