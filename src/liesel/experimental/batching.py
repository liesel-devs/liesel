from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.flatten_util
import jax.numpy as jnp

from ..goose import LieselInterface
from ..goose.types import ModelState, Position

Array = Any


@dataclass
class BatchIndices:
    position_keys: Sequence[str]
    n: int
    batch_size: int | None
    shuffle: bool = True
    axes: dict[str, int] | None = None
    default_axis: int = 0
    batch_number: int = 0

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

    def get_batched_position(self, position: Position) -> Position:
        idx = self.batch_indices[self.batch_number]
        batched_position = {}
        assert isinstance(self.axes, dict)
        for key in self.position_keys:
            axis = self.axes.get(key, self.default_axis)
            batched = jnp.take(position[key], idx, axis=axis)
            batched_position[key] = batched

        return Position(batched_position)

    def _tree_flatten(self):
        children = (self.batch_number, self.indices)
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
        bi = cls(batch_number=children[0], **aux_data)
        bi.indices = children[1]
        return bi


jax.tree_util.register_pytree_node(
    BatchIndices, BatchIndices._tree_flatten, BatchIndices._tree_unflatten
)


class BatchedLieselInterface(LieselInterface):
    def batched_state(
        self, position: Position, batch_indices: BatchIndices, model_state: ModelState
    ) -> ModelState:
        pos = self._model.extract_position(batch_indices.position_keys, model_state)
        batched_position = batch_indices.get_batched_position(Position(pos))

        new_position = Position(batched_position | position)
        updated_state = self.update_state(new_position, model_state)
        return updated_state

    def batched_log_prob(
        self,
        position: Position,
        batch_indices: BatchIndices,
        model_state: ModelState,
    ):
        batched_state = self.batched_state(position, batch_indices, model_state)
        log_lik = (
            batch_indices.likelihood_scalar * batched_state["_model_log_lik"].value
        )
        log_prior = batched_state["_model_log_prior"].value
        return log_lik + log_prior


class BatchedLogProb:
    def __init__(
        self,
        model: BatchedLieselInterface,
        model_state: ModelState,
        batch_indices: BatchIndices,
        diff_mode: Literal["forward", "reverse"] = "forward",
    ):
        self.model = model
        self.model_state = model_state
        self.batch_indices = batch_indices

        self._grad_fn = jax.grad(self.log_prob, argnums=0)

        if diff_mode == "forward":
            self._hessian_fn = jax.jacfwd(self._grad_fn)
        elif diff_mode == "reverse":
            self._hessian_fn = jax.jacrev(self._grad_fn)
        else:
            raise ValueError(f"Unrecognized argument value {diff_mode=}")
        self.diff_mode = diff_mode

    def __call__(self, position: Position) -> Array:
        return self.log_prob(position=position)

    def log_prob(self, position: Position) -> Array:
        """
        Log probability function evaluated at provided ``position``.
        """
        log_prob = self.model.batched_log_prob(
            position, batch_indices=self.batch_indices, model_state=self.model_state
        )
        return log_prob

    def grad(self, position: Position) -> dict[str, Array]:
        """
        Gradient of the log probability function with respect to the ``position``.
        """
        return self._grad_fn(position)

    def hessian(self, position: Position) -> dict[str, Array]:
        """
        Hessian of the log probability function with respect to the ``position``.
        """
        return self._hessian_fn(position)


class FlatBatchedLogProb:
    def __init__(
        self,
        model: BatchedLieselInterface,
        model_state: ModelState,
        position_keys: Sequence[str],
        batch_indices: BatchIndices,
        diff_mode: Literal["forward", "reverse"] = "forward",
    ):
        self.model = model
        self.model_state = model_state
        self.batch_indices = batch_indices

        position = self.model.extract_position(position_keys, model_state)
        _, unravel_fn = jax.flatten_util.ravel_pytree(position)
        self.unravel_fn = unravel_fn

        self._grad_fn = jax.grad(self.log_prob, argnums=0)

        if diff_mode == "forward":
            self._hessian_fn = jax.jacfwd(self._grad_fn)
        elif diff_mode == "reverse":
            self._hessian_fn = jax.jacrev(self._grad_fn)
        else:
            raise ValueError(f"Unrecognized argument value {diff_mode=}")
        self.diff_mode = diff_mode

    def __call__(self, flat_position: Position) -> Array:
        return self.log_prob(flat_position=flat_position)

    def log_prob(self, flat_position: Position) -> Array:
        """
        Log probability function evaluated at provided ``position``.
        """
        position = self.unravel_fn(flat_position)
        log_prob = self.model.batched_log_prob(
            position, batch_indices=self.batch_indices, model_state=self.model_state
        )
        return log_prob

    def grad(self, flat_position: Position) -> dict[str, Array]:
        """
        Gradient of the log probability function with respect to the ``position``.
        """
        return self._grad_fn(flat_position)

    def hessian(self, flat_position: Position) -> dict[str, Array]:
        """
        Hessian of the log probability function with respect to the ``position``.
        """
        return self._hessian_fn(flat_position)
