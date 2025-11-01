from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ...model import Model
from .types import ModelInterface, ModelState, Position


@dataclass
class PositionSplit:
    train: Position
    validate: Position
    test: Position

    n_train: int
    n_validate: int
    n_test: int

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(train={self.n_train}, "
            f"validate={self.n_validate}, test={self.n_test})"
        )
        return out


@dataclass
class StateSplit:
    train_state: ModelState
    pos_validate: Position
    pos_test: Position

    n_train: int
    n_validate: int
    n_test: int

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(train={self.n_train}, "
            f"validate={self.n_validate}, test={self.n_test})"
        )
        return out


@dataclass
class Split:
    position_keys: Sequence[str]
    n: int
    n_validate: int = 0
    n_test: int = 0
    n_train: int | None = None
    axes: dict[str, int] | None = None
    default_axis: int = 0

    def __post_init__(self):
        if self.axes is None:
            self.axes = {}

        if self.n_train is None:
            self.n_train = self.n - self.n_validate - self.n_test

        self.indices = jnp.arange(self.n)

        if self.n_train < 0:
            raise ValueError(
                f"The given {self.n_validate=} and {self.n_test=} imply "
                f"a total of {self.n_train=}, which < 0."
            )

        n = self.n_train + self.n_validate + self.n_test
        if n > self.n:
            raise ValueError(
                f"The given {self.n_train=}, {self.n_validate=}, and {self.n_test=} "
                f"imply a total of {n=}, which is > the provided {self.n=}."
            )

        if self.share_validate < 0.0 or self.share_test < 0.0:
            raise ValueError(
                f"One of {self.share_validate=} or {self.share_test=} is negative, "
                "which is not allowed."
            )

    @property
    def share_validate(self) -> float:
        return self.n_validate / self.n

    @property
    def share_test(self) -> float:
        return self.n_test / self.n

    @classmethod
    def from_share(
        cls,
        position_keys: Sequence[str],
        n: int,
        share_validate: float = 0.0,
        share_test: float = 0.0,
        axes: dict[str, int] | None = None,
        default_axis: int = 0,
    ) -> Split:
        n_validate = int(n * share_validate)
        n_test = int(n * share_test)

        return cls(
            position_keys=position_keys,
            n=n,
            n_validate=n_validate,
            n_test=n_test,
            axes=axes,
            default_axis=default_axis,
        )

    def permute_indices(self, key: jax.Array) -> jax.Array:
        return jax.random.permutation(key, self.indices)

    @property
    def indices_train(self) -> jax.Array:
        return self.indices[: self.n_train]

    @property
    def indices_validate(self) -> jax.Array:
        start = self.n_train
        end = self.n_train + self.n_validate
        return self.indices[start:end]

    @property
    def indices_test(self) -> jax.Array:
        start = self.n_train + self.n_validate
        end = self.n_train + self.n_validate + self.n_test
        return self.indices[start:end]

    def split_position(self, position: Position) -> PositionSplit:
        train_position = {}
        validation_position = {}
        test_position = {}

        assert isinstance(self.axes, dict)
        for key in self.position_keys:
            axis = self.axes.get(key, self.default_axis)
            train_values = jnp.take(position[key], self.indices_train, axis=axis)
            validation_values = jnp.take(
                position[key], self.indices_validate, axis=axis
            )
            test_values = jnp.take(position[key], self.indices_test, axis=axis)

            train_position[key] = train_values
            validation_position[key] = validation_values
            test_position[key] = test_values

        split = PositionSplit(
            train=Position(train_position),
            validate=Position(validation_position),
            test=Position(test_position),
            n_train=self.n_train,
            n_validate=self.n_validate,
            n_test=self.n_test,
        )
        return split

    def split_state(
        self, interface: ModelInterface | Model, model_state: ModelState
    ) -> StateSplit:
        obs = interface.extract_position(self.position_keys, model_state)
        split = self.split_position(obs)
        train_state = interface.update_state(split.train, model_state)
        return StateSplit(
            train_state,
            split.validate,
            split.test,
            n_train=self.n_train,
            n_validate=self.n_validate,
            n_test=self.n_test,
        )

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(train={self.n_train}, "
            f"validate={self.n_validate}, test={self.n_test})"
        )
        return out
