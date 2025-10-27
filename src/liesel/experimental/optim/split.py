from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ...model import Model
from .types import ModelInterface, ModelState, Position


@dataclass
class PositionSplit:
    train: Position
    validation: Position
    test: Position

    n_train: int
    n_validation: int
    n_test: int

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(train={self.n_train}, "
            f"validation={self.n_validation}, test={self.n_test})"
        )
        return out


@dataclass
class StateSplit:
    train_state: ModelState
    validation_position: Position
    test_position: Position

    n_train: int
    n_validation: int
    n_test: int

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(train={self.n_train}, "
            f"validation={self.n_validation}, test={self.n_test})"
        )
        return out


@dataclass
class Split:
    position_keys: Sequence[str]
    n: int
    share_validation: float = 0.0
    share_test: float = 0.0
    axes: dict[str, int] | None = None
    default_axis: int = 0

    def __post_init__(self):
        if self.axes is None:
            self.axes = {}

        self.n_validation = int(self.n * self.share_validation)
        self.n_test = int(self.n * self.share_test)
        self.n_train = self.n - self.n_validation - self.n_test

        self.indices = jnp.arange(self.n)

        if self.n_train < 0:
            raise ValueError(
                f"The given {self.share_validation=} and {self.share_test=} imply "
                f"a total of {self.n_train=}, which < 0."
            )

        if self.share_validation < 0.0 or self.share_test < 0.0:
            raise ValueError(
                f"One of {self.share_validation=} or {self.share_test=} is negative, "
                "which is not allowed."
            )

    def permute_indices(self, key: jax.Array) -> jax.Array:
        return jax.random.permutation(key, self.indices)

    @property
    def indices_train(self) -> jax.Array:
        return self.indices[: self.n_train]

    @property
    def indices_validation(self) -> jax.Array:
        start = self.n_train
        end = self.n_train + self.n_validation
        return self.indices[start:end]

    @property
    def indices_test(self) -> jax.Array:
        start = self.n_train + self.n_validation
        end = self.n_train + self.n_validation + self.n_test
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
                position[key], self.indices_validation, axis=axis
            )
            test_values = jnp.take(position[key], self.indices_test, axis=axis)

            train_position[key] = train_values
            validation_position[key] = validation_values
            test_position[key] = test_values

        split = PositionSplit(
            train=Position(train_position),
            validation=Position(validation_position),
            test=Position(test_position),
            n_train=self.n_train,
            n_validation=self.n_validation,
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
            split.validation,
            split.test,
            n_train=self.n_train,
            n_validation=self.n_validation,
            n_test=self.n_test,
        )

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(train={self.n_train}, "
            f"validation={self.n_validation}, test={self.n_test})"
        )
        return out
