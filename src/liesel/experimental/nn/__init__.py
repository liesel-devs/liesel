from __future__ import annotations

import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.typing import ArrayLike

from ...goose.types import Position
from ...model import Calc, TransientCalc, Var


class Bias(Var):
    pass


class Weights(Var):
    pass


class WeightedSum(Var):
    pass


class Feature(Var):
    def __init__(
        self,
        x: ArrayLike,
        fn: Callable[[jax.Array], jax.Array] = lambda x: x,
        standardize: bool = True,
        train_x: ArrayLike | None = None,
        name: str = "",
    ) -> None:
        if train_x is None:
            train_x = x

        x = jnp.asarray(x)
        train_x = jnp.asarray(train_x)

        mean = fn(train_x).mean()
        scale = fn(train_x).std()

        def standardization(x):
            x_transformed = fn(x)
            x_standardized = (x_transformed - mean) / scale
            return x_standardized

        fn_ = standardization if standardize else fn

        xvar = Var.new_obs(x, name=name)

        super().__init__(TransientCalc(fn_, xvar), name="fn(" + name + ")")

    @classmethod
    def from_data(
        cls,
        batch: Position,
        train_data: Position,
        name: str,
        fn: Callable[[jax.Array], jax.Array] = lambda x: x,
        standardize: bool = True,
    ) -> Feature:
        x = batch[name]
        train_x = train_data[name]
        return cls(x=x, fn=fn, standardize=standardize, train_x=train_x, name=name)


class Perceptron(Var):
    def __init__(
        self,
        x: Var,
        activation_fn: Callable[[jax.Array], jax.Array] = lambda x: x,
        name: str = "",
        init_seed: int | None = None,
    ) -> None:
        self.activation_fn = activation_fn
        self.x = x

        self.bias = Bias.new_param(0.001)
        n_in = self.x.value.shape[-1]
        if init_seed is None:
            init_seed = int(time.time())
        key = jax.random.key(init_seed)
        init_weights = tfd.Normal(loc=0.0, scale=jnp.sqrt(2 / n_in)).sample(
            (n_in,), seed=key
        )
        self.weights = Weights.new_param(init_weights)

        self.weighted_sum = WeightedSum.new_calc(
            lambda bias, weights, X: bias + jnp.dot(X, weights),
            bias=self.bias,
            weights=self.weights,
            X=self.x,
        )

        out = Calc(self.activation_fn, self.weighted_sum)

        super().__init__(out, name=name)

    @classmethod
    def from_layer(cls, layer: DenseLayer, **kwargs) -> Perceptron:
        X = Calc(lambda *inputs: jnp.vstack(inputs).T, *layer.outputs)
        return cls(X, **kwargs)


class DenseLayer:
    def __init__(
        self,
        X: Var,
        size: int,
        activation_fn: Callable[[jax.Array], jax.Array] = lambda x: x,
    ) -> None:
        self.X = X
        self.size = size
        self.activation_fn = activation_fn
        self.outputs: list[Perceptron] = []

        for _ in range(self.size):
            per = Perceptron(self.X, self.activation_fn)
            self.outputs.append(per)

    @classmethod
    def from_features(cls, *inputs: Var | Feature, **kwargs) -> DenseLayer:
        def fn(*inputs):
            return jnp.vstack(inputs).T

        X = Calc(fn, *inputs)
        return cls(X=X, **kwargs)

    @classmethod
    def from_layer(cls, layer: DenseLayer, **kwargs) -> DenseLayer:
        return DenseLayer.from_features(*layer.outputs, **kwargs)
