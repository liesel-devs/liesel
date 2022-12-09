"""
Imitates the API from v0.1.
"""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp

from .nodes import Bijector as TFPBijector
from .nodes import Calc, Dist, Node, Obs, Param, Var

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Strong variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def DesignMatrix(
    value: Any | Calc, distribution: Dist | None = None, name: str = ""
) -> Var:
    """A strong variable representing a design matrix."""
    var = Obs(value, distribution, name)
    var.role = "DesignMatrix"
    return var


def Hyperparameter(
    value: Any | Calc, distribution: Dist | None = None, name: str = ""
) -> Var:
    """A strong variable representing a hyperparameter."""
    var = Var(value, distribution, name)
    var.role = "Hyperparameter"
    return var


def Parameter(
    value: Any | Calc, distribution: Dist | None = None, name: str = ""
) -> Var:
    """A strong variable representing a model parameter."""
    var = Param(value, distribution, name)
    var.role = "Parameter"
    return var


def RegressionCoef(
    value: Any | Calc, distribution: Dist | None = None, name: str = ""
) -> Var:
    """A strong variable representing a vector of regression coefficients."""
    var = Param(value, distribution, name)
    var.role = "RegressionCoef"
    return var


def Response(
    value: Any | Calc, distribution: Dist | None = None, name: str = ""
) -> Var:
    """A strong variable representing a response vector."""
    var = Obs(value, distribution, name)
    var.role = "Response"
    return var


def SmoothingParam(
    value: Any | Calc, distribution: Dist | None = None, name: str = ""
) -> Var:
    """A strong variable representing a smoothing parameter."""
    var = Param(value, distribution, name)
    var.role = "SmoothingParam"
    return var


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Weak variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def Addition(
    *inputs: Var | Node, distribution: Dist | None = None, name: str = ""
) -> Var:
    """A weak variable calculating an element-wise sum."""
    calc = Calc(lambda *args, **kwargs: sum(args) + sum(kwargs.values()), *inputs)
    var = Var(calc, distribution, name)
    var.role = "Addition"
    return var


def Bijector(
    _input: Var | Node,
    bijector: type[TFPBijector],
    inverse: bool = False,
    distribution: Dist | None = None,
    name: str = "",
) -> Var:
    """
    A weak variable evaluating the ``forward()`` or ``inverse()`` method
    of a TFP bijector.
    """

    def fn(x):
        return bijector().forward(x) if not inverse else bijector().inverse(x)

    calc = Calc(fn, _input)
    var = Var(calc, distribution, name)
    var.role = "Bijector"
    return var


def ColumnStack(
    *inputs: Var | Node, distribution: Dist | None = None, name: str = ""
) -> Var:
    """A weak variable stacking its inputs column-wise."""
    calc = Calc(lambda *args: jnp.column_stack(args), *inputs)
    var = Var(calc, distribution, name)
    var.role = "ColumnStack"
    return var


def InverseLink(
    _input: Var | Node,
    bijector: type[TFPBijector],
    inverse: bool = False,
    distribution: Dist | None = None,
    name: str = "",
) -> Var:
    """A weak variable representing an inverse link function."""
    var = Bijector(_input, bijector, inverse, distribution, name)
    var.role = "InverseLink"
    return var


class PITCalc(Node):
    """A probability integral transform (PIT) calculator node."""

    def __init__(
        self,
        _input: Dist,
        _name: str = "",
        _needs_seed: bool = False,
    ):
        super().__init__(_input, _name=_name, _needs_seed=_needs_seed)

    def update(self) -> PITCalc:
        dist = cast(Dist, self.inputs[0])

        if not dist.at:
            raise RuntimeError(
                f"Cannot evaluate PIT on {repr(dist)}, property `at` not set"
            )

        self._value = dist.init_dist().cdf(dist.at.value)
        self._outdated = False
        return self


def PIT(_input: Var | Dist, distribution: Dist | None = None, name: str = "") -> Var:
    """A weak variable evaluating a probability integral transform (PIT)."""
    dist = _input.dist_node if isinstance(_input, Var) else _input

    if not dist:
        raise RuntimeError(f"Cannot evaluate PIT on {repr(_input)}, has no dist node")

    if _input.name and not name:
        name = f"{_input.name}_pit"

    calc = PITCalc(dist)
    var = Var(calc, distribution, name)
    var.role = "PIT"
    return var


def Predictor(
    *inputs: Var | Node, distribution: Dist | None = None, name: str = ""
) -> Var:
    """A weak variable representing a regression predictor."""
    var = Addition(*inputs, distribution=distribution, name=name)
    var.role = "Predictor"
    return var


def Smooth(
    x: Var | Node, beta: Var | Node, distribution: Dist | None = None, name: str = ""
) -> Var:
    """A weak variable calculating the matrix-vector product ``x @ beta``."""
    calc = Calc(lambda x, beta: x @ beta, x=x, beta=beta)
    var = Var(calc, distribution, name)
    var.role = "Smooth"
    return var
