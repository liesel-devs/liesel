"""Optimizer wrappers used by :class:`.OptimEngine`.

The classes in this module adapt Optax gradient transformations to the
position-dictionary workflow used by the experimental optimizer engine. Each
optimizer owns a subset of parameter keys and updates only that subset during an
engine step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax

from .types import Position

if TYPE_CHECKING:
    from .loss import Loss
    from .state import OptimCarry


@dataclass
class Optimizer:
    """
    Wraps an Optax gradient transformation for selected position entries.

    ``Optimizer`` is the default adapter used by :class:`.OptimEngine`. It extracts
    the entries named in :attr:`position_keys`, differentiates the configured loss
    with respect to only those entries, applies an Optax update, and merges the
    updated subset back into the full engine position.

    Parameters
    ----------
    position_keys
        Names of the parameter entries owned by this optimizer.
    optimizer
        Optax gradient transformation, for example ``optax.adam(...)`` or
        ``optax.sgd(...)``.
    identifier
        Optional identifier used to store this optimizer's state in
        :class:`.OptimCarry`. Missing identifiers are filled by
        :class:`.OptimEngine`.

    Notes
    -----
    Multiple optimizers can be used in the same engine, but their
    :attr:`position_keys` and identifiers must be disjoint after automatic naming.
    ``position_keys`` are normalized to a tuple during initialization.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import optax
    >>> from liesel.experimental.optim import Optimizer
    >>> from liesel.experimental.optim.types import Position
    >>> optimizer = Optimizer(["x"], optax.sgd(0.1), identifier="x_opt")
    >>> position = Position({"x": jnp.array(1.0), "y": jnp.array(2.0)})
    >>> optimizer.position(position)["x"].tolist()
    1.0
    >>> sorted(optimizer.not_position(position))
    ['y']
    >>> repr(optimizer)
    "Optimizer(('x',), identifier=x_opt)"
    """

    position_keys: Sequence[str]
    optimizer: optax.GradientTransformation
    identifier: str = ""

    def __post_init__(self) -> None:
        """
        Validates and normalizes :attr:`position_keys`.

        Raises
        ------
        ValueError
            If no position keys are supplied or if any key is duplicated.
        """
        self.position_keys = tuple(self.position_keys)

        if len(self.position_keys) == 0:
            raise ValueError("position_keys must not be empty.")

        duplicates = sorted(
            {key for key in self.position_keys if self.position_keys.count(key) > 1}
        )
        if duplicates:
            raise ValueError(f"Duplicate position_keys are not allowed: {duplicates}.")

    def position(self, position: Position) -> Position:
        """
        Extracts the subset of ``position`` owned by this optimizer.

        Parameters
        ----------
        position
            Full optimizer position.

        Returns
        -------
        Position
            Mapping containing only keys listed in :attr:`position_keys`. Values are
            converted with :func:`jax.numpy.asarray`.

        Raises
        ------
        KeyError
            If any key listed in :attr:`position_keys` is missing from ``position``.
        """
        missing = [key for key in self.position_keys if key not in position]
        if missing:
            raise KeyError(f"Position is missing keys claimed by optimizer: {missing}.")

        pos = Position({k: jnp.asarray(position[k]) for k in self.position_keys})

        return pos

    def not_position(self, position: Position) -> Position:
        """
        Extracts the subset of ``position`` not owned by this optimizer.

        Parameters
        ----------
        position
            Full optimizer position.

        Returns
        -------
        Position
            Mapping containing entries whose keys are not in :attr:`position_keys`.
            During engine updates, this subset is exposed as
            ``carry.fixed_position`` so losses can still evaluate the full position.
        """
        pos = Position(
            {k: v for k, v in position.items() if k not in self.position_keys}
        )
        return pos

    def init(self, position: Position) -> optax.OptState:
        """
        Initializes the wrapped Optax transformation.

        Parameters
        ----------
        position
            Full optimizer position. Only :attr:`position_keys` are passed to the
            Optax transformation.

        Returns
        -------
        optax.OptState
            Initial optimizer state.
        """
        pos = self.position(position)
        return self.optimizer.init(pos)

    def step(self, position: Position, loss: Loss, carry: OptimCarry) -> OptimCarry:
        """
        Runs one optimizer step on ``position``.

        Parameters
        ----------
        position
            Parameter subset owned by this optimizer.
        loss
            Loss object providing :meth:`grad`.
        carry
            Current optimizer carry. The optimizer state for this object is read from
            and written to ``carry.optimizer_states[self.identifier]``.

        Returns
        -------
        OptimCarry
            Updated carry with the new parameter subset merged into
            ``carry.position``.
        """
        pos = position

        opt_state = carry.optimizer_states[self.identifier]
        grad = loss.grad(pos, carry)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=pos)
        updated_position = Position(optax.apply_updates(pos, updates))

        carry.position = Position(carry.position | updated_position)
        carry.optimizer_states[self.identifier] = opt_state
        return carry

    def _tree_flatten(self):
        """Flattens the optimizer as a JAX pytree node with static metadata."""
        children = tuple()
        aux_data = {
            "position_keys": self.position_keys,
            "identifier": self.identifier,
            "optimizer": self.optimizer,
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """Reconstructs the optimizer from JAX pytree metadata."""
        bi = cls(*children, **aux_data)
        return bi

    def __repr__(self) -> str:
        """Returns a compact representation with position keys and identifier."""
        name = type(self).__name__
        out = f"{name}({self.position_keys}, identifier={self.identifier})"
        return out


jax.tree_util.register_pytree_node(
    Optimizer, Optimizer._tree_flatten, Optimizer._tree_unflatten
)


@dataclass(repr=False)
class LBFGS(Optimizer):
    """
    Optimizer wrapper using Optax L-BFGS.

    ``LBFGS`` behaves like :class:`Optimizer` but uses
    :func:`optax.value_and_grad_from_state` inside :meth:`step`, which lets Optax
    reuse value/gradient information stored by the L-BFGS transformation.

    Parameters
    ----------
    position_keys
        Names of the parameter entries owned by this optimizer.
    optimizer
        Optax L-BFGS transformation. Defaults to ``optax.lbfgs()``.
    identifier
        Optional optimizer-state identifier filled by :class:`.OptimEngine` when
        omitted.

    Examples
    --------
    >>> from liesel.experimental.optim import LBFGS
    >>> lbfgs = LBFGS(["loc"], identifier="loc_lbfgs")
    >>> lbfgs.position_keys
    ('loc',)
    >>> lbfgs.identifier
    'loc_lbfgs'
    >>> repr(lbfgs)
    "LBFGS(('loc',), identifier=loc_lbfgs)"
    """

    position_keys: Sequence[str]
    optimizer: optax.GradientTransformation = optax.lbfgs()
    identifier: str = ""

    def step(self, position: Position, loss: Loss, carry: OptimCarry) -> OptimCarry:
        """
        Runs one L-BFGS optimizer step on ``position``.

        Parameters
        ----------
        position
            Parameter subset owned by this optimizer.
        loss
            Loss object providing :meth:`loss_train_batched`.
        carry
            Current optimizer carry. The L-BFGS state is read from and written to
            ``carry.optimizer_states[self.identifier]``.

        Returns
        -------
        OptimCarry
            Updated carry with the new parameter subset merged into
            ``carry.position``.
        """
        pos = position
        opt_state = carry.optimizer_states[self.identifier]

        def loss_fn(pos: Position) -> jax.Array:
            return loss.loss_train_batched(pos, carry)

        value_and_grad = optax.value_and_grad_from_state(loss_fn)
        value, grad = value_and_grad(pos, state=opt_state)
        updates, opt_state = self.optimizer.update(
            grad, opt_state, params=pos, value=value, grad=grad, value_fn=loss_fn
        )

        updated_position = Position(optax.apply_updates(pos, updates))

        carry.position = Position(carry.position | updated_position)
        carry.optimizer_states[self.identifier] = opt_state
        return carry


jax.tree_util.register_pytree_node(LBFGS, LBFGS._tree_flatten, LBFGS._tree_unflatten)
