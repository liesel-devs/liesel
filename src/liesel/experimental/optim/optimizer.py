from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from .types import Position


@dataclass
class Optimizer:
    position_keys: Sequence[str]
    optimizer: optax.GradientTransformation
    identifier: str = ""

    def position(self, position: Position) -> dict[str, jax.Array]:
        pos = {
            k: jnp.asarray(v) for k, v in position.items() if k in self.position_keys
        }

        return pos

    def not_position(self, position: Position) -> dict[str, jax.Array]:
        pos = {k: v for k, v in position.items() if k not in self.position_keys}
        return pos

    def init(self, position: Position) -> optax.OptState:
        pos = self.position(position)
        return self.optimizer.init(pos)

    def step(self, position: Position, loss, carry):
        pos = position

        opt_state = carry.optimizer_states[self.identifier]
        grad = loss.grad(pos, carry)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=pos)
        updated_position = optax.apply_updates(pos, updates)

        carry.position = carry.position | updated_position
        carry.optimizer_states[self.identifier] = opt_state
        return carry

    def _tree_flatten(self):
        children = tuple()
        aux_data = {
            "position_keys": self.position_keys,
            "identifier": self.identifier,
            "optimizer": self.optimizer,
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        bi = cls(*children, **aux_data)
        return bi

    def __repr__(self) -> str:
        name = type(self).__name__
        out = f"{name}({self.position_keys}, identifier={self.identifier})"
        return out


jax.tree_util.register_pytree_node(
    Optimizer, Optimizer._tree_flatten, Optimizer._tree_unflatten
)


@dataclass
class LBFGS(Optimizer):
    position_keys: Sequence[str]
    optimizer: optax.GradientTransformation = optax.lbfgs()
    identifier: str = ""

    def step(self, position: Position, loss, carry):
        pos = position
        opt_state = carry.optimizer_states[self.identifier]

        def loss_fn(pos):
            return loss.loss_train_batched(pos, carry)

        value_and_grad = optax.value_and_grad_from_state(loss_fn)
        value, grad = value_and_grad(pos, state=opt_state)
        updates, opt_state = self.optimizer.update(
            grad, opt_state, params=pos, value=value, grad=grad, value_fn=loss_fn
        )

        updated_position = optax.apply_updates(pos, updates)

        carry.position = carry.position | updated_position
        carry.optimizer_states[self.identifier] = opt_state
        return carry


jax.tree_util.register_pytree_node(LBFGS, LBFGS._tree_flatten, LBFGS._tree_unflatten)
