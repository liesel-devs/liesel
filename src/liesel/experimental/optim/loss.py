from collections.abc import Sequence
from typing import Literal, Protocol

import jax

from ...model import Model
from .split import PositionSplit, StateSplit
from .state import OptimCarry
from .types import ModelInterface, ModelState, Position


class Loss(Protocol):
    split: StateSplit

    @property
    def obs_validation(self) -> Position: ...

    @property
    def scale_validation(self) -> float: ...

    @property
    def n_validation(self) -> int: ...

    @property
    def model(self) -> Model | ModelInterface: ...

    @property
    def model_state(self) -> ModelState: ...

    def loss_train(self, params: Position, carry: OptimCarry) -> jax.Array: ...

    def loss_train_batched(self, params: Position, carry: OptimCarry) -> jax.Array: ...

    def loss_validation(self, params: Position, carry: OptimCarry) -> jax.Array: ...

    def value_and_grad(self, params: Position, carry: OptimCarry): ...

    def grad(self, params: Position, carry: OptimCarry): ...


class LossMixin:
    @property
    def obs_validation(self) -> Position:
        if self.split.n_validation == 0:
            return self.split.train

        return self.split.validation

    @property
    def scale_validation(self) -> float:
        if self.split.n_validation == 0:
            return 1.0

        return self.split.n_train / self.n_validation

    @property
    def n_validation(self) -> int:
        if self.split.n_validation == 0:
            return self.split.n_train

        return self.split.n_validation

    def value_and_grad(self, params: Position, carry: OptimCarry):
        grad_ = jax.value_and_grad(self.loss_train_batched, argnums=0)
        value, grad_tree = grad_(params, carry)
        return value, grad_tree

    def grad(self, params: Position, carry: OptimCarry):
        grad_ = jax.grad(self.loss_train_batched, argnums=0)
        grad_tree = grad_(params, carry)
        return grad_tree


class NegLogProbLoss(LossMixin):
    def __init__(
        self,
        model: Model,
        split: PositionSplit,
        validation_strategy: Literal["log_lik", "log_prob"] = "log_lik",
        scale: bool = False,
    ):
        self._model = model
        self.split = split
        self.validation_strategy = validation_strategy
        self.scale = scale
        self.scalar = self.split.n_train if self.scale else 1.0

    @property
    def model(self) -> Model:
        return self._model

    def position(self, position_keys: Sequence[str]) -> Position:
        return self.model.extract_position(position_keys)

    def loss_train_batched(self, params: Position, carry: OptimCarry):
        position = Position(params | carry.batch | carry.fixed_position)
        new_state = self.model.update_state(position, carry.model_state)

        scale_log_lik_by = carry.batches.n / carry.batches.batch_size

        log_lik = scale_log_lik_by * new_state["_model_log_lik"].value
        log_prior = new_state["_model_log_prior"].value
        return -(log_lik + log_prior) / self.scalar

    def loss_train(self, params: Position, carry: OptimCarry):
        position = Position(params | carry.batch | carry.fixed_position)
        new_state = self.model.update_state(position, carry.model_state)

        log_lik = new_state["_model_log_lik"].value
        log_prior = new_state["_model_log_prior"].value
        return -(log_lik + log_prior) / self.scalar

    def loss_validation(self, params: Position, carry: OptimCarry):
        position = Position(params | self.split.validate | carry.fixed_position)
        new_state = self.model.update_state(position, carry.model_state)
        loss = -self.scale_validation * new_state["_model_log_lik"].value
        if self.validation_strategy == "log_prob":
            loss -= new_state["_model_log_prior"].value

        return loss / self.scalar

    def __repr__(self) -> str:
        name = type(self).__name__
        out = f"{name}(validation_strategy={self.validation_strategy})"
        return out
