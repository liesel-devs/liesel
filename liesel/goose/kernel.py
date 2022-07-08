"""
# Kernel-related info, outcome and mixin classes
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic

import jax

from .epoch import EpochState, EpochType
from .pytree import register_dataclass_as_pytree
from .types import (
    KeyArray,
    ModelInterface,
    ModelState,
    Position,
    TKernelState,
    TTransitionInfo,
    TTuningInfo,
)


@register_dataclass_as_pytree
@dataclass
class DefaultTransitionInfo:
    error_code: int
    acceptance_prob: float
    position_moved: int

    def minimize(self) -> "DefaultTransitionInfo":
        return self


@register_dataclass_as_pytree
@dataclass
class DefaultTuningInfo:
    error_code: int
    time: int


@register_dataclass_as_pytree
@dataclass
class TransitionOutcome(Generic[TKernelState, TTransitionInfo]):
    """
    A dataclass for the return value of the kernel method `transition`.
    Different kernels can use different types of `liesel.goose.types.KernelState`'s
    and `liesel.goose.types.TransitionInfo`'s.
    """

    info: TTransitionInfo
    kernel_state: TKernelState
    model_state: ModelState


@register_dataclass_as_pytree
@dataclass
class TuningOutcome(Generic[TKernelState, TTuningInfo]):
    """
    A dataclass for the return value of the kernel method `tune`.
    Different kernels can use different types of `liesel.goose.types.KernelState`'s
    and `liesel.goose.types.TuningInfo`'s.
    """

    info: TTuningInfo
    kernel_state: TKernelState


@register_dataclass_as_pytree
@dataclass
class WarmupOutcome(Generic[TKernelState]):
    """
    A dataclass for the return value of the kernel method `end_warmup`.
    Different kernels can use different types of `liesel.goose.types.KernelState`'s.
    """

    error_code: int
    kernel_state: TKernelState


class ModelMixin:
    """
    A mixin facilitating the interaction with the model interface.
    """

    position_keys: tuple[str, ...]
    _model: ModelInterface | None

    @property
    def model(self) -> ModelInterface:
        """Returns the model interface if it is set. Raises error otherwise."""

        if self._model is None:
            raise RuntimeError("Model interface not set")

        return self._model

    def set_model(self, model: ModelInterface):
        self._model = model

    def has_model(self) -> bool:
        return self._model is not None

    def position(self, model_state: ModelState) -> Position:
        """
        Extracts the position from a model state.
        """

        return self.model.extract_position(self.position_keys, model_state)

    def log_prob_fn(self, model_state: ModelState) -> Callable[[Position], float]:
        """
        Returns the log-probability function with the position as the only argument.
        """

        def log_prob_fn(position: Position) -> float:
            new_model_state = self.model.update_state(position, model_state)
            return self.model.log_prob(new_model_state)

        return log_prob_fn


class TransitionMixin(Generic[TKernelState, TTransitionInfo]):
    """
    An abstract mixin defining two transition methods with and without adaptation.
    """

    def transition(
        self,
        prng_key: KeyArray,
        kernel_state: TKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[TKernelState, TTransitionInfo]:
        is_adaptation = EpochType.is_adaptation(epoch.config.type)

        outcome: TransitionOutcome[TKernelState, TTransitionInfo] = jax.lax.cond(
            is_adaptation,
            self._adaptive_transition,
            self._standard_transition,
            prng_key,
            kernel_state,
            model_state,
            epoch,
        )

        return outcome

    @abstractmethod
    def _standard_transition(
        self,
        prng_key: KeyArray,
        kernel_state: TKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[TKernelState, TTransitionInfo]:
        """
        Performs an MCMC transition *outside* an adaptation epoch. Must be jittable.
        """

        raise NotImplementedError

    @abstractmethod
    def _adaptive_transition(
        self,
        prng_key: KeyArray,
        kernel_state: TKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[TKernelState, TTransitionInfo]:
        """
        Performs an MCMC transition *in* an adaptation epoch. Must be jittable.
        """

        raise NotImplementedError


class TuningMixin(Generic[TKernelState, TTuningInfo]):
    """
    An abstract mixin defining two tuning methods after a slow and a fast
    adaptation epoch.
    """

    def tune(
        self,
        prng_key: KeyArray,
        kernel_state: TKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None,
    ) -> TuningOutcome[TKernelState, TTuningInfo]:
        is_slow = epoch.config.type == EpochType.SLOW_ADAPTATION

        outcome: TuningOutcome[TKernelState, TTuningInfo] = jax.lax.cond(
            is_slow,
            self._tune_slow,
            self._tune_fast,
            prng_key,
            kernel_state,
            model_state,
            epoch,
            history,
        )

        return outcome

    @abstractmethod
    def _tune_fast(
        self,
        prng_key: KeyArray,
        kernel_state: TKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None,
    ) -> TuningOutcome[TKernelState, TTuningInfo]:
        """
        Tunes a kernel after a *fast* adaptation epoch. Must be jittable.
        """

        raise NotImplementedError

    @abstractmethod
    def _tune_slow(
        self,
        prng_key: KeyArray,
        kernel_state: TKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None,
    ) -> TuningOutcome[TKernelState, TTuningInfo]:
        """
        Tunes a kernel after a *slow* adaptation epoch. Must be jittable.
        """

        raise NotImplementedError
