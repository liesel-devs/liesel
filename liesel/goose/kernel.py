"""
Kernel-related info, outcome and mixin classes.
"""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic

import jax

from ..docs import usedocs
from .epoch import EpochState, EpochType
from .pytree import register_dataclass_as_pytree
from .types import (
    Kernel,
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
    """A default template for a transition information object."""

    error_code: int
    """Error code for the transition."""
    acceptance_prob: float
    """Acceptance probability of the transition."""
    position_moved: int
    """Indicates whether the transition resulted in acceptance or not."""

    def minimize(self) -> "DefaultTransitionInfo":
        """Minimizes the transitioninfo."""
        return self


@register_dataclass_as_pytree
@dataclass
class DefaultTuningInfo:
    """A default template for a tuning information object."""

    error_code: int
    """Error code for error during tuning."""
    time: int
    """MCMC time when the tuning happend."""


@register_dataclass_as_pytree
@dataclass
class TransitionOutcome(Generic[TKernelState, TTransitionInfo]):
    """
    A dataclass for the return value of the kernel method :meth:`.Kernel.transition`.
    Different kernels can use different types of :class:`.KernelState`'s and
    :class:`.TransitionInfo`'s.
    """

    info: TTransitionInfo
    """
    A transition info object, see :class:`.DefaultTransitionInfo`.
    """
    kernel_state: TKernelState
    """
    A kernel state object, see :class:`.DAKernelState`.
    """
    model_state: ModelState
    """
    Model state that results from the transition.

    The exact definition depends on the model being used. See, for example,
    :class:`.DictModel`.

    """


@register_dataclass_as_pytree
@dataclass
class TuningOutcome(Generic[TKernelState, TTuningInfo]):
    """
    A dataclass for the return value of the kernel method :meth:`.Kernel.tune`.
    Different kernels can use different types of :class:`.KernelState`'s and
    :class:`.TuningInfo`'s.
    """

    info: TTuningInfo
    """
    A tuning info object, see :class:`.DefaultTuningInfo`.
    """
    kernel_state: TKernelState
    """
    A kernel state object, see :class:`.DAKernelState`.
    """


@register_dataclass_as_pytree
@dataclass
class WarmupOutcome(Generic[TKernelState]):
    """
    A dataclass for the return value of the kernel method :meth:`.Kernel.end_warmup`.
    Different kernels can use different types of :class:`.KernelState`'s.
    """

    error_code: int
    """Error code for the transition."""
    kernel_state: TKernelState
    """
    A kernel state object, see :class:`.DAKernelState`.
    """


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
        """Sets the model interface."""
        self._model = model

    def has_model(self) -> bool:
        """Whether the model interface is set."""
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

    @usedocs(Kernel.transition)
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
    An abstract mixin defining two tuning methods after a slow and a fast adaptation
    epoch.
    """

    @usedocs(Kernel.tune)
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
