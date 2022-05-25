"""
# Type aliases, type variables and protocols
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, NewType, Protocol, Sequence, TypeVar

import jax.numpy

if TYPE_CHECKING:
    from .epoch import EpochState
    from .kernel import (
        DefaultTransitionInfo,
        TransitionOutcome,
        TuningOutcome,
        WarmupOutcome,
    )


# simple type aliases

PyTree = Any
Array = Any

ModelState = PyTree
Position = NewType("Position", dict[str, Array])
KernelState = PyTree
KeyArray = jax.random.KeyArray


class TuningInfo(Protocol):
    error_code: int
    time: int


class TransitionInfo(Protocol):
    error_code: int
    """An error code defined in the error book of the kernel.
    0 if no errors occurred during the transition."""

    acceptance_prob: float
    """The acceptance probability of a proposal of a Metropolis-Hastings kernel
    or the average acceptance probability across a trajectory of a NUTS-type kernel.
    99.0 if not used by the kernel."""

    position_moved: int
    """0 if the position did not move during the transition, 1 if it *did* move,
    and 99 if unknown."""

    @abstractmethod
    def minimize(self) -> DefaultTransitionInfo:
        raise NotImplementedError


TKernelState = TypeVar("TKernelState", bound=KernelState)
TTransitionInfo = TypeVar("TTransitionInfo", bound=TransitionInfo)
TTuningInfo = TypeVar("TTuningInfo", bound=TuningInfo)


class ModelInterface(Protocol):
    """
    The model interface defines a standardized way for Goose to communicate
    with a statistical model, that is, to update the model state and to compute
    the log-probability.
    """

    @abstractmethod
    def extract_position(
        self, position_keys: Sequence[str], model_state: ModelState
    ) -> Position:
        """
        Extracts the position from the model state given a sequence of position keys.

        The returned position must have the same order as the supplied key sequence.
        """

        raise NotImplementedError

    @abstractmethod
    def update_state(self, position: Position, model_state: ModelState) -> ModelState:
        """Updates the model state with the values in the position."""

        raise NotImplementedError

    @abstractmethod
    def log_prob(self, model_state: ModelState) -> float:
        """Computes the unnormalized log-probability given the model state."""

        raise NotImplementedError


class Kernel(Protocol[TKernelState, TTransitionInfo, TTuningInfo]):
    """Protocol for a transition kernel."""

    error_book: ClassVar[dict[int, str]]
    """
    Maps error codes to error messages.

    We use -1 if an error code is required but the kernel is skipped.
    """

    needs_history: ClassVar[bool] = False
    """Is set to true if the kernel expects the history for tuning."""

    position_keys: tuple[str, ...]
    """Keys for which the kernel handles the transition."""

    identifier: str = ""
    """
    An identifier for the kernel object that is set by the `EngineBuilder`
    if it is an empty string.
    """

    @abstractmethod
    def set_model(self, model: ModelInterface):
        ...

    @abstractmethod
    def has_model(self) -> bool:
        ...

    @abstractmethod
    def init_state(self, prng_key: KeyArray, model_state: ModelState) -> KernelState:
        """Creates the initial kernel state."""

        raise NotImplementedError

    @abstractmethod
    def transition(
        self,
        prng_key: KeyArray,
        kernel_state: TKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[TKernelState, TTransitionInfo]:
        """Handles one transition. Must be jittable."""

        raise NotImplementedError

    @abstractmethod
    def tune(
        self,
        prng_key: KeyArray,
        kernel_state: TKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None,
    ) -> TuningOutcome[TKernelState, TTuningInfo]:
        """
        Called after each warmup epoch.

        `history` may be `None` if class variable `needs_history` is `False`.

        Must be jittable.
        """

        raise NotImplementedError

    @abstractmethod
    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: TKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TKernelState:
        """
        Called at the beginning of an epoch. Must be jittable.
        """

        raise NotImplementedError

    @abstractmethod
    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: TKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TKernelState:
        """
        Called at the end of an epoch. Must be jittable.
        """

        raise NotImplementedError

    @abstractmethod
    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_state: TKernelState,
        model_state: ModelState,
        tuning_history: TTuningInfo | None,
    ) -> WarmupOutcome[TKernelState]:
        """
        Asks the kernel to inspect the warmup history and react to it.

        This method is executed once the first non-warmup epoch is encountered
        and before `start_epoch` is called.

        `tuning_history` is `None` if no tuning has happened prior to the first
        non-warmup epoch. Otherwise `tuning_history` has the same structure as returned
        from `tune()` but each leaf has an additional dimension. The first dimension
        refers to the n-the tuning.

        Must be jittable.

        The signature is likely to change.
        """

        raise NotImplementedError


class GeneratedQuantity(Protocol):
    error_code: int


TGeneratedQuantity = TypeVar(
    "TGeneratedQuantity", bound=GeneratedQuantity, covariant=True
)


class QuantityGenerator(Protocol[TGeneratedQuantity]):
    error_book: ClassVar[dict[int, str]]
    identifier: str

    @abstractmethod
    def set_model(self, model: ModelInterface):
        raise NotImplementedError

    @abstractmethod
    def has_model(self) -> bool:
        raise NotImplementedError

    def generate(
        self,
        prng_key: KeyArray,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TGeneratedQuantity:
        """Generates a new quantity based on the model and PRNG state."""
