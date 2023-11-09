"""
Type aliases, type variables and protocols.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, NewType, Protocol, TypeVar

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
KeyArray = Any
JitterFunction = Callable[[KeyArray, Array], Array]
JitterFunctions = dict[str, JitterFunction]


class TuningInfo(Protocol):
    """Holds information about sampler tuning."""

    error_code: int
    time: int


class TransitionInfo(Protocol):
    """Holds information about MCMC transitions."""

    error_code: int
    """
    An error code defined in the error book of the kernel.
    0 if no errors occurred during the transition.
    """

    acceptance_prob: float
    """
    The acceptance probability of a proposal of a Metropolis-Hastings kernel
    or the average acceptance probability across a trajectory of a NUTS-type kernel.
    99.0 if not used by the kernel.
    """

    position_moved: int
    """
    0 if the position did not move during the transition, 1 if it *did* move,
    and 99 if unknown.
    """

    @abstractmethod
    def minimize(self) -> DefaultTransitionInfo:
        raise NotImplementedError


TKernelState = TypeVar("TKernelState", bound=KernelState)
TTransitionInfo = TypeVar("TTransitionInfo", bound=TransitionInfo)
TTuningInfo = TypeVar("TTuningInfo", bound=TuningInfo)


class ModelInterface(Protocol):
    """
    Defines a standardized way for Goose to communicate with a statistical model.

    This means predominantly, to update the model state and to compute the
    log-probability.
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
        """Sets the model interface."""
        ...

    @abstractmethod
    def has_model(self) -> bool:
        """Whether the model interface is set."""
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
        The method can perform automatic tuning of the kernel and is called
        after each adaptation epoch.

        To tune the kernel, the method can return an altered kernel state.

        Must be jittable.

        Parameters
        ----------
        prng_key
            The key for JAX' pseudo-random number generator.
        model_state
            Current model state.
        kernel_state
            Current kernel state.
        epoch
            Current epoch state.
        history
            Holds the history of the position of the current epoch, i.e., that
            is the position but each leave in the pytree is enhanced by one
            dimension (axis = 0) which represents the time or MCMC iteration.
            The value may be ``None`` if the class variable :py:attr:`~needs_histroy` is
            ``False``.
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

        This method is executed once the first non-warmup epoch is encountered and
        before :meth:`.start_epoch` is called.

        ``tuning_history`` is ``None`` if no tuning has happened prior to the first
        non-warmup epoch. Otherwise ``tuning_history`` has the same structure as
        returned from :meth:`.tune` but each leaf has an additional dimension. The first
        dimension refers to the n-the tuning.

        Must be jittable.

        The signature is likely to change.
        """

        raise NotImplementedError


class GeneratedQuantity(Protocol):
    """
    Protocol representing the data structure for quantities generated via
    implmentations of :class:`.QuantityGenerator`.

    Concrete implementations should add additional attributes.

    The attribute ``error_code`` is reserved to store integers that map to error
    messages via the error book provided in the implmentation of
    :class:`.QuantityGenerator`.
    """

    error_code: int


TGeneratedQuantity = TypeVar(
    "TGeneratedQuantity", bound=GeneratedQuantity, covariant=True
)


class QuantityGenerator(Protocol[TGeneratedQuantity]):
    """
    Protocol for a class that calculates a quantity based on the model
    state and a random seed.
    """

    error_book: ClassVar[dict[int, str]]
    identifier: str

    @abstractmethod
    def set_model(self, model: ModelInterface):
        """
        Sets a model.

        Parameters
        ----------
        model
            The model to be set.
        """
        raise NotImplementedError

    @abstractmethod
    def has_model(self) -> bool:
        """*True*, if a model has been set with :meth:`.set_model`."""
        raise NotImplementedError

    def generate(
        self,
        prng_key: KeyArray,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TGeneratedQuantity:
        """
        Generates a new quantity based on the model and PRNG state.

        Parameters
        ----------
        prng_key
            The key for JAX' pseudo-random number generator.
        model_state
            Current model state.
        epoch
            Current epoch state.
        """
