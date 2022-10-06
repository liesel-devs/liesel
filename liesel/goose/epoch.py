"""
MCMC epochs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, cast

from .pytree import register_dataclass_as_pytree
from .types import PyTree

__docformat__ = "numpy"


class EpochType(IntEnum):
    """Indicates which MCMC phase the epoch is part of."""

    INITIAL_VALUES = 0
    FAST_ADAPTATION = 1
    SLOW_ADAPTATION = 2
    BURNIN = 3
    POSTERIOR = 4

    @staticmethod
    def is_adaptation(epoch_type: EpochType) -> bool:
        """
        Returns ``True`` if the epoch is part of the adaptation phase. Implemented as a
        static method to make it jittable.
        """
        lhs = EpochType.INITIAL_VALUES < epoch_type
        rhs = epoch_type < EpochType.BURNIN
        # using `*` instead of `and` for jax.jit
        # `cast` is a no-op used to satisfy mypy
        return cast(bool, lhs * rhs)

    @staticmethod
    def is_warmup(epoch_type: EpochType) -> bool:
        """
        Returns ``True`` if the epoch is part of the warmup phase.
        Implemented as a static method to make it jittable.
        """
        rhs = EpochType.INITIAL_VALUES < epoch_type
        lhs = epoch_type < EpochType.POSTERIOR
        # using `*` instead of `and` for jax.jit
        # `cast` is a no-op used to satisfy mypy
        return cast(bool, lhs * rhs)


@register_dataclass_as_pytree
@dataclass
class EpochConfig:
    """Defines an Epoch in an MCMC algorithm."""

    type: EpochType
    """
    Type of this epoch.
    """
    duration: int
    """Duration of this epoch in terms of MCMC interations."""
    thinning: int
    """Thinning applied in this epoch."""
    optional: PyTree | None
    """Optional data attachted to this epoch. Must be a PyTree or None."""

    def to_state(self, nth_epoch: int, time_before_epoch: int) -> EpochState:
        """
        Creates initailized :class:`.EpochState` object based on this of
        :class:`.EpochConfig` object.

        Parameters
        ----------
        nth_epoch: position of the epoch in the epoch sequence.
        time_before_epoch
            time (i.e., mcmc iterations) that passed before this epoch; i.e.,
            the sum of all previous epoch durations.
        """
        return EpochState(
            config=self,
            nth_epoch=nth_epoch,
            time=time_before_epoch,
            time_before_epoch=time_before_epoch,
            time_in_epoch=0,
        )


@register_dataclass_as_pytree
@dataclass
class EpochState:
    """
    EpochState describes the state in the current epoch.

    In particular, it calculates how much time is left in the current epoch and
    advances time. It also, provides access to the :class:`.EpochConfig` of the
    current epoch.
    """

    config: EpochConfig
    """:class:`.EpochConfig` of this epoch."""
    nth_epoch: int
    """Position of this epoch in the epoch sequence."""
    time: int
    """Current time."""
    time_before_epoch: int
    """Time passed before this epoch started."""
    time_in_epoch: int
    """Current time within this epoch."""

    def time_left(self) -> int:
        """Returns how much time is left in this epoch."""
        return self.config.duration - self.time_in_epoch

    def advance_time(self, by: int):
        "Advances time by ``by`` units (mcmc iterations)."
        self.time = self.time + by
        self.time_in_epoch = self.time_in_epoch + by


class EpochManager:
    """
    Manages :class:`.EpochConfig` objects.

    A sequence of :class:`.EpochConfig` objects can be handed to the manager either
    during initialization or later with the `append` method. The manager creates
    a new :class:`.EpochState` object with properly initialized time values.

    Furthermore, the :class:`.EpochManager` enforces this invariant:

    - An epoch must be of duration at least 1.
    - An initial values epoch must be of duration exactly 1.
    - The first epoch must be an initial values epoch.
    - No other than the first epoch may be an initial values epoch.
    - A posterior epoch may not be followed by a non-posterior epoch.
    - Temporary: Thinning must be 1.

    The limitation that thinning must be 1 will probably be removed in the
    future. To do so, we need to figure out how kernels relying on history
    tuning can be tuned if the history is thinned.
    """

    def __init__(self, configs: Iterable[EpochConfig] | None):
        self._configs: list[EpochConfig] = []
        self._next_epoch_ptr = 0
        self._next_start_time: int = 0
        self._nth_epoch: int = 0

        if configs is not None:
            for config in configs:
                self.append(config)

    def append(self, config: EpochConfig):
        """Appends an :class:`.EpochConfig` to the list of epochs."""
        if not self._configs and config.type != EpochType.INITIAL_VALUES:
            raise RuntimeError("First epoch must be of type INITIAL_VALUES")

        if config.type == EpochType.INITIAL_VALUES:
            if self._configs:
                raise RuntimeError("Only the first epoch may be of type INITIAL_VALUES")

            if config.duration != 1:
                raise RuntimeError("Epochs of type INITIAL_VALUES must have duration 1")

        if (
            EpochType.is_warmup(config.type)
            and self._configs[-1].type == EpochType.POSTERIOR
        ):
            raise RuntimeError(
                "Warmup epochs may not follow an epoch of type POSTERIOR"
            )

        if config.duration < 1:
            raise RuntimeError("Duration must be greater than or equal to 1")

        if config.thinning < 1:
            raise RuntimeError("Thinning must be greater than or equal to 1")

        if config.thinning != 1:
            if config.duration < config.thinning:
                raise RuntimeError("Duration must be greater than or equal to thinning")

            if config.type == EpochType.POSTERIOR:
                if config.duration % config.thinning != 0:
                    raise RuntimeError("Duration must be a multiple of thinning")

        self._configs.append(config)

    def has_more(self) -> bool:
        """
        Returns ``True`` if there are epoch configs that have not been returned yet by
        the :func:`.next` method, and false otherwise.
        """
        return self._next_epoch_ptr < len(self._configs)

    def next(self) -> EpochState:
        """
        Returns the next epoch with an initialized state.

        Raises a :class:`.RuntimeError` if there are no more epoch configs to return.
        """
        if self.has_more():
            config = self._configs[self._next_epoch_ptr]
            start_time = self._next_start_time
            self._next_start_time += config.duration
            state = config.to_state(self._next_epoch_ptr, start_time)
            self._next_epoch_ptr += 1
            return state
        else:
            raise RuntimeError("No epochs in manager")
