"""
MCMC engine

This module is experimental. Expect API changes.
"""

# mypy: check-untyped-defs

from __future__ import annotations

import logging
import math
import pickle
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from numbers import Integral, Real
from time import monotonic
from typing import Any, Literal, NamedTuple, NoReturn, cast

import jax
import jax.lax
import jax.numpy as jnp
import jax.random
import jax.tree_util
import numpy as np
from tqdm import tqdm

from liesel.option import Option

from .chain import Chain, EpochChainManager, ListChain
from .epoch import EpochConfig, EpochManager, EpochState, EpochType
from .kernel_sequence import KernelSequence, KernelStates, TransitionInfos, TuningInfos
from .pytree import as_strong_pytree, register_dataclass_as_pytree
from .types import (
    Array,
    GeneratedQuantity,
    Kernel,
    KernelState,
    KeyArray,
    ModelInterface,
    ModelState,
    Position,
    PyTree,
    QuantityGenerator,
    TransitionInfo,
)

logger = logging.getLogger(__name__)

type KernelClass = type[Kernel[Any, Any, Any]]


class KernelErrorLog(NamedTuple):
    """
    Holds the number of the transitions in which an error in at least one chain occured
    and an array with the error code for each chain.

    Additionally, the kernel identifier is specified and optionally the cls of the
    kernel.
    """

    kernel_ident: str
    kernel_cls: Option[KernelClass]  # needed to use the error book

    transition: np.ndarray
    """1-D array (time)."""

    error_codes: np.ndarray
    """2-D array (chain, time)."""


ErrorLog = dict[str, KernelErrorLog]


@partial(jax.jit, static_argnums=1)
def _split_keys(keys, n):
    keys = jax.lax.map(lambda key: jax.random.split(key, n), keys)
    return keys


def _initialze_prng(seed: int | KeyArray) -> KeyArray:
    if jnp.isscalar(seed):
        return jax.random.PRNGKey(seed)
    elif jnp.shape(seed) == (2,):
        return seed
    else:
        raise ValueError("Seed has an unsupported shape")


def _add_time_dimension(x: PyTree) -> PyTree:
    """
    Adds a new dimension for time to each leaf.

    The returned tree has the same structure with one additional dimension of
    size 1. The new dimension is ``axis=1``. Each leaf must have at least one
    dimension (representing the chain index).
    """
    initial_position = jax.tree_util.tree_map(
        lambda y, *_ys: jnp.expand_dims(y, 1),
        x,
    )
    return initial_position


def _arguments_signature(*args: PyTree) -> tuple[Any, ...]:
    leaves, treedef = jax.tree_util.tree_flatten(args)
    return (
        treedef,
        *(
            (leaf.shape, leaf.dtype, getattr(leaf, "weak_type", False))
            for leaf in leaves
        ),
    )


@register_dataclass_as_pytree
@dataclass(frozen=True)
class Carry:
    """
    Holds the state that needs to be carried between MCMC interations.
    """

    kernel_states: KernelStates
    model_state: ModelState
    epoch: EpochState


@dataclass
class SamplingResults:
    """
    Contains the results of the MCMC engine.

    Easy access to the samples is provided via the methods
    :meth:`.get_samples` and :meth:`.get_posterior_samples`.
    """

    positions: EpochChainManager
    """EpochChainManager giving access to monitored variables."""
    transition_infos: EpochChainManager
    """EpochChainManager storing all transition infos."""
    generated_quantities: Option[EpochChainManager]
    """
    Option[EpochChainManager] storing all generated_quantities.

    is_none(), if no quantities have been generated.
    """
    tuning_infos: Option[Chain]
    """
    Option[Chain] storing all tuning infos.

    is_none(), if no tuning was executed
    """

    kernel_states: Option[EpochChainManager]
    """
    Option[EpochChainManager] holds all kernel states.

    is_none(), if monitoring kernel states was not requested.
    """

    full_model_states: Option[EpochChainManager]
    """
    Option[EpochChainManager] holds the full model state of each iteration.

    is_none(), if monitoring was not explicitly requested.
    """

    kernel_classes: Option[dict[str, KernelClass]]
    """
    Optional map of kernel identifier to the respective kernel type.
    """

    kernels_by_pos_key: Option[dict[str, str]]
    """
    Optional map of position key to identifier of the for sampling responsible
    kernel.
    """

    elapsed_wall_time: float | None = None
    """
    Elapsed seconds in the latest top-level Engine sampling call.

    Includes first-use compilation and synchronized epoch finalization, but is
    ``None`` for legacy results or before a sampling call has completed or stopped.
    """

    stop_reason: (
        Literal["completed", "wall_time_reached", "max_wall_time_reached"] | None
    ) = None
    """
    Reason why the latest top-level Engine sampling call stopped.

    ``"completed"`` means the requested finite schedule (or the one epoch requested
    by :meth:`Engine.sample_next_epoch`) completed. ``"wall_time_reached"`` is a
    normal timed stop, and ``"max_wall_time_reached"`` records a safety stop before
    :class:`TimeoutError` is raised. ``None`` denotes legacy results or no completed
    top-level sampling call. Stored arrays remain authoritative for sample counts.
    """

    def get_samples(self) -> Position:
        """
        Returns a dictionary of all samples for all parameters included in the
        position.
        """
        opt: Option[Position] = self.positions.combine_all()
        return opt.expect(f"No samples in {self!r}")

    def get_warmup_samples(self) -> Position:
        """
        Returns a dictionary of adaptation samples for all parameters included in the
        position.
        """
        opt = self.positions.combine_filtered(
            lambda config: EpochType.is_warmup(config.type)
        )
        return opt.expect(f"No warmup samples in {self!r}")

    def get_adaptation_samples(self) -> Position:
        """
        Returns a dictionary of adaptation samples for all parameters included in the
        position.
        """
        opt = self.positions.combine_filtered(
            lambda config: EpochType.is_adaptation(config.type)
        )
        return opt.expect(f"No adaptation samples in {self!r}")

    def get_posterior_samples(self) -> Position:
        """
        Returns a dictionary of posterior samples for all parameters included in the
        position.
        """
        opt = self.positions.combine_filtered(
            lambda config: config.type == EpochType.POSTERIOR
        )
        return opt.expect(f"No posterior samples in {self!r}")

    def get_kernels_by_pos_key(self) -> dict[str, str]:
        """
        Returns a dict, identifying the kernel used to sample each position.

        The dict has the format ``{"position name": "kernel identifier"}``.
        """
        return self.kernels_by_pos_key.expect(
            f"No position-kernel associations in {self!r}"
        )

    def get_pos_keys_by_kernels(self) -> dict[str, list[str]]:
        """
        Returns a dict, identifying the position keys governed by each kernel.

        The dict has the format
        ``{"kernel identifier": ["position key 1", "position key 2"]}``.
        """
        pos_key_by_kernels = {}
        for posname, kernelname in self.get_kernels_by_pos_key().items():
            if kernelname not in pos_key_by_kernels:
                pos_key_by_kernels[kernelname] = [posname]
            else:
                pos_key_by_kernels[kernelname].append(posname)
        return pos_key_by_kernels

    def get_posterior_transition_infos(self) -> dict[str, TransitionInfo]:
        """
        Returns a dictionary of posterior transition information for all parameters
        included in the position.
        """
        opt = self.transition_infos.combine_filtered(
            lambda config: config.type == EpochType.POSTERIOR
        )
        return opt.expect(f"No posterior transition infos in {self!r}")

    def get_warmup_transition_infos(self) -> dict[str, TransitionInfo]:
        """
        Returns a dictionary of posterior transition information for all parameters
        included in the position.
        """
        opt = self.transition_infos.combine_filtered(
            lambda config: EpochType.is_warmup(config.type)
        )
        return opt.expect(f"No warmup transition infos in {self!r}")

    def get_warmup_acceptance_probabilities(self) -> dict[str, Array]:
        """
        Returns dictionary of acceptance probabilities during warmup by kernel.
        """
        transition_infos = self.get_warmup_transition_infos()
        data = {}
        for k, tinfo in transition_infos.items():
            data[k] = jnp.asarray(tinfo.acceptance_prob)
        return data

    def get_warmup_position_moved(self) -> dict[str, Array]:
        """
        Returns dictionary of transition movements (0: no move, 1: move)
        during warmup by kernel.
        """
        transition_infos = self.get_warmup_transition_infos()
        data = {}
        for k, tinfo in transition_infos.items():
            data[k] = jnp.asarray(tinfo.position_moved)
        return data

    def get_posterior_acceptance_probabilities(self) -> dict[str, Array]:
        """
        Returns dictionary of acceptance probabilities during posterior by kernel.
        """
        transition_infos = self.get_posterior_transition_infos()
        data = {}
        for k, tinfo in transition_infos.items():
            data[k] = jnp.asarray(tinfo.acceptance_prob)
        return data

    def get_posterior_position_moved(self) -> dict[str, Array]:
        """
        Returns dictionary of transition movements (0: no move, 1: move)
        during posterior by kernel.
        """
        transition_infos = self.get_posterior_transition_infos()
        data = {}
        for k, tinfo in transition_infos.items():
            data[k] = jnp.asarray(tinfo.position_moved)
        return data

    def get_tuning_times(self) -> Option[Array]:
        """
        Returns array of tuning times.
        """
        if self.tuning_infos.is_none():
            return Option.none()

        # opt_tis is not None since self.tuning_infos is not None
        opt_tis = self.tuning_infos.unwrap().get().unwrap()

        time: Array = next(iter(opt_tis.values())).time

        return Option(time)

    def get_warmup_kernel_states(
        self, process_state: Callable[[KernelState], Any] = asdict
    ) -> dict[str, Any]:
        """
        If available, returns a dictionary of kernel states recorded during
        warmup, organized by kernel.

        The argument ``process_state`` is a callable that is used to process the kernel
        states. The default kernel states in Liesel are dataclasses, which is why the
        default here is ``dataclasses.asdict``.
        """
        kernels = list(self.tuning_infos.expect("none").get().expect("none"))
        states = (
            self.kernel_states.expect("Kernel states not recorded.")
            .combine_filtered(lambda config: EpochType.is_warmup(config.type))
            .expect("none")
        )

        assert len(kernels) == len(states)

        out = {}
        for kernel, state in zip(kernels, states):
            out[kernel] = process_state(state)
        return out

    def get_error_log(self, posterior_only=False) -> Option[ErrorLog]:
        """
        Returns the error log that is an dict[kernel_name, KernelErrorLog]
        """
        opt: Option[TransitionInfos]
        if posterior_only:
            opt = self.transition_infos.combine_filtered(
                lambda config: config.type == EpochType.POSTERIOR
            )
            if opt.is_none():
                return Option(None)
            else:
                tis = opt.expect(f"No posterior transition infos in {self!r}")
        else:
            opt = self.transition_infos.combine_all()
            tis = opt.expect(f"No transition infos in {self!r}")

        error_log: ErrorLog = {}
        for ker_name in tis:
            mask = np.any(tis[ker_name].error_code != 0, axis=0)
            transition: np.ndarray = np.where(mask)[0]
            # cast is ok since the object has more dimensions in the leaf
            error_codes: np.ndarray = cast(np.ndarray, tis[ker_name].error_code)[
                :, mask
            ]
            if self.kernel_classes.is_some():
                cls = Option(self.kernel_classes.unwrap()[ker_name])
            else:
                cls = Option(None)
            error_log[ker_name] = KernelErrorLog(ker_name, cls, transition, error_codes)
        return Option(error_log)

    def pkl_save(self, path) -> None:
        """Save result as a pickled object under :attr:`.path`."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def pkl_load(path) -> SamplingResults:
        """Loads the pickled object from :attr:`.path`."""
        with open(path, "rb") as f:
            return pickle.load(f)


class Engine:
    """
    MCMC engine capable of combining multiple transition kernels.

    Sampling is dispatched in blocks of :attr:`.jit_block_size` transitions per
    chain. Both normal time-based sampling and the exceptional
    :attr:`.max_wall_time` safety limit check elapsed time only after a completed,
    synchronized block. Consequently, limits are soft and can overshoot by one block
    plus any required epoch-end hooks and adaptation tuning. A partial epoch remains
    active and is resumed by the next sampling call; it is not finalized or tuned
    early.

    Notes
    -----
    The constructor argument ``jitted_sample_duration`` is retained for backwards
    compatibility. New code should configure ``EngineBuilder.jit_block_size`` and
    construct Engines through :class:`.EngineBuilder`.
    """

    def __init__(
        self,
        seeds: KeyArray,
        model_states: ModelState,
        kernel_sequence: KernelSequence,
        epoch_configs: Sequence[EpochConfig],
        jitted_sample_duration: int,
        model: ModelInterface,
        position_keys: Sequence[str] | None,
        minimize_transition_infos: bool = False,
        store_kernel_states: bool = False,
        quantity_generators: Sequence[QuantityGenerator] = [],
        show_progress: bool = True,
        max_wall_time: float | None = None,
        _jit_block_size_is_explicit: bool = True,
    ):
        if (
            isinstance(jitted_sample_duration, bool)
            or not isinstance(jitted_sample_duration, Integral)
            or jitted_sample_duration < 1
        ):
            raise ValueError("jit_block_size must be an integer greater than zero")
        jitted_sample_duration = int(jitted_sample_duration)
        for config in epoch_configs:
            if (
                config.type != EpochType.INITIAL_VALUES
                and config.duration % jitted_sample_duration
            ):
                raise ValueError(
                    f"jit_block_size {jitted_sample_duration} must divide epoch "
                    f"duration {config.duration}"
                )

        # fill slots that can be filled directly
        self._inital_states = model_states
        self._seeds = seeds
        self._jit_block_size = jitted_sample_duration
        self._jit_block_size_is_explicit = _jit_block_size_is_explicit
        self._max_wall_time: float | None = None
        self.max_wall_time = max_wall_time
        self._elapsed_wall_time: float | None = None
        self._stop_reason: (
            Literal["completed", "wall_time_reached", "max_wall_time_reached"] | None
        ) = None
        self._minimize_transition_infos = minimize_transition_infos
        self._store_kernel_states = store_kernel_states
        self._model_states = model_states
        self._quantity_generators = quantity_generators
        self._show_progress = show_progress

        self._kernel_sequence = kernel_sequence
        self._epoch_manager = EpochManager(epoch_configs)
        self._warmup_has_ended = False

        if not position_keys:
            position_keys = [
                key
                for ker in self._kernel_sequence._kernels  # FIXME: use of private field
                for key in ker.position_keys
            ]

        self._position_keys = position_keys
        self._model = model

        # feed in history if at least one kernel requires history for tuning
        #
        # FIXME: automatically fetch position keys
        #
        # fetch kernels' position keys and add them automatically to track them
        # in the position chain
        self._history_required_for_tuning = any(
            ker.needs_history for ker in self._kernel_sequence._kernels
        )  # FIXME: use of private field

        self._prng_key = seeds

        # setup storage
        self._position_chain: EpochChainManager = EpochChainManager(apply_thinning=True)
        self._transition_info_chain: EpochChainManager = EpochChainManager()
        self._tuning_info_chain: ListChain = ListChain()
        self._kernel_state_chain: EpochChainManager = EpochChainManager()
        self._quantities_chain: EpochChainManager = EpochChainManager(
            apply_thinning=True
        )

        logger.info("Initializing kernels...")
        # initialize kernel state
        keys = self._split_prng_key_one()
        self._kernel_states = jax.vmap(self._kernel_sequence.init_states)(
            keys, self._model_states
        )
        logger.info("Done")

        # current epoch
        self._epoch: EpochState | None = None

        # jit sample function
        self._sample_many_jitted = jax.jit(
            jax.vmap(
                self._sample_many,
                in_axes=(0, None, 0, 0),
                out_axes=(None, 0, 0, 0, 0, 0, 0),
            )
        )
        self._sample_many_compiled = None
        self._sample_many_compiled_signature = None

    @property
    def jit_block_size(self) -> int:
        """
        Number of transitions per chain executed by one JIT call.

        Smaller values improve wall-clock responsiveness at the cost of more host
        dispatch and synchronization overhead. The value is fixed after construction.
        """
        return self._jit_block_size

    @property
    def max_wall_time(self) -> float | None:
        """
        Optional wall-clock safety limit in seconds for each sampling call.

        Reaching this soft limit at a synchronized JIT-block boundary records
        ``"max_wall_time_reached"`` and raises :class:`TimeoutError`. Assign ``None``
        to disable the limit before resuming. A non-``None`` value requires an
        explicitly configured :attr:`.jit_block_size`.
        """
        return self._max_wall_time

    @max_wall_time.setter
    def max_wall_time(self, value: float | None) -> None:
        if value is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    "max_wall_time must be a finite number greater than zero"
                )
            if not self._jit_block_size_is_explicit:
                raise ValueError(
                    "max_wall_time requires an explicitly configured jit_block_size"
                )
            value = float(value)

        self._max_wall_time = value

    @property
    def current_epoch(self) -> EpochState:
        """
        Returns the current epoch.

        Raises a :exc:`.RuntimeError` if no epoch is active.
        """
        if self._epoch is None:
            raise RuntimeError("No active epoch")

        return self._epoch

    def compile(self) -> None:
        """
        Compile and retain the sampling executable without advancing Engine state.

        Timed sampling otherwise includes lazy JIT compilation. Call this method on
        each Engine before starting its timer when comparing configurations under
        equal sampling budgets. The method is idempotent for an unchanged compilation
        signature. It does not promise to eliminate first-dispatch, allocation, or
        hardware-cache costs.
        """
        epoch = self._epoch
        if epoch is None:
            configs = self._epoch_manager._configs
            next_epoch = self._epoch_manager._next_epoch_ptr
            time_before_epoch = sum(config.duration for config in configs[:next_epoch])
            for nth_epoch, config in enumerate(configs[next_epoch:], next_epoch):
                if config.type != EpochType.INITIAL_VALUES:
                    epoch = config.to_state(nth_epoch, time_before_epoch)
                    break
                time_before_epoch += config.duration

        if epoch is None:
            return

        args = (
            _split_keys(self._prng_key, self._jit_block_size + 1)[:, 1:, :],
            as_strong_pytree(epoch),
            as_strong_pytree(self._kernel_states),
            as_strong_pytree(self._model_states),
        )
        signature = _arguments_signature(*args)
        if (
            self._sample_many_compiled is not None
            and self._sample_many_compiled_signature == signature
        ):
            return

        compiled = self._sample_many_jitted.lower(*args).compile()
        self._sample_many_compiled = compiled
        self._sample_many_compiled_signature = signature

    def sample_all_epochs(self):
        """
        Runs sampling for all remaining epochs.

        Auto-tuning methods are called automatically. If :attr:`.max_wall_time` is
        reached, the method raises :class:`TimeoutError` at a completed JIT-block
        boundary. Completed work and any active partial epoch remain available for
        inspection or a later call that resumes sampling.
        """
        start_time = monotonic()
        safety_limit = self._max_wall_time
        limit_reached, elapsed = self._sample_epochs(start_time, safety_limit)
        if limit_reached:
            assert safety_limit is not None
            self._raise_timeout(safety_limit, elapsed)

        self._elapsed_wall_time = elapsed
        self._stop_reason = "completed"

    def sample_next_epoch(self):
        """
        Run sampling for the next or currently active epoch.

        The configured :attr:`.max_wall_time` applies to this call and raises
        :class:`TimeoutError` at a completed JIT-block boundary. An interrupted epoch
        remains active and is resumed on the next sampling call.
        """
        start_time = monotonic()
        safety_limit = self._max_wall_time
        limit_reached, elapsed = self._sample_next_epoch(start_time, safety_limit)
        if limit_reached:
            assert safety_limit is not None
            assert elapsed is not None
            self._raise_timeout(safety_limit, elapsed)

        self._synchronize_top_level()
        self._elapsed_wall_time = monotonic() - start_time
        self._stop_reason = "completed"

    def sample_for_time(self, wall_time: float) -> None:
        """
        Run sampling until ``wall_time`` or the finite epoch schedule is exhausted.

        ``wall_time`` must be a positive, finite number of seconds, and the Engine
        must have an explicitly configured :attr:`.jit_block_size`. When work exists,
        at least one block is executed. Adaptation, burn-in, and posterior epochs all
        count when present in the remaining schedule; completed epochs are never
        repeated to fill the budget.

        The timer includes lazy compilation, synchronized blocks, and required epoch
        finalization. Checks occur only after blocks, so the horizon can be exceeded by
        one block plus finalization work. Reaching the normal horizon returns with
        ``stop_reason="wall_time_reached"``. Exhausting the schedule first returns
        with ``stop_reason="completed"``.

        If :attr:`.max_wall_time` is shorter than ``wall_time``, a warning is logged
        and the safety limit controls: reaching it records
        ``"max_wall_time_reached"`` and raises :class:`TimeoutError`. Equal limits use
        normal time-based completion.

        Parameters
        ----------
        wall_time
            Positive, finite sampling horizon in seconds.

        See Also
        --------
        compile : Compile without consuming the timed sampling budget.
        """
        start_time = monotonic()
        if (
            isinstance(wall_time, bool)
            or not isinstance(wall_time, Real)
            or not math.isfinite(wall_time)
            or wall_time <= 0
        ):
            raise ValueError("wall_time must be a finite number greater than zero")
        if not self._jit_block_size_is_explicit:
            raise ValueError(
                "sample_for_time requires the caller to configure jit_block_size "
                "because wall-clock overshoot is controlled at JIT-block boundaries"
            )

        wall_time = float(wall_time)
        safety_controls = (
            self._max_wall_time is not None and self._max_wall_time < wall_time
        )
        if safety_controls:
            assert self._max_wall_time is not None
            limit = self._max_wall_time
            logger.warning(
                "max_wall_time %s is shorter than wall_time %s; the safety "
                "limit controls this sampling call",
                limit,
                wall_time,
            )
        else:
            limit = wall_time

        limit_reached, elapsed = self._sample_epochs(start_time, limit)
        if limit_reached:
            if safety_controls:
                self._raise_timeout(limit, elapsed)
            self._elapsed_wall_time = elapsed
            self._stop_reason = "wall_time_reached"
            return

        self._elapsed_wall_time = elapsed
        self._stop_reason = "completed"

    def _sample_epochs(
        self, start_time: float, limit: float | None
    ) -> tuple[bool, float]:
        while self._epoch is not None or self._epoch_manager.has_more():
            limit_reached, elapsed = self._sample_next_epoch(start_time, limit)
            if limit_reached:
                assert elapsed is not None
                return True, elapsed

        self._synchronize_top_level()
        return False, monotonic() - start_time

    def _raise_timeout(self, safety_limit: float, elapsed: float) -> NoReturn:
        self._elapsed_wall_time = elapsed
        self._stop_reason = "max_wall_time_reached"
        raise TimeoutError(
            f"Sampling exceeded max_wall_time {safety_limit} seconds after "
            f"{elapsed} seconds with jit_block_size {self._jit_block_size}; "
            "the Engine can be resumed."
        )

    def _synchronize_top_level(self) -> None:
        outputs = [
            self._kernel_states,
            self._model_states,
            self._tuning_info_chain.get().value,
        ]
        chains = [self._position_chain, self._transition_info_chain]
        if self._store_kernel_states:
            chains.append(self._kernel_state_chain)
        if self._quantity_generators:
            chains.append(self._quantities_chain)

        outputs.extend(
            chain.get_current_chain().get().value
            for chain in chains
            if chain.get_epochs()
        )
        jax.block_until_ready(outputs)

    def _sample_next_epoch(
        self, start_time: float | None = None, safety_limit: float | None = None
    ) -> tuple[bool, float | None]:
        """Runs sampling for the next or currently active epoch."""
        resuming = self._epoch is not None
        if not resuming:
            self._start_epoch()

        # special treatment for the initial values
        if self.current_epoch.config.type == EpochType.INITIAL_VALUES:
            self._handle_inital_values_epoch()
            return False, None

        if not resuming:
            self._kernel_start_epoch()

        duration = int(self.current_epoch.time_left())
        epoch_type = EpochType(int(self.current_epoch.config.type)).name
        jitted = self._jit_block_size

        if self._show_progress:
            logger.info(
                f"Starting epoch: {epoch_type}, {duration} transitions, "
                f"{jitted} jitted together"
            )

        limit_reached, elapsed = self._sample_for_duration(
            duration=duration,
            start_time=start_time,
            safety_limit=safety_limit,
        )
        if self.current_epoch.time_left() == 0:
            tuning_infos = self._end_epoch()
            if safety_limit is not None:
                assert start_time is not None
                jax.block_until_ready((self._kernel_states, tuning_infos))
                elapsed = monotonic() - start_time
                limit_reached = elapsed >= safety_limit

        return limit_reached, elapsed

    def append_epoch(self, epoch: EpochConfig):
        """
        Append an epoch to the remaining sampling schedule.

        Its duration must be divisible by :attr:`.jit_block_size`. This can be used to
        finish adaptation and burn-in first, then append a posterior epoch for a
        posterior-only timed call.
        """
        if (
            epoch.type != EpochType.INITIAL_VALUES
            and epoch.duration % self._jit_block_size
        ):
            raise ValueError(
                f"jit_block_size {self._jit_block_size} must divide epoch duration "
                f"{epoch.duration}"
            )
        self._epoch_manager.append(epoch)

    def is_sampling_done(self) -> bool:
        """Returns true if all configured epochs have been sampled."""
        return self._epoch is None and not self._epoch_manager.has_more()

    def get_results(self) -> SamplingResults:
        """
        Return the currently stored sampling results.

        The result includes ``elapsed_wall_time`` and ``stop_reason`` for the latest
        top-level sampling call, including metadata recorded before a safety
        :class:`TimeoutError` was raised.
        """
        if self._store_kernel_states:
            ksc = self._kernel_state_chain
        else:
            ksc = None

        if self._quantity_generators:
            gqs = self._quantities_chain
        else:
            gqs = None

        kernels = self._kernel_sequence.get_kernels()
        kernels_cls: dict[str, KernelClass] = {
            ker.identifier: type(ker) for ker in kernels
        }

        kernels_by_position: dict[str, str] = {}
        for kernel in kernels:
            kernels_by_position.update(
                {key: kernel.identifier for key in kernel.position_keys}
            )

        return SamplingResults(
            positions=self._position_chain,
            transition_infos=self._transition_info_chain,
            generated_quantities=Option(gqs),
            tuning_infos=Option(self._tuning_info_chain),
            kernel_states=Option(ksc),
            full_model_states=Option(None),
            kernel_classes=Option(kernels_cls),
            kernels_by_pos_key=Option(kernels_by_position),
            elapsed_wall_time=self._elapsed_wall_time,
            stop_reason=self._stop_reason,
        )

    def _split_prng_key(self, n: int = 1) -> KeyArray:
        keys = _split_keys(self._prng_key, n + 1)
        self._prng_key = keys[:, 0, :]
        return keys[:, 1:, :]

    def _split_prng_key_one(self) -> KeyArray:
        key = self._split_prng_key(1)
        return key[:, 0, :]

    def _generate_quantity(self):
        if not self._quantity_generators:
            return None

        quants = {}

        for qg in self._quantity_generators:
            key = self._split_prng_key_one()
            gen_f = jax.vmap(qg.generate, in_axes=(0, 0, None))
            quant = gen_f(key, self._model_states, self.current_epoch)
            quants[qg.identifier] = quant

        return quants

    def _handle_inital_values_epoch(self):
        assert self.current_epoch.config.type == EpochType.INITIAL_VALUES
        self.current_epoch.advance_time(1)

        initial_position = _add_time_dimension(
            x=jax.vmap(self._model.extract_position, in_axes=(None, 0))(
                self._position_keys, self._model_states
            ),
        )
        self._position_chain.append(initial_position)

        if self._store_kernel_states:
            ks = _add_time_dimension(x=self._kernel_states)
            self._kernel_state_chain.append(ks)

        if self._quantity_generators:
            quants = self._generate_quantity()
            quants = _add_time_dimension(x=quants)
            self._quantities_chain.append(quants)

        self._epoch = None

    def _start_epoch(self):
        """Advances to the next epoch."""
        if self._epoch is not None:
            raise RuntimeError("Epoch is active and not completed")

        self._epoch = self._epoch_manager.next()

        # invoke end_warmup() for the first non-warmup epoch
        if (
            not self._warmup_has_ended
            and self.current_epoch.config.type == EpochType.POSTERIOR
        ):
            self._end_warmup()

        # advance chains to next epoch
        self._position_chain.advance_epoch(self.current_epoch.config)
        self._transition_info_chain.advance_epoch(self.current_epoch.config)
        self._kernel_state_chain.advance_epoch(self.current_epoch.config)
        self._quantities_chain.advance_epoch(self.current_epoch.config)

    def _kernel_start_epoch(self):
        """Inform kernels about new epoch."""
        keys = self._split_prng_key_one()
        self._kernel_states = jax.vmap(
            self._kernel_sequence.start_epoch, in_axes=(0, 0, 0, None)
        )(keys, self._kernel_states, self._model_states, self.current_epoch)

    def _end_warmup(self):
        """
        Ends the warmup sequence.

        Calls :func:`.end_warmup` for each kernel. From now on, only epochs of type
        posterior can follow.
        """
        keys = self._split_prng_key_one()
        tuning_infos: TuningInfos | None = self._tuning_info_chain.get().value

        end_warmup_output = jax.vmap(self._kernel_sequence.end_warmup)(
            keys, self._kernel_states, self._model_states, tuning_infos
        )
        self._kernel_states = end_warmup_output.kernel_states

        # add warnings for the user if there are any non-zero error-code
        for kid, ec in end_warmup_output.error_codes.items():
            if jnp.any(ec != 0):
                logger.warning(f"Warmup error code for {kid}: {ec}")

        logger.info("Finished warmup")

    def _end_epoch(self) -> TuningInfos | None:
        """
        End epoch.

        Informs kernels about the end of the epoch and initializes the tuning
        if required.
        """
        # ensure that an epoch is active
        epoch = self.current_epoch

        # inform kernels about end of epoch
        end_keys = self._split_prng_key_one()
        self._kernel_states = jax.vmap(
            self._kernel_sequence.end_epoch, in_axes=(0, 0, 0, None)
        )(end_keys, self._kernel_states, self._model_states, epoch)

        tuning_infos = self._tune_kernels(epoch)

        if self._show_progress:
            ti_option = self._transition_info_chain.get_current_chain().get()

            def count_non_zero_error_codes(tis: TransitionInfos):
                cts = {}
                for kernel_id, ti in tis.items():
                    error_code = jnp.asarray(ti.error_code)
                    nzero = jnp.sum(error_code != 0, axis=1)
                    ntrans = error_code.shape[1]
                    cts[kernel_id] = (nzero, ntrans)
                return cts

            error_info: dict[str, tuple[Array, int]] = ti_option.map_or(
                {}, count_non_zero_error_codes
            )

            for kid, kcts in error_info.items():
                if jnp.any(kcts[0] != 0):
                    logger.warning(
                        f"Errors per chain for {kid}: "
                        f"{', '.join(map(str, kcts[0]))} / {kcts[1]} transitions"
                    )

            logger.info("Finished epoch")

        # no epoch is active anymore
        self._epoch = None
        return tuning_infos

    def _tune_kernels(self, epoch: EpochState) -> TuningInfos | None:
        """Trigger tuning if epoch is an adaptation phase."""
        if EpochType.is_adaptation(epoch.config.type):
            tune_keys = self._split_prng_key_one()
            if self._history_required_for_tuning:
                history = (
                    self._position_chain.get_current_chain()
                    .get()
                    .expect("The history must contain samples.")
                )
            else:
                history = None

            tune_output = jax.vmap(
                self._kernel_sequence.tune, in_axes=(0, 0, 0, None, 0)
            )(tune_keys, self._kernel_states, self._model_states, epoch, history)
            self._kernel_states = tune_output.kernel_states

            # we need to add the time dimension
            tuning_infos = _add_time_dimension(x=tune_output.infos)
            self._tuning_info_chain.append(tuning_infos)
            return tuning_infos

        return None

    def _sample_many(
        self,
        keys: KeyArray,
        epoch: EpochState,
        kernel_states: KernelStates,
        model_state: ModelState,
    ) -> tuple[
        EpochState,
        KernelStates,
        ModelState,
        Position,
        TransitionInfos,
        None | KernelStates,
        None | dict[str, GeneratedQuantity],
    ]:
        def scan_f(
            carry: Carry, key: KeyArray
        ) -> tuple[
            Carry,
            tuple[
                Position,
                TransitionInfos,
                None | KernelStates,
                None | dict[str, GeneratedQuantity],
            ],
        ]:
            key_trans, key_quants = jax.random.split(key)
            epoch = carry.epoch
            out = self._kernel_sequence.transition(
                key_trans, carry.kernel_states, carry.model_state, epoch
            )
            epoch.advance_time(1)
            new_carry = Carry(out.kernel_states, out.model_state, epoch)

            # extract the position specified to store in chain
            position = self._model.extract_position(
                self._position_keys, out.model_state
            )

            # minimize transition infos if requested
            tinfos = out.infos
            if self._minimize_transition_infos:
                for id in tinfos:
                    tinfos[id] = tinfos[id].minimize()

            ks = None
            if self._store_kernel_states:
                ks = new_carry.kernel_states

            quants = None
            if self._quantity_generators:
                quants = {}
                keys = jax.random.split(key_quants, len(self._quantity_generators))
                for i, qg in enumerate(self._quantity_generators):
                    key = keys[i]
                    quant = qg.generate(key, out.model_state, epoch)
                    quants[qg.identifier] = quant

            return new_carry, (position, tinfos, ks, quants)

        inital_carry = Carry(kernel_states, model_state, epoch)
        carry, chain = jax.lax.scan(scan_f, inital_carry, keys)
        kernel_states = carry.kernel_states
        model_state = carry.model_state
        epoch = carry.epoch

        return (
            epoch,
            kernel_states,
            model_state,
            chain[0],
            chain[1],
            chain[2],
            chain[3],
        )

    def _sample_for_duration(
        self,
        duration: int,
        start_time: float | None = None,
        safety_limit: float | None = None,
    ) -> tuple[bool, float | None]:
        if self.current_epoch.time_left() < duration:
            raise RuntimeError("Not enough time left in epoch")

        if duration % self._jit_block_size:
            raise RuntimeError(
                f"Duration {duration} is not a multiple of the "
                f"jit_block_size {self._jit_block_size}"
            )

        # convert to non-weak device arrays to avoid recompilation
        self._epoch = as_strong_pytree(self._epoch)
        self._kernel_states = as_strong_pytree(self._kernel_states)
        self._model_states = as_strong_pytree(self._model_states)

        it = range(duration // self._jit_block_size)

        if self._show_progress:
            it = tqdm(it, ncols=80, unit="chunk")

        for dur_i in it:
            # FIXME: split for entire duration instead of each loop iteration
            keys = self._split_prng_key(self._jit_block_size)
            args = (
                keys,
                self.current_epoch,
                self._kernel_states,
                self._model_states,
            )
            sample_many = self._sample_many_jitted
            if (
                self._sample_many_compiled is not None
                and self._sample_many_compiled_signature == _arguments_signature(*args)
            ):
                sample_many = self._sample_many_compiled
            (
                new_epoch,
                new_ks,
                new_ms,
                position_chain,
                infos,
                ksc,
                quants,
            ) = sample_many(*args)
            self._epoch = new_epoch
            self._kernel_states = new_ks
            self._model_states = new_ms
            self._position_chain.append(position_chain)
            self._transition_info_chain.append(infos)
            if self._store_kernel_states:
                self._kernel_state_chain.append(ksc)
            if self._quantity_generators:
                self._quantities_chain.append(quants)

            if safety_limit is not None:
                assert start_time is not None
                jax.block_until_ready(
                    (new_epoch, new_ks, new_ms, position_chain, infos, ksc, quants)
                )
                elapsed = monotonic() - start_time
                if elapsed >= safety_limit:
                    return True, elapsed

        return False, None
