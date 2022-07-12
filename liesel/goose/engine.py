"""
# MCMC engine

This module is experimental. Expect API changes.
"""

# mypy: check-untyped-defs

import logging
import pickle
import warnings
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Sequence, cast

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
    KeyArray,
    ModelInterface,
    ModelState,
    Position,
    PyTree,
    QuantityGenerator,
    TransitionInfo,
)

logger = logging.getLogger(__name__)


class KernelErrorLog(NamedTuple):
    """
    Holds the number of the transitions in which an error in at least one chain
    occured and an array with the error code for each chain. Additionally, the
    kernel identifier is specified and optionally the cls of the kernel.

    - transition is an 1-D array (time)
    - error_codes is a 2-D array (chain, time)
    """

    kernel_ident: str
    kernel_cls: Option[type]  # needed to use the error book
    transition: np.ndarray
    error_codes: np.ndarray


ErrorLog = dict[str, KernelErrorLog]


def _expand_and_stack(chunk, *rest):
    chunks = [chunk]
    chunks.extend(rest)
    expended_chunks = [jnp.expand_dims(chunk, 0) for chunk in chunks]
    return jnp.concatenate(expended_chunks, axis=0)


def stack_for_multi(chunks: list):
    """
    Combine identically structured pytrees to be used in multichain.

    The function adds a new dimension (axis 0) to each leaf and stacks the leafs
    along the new axis.

    **deprecated**
    """

    warnings.warn(
        "`stack_for_multi` is deprecated. Please use the functions"
        " in the `pytree` module.",
        DeprecationWarning,
    )

    return jax.tree_util.tree_map(
        lambda x, *xs: _expand_and_stack(x, *xs), chunks[0], *chunks[1:]
    )


@partial(jax.jit, static_argnums=1)
def _split_keys(keys, n):
    keys = jax.lax.map(lambda key: jax.random.split(key, n), keys)
    return keys


def _initialze_prng(seed: int | KeyArray) -> KeyArray:
    if jnp.isscalar(seed):
        return jax.random.PRNGKey(seed)  # type: ignore
    elif jnp.shape(seed) == (2,):  # type: ignore
        return seed  # type: ignore
    else:
        raise ValueError("Seed has an unsupported shape")


def _add_time_dimension(x: PyTree) -> PyTree:
    """
    Adds a new dimension for time to each leaf.

    The returned tree has the same structure with one additional dimension of
    size 1. The new dimension is `axis=1`. Each leaf must have at least one
    dimension (representing the chain index).
    """
    initial_position = jax.tree_util.tree_map(
        lambda y, *_ys: jnp.expand_dims(y, 1),
        x,
    )
    return initial_position


@register_dataclass_as_pytree
@dataclass(frozen=True)
class Carry:
    kernel_states: KernelStates
    model_state: ModelState
    epoch: EpochState


@dataclass
class SamplingResult:
    """
    Contains the results of the MCMC engine.

    Easy access to the samples is provided via the methods
    `get_samples` and `get_posterior_samples`.
    """

    positions: EpochChainManager
    transition_infos: EpochChainManager
    generated_quantities: Option[EpochChainManager]
    tuning_infos: Option[Chain]
    kernel_states: Option[EpochChainManager]
    full_model_states: Option[EpochChainManager]
    kernel_classes: Option[dict[str, type]]

    def get_samples(self) -> Position:
        opt: Option[Position] = self.positions.combine_all()
        return opt.expect(f"No samples in {repr(self)}")

    def get_posterior_samples(self) -> Position:
        opt = self.positions.combine_filtered(
            lambda config: config.type == EpochType.POSTERIOR
        )
        return opt.expect(f"No posterior samples in {repr(self)}")

    def get_posterior_transition_infos(self) -> dict[str, TransitionInfo]:
        opt = self.transition_infos.combine_filtered(
            lambda config: config.type == EpochType.POSTERIOR
        )
        return opt.expect(f"No posterior transition infos in {repr(self)}")

    def get_tuning_times(self) -> Option[Array]:
        if self.tuning_infos.is_none():
            return Option.none()

        # opt_tis is not None since self.tuning_infos is not None
        opt_tis = self.tuning_infos.unwrap().get().unwrap()

        time: Array = next(iter(opt_tis.values())).time

        return Option(time)

    def get_error_log(self, posterior_only=False) -> Option[ErrorLog]:
        """
        returns the error log

        that is an dict[kernel_name, KernelErrorLog]
        """
        opt: Option[TransitionInfos]
        if posterior_only:
            opt = self.transition_infos.combine_filtered(
                lambda config: config.type == EpochType.POSTERIOR
            )
            if opt.is_none():
                return Option(None)
            else:
                tis = opt.expect(f"No posterior transition infos in {repr(self)}")
        else:
            opt = self.transition_infos.combine_all()
            tis = opt.expect(f"No transition infos in {repr(self)}")

        error_log: ErrorLog = {}
        for ker_name in tis:
            mask = np.any(tis[ker_name].error_code != 0, axis=0)
            transition: np.ndarray = np.where(mask)[0]
            # cast is ok since the object has more dimensions in the leaf
            error_codes: np.ndarray = cast(np.ndarray, tis[ker_name].error_code)[
                :, mask
            ]
            cls = self.kernel_classes.map(lambda d: d[ker_name])
            error_log[ker_name] = KernelErrorLog(ker_name, cls, transition, error_codes)
        return Option(error_log)

    def pkl_save(self, path) -> None:
        """Save result as a pickled object under `path`."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def pkl_load(path) -> "SamplingResult":
        """Loads the pickled object from `path`."""
        with open(path, "rb") as f:
            return pickle.load(f)


class Engine:
    """MCMC engine capable of combining multiple transition kernels."""

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
    ):
        # fill slots that can be filled directly
        self._inital_states = model_states
        self._seeds = seeds
        self._jitted_sample_duration = jitted_sample_duration
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
            [ker.needs_history for ker in self._kernel_sequence._kernels]
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

        # initialize kernel state
        keys = self._split_prng_key_one()
        self._kernel_states = jax.vmap(self._kernel_sequence.init_states)(
            keys, self._model_states
        )

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

    @property
    def current_epoch(self) -> EpochState:
        """
        Returns the current epoch.

        Raises a `RuntimeError` if no epoch is active.
        """
        if self._epoch is None:
            raise RuntimeError("No active epoch")

        return self._epoch

    def sample_all_epochs(self):
        """
        Runs sampling for all remaining epochs.

        Auto-tuning methods are called automatically.
        """
        while self._epoch_manager.has_more():
            self.sample_next_epoch()

    def sample_next_epoch(self):
        """Runs sampling for the next epoch assuming no epoch is active."""
        self._start_epoch()

        # special treatment for the initial values
        if self.current_epoch.config.type == EpochType.INITIAL_VALUES:
            self._handle_inital_values_epoch()
            return

        self._kernel_start_epoch()

        duration = self.current_epoch.config.duration
        epoch_type = self.current_epoch.config.type.name
        jitted = self._jitted_sample_duration

        if self._show_progress:
            logger.info(
                f"Starting epoch: {epoch_type}, {duration} transitions, "
                f"{jitted} jitted together"
            )

        self._sample_for_duration(duration=duration)
        self._end_epoch()

    def append_epoch(self, epoch: EpochConfig):
        """Appends an epoch to the epochs that should be sampled."""
        self._epoch_manager.append(epoch)

    def is_sampling_done(self) -> bool:
        """Returns true if all configured epochs have been sampled."""
        return not self._epoch_manager.has_more()

    def get_results(self) -> SamplingResult:
        """Returns the results of the sampling process."""
        if self._store_kernel_states:
            ksc = self._kernel_state_chain
        else:
            ksc = None

        if self._quantity_generators:
            gqs = self._quantities_chain
        else:
            gqs = None

        kernels_cls: dict[str, type] = {
            ker.identifier: type(ker) for ker in self._kernel_sequence.get_kernels()
        }

        return SamplingResult(
            positions=self._position_chain,
            transition_infos=self._transition_info_chain,
            generated_quantities=Option(gqs),
            tuning_infos=Option(self._tuning_info_chain),
            kernel_states=Option(ksc),
            full_model_states=Option(None),
            kernel_classes=Option(kernels_cls),
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

        Calls `end_warmup` for each kernel. From now on, only epochs of type
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

    def _end_epoch(self):
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

        self._tune_kernels(epoch)

        if self._show_progress:
            ti_option = self._transition_info_chain.get_current_chain().get()

            def count_non_zero_error_codes(tis: TransitionInfos):
                cts = {}
                for kernel_id, ti in tis.items():
                    nzero = jnp.sum(ti.error_code != 0, axis=1)
                    ntrans = ti.error_code.shape[1]  # type: ignore
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

    def _tune_kernels(self, epoch: EpochState):
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
            self._tuning_info_chain.append(_add_time_dimension(x=tune_output.infos))

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
                for id in tinfos.keys():
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

    def _sample_for_duration(self, duration: int):
        if self.current_epoch.time_left() < duration:
            raise RuntimeError("Not enough time left in epoch")

        if duration % self._jitted_sample_duration:
            raise RuntimeError(
                f"Duration {duration} is not a multiple of the "
                f"jitted sampling duration {self._jitted_sample_duration}"
            )

        # convert to non-weak device arrays to avoid recompilation
        self._epoch = as_strong_pytree(self._epoch)
        self._kernel_states = as_strong_pytree(self._kernel_states)
        self._model_states = as_strong_pytree(self._model_states)

        it = range(duration // self._jitted_sample_duration)

        if self._show_progress:
            it = tqdm(it, ncols=80, disable=None, unit="chunk")

        for dur_i in it:
            # FIXME: split for entire duration instead of each loop iteration
            keys = self._split_prng_key(self._jitted_sample_duration)
            (
                new_epoch,
                new_ks,
                new_ms,
                position_chain,
                infos,
                ksc,
                quants,
            ) = self._sample_many_jitted(
                keys, self.current_epoch, self._kernel_states, self._model_states
            )

            self._epoch = new_epoch
            self._kernel_states = new_ks
            self._model_states = new_ms
            self._position_chain.append(position_chain)
            self._transition_info_chain.append(infos)
            if self._store_kernel_states:
                self._kernel_state_chain.append(ksc)
            if self._quantity_generators:
                self._quantities_chain.append(quants)
