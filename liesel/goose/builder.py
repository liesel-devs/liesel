"""
# MCMC engine builder

The purpose of the engine builder is to provide a simple API to gradually assemble
the components needed by the MCMC engine. The builder is responsible of returning an
engine in a well-defined state. Furthermore, the builder can return different engine
implementations.
"""

import math
from collections.abc import Iterable
from typing import cast

import jax
import jax.numpy as jnp

from liesel.option import Option

from .engine import Engine
from .epoch import EpochConfig, EpochManager
from .kernel_sequence import KernelSequence
from .pytree import stack_leaves
from .types import Kernel, KeyArray, ModelInterface, ModelState, QuantityGenerator
from .warmup import stan_epochs


def _find_duplicate(xs: list[str]) -> Option[str]:
    """Checks if a list of strings contains any duplicates."""
    set_ = set()

    for x in xs:
        if x in set_:
            return Option(x)
        else:
            set_.add(x)

    return Option(None)


class EngineBuilder:
    """
    The `EngineBuilder` is used to construct an MCMC Engine.

    Currently, the `EngineBuilder` builds an object of the class `Engine`.

    By default, every position key associated with an MCMC kernel is tracked.
    This behavior can be adjusted with the fields `positions_included`
    and `positions_excluded`.

    ## Parameters

    - `seed`: Used to initialize the PRNG for the building process. The PRNG state
      is also used to initialize the PRNG state of the engine. A user-specific seed
      to initialize the engine can be set with the method `set_engine_seed`.
    - `num_chains`: The number of chains to be used.
    """

    def __init__(self, seed: int, num_chains: int):
        keys = jax.random.split(jax.random.PRNGKey(seed))
        self._prng_key: KeyArray = keys[0]
        self._engine_key: KeyArray = keys[1]
        self._num_chains: int = num_chains
        self._kernels: list[Kernel] = []
        self._quantity_generators: list[QuantityGenerator] = []
        self._model_state: Option[ModelState] = Option(None)

        self._model: Option[ModelInterface] = Option(None)

        # public fields, only simple states
        self.store_kernel_states: bool = False
        self.minimize_transition_infos: bool = False
        self.show_progress: bool = True

        self.positions_included: list[str] = []
        """List of additional position keys that should be tracked."""

        self.positions_excluded: list[str] = []
        """
        List of position keys that should not be tracked.

        Excluded keys override additional keys.
        """

    def set_engine_seed(self, seed: int | KeyArray):
        """Sets a seed used to initialize the MCMC engine."""
        if jnp.isscalar(seed):
            seed_int: int = cast(int, seed)
            self._engine_key = jax.random.PRNGKey(seed_int)
        else:
            seed_keyarray = cast(KeyArray, seed)
            self._engine_key = seed_keyarray

    @property
    def engine_seed(self) -> KeyArray:
        return self._engine_key

    def add_kernel(self, kernel: Kernel):
        """Adds a `Kernel`."""
        self._kernels.append(kernel)

    @property
    def kernels(self) -> tuple[Kernel, ...]:
        return tuple(self._kernels)

    def add_quantity_generator(self, generator: QuantityGenerator):
        """Adds a `QuantityGenerator`."""
        self._quantity_generators.append(generator)

    @property
    def quantity_generators(self) -> tuple[QuantityGenerator, ...]:
        return tuple(self._quantity_generators)

    def set_initial_values(self, model_state: ModelState, multiple_chains=False):
        """
        Sets the initial model state.

        If `multiple_chains` is true the `model_state` will be used as is;
        otherwise `model_state` will be used as the initial values for each chain.
        Note that if `multiple_chains` is true, the first axis of each leaf of
        `model_state` refers to the chain.
        """
        if not multiple_chains:
            model_states = stack_leaves(model_state for _ in range(self._num_chains))

        self._model_state = Option(model_states)

    @property
    def model_state(self) -> Option[ModelState]:
        return self._model_state

    def set_epochs(self, epochs: Iterable[EpochConfig]):
        """Sets epochs."""
        self._epochs = EpochManager(epochs)

    def set_duration(
        self,
        warmup_duration: int,
        posterior_duration: int,
        term_duration: int = 50,
        thinning_posterior: int = 1,
        thinning_warmup: int = 1,
    ):
        """
        Sets epochs using the `stan_epochs` function.

        Note that `term_duration` needs to be long enough that tuning algorithms
        like dual averaging can converge.
        """
        epochs = stan_epochs(
            warmup_duration,
            posterior_duration,
            term_duration=term_duration,
            thinning_posterior=thinning_posterior,
            thinning_warmup=thinning_warmup,
        )
        self._epochs = EpochManager(epochs)

    @property
    def epochs(self) -> tuple[EpochConfig, ...]:
        return tuple(self._epochs._configs)

    def set_model(self, model: ModelInterface):
        """Sets the model interface for all kernels and quantity generators."""
        self._model = Option(model)

    def build(self) -> Engine:
        """Builds the MCMC engine with the provided setup."""
        # build list of position keys
        pos_keys: list[str] = []
        for ker in self._kernels:
            pos_keys.extend(ker.position_keys)
        dupl = _find_duplicate(pos_keys)
        if dupl.is_some():
            raise RuntimeError(
                f"The position key {dupl.unwrap()} is claimed by multiple kernels"
            )

        pos_keys.extend(self.positions_included)
        pos_keys = [key for key in pos_keys if key not in self.positions_excluded]

        # find good jittable number
        epochs = self._epochs._configs  # FIXME: use of private field
        durations = [e.duration for e in epochs[1:]]
        jit_duration = math.gcd(*durations)

        # seeds
        seeds = self._engine_key
        if seeds.shape == (2,):
            # no multi-chain key
            seeds = jax.random.split(seeds, self._num_chains)
        if seeds.shape != (self._num_chains, 2):
            raise RuntimeError(
                f"MCMC seed has the wrong dimensions {seeds}. "
                f"Expected is {(self._num_chains, 2)}"
            )

        # check for duplicated identifiers in self._quantity_generators
        idents = []
        for qg in self._quantity_generators:
            idents.append(qg.identifier)

        dupl = _find_duplicate(idents)
        if dupl.is_some():
            raise RuntimeError(
                f"The identifier {dupl.unwrap()} is used by multiple "
                "quantity generators"
            )

        # set model interface for all kernels and quantity generators
        model = self._model.expect("Model interface must be set")
        for ker in self.kernels:
            if not ker.has_model():
                ker.set_model(model)
        for qg in self.quantity_generators:
            if not qg.has_model():
                qg.set_model(model)

        # assign identifiers to kernels
        for idx, ker in enumerate(self.kernels):
            if not ker.identifier:
                ker.identifier = f"kernel_{idx:02d}"

        return Engine(
            seeds=seeds,
            model_states=self._model_state.expect("Model state must be set"),
            kernel_sequence=KernelSequence(self.kernels),
            epoch_configs=epochs,
            jitted_sample_duration=jit_duration,
            model=model,
            position_keys=pos_keys,
            minimize_transition_infos=self.minimize_transition_infos,
            store_kernel_states=self.store_kernel_states,
            quantity_generators=self.quantity_generators,
            show_progress=self.show_progress,
        )
