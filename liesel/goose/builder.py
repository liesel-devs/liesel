"""
MCMC engine builder

The purpose of the engine builder is to provide a simple API to gradually assemble
the components needed by the MCMC engine. The builder is responsible of returning an
engine in a well-defined state. Furthermore, the builder can return different engine
implementations.
"""

import logging
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
from .types import (
    JitterFunctions,
    Kernel,
    KeyArray,
    ModelInterface,
    ModelState,
    QuantityGenerator,
)
from .warmup import stan_epochs

logger = logging.getLogger(__name__)


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
    The :class:`.EngineBuilder` is used to construct an MCMC Engine.

    .. rubric:: Workflow

    The general workflow usually looks something like this:

    #. Create a builder with :class:`.EngineBuilder`.
    #. Set the desired number of warmup and posterior samples :meth:`.set_duration`.
    #. Set the model interface with :meth:`.set_model`.
    #. Set the initial values with :meth:`.set_initial_values`.
    #. Add MCMC kernels with :meth:`.add_kernel`.
    #. Build an :class:`~.goose.Engine` with :meth:`.build`.

    Optionally, you can also:

    - Add position keys to :attr:`.positions_included` for tracking. If you are using a
      :class:`.Model` object, position keys are the names of variables or nodes in the
      model. Refer to :attr:`.positions_included` for more information.
    - Add custom jittering for start values with :meth:`.set_jitter_fns`.

    Parameters
    ----------
    seed
        Either an int or a key generated from jax.random.PRNGKey. Used for jittering
        initial values and MCMC sampling.
    num_chains
        The number of chains to be used.

    See Also
    --------
    ~.goose.Engine : The MCMC engine, output of :meth:`.build`. ~.goose.LieselInterface
    : Interface for a :class:`~liesel.model.model.Model` object. ~.goose.NUTSKernel :
    The NUTS kernel. ~.goose.HMCKernel : The HMC kernel. ~.goose.IWLSKernel : The IWLS
    kernel. ~.goose.RWKernel : The random walk kernel.

    Notes
    -----

    By default, only position keys associated with an MCMC kernel is tracked. This
    behavior can be adjusted with the fields :attr:`.positions_included` and
    :attr:`.positions_excluded`.

    Examples
    --------

    For this example, we import ``tensorflow_probability`` as follows:

    >>> import tensorflow_probability.substrates.jax.distributions as tfd

    First, we set up a minimal model:

    >>> mu = lsl.param(0.0, name="mu")
    >>> dist = lsl.Dist(tfd.Normal, loc=mu, scale=1.0)
    >>> y = lsl.obs(jnp.array([1.0, 2.0, 3.0]), dist, name="y")
    >>> model = lsl.GraphBuilder().add(y).build_model()

    Now we initialize the EngineBuilder and set the desired number of warmup and
    posterior samples:

    >>> builder = gs.EngineBuilder(seed=1, num_chains=4)
    >>> builder.set_duration(warmup_duration=1000, posterior_duration=1000)

    Next, we set the model interface and initial values:

    >>> builder.set_model(gs.LieselInterface(model))
    >>> builder.set_initial_values(model.state)

    We add a NUTS kernel for the parameter ``"mu"``:

    >>> builder.add_kernel(gs.NUTSKernel(["mu"]))

    Finally, we build the engine:

    >>> engine = builder.build()

    From here, you can continue with :meth:`~.goose.Engine.sample_all_epochs` to draw
    samples from your posterior distribution.
    """

    def __init__(self, seed: int | KeyArray, num_chains: int):
        if isinstance(seed, int):
            keys = jax.random.split(jax.random.PRNGKey(seed), 3)
        elif isinstance(seed, jax.Array):
            keys = jax.random.split(seed, 3)
        else:
            raise TypeError(
                "Provide either an int or a key from jax.random.PRNGKey as seed."
            )

        self._prng_key: KeyArray = keys[0]
        self._engine_key: KeyArray = keys[1]
        self._jitter_key: KeyArray = keys[2]
        self._num_chains: int = num_chains
        self._kernels: list[Kernel] = []
        self._quantity_generators: list[QuantityGenerator] = []
        self._model_state: Option[ModelState] = Option(None)
        self._model: Option[ModelInterface] = Option(None)
        self._jitter_fns: Option[JitterFunctions] = Option(None)

        # public fields, only simple states
        self.store_kernel_states: bool = False
        self.minimize_transition_infos: bool = False
        self.show_progress: bool = True
        """Whether to show progress bars during sampling."""

        self.positions_included: list[str] = []
        """
        List of additional position keys that should be tracked.

        If a position key is tracked that means the correspond element of the model
        state will be saved and included in the :class:`~.goose.SamplingResults` and the
        posterior samples returned by
        :meth:`~.goose.SamplingResults.get_posterior_samples`.

        By default, only position keys associated with an MCMC kernel are tracked. You
        can easily add additional position keys by appending to this list.

        Examples
        --------

        For this example, we import ``tensorflow_probability`` as follows:

        >>> import tensorflow_probability.substrates.jax.distributions as tfd

        Consider the following simple model, in which we use the logarithm of the scale
        parameter in a normal distribution and take the exponential value for including
        the actual scale:

        >>> log_scale = lsl.param(0.0, name="log_scale")
        >>> scale = lsl.Calc(jnp.exp, variance, _name="scale")
        >>> dist = lsl.Dist(tfd.Normal, loc=0.0, scale=scale)
        >>> y = lsl.obs(jnp.array([1.0, 2.0, 3.0]), dist, name="y")
        >>> model = lsl.GraphBuilder().add(y).build_model()

        Now we might want to set up an engine builder with a NUTS kernel for the
        parameter ``"log_scale"``:

        >>> builder = gs.EngineBuilder(seed=1, num_chains=4)
        >>> builder.set_model(gs.LieselInterface(model))
        >>> builder.add_kernel(gs.NUTSKernel(["log_scale"]))

        By default, only the position key ``"log_scale"`` is tracked and will be
        included in the results. Now, if you also want the value of ``"scale"`` to be
        included, you can add it to the list of included position keys:

        >>> builder.position_keys.append("scale")
        >>> builder.position_keys
        ['scale']

        Beware however that including many intermediate position keys can lead to large
        results. In some cases it may be preferable to keep the tracked positions to a
        minimum and recompute the intermediate values from the posterior samples.

        """
        self.positions_excluded: list[str] = []
        """List of position keys that should not be tracked. Excluded keys override
        additional keys."""

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
        """The seed for the engine's pseudo-random number generation."""
        return self._engine_key

    def add_kernel(self, kernel: Kernel):
        """Adds a :class:`.Kernel`."""
        self._kernels.append(kernel)

    @property
    def kernels(self) -> tuple[Kernel, ...]:
        """Tuple of all Kernels that are present in the builder."""
        return tuple(self._kernels)

    def add_quantity_generator(self, generator: QuantityGenerator):
        """Adds a :class:`.QuantityGenerator`."""
        self._quantity_generators.append(generator)

    @property
    def quantity_generators(self) -> tuple[QuantityGenerator, ...]:
        """Tuple of all quantity generators present in the builder."""
        return tuple(self._quantity_generators)

    def set_initial_values(self, model_state: ModelState, multiple_chains=False):
        """
        Sets the initial model state.

        If :attr:`.multiple_chains` is true the :attr:`.model_state` will be used as is;
        otherwise :attr:`.model_state` will be used as the initial values for each
        chain. Note that if :attr:`.multiple_chains` is true, the first axis of each
        leaf of :attr:`.model_state` refers to the chain.
        """
        if not multiple_chains:
            model_states = stack_leaves(model_state for _ in range(self._num_chains))

        self._model_state = Option(model_states)

    def set_jitter_fns(self, jitter_fns: JitterFunctions | None):
        """
        Set the jittering functions.

        A jittering function is a function that takes as input a key and a value,
        and applies a random jittering (noise) to the input value
        based on the given key.
        If no jitter function is provided a `Warning` will be raised and the
        values won't be jittered.

        Parameters
        ----------
        jitter_fns
            A dictionary where a jittering function is assigned to each position key.

        Examples
        --------

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import tensorflow_probability.substrates.jax.distributions as tfd

        >>> key = jax.random.PRNGKey(42)

        In this example, we show how to use the method
        :meth:`.EngineBuilder.set_jitter_fns` to apply jittering
        to the initial values of each chain.

        First, we sample 500 data points from a Normal Distrbution
        with mean 2.0 and standard deviation 1.0.

        >>> n = 500
        >>> true_mu = 2.0
        >>> true_sigma = 1.0

        >>> x_vec = tfd.Normal(loc=true_mu, scale=true_sigma).sample((n, ), key)

        Then, we define the distribution we want to sample from, which is
        parametrized by a single parameter `mu`.

        >>> mu = lsl.param(1.0, name="mu")
        >>> x_dist = lsl.Dist(tfd.Normal, loc=true_mu, scale=true_sigma)
        >>> x = lsl.Var(x_vec, distribution=x_dist, name="x")

        Now, we can create the model with :class:`.GraphBuilder`.

        >>> gb = lsl.GraphBuilder().add(x)
        >>> model = gb.build_model()

        Finally, we build the model with :class:`.EngineBuilder`. We will use 4
        parallel chains and sample our varaible using a :class:`~.goose.NUTSKernel`.

        >>> builder = gs.EngineBuilder(seed=1337, num_chains=4)
        >>> builder.set_model(gs.LieselInterface(model))
        >>> builder.set_initial_values(model.state)
        >>> builder.add_kernel(gs.NUTSKernel(["mu"]))

        A jitter function takes as input a key and value and applies random jittering
        to the given value using the key. In this case, we apply a uniform noise with
        a minimum value of -1.0 and maximum value of 1.0. Notice that the shape of `val`
        is `(4, 1)`, where the first dimension corresponds to the number of chains.

        >>> def jitter_fn(key, val):
        ...     jitter = jax.random.uniform(key, val.shape, val.dtype, -1.0, 1.0)
        ...     return val + jitter

        The method takes as input a dictionary where a
        jittering function is assigned to each position key.

        >>> builder.set_jitter_fns({"mu": jitter_fn})
        """
        self._jitter_fns = Option(jitter_fns)

    @property
    def jitter_fns(self) -> Option[JitterFunctions]:
        """Jittering functions."""
        return self._jitter_fns

    @property
    def model_state(self) -> Option[ModelState]:
        """Model state."""
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
        Sets epochs using the :func:`.stan_epochs` function.

        Note that :attr:`.term_duration` needs to be long enough that tuning algorithms
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
        """Tuple of epoch configurations."""
        return tuple(self._epochs._configs)

    def set_model(self, model: ModelInterface):
        """Sets the model interface for all kernels and quantity generators."""
        # avoid circular import
        from liesel.model import Model as LieselModel

        if isinstance(model, LieselModel):
            raise TypeError(
                f"{model=} is a `lsl.Model` instance. Please wrap it in a"
                " `gs.LieselInterface`."
            )
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

        # set initial values
        model_states = self._model_state.expect("Model state must be set")

        if self._jitter_fns.is_some():
            jitter_fns = self._jitter_fns.unwrap()
            jitter_fns_pos_keys = list(jitter_fns.keys())
            missing_keys = []

            for pos_key in pos_keys:
                if pos_key not in jitter_fns_pos_keys:
                    missing_keys.append(pos_key)

            if missing_keys:
                pretty_keys = str(missing_keys)[1:-1]
                logger.warning(
                    f"No jitter functions provided for position keys {pretty_keys}. "
                    "The initial values for these keys won't be jittered"
                )

            jitter_keys = jax.random.split(self._jitter_key, len(jitter_fns_pos_keys))
            current_position = model.extract_position(jitter_fns_pos_keys, model_states)
            jittered_position = {}

            for i, pos_key in enumerate(jitter_fns_pos_keys):
                jittered_position[pos_key] = jax.vmap(jitter_fns[pos_key])(
                    jax.random.split(jitter_keys[i], self._num_chains),
                    current_position[pos_key],
                )

            model_states = jax.vmap(model.update_state)(jittered_position, model_states)
        else:
            logger.warning(
                "No jitter functions provided. The initial values won't be jittered"
            )

        # extending position keys
        pos_keys.extend(self.positions_included)
        pos_keys = [key for key in pos_keys if key not in self.positions_excluded]

        return Engine(
            seeds=seeds,
            model_states=model_states,
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
