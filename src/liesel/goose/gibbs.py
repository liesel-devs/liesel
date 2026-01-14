"""
Gibbs sampler.
"""

from collections.abc import Callable, Sequence
from typing import ClassVar

from .epoch import EpochState
from .kernel import (
    DefaultTransitionInfo,
    DefaultTuningInfo,
    ModelMixin,
    ReprMixin,
    TransitionOutcome,
    TuningOutcome,
    WarmupOutcome,
)
from .types import Kernel, KernelState, KeyArray, ModelState, Position, TuningInfo

GibbsKernelState = KernelState
GibbsTransitionInfo = DefaultTransitionInfo
GibbsTuningInfo = DefaultTuningInfo


class GibbsKernel(
    ModelMixin,
    Kernel[GibbsKernelState, GibbsTransitionInfo, GibbsTuningInfo],
    ReprMixin,
):
    """
    A Gibbs kernel implementing the :class:`.Kernel` protocol.

    Parameters
    ----------

    position_keys
        Sequence of position keys (variable names) handled by this kernel.
    transition_fn
        Custom transition function that needs to be provided by the user.
    identifier
        A string acting as a unique identifier for this kernel.

    Examples
    --------

    For this example, we import ``tensorflow_probability``, ``jax`` and ``jax.numpy``
    as follows:

    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> import jax
    >>> import jax.numpy as jnp


    First, we set up a minimal model:

    >>> mu = lsl.Var.new_param(0.0, name="mu")
    >>> dist = lsl.Dist(tfd.Normal, loc=mu, scale=1.0)
    >>> y = lsl.Var.new_obs(jnp.array([1.0, 2.0, 3.0]), dist, name="y")
    >>> model = lsl.Model([y])

    Now we initialize the EngineBuilder and set the desired number of warmup and
    posterior samples:

    >>> builder = gs.EngineBuilder(seed=1, num_chains=4)
    >>> builder.set_duration(warmup_duration=1000, posterior_duration=1000)

    Next, we set the model interface and initial values:

    >>> interface = gs.LieselInterface(model)
    >>> builder.set_model(interface)
    >>> builder.set_initial_values(model.state)

    We define a function to sample from the full conditional for the parameter ``"mu"``:

    >>> def sample_mu(prng_key, model_state):
    >>> # extract relevant values from model state
    >>>     pos = interface.extract_position(
    >>>     position_keys=["y", "mu"],
    >>>     model_state=model_state
    >>>     )
    >>> # calculate relevant intermediate quantities
    >>>     n = len(pos["y"])
    >>>     y_mean = pos["y"].mean()
    >>>     mu_new = (n*y_mean + pos["mu"]) / (n + 1)
    >>> # draw new value from full conditional
    >>>     draw = mu_new + jax.random.normal(prng_key)
    >>> # return key-value pair of variable name and new value
    >>>     return {"mu": draw}

    >>> builder.add_kernel(gs.GibbsKernel(["mu"], sample_mu))

    Finally, we build the engine:

    >>> engine = builder.build()

    From here, you can continue with :meth:`~.goose.Engine.sample_all_epochs` to draw
    samples from your posterior distribution.

    See Also
    --------
    :doc:`/tutorials/md/01d-gibbs-sampling`
    """

    error_book: ClassVar[dict[int, str]] = {0: "no errors"}
    needs_history: ClassVar[bool] = False
    identifier: str = ""
    position_keys: tuple[str, ...]

    def __init__(
        self,
        position_keys: Sequence[str],
        transition_fn: Callable[[KeyArray, ModelState], Position],
        identifier: str = "",
    ):
        self._model = None
        self.position_keys = tuple(position_keys)
        self._transition_fn = transition_fn
        self.identifier = identifier

    def init_state(self, prng_key, model_state):
        """
        Initializes an (empty) kernel state.
        """

        return {}

    def transition(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[KernelState, GibbsTransitionInfo]:
        """
        Performs an MCMC transition.
        """

        info = GibbsTransitionInfo(
            error_code=0,
            acceptance_prob=1.0,
            position_moved=1,
        )

        position = self._transition_fn(prng_key, model_state)
        model_state = self.model.update_state(position, model_state)
        return TransitionOutcome(info, kernel_state, model_state)

    def tune(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None = None,
    ) -> TuningOutcome[KernelState, GibbsTuningInfo]:
        """
        Currently does nothing.
        """

        info = GibbsTuningInfo(error_code=0, time=epoch.time)
        return TuningOutcome(info, kernel_state)

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> KernelState:
        """
        Currently does nothing.
        """

        return kernel_state

    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> KernelState:
        """
        Currently does nothing.
        """

        return kernel_state

    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        tuning_history: TuningInfo | None,
    ) -> WarmupOutcome[KernelState]:
        """
        Currently does nothing.
        """

        return WarmupOutcome(error_code=0, kernel_state=kernel_state)
