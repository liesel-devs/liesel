"""
Metroplis Hastings kernel. This kernel allows for a user-defined proposal functions and
adds the MH step. Optional, the kernel supports a stepsize adaptation.
"""

from collections.abc import Callable, Sequence
from typing import ClassVar, NamedTuple

import jax

from .da import da_finalize, da_init, da_step
from .epoch import EpochState
from .kernel import (
    DefaultTransitionInfo,
    DefaultTuningInfo,
    ModelMixin,
    ReprMixin,
    TransitionMixin,
    TransitionOutcome,
    TuningOutcome,
    WarmupOutcome,
)
from .mh import mh_step
from .rw import RWKernelState
from .types import KeyArray, ModelState, Position, TuningInfo


class MHProposal(NamedTuple):
    """
    Encapsulates a proposed state and the log-correction for a
    Metropolis-Hastings transition.

    Parameters
    ----------
    position
        A dictionary mapping parameter names to their newly proposed values.
    log_correction
        The Metropolis-Hastings correction in the case of an asymmetric proposal
        distribution. Let :math:`q(x' | x)` be the density of the proposal ``x'`` given
        the current state ``x``, then the ``log_correction`` is defined as
        :math:`log[q(x | x') / q(x' | x)]`.

    See Also
    --------
    :class:`.MHKernel`

    """

    position: Position
    log_correction: float
    """
    Let :math:`q(x' | x)` be the proposal density, then
    :math:`log(q(x | x') / q(x' | x))` is the log_mh_correction.
    """


MHTransitionInfo = DefaultTransitionInfo
MHTuningInfo = DefaultTuningInfo
MHProposalFn = Callable[[KeyArray, ModelState, float], MHProposal]


class MHKernel(ModelMixin, TransitionMixin[RWKernelState, MHTransitionInfo], ReprMixin):
    """
    A Metropolis-Hastings kernel implementing the :class:`.Kernel` protocol.

    Parameters
    ----------
    position_keys
        Sequence of position keys (variable names) handled by this kernel.
    proposal_fn
        Custom proposal function that proposes a new state.
        Needs to be provided by the user.
    initial_step_size
        Value at which to start step size tuning.
    da_tune_step_size
        If ``True``, the step size passed as an argument to the
        proposal function is tuned using the dual averaging algorithm.
        Step size is tuned on the fly during all adaptive epochs.
    da_target_accept
        Target acceptance probability for dual averaging algorithm.
    da_gamma
        The adaptation regularization scale.
    da_kappa
        The adaptation relaxation exponent.
    da_t0
        The adaptation iteration offset.
    identifier
        A string acting as a unique identifier for this kernel.

    Examples
    --------

    To begin, we import ``tensorflow_probability``, ``jax`` and ``jax.numpy``
    as follows:

    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> import jax
    >>> import jax.numpy as jnp


    Then, we set up a minimal model:

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

    We define a function to propose new values for the parameter ``"mu"``:

    >>> def mu_proposal(key, model_state, step_size):
    ...     # extract relevant values from model state
    ...     pos = interface.extract_position(
    ...         position_keys=["mu"], model_state=model_state
    ...     )
    ...     mu_current = pos["mu"]
    ...     # draw epsilon
    ...     epsilon = jax.random.uniform(key, minval=-0.5, maxval=0.5)
    ...     mu_proposed = mu_current + epsilon
    ...     pos = {"mu": mu_proposed}
    ...     return gs.MHProposal(pos, log_correction=0.0)


    >>> builder.add_kernel(gs.MHKernel(["mu"], mu_proposal))

    Finally, we build the engine:

    >>> engine = builder.build()

    From here, you can continue with :meth:`~.goose.Engine.sample_all_epochs` to draw
    samples from your posterior distribution.

    See Also
    --------
    :class:`.MHProposal`

    """

    error_book: ClassVar[dict[int, str]] = {0: "no errors", 90: "nan acceptance prob"}
    """Dict of error codes and their meaning."""
    needs_history: ClassVar[bool] = False
    """Whether this kernel needs its history for tuning."""
    identifier: str = ""
    """Kernel identifier, set by :class:`.EngineBuilder`"""
    position_keys: tuple[str, ...]
    """Tuple of position keys handled by this kernel."""

    def __init__(
        self,
        position_keys: Sequence[str],
        proposal_fn: MHProposalFn,
        initial_step_size: float = 1.0,
        da_tune_step_size=False,
        da_target_accept: float = 0.234,
        da_gamma: float = 0.05,
        da_kappa: float = 0.75,
        da_t0: int = 10,
        identifier: str = "",
    ):
        self._model = None
        self.position_keys = tuple(position_keys)
        self._proposal_fn = proposal_fn
        self.initial_step_size = initial_step_size
        self.da_tune_step_size = da_tune_step_size
        self.da_target_accept = da_target_accept
        self.da_gamma = da_gamma
        self.da_kappa = da_kappa
        self.da_t0 = da_t0
        self.identifier = identifier

    def init_state(self, prng_key, model_state):
        """Initializes the kernel state."""

        return RWKernelState(step_size=self.initial_step_size)

    def _standard_transition(
        self,
        prng_key: KeyArray,
        kernel_state: RWKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[RWKernelState, DefaultTransitionInfo]:
        """Performs an MCMC transition *without* dual averaging."""

        key, subkey = jax.random.split(prng_key)
        step_size = kernel_state.step_size

        # generate a proposal
        proposal = self._proposal_fn(key, model_state, step_size)

        # metropolis-hastings calibration
        info, model_state = mh_step(
            subkey,
            self.model,
            proposal.position,
            model_state,
            proposal.log_correction,
        )
        return TransitionOutcome(info, kernel_state, model_state)

    def _adaptive_transition(
        self,
        prng_key: KeyArray,
        kernel_state: RWKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[RWKernelState, DefaultTransitionInfo]:
        """Performs an MCMC transition *with* dual averaging."""

        outcome = self._standard_transition(prng_key, kernel_state, model_state, epoch)

        if self.da_tune_step_size:
            da_step(
                outcome.kernel_state,
                outcome.info.acceptance_prob,
                epoch.time_in_epoch,
                self.da_target_accept,
                self.da_gamma,
                self.da_kappa,
                self.da_t0,
            )

        return outcome

    def tune(
        self,
        prng_key: KeyArray,
        kernel_state: RWKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None = None,
    ) -> TuningOutcome[RWKernelState, DefaultTuningInfo]:
        """Currently does nothing."""

        info = MHTuningInfo(error_code=0, time=epoch.time)
        return TuningOutcome(info, kernel_state)

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: RWKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> RWKernelState:
        """Resets the state of the dual averaging algorithm."""

        da_init(kernel_state)
        return kernel_state

    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: RWKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> RWKernelState:
        """
        Sets the step size as found by the dual averaging algorithm.
        """

        da_finalize(kernel_state)
        return kernel_state

    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_state: RWKernelState,
        model_state: ModelState,
        tuning_history: TuningInfo | None,
    ) -> WarmupOutcome[RWKernelState]:
        """Currently does nothing."""

        return WarmupOutcome(error_code=0, kernel_state=kernel_state)
