"""
Gibbs sampler.
"""
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp

from .epoch import EpochState
from .kernel import (
    DefaultTransitionInfo,
    DefaultTuningInfo,
    ModelMixin,
    TransitionOutcome,
    TuningOutcome,
    WarmupOutcome,
)
from .types import (
    Array,
    Kernel,
    KernelState,
    KeyArray,
    ModelState,
    Position,
    TuningInfo,
)

if TYPE_CHECKING:
    from ..model.model import Model

GibbsKernelState = KernelState
GibbsTransitionInfo = DefaultTransitionInfo
GibbsTuningInfo = DefaultTuningInfo


class GibbsKernel(
    ModelMixin, Kernel[GibbsKernelState, GibbsTransitionInfo, GibbsTuningInfo]
):
    """
    A Gibbs kernel implementing the :class:`.Kernel` protocol.
    """

    error_book: ClassVar[dict[int, str]] = {0: "no errors"}
    needs_history: ClassVar[bool] = False
    identifier: str = ""
    position_keys: tuple[str, ...]

    def __init__(
        self,
        position_keys: Sequence[str],
        transition_fn: Callable[[KeyArray, ModelState], Position],
    ):
        self._model = None
        self.position_keys = tuple(position_keys)
        self._transition_fn = transition_fn

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


def create_categorical_gibbs_kernel(
    name: str, outcomes: Sequence[Array], model: "Model"
) -> GibbsKernel:
    """
    Creates a categorical Gibbs kernel.

    The prior distribution of the variable to sample must be a categorical distribution,
    usually implemented via :class:`tfd.FiniteDiscrete`.

    This kernel evaluates the full conditional log probability of the model for each
    possible value of the variable to sample. It then draws a new value for the variable
    from the categorical distribution defined by the full conditional log probabilities.

    Usually, you can define more efficient specialized kernels if you know the actual
    full conditional distribution of the variable to sample.

    Parameters
    ----------
    name
        The name of the variable to sample.
    outcomes
        List of possible outcomes.
    model
        The model to sample from.

    Examples
    --------

    In the following example, we create a categorical Gibbs kernel for a variable
    with three possible values. The prior distribution of the variable is a categorical
    distribution with probabilities ``[0.1, 0.2, 0.7]``.

    You can then use the kernel to sample from the model.

    >>> import tensorflow_probability.substrates.jax.distributions as tfd

    >>> values = [0, 1, 2]
    >>> prior_probs = [0.1, 0.2, 0.7]
    >>> value_grid = lsl.Var(values, name="value_grid")

    >>> prior = lsl.Dist(tfd.FiniteDiscrete, outcomes=value_grid, probs=prior_probs)
    >>> categorical_var = lsl.Var(
    ...     value=values[0],
    ...     distribution=prior,
    ...     name="categorical_var",
    ... )

    >>> model = lsl.GraphBuilder().add(categorical_var).build_model()
    >>> kernel = create_categorical_gibbs_kernel("categorical_var", values, model)
    >>> type(kernel)
    <class 'liesel.goose.gibbs.GibbsKernel'>

    """

    model = model._copy_computational_model()

    def transition_fn(prng_key, model_state):

        model.state = model_state
        for node in model.nodes.values():
            node._outdated = False

        def conditional_log_prob_fn(value: Array):
            """
            Evaluates the full conditional log probability of the model
            given the input value.
            """
            model.vars[name].value = value
            return model.log_prob

        conditional_log_probs = jax.tree_map(conditional_log_prob_fn, outcomes)

        draw_index = jax.random.categorical(
            prng_key, logits=jnp.stack(conditional_log_probs)
        )
        draw = jnp.array(outcomes)[draw_index]

        return {name: draw}

    return GibbsKernel([name], transition_fn)
