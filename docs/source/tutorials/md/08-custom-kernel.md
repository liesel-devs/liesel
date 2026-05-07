
# Defining a custom MCMC kernel

## Custom Metropolis-Hastings kernel

The easiest way to use a custom MCMC kernel in `liesel.goose` is to
provide a proposal function for a {class}`.MHKernel`. The function must
accept a pseudo-random number key, a model state and a step size as
arguments, and be compatible with just-in-time compilation via `jax`
(i.e., pure, without side-effects). It returns a {class}`.MHProposal`,
which simply wraps the proposed value and the Metropolis-Hastings
log-correction factor. The {class}`.MHKernel` handles the
acceptance/rejection logic and is fully equipped with dual averaging
functionality for step size tuning, which can be switched on by passing
`da_tune_step_size` as a keyword argument to the kernel. In this case,
users should ensure that their settings for the initial step size
(default: $1$) and the target acceptance probability (default: $.234$)
are suitable.

As an example, a random walk kernel (like {class}`.RWKernel`) can be
implemented with

``` python
>>> param_name = ... # name of the parameter variable to be sampled
>>> def rw_proposal(prng_key, model_state, step_size):
...     pos = model.extract_position([param_name], model_state)
...     current = pos[param_name]
...
...     proposal_dist = tfd.Normal(loc=current, scale=step_size)
...     proposed = proposal_dist.sample(seed=prng_key)
...
...     backward_dist = tfd.Normal(loc=proposed, scale=step_size)
...     backward_log_prob = backward_dist.log_prob(current)
...     forward_log_prob = proposal_dist.log_prob(proposed)
...     log_correction = (backward_log_prob - forward_log_prob).sum()
...     return gs.MHProposal({param_name: proposed}, log_correction)
```

It can then be attached to the coefficient variable with

``` python
>>> model.vars[param_name].coef.inference = gs.MCMCSpec(
...     gs.MHKernel,
...     kernel_kwargs={"proposal_fn": rw_proposal, "da_tune_step_size": True},
... )
```

In this case, the proposal distribution is symmetric, so the log
correction factor is zero by definition. We still compute it here
explicitly for the purpose of demonstration.

While a custom proposal function for a {class}`.MHKernel` can be written
conveniently, it may not cover cases in which a custom MCMC kernel
requires additional hyperparameters or specialized tuning. For such
cases, `liesel.goose` provides tools for users to write their own
classes, implementing the {class}`.Kernel` protocol.

The next section shows you how to write such a fully custom kernel
class.

## Fully customized MCMC kernel

Any Python class that implements the {class}`.Kernel` protocol can be
used as an MCMC kernel class in `liesel.goose`. The protocol requires
the implementation of several attributes and methods, the most important
of which are {meth}`.Kernel.transition` and {meth}`.Kernel.tune`. These
methods are called by the engine and need to be pure and jittable.

### Overview

**The transition method.** The purpose of the transition method is to
move the subset of the model state handled by the kernel using a valid
MCMC step, e.g.~a Metropolis-Hastings algorithm. Its signature is:

``` python
>>> class Kernel:
...
...     def transition(
...         self,
...         prng_key: KeyArray,
...         kernel_state: KernelState,
...         model_state: ModelState,
...         epoch: EpochState,
...     ) -> TransitionOutcome[KernelState, TransitionInfo]:
...         ...
```

Since the {meth}`.Kernel.transition` method must be pure, and MCMC
transitions generally involve the generation of random numbers, a key
for pseudo-random number generation (PRNG) needs to be provided as an
argument. In addition, the {meth}`.Kernel.transition` method receives
the kernel state, the model state and the epoch state as arguments, and
returns a {class}`.TransitionOutcome` object, which wraps the new kernel
state, the new model state and some meta-information about the
transition, e.g.~an error code or the acceptance probability (in a
{class}`.TransitionInfo` object). An error code of zero indicates that
the transition did not produce an error.

All inputs and outputs must be valid *pytrees* (i.e.~arrays or nested
lists, tuples or dicts of arrays). The structure of these objects,
e.g.~the shape of the arrays in the kernel state, must not change
between transitions. This allows the kernels to have specialized
{class}`.KernelState` and {class}`.TransitionInfo` classes. A kernel
state can be any pytree.

**The tune method.** The {meth}`.Kernel.tune` method is updates the
kernel hyperparameters at the end of an adaptation epoch. The method
receives the PRNG key, the model state, the kernel state, the epoch
state, and (optionally) the *history*, i.e.~the samples from the
previous epoch, as arguments. It returns a {class}`.TuningOutcome`
object that wraps the new kernel state and some meta-information about
the tuning process, e.g.~an error code. As for the transition, the
{class}`.TuningInfo` class can be kernel-specific but must be a valid
pytree.

The signature of the {meth}`.Kernel.tune` method is as follows:

``` python
>>> class Kernel:
...
...     def tune(
...         self,
...         prng_key: KeyArray,
...         kernel_state: KernelState,
...         model_state: ModelState,
...         epoch: EpochState,
...         history: Position | None,
...     ) -> TuningOutcome[KernelState, TuningInfo]:
...         ...
```

### Step-by-step tutorial

We will now go through the definition of the {class}`.RWKernel`
step-by-step.

#### The kernel state

First, we define the {class}`.KernelState`. Since we plan to use dual
averaging for step size tuning in this kernel class, we define a kernel
state that follows the {class}`.DAKernelState` protocol.

``` python
from dataclasses import dataclass, field  # general dataclass functionalty
from liesel.goose.pytree import (
    register_dataclass_as_pytree,  # dataclasses must be registered as pytrees with jax
)
from liesel.goose import da  # dual averaging functionality


@register_dataclass_as_pytree
@dataclass
class RWKernelState:
    """
    A dataclass for the state of a ``RWKernel``, implementing the
    :class:`.DAKernelState` protocol.
    """

    step_size: float
    error_sum: float = field(default=0.0, init=False)
    log_avg_step_size: float = field(default=0.0, init=False)
    mu: float = field(init=False)

    def __post_init__(self):
        da.da_init(self)
```

#### The kernel class

We now define the actual kernel class. The class inherits from two
mixins provided by `liesel.goose`.

The {class}`.ModelMixin` gives the kernel access to the model and
provides convenience methods such as {meth}`.ModelMixin.position`, which
extracts the part of the model state handled by this kernel.

The {class}`.TransitionMixin` provides the public
{meth}`.TransitionMixin.transition` method. Internally, it dispatches to
`_standard_transition` or `_adaptive_transition`, depending on the
current epoch. This means that we only have to implement these two
methods.

``` python
import jax
import liesel.goose as gs


class RWKernel(
    gs.ModelMixin, gs.TransitionMixin[RWKernelState, gs.DefaultTransitionInfo]
):
    error_book = {0: "no errors", 90: "nan acceptance prob"}
    """Dict of error codes and their meaning."""

    needs_history = False
    """Whether this kernel needs its history for tuning."""

    identifier: str = ""
    """Kernel identifier, set by :class:`~.goose.EngineBuilder`"""

    position_keys: tuple[str, ...]
    """Tuple of position keys handled by this kernel."""
```

At the beginning of the class, we define a few class attributes required
by the kernel protocol.

The `error_book` maps error codes to human-readable messages. By
convention, an error code of zero means that no error occurred.

The `needs_history` attribute tells the engine whether the kernel
requires the samples from the previous epoch for tuning. This random
walk kernel does not use the history, so we set it to `False`.

The `identifier` is set by the {class}`~.goose.EngineBuilder` and can be
used to distinguish between kernels. Finally, `position_keys` stores the
names of the model variables handled by this kernel.

The constructor stores the user-supplied settings. The most important
argument is `position_keys`, which determines which model variables are
updated by this kernel.

The remaining arguments configure the initial step size and the dual
averaging algorithm. These values are stored on the kernel object, but
they are not part of the kernel state. The mutable, chain-specific part
of the kernel is stored separately in the {class}`.RWKernelState`.

``` python
    def __init__(
        self,
        position_keys: list[str] | tuple[str, ...],
        initial_step_size: float = 1.0,
        da_target_accept: float = 0.234,
        da_gamma: float = 0.05,
        da_kappa: float = 0.75,
        da_t0: int = 10,
        identifier: str = "",
    ):
        self._model = None
        self.position_keys = tuple(position_keys)
        self.initial_step_size = initial_step_size
        self.da_target_accept = da_target_accept
        self.da_gamma = da_gamma
        self.da_kappa = da_kappa
        self.da_t0 = da_t0
        self.identifier = identifier
```

Before sampling starts, the engine calls `init_state`. This method
creates the initial kernel state for one chain. In our case, the only
user-facing state variable is the current step size.

``` python
    def init_state(self, prng_key, model_state: gs.ModelState) -> RWKernelState:
        """
        Initializes the kernel state.
        """
        return RWKernelState(step_size=self.initial_step_size)
```

Next, we implement the non-adaptive transition. This method performs one
ordinary Metropolis-Hastings random walk step.

First, we split the pseudo-random number key. One key is used to
generate the proposal, and the other key is used inside the
Metropolis-Hastings accept/reject step.

``` python
    def _standard_transition(
        self,
        prng_key,
        kernel_state: RWKernelState,
        model_state: gs.ModelState,
        epoch: gs.EpochState,
    ) -> gs.TransitionOutcome[RWKernelState, gs.DefaultTransitionInfo]:
        """
        Performs an MCMC transition *without* dual averaging.
        """

        key, subkey = jax.random.split(prng_key)
        step_size = kernel_state.step_size
        ...
```

The current position is extracted from the model state. Since the
position can be a pytree, we flatten it into a single vector before
adding Gaussian noise. After the proposal has been generated, we
transform it back into the original pytree structure.

This lets the same implementation work for scalar, vector-valued, or
structured model positions.

``` python
def _standard_transition(...):
        # ... (continued)
        # random walk proposal
        position = self.position(model_state)
        flat_position, unravel_fn = jax.flatten_util.ravel_pytree(position)
        step = step_size * jax.random.normal(key, flat_position.shape)
        flat_proposal = flat_position + step
        proposal = unravel_fn(flat_proposal)
```

Finally, we pass the proposal to {func}`.mh_step`. This function
evaluates the proposed model state and performs the Metropolis-Hastings
accept/reject step.

The result is returned as a {class}`.TransitionOutcome`, which contains
the transition information, the kernel state, and the updated model
state.

``` python
def _standard_transition(...):
        # ... (continued)
        # metropolis-hastings calibration
        info, model_state = gs.mh_step(subkey, self.model, proposal, model_state)
        return gs.TransitionOutcome(info, kernel_state, model_state)
```

The adaptive transition starts by performing the same
Metropolis-Hastings step as above. It then updates the dual averaging
state using the observed acceptance probability from the transition.

The dual averaging update modifies the kernel state in place. It uses
the current acceptance probability, the time within the current epoch,
and the dual averaging hyperparameters stored on the kernel object.

``` python
    def _adaptive_transition(
        self,
        prng_key,
        kernel_state: RWKernelState,
        model_state: gs.ModelState,
        epoch: gs.EpochState,
    ) -> gs.TransitionOutcome[RWKernelState, gs.DefaultTransitionInfo]:
        """
        Performs an MCMC transition *with* dual averaging.
        """

        outcome = self._standard_transition(prng_key, kernel_state, model_state, epoch)

        da.da_step(
            outcome.kernel_state,
            outcome.info.acceptance_prob,
            epoch.time_in_epoch,
            self.da_target_accept,
            self.da_gamma,
            self.da_kappa,
            self.da_t0,
        )

        return outcome
```

The `tune` method is called by the engine at the end of a tuning epoch.
This particular kernel does not perform any additional tuning at the end
of an epoch, because the step size adaptation already happens during the
adaptive transitions.

Still, the method must be implemented to satisfy the kernel protocol. We
therefore return a successful {class}`.TuningOutcome` with the unchanged
kernel state.

``` python
    def tune(
        self,
        prng_key,
        kernel_state: RWKernelState,
        model_state: gs.ModelState,
        epoch: gs.EpochState,
        history: gs.Position | None = None,
    ) -> gs.TuningOutcome[RWKernelState, gs.DefaultTuningInfo]:
        """
        Currently does nothing.
        """

        info = gs.DefaultTuningInfo(error_code=0, time=epoch.time)
        return gs.TuningOutcome(info, kernel_state)
```

At the beginning of each adaptation epoch, we reset the dual averaging
state. This is done in `start_epoch`.

This reset does not discard the current step size itself. Instead, it
reinitializes the auxiliary quantities used internally by the dual
averaging algorithm.

``` python
    def start_epoch(
        self,
        prng_key,
        kernel_state: RWKernelState,
        model_state: gs.ModelState,
        epoch: gs.EpochState,
    ) -> RWKernelState:
        """
        Resets the state of the dual averaging algorithm.
        """

        da.da_init(kernel_state)
        return kernel_state
```

At the end of an adaptation epoch, we finalize the dual averaging
update. This replaces the current step size by the averaged step size
found during the epoch.

``` python
    def end_epoch(
        self,
        prng_key,
        kernel_state: RWKernelState,
        model_state: gs.ModelState,
        epoch: gs.EpochState,
    ) -> RWKernelState:
        """
        Sets the step size as found by the dual averaging algorithm.
        """

        da.da_finalize(kernel_state)
        return kernel_state
```

Finally, the engine calls `end_warmup` after all warmup epochs have
finished. This hook can be used for final warmup-specific adjustments.
Our random walk kernel does not need any such adjustment, so we simply
return the unchanged kernel state.

``` python
    def end_warmup(
        self,
        prng_key,
        kernel_state: RWKernelState,
        model_state: gs.ModelState,
        tuning_history: gs.TuningInfo | None,
    ) -> gs.WarmupOutcome[RWKernelState]:
        """
        Currently does nothing.
        """

        return gs.WarmupOutcome(error_code=0, kernel_state=kernel_state)
```

This completes the kernel class. The main logic is contained in
`_standard_transition`, which constructs a random walk proposal and
delegates the Metropolis-Hastings correction to {func}`.mh_step`. The
adaptive version adds one more step: it updates the step size using dual
averaging based on the observed acceptance probability.

We now simply restate the full code-block here:

``` python
import jax
import liesel.goose as gs


class RWKernel(
    gs.ModelMixin, gs.TransitionMixin[RWKernelState, gs.DefaultTransitionInfo]
):
    error_book = {0: "no errors", 90: "nan acceptance prob"}
    """Dict of error codes and their meaning."""

    needs_history = False
    """Whether this kernel needs its history for tuning."""

    identifier: str = ""
    """Kernel identifier, set by :class:`~.goose.EngineBuilder`"""

    position_keys: tuple[str, ...]
    """Tuple of position keys handled by this kernel."""

    def __init__(
        self,
        position_keys: list[str] | tuple[str, ...],
        initial_step_size: float = 1.0,
        da_target_accept: float = 0.234,
        da_gamma: float = 0.05,
        da_kappa: float = 0.75,
        da_t0: int = 10,
        identifier: str = "",
    ):
        self._model = None
        self.position_keys = tuple(position_keys)
        self.initial_step_size = initial_step_size
        self.da_target_accept = da_target_accept
        self.da_gamma = da_gamma
        self.da_kappa = da_kappa
        self.da_t0 = da_t0
        self.identifier = identifier

    def init_state(self, prng_key, model_state: gs.ModelState) -> RWKernelState:
        """
        Initializes the kernel state.
        """
        return RWKernelState(step_size=self.initial_step_size)

    def _standard_transition(
        self,
        prng_key,
        kernel_state: RWKernelState,
        model_state: gs.ModelState,
        epoch: gs.EpochState,
    ) -> gs.TransitionOutcome[RWKernelState, gs.DefaultTransitionInfo]:
        """
        Performs an MCMC transition *without* dual averaging.
        """

        key, subkey = jax.random.split(prng_key)
        step_size = kernel_state.step_size

        # random walk proposal
        position = self.position(model_state)
        flat_position, unravel_fn = jax.flatten_util.ravel_pytree(position)
        step = step_size * jax.random.normal(key, flat_position.shape)
        flat_proposal = flat_position + step
        proposal = unravel_fn(flat_proposal)

        # metropolis-hastings calibration
        info, model_state = gs.mh_step(subkey, self.model, proposal, model_state)
        return gs.TransitionOutcome(info, kernel_state, model_state)

    def _adaptive_transition(
        self,
        prng_key,
        kernel_state: RWKernelState,
        model_state: gs.ModelState,
        epoch: gs.EpochState,
    ) -> gs.TransitionOutcome[RWKernelState, gs.DefaultTransitionInfo]:
        """
        Performs an MCMC transition *with* dual averaging.
        """

        outcome = self._standard_transition(prng_key, kernel_state, model_state, epoch)

        da.da_step(
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
        prng_key,
        kernel_state: RWKernelState,
        model_state: gs.ModelState,
        epoch: gs.EpochState,
        history: gs.Position | None = None,
    ) -> gs.TuningOutcome[RWKernelState, gs.DefaultTuningInfo]:
        """
        Currently does nothing.
        """

        info = gs.DefaultTuningInfo(error_code=0, time=epoch.time)
        return gs.TuningOutcome(info, kernel_state)

    def start_epoch(
        self,
        prng_key,
        kernel_state: RWKernelState,
        model_state: gs.ModelState,
        epoch: gs.EpochState,
    ) -> RWKernelState:
        """
        Resets the state of the dual averaging algorithm.
        """

        da.da_init(kernel_state)
        return kernel_state

    def end_epoch(
        self,
        prng_key,
        kernel_state: RWKernelState,
        model_state: gs.ModelState,
        epoch: gs.EpochState,
    ) -> RWKernelState:
        """
        Sets the step size as found by the dual averaging algorithm.
        """

        da.da_finalize(kernel_state)
        return kernel_state

    def end_warmup(
        self,
        prng_key,
        kernel_state: RWKernelState,
        model_state: gs.ModelState,
        tuning_history: gs.TuningInfo | None,
    ) -> gs.WarmupOutcome[RWKernelState]:
        """
        Currently does nothing.
        """

        return gs.WarmupOutcome(error_code=0, kernel_state=kernel_state)
```

#### Trying out our new kernel

Here, we just take a very simple model to confirm that our kernel runs.

``` python
import liesel.model as lsl
import tensorflow_probability.substrates.jax.distributions as tfd

mu = lsl.Var.new_param(0.0, name="mu", inference=gs.MCMCSpec(RWKernel))
y = lsl.Var.new_obs(
    value=jax.random.normal(jax.random.key(13), (100,)) + 0.5,
    dist=lsl.Dist(tfd.Normal, loc=mu, scale=1.0),
    name="y",
)
model = lsl.Model(y)

results = gs.LieselMCMC(model).run_for_epochs(
    seed=7, num_chains=4, adaptation=500, posterior=500
)
```


      0%|                                                  | 0/2 [00:00<?, ?chunk/s]
     50%|#####################                     | 1/2 [00:00<00:00,  2.38chunk/s]
    100%|##########################################| 2/2 [00:00<00:00,  4.75chunk/s]

      0%|                                                  | 0/1 [00:00<?, ?chunk/s]
    100%|########################################| 1/1 [00:00<00:00, 2258.65chunk/s]

      0%|                                                  | 0/2 [00:00<?, ?chunk/s]
    100%|########################################| 2/2 [00:00<00:00, 4213.26chunk/s]

      0%|                                                 | 0/11 [00:00<?, ?chunk/s]
    100%|#######################################| 11/11 [00:00<00:00, 863.01chunk/s]

      0%|                                                  | 0/4 [00:00<?, ?chunk/s]
    100%|########################################| 4/4 [00:00<00:00, 5215.17chunk/s]

      0%|                                                 | 0/20 [00:00<?, ?chunk/s]
    100%|######################################| 20/20 [00:00<00:00, 3343.27chunk/s]

``` python
gs.Summary(results)
```

<p>
<strong>Parameter summary:</strong>
</p>
<table border="0" class="dataframe">
<thead>
<tr style="text-align: right;">
<th>
</th>
<th>
</th>
<th>
kernel
</th>
<th>
mean
</th>
<th>
sd
</th>
<th>
q_0.05
</th>
<th>
q_0.5
</th>
<th>
q_0.95
</th>
<th>
sample_size
</th>
<th>
ess_bulk
</th>
<th>
ess_tail
</th>
<th>
rhat
</th>
</tr>
<tr>
<th>
parameter
</th>
<th>
index
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
</tr>
</thead>
<tbody>
<tr>
<th>
mu
</th>
<th>
()
</th>
<td>
kernel_00
</td>
<td>
0.461
</td>
<td>
0.095
</td>
<td>
0.298
</td>
<td>
0.456
</td>
<td>
0.634
</td>
<td>
2000
</td>
<td>
319.798
</td>
<td>
212.696
</td>
<td>
1.016
</td>
</tr>
</tbody>
</table>
<p>
<strong>Acceptance probabilities:</strong>
</p>
<table border="0" class="dataframe">
<thead>
<tr style="text-align: right;">
<th>
</th>
<th>
</th>
<th>
</th>
<th>
acceptance_probability
</th>
<th>
position_moved
</th>
</tr>
<tr>
<th>
kernel
</th>
<th>
positions
</th>
<th>
phase
</th>
<th>
</th>
<th>
</th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="2" valign="top">
kernel_00
</th>
<th rowspan="2" valign="top">
mu
</th>
<th>
posterior
</th>
<td>
0.204
</td>
<td>
0.204
</td>
</tr>
<tr>
<th>
warmup
</th>
<td>
0.222
</td>
<td>
0.224
</td>
</tr>
</tbody>
</table>
