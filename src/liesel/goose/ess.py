"""
Elliptical Slice Sampling (ESS) kernel.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, cast

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import lax, random

from .epoch import EpochState
from .kernel import (
    DefaultTransitionInfo,
    DefaultTuningInfo,
    ModelMixin,
    TransitionMixin,
    TransitionOutcome,
    TuningMixin,
    TuningOutcome,
    WarmupOutcome,
)
from .pytree import register_dataclass_as_pytree
from .types import Array, KeyArray, ModelState, Position


@register_dataclass_as_pytree
@dataclass
class ESSKernelState:
    """
    A dataclass for the state of an :class:`.ESSKernel`.
    """

    # Kernel state is minimal for ESS
    pass


class ESSKernel(
    ModelMixin,
    TransitionMixin[ESSKernelState, DefaultTransitionInfo],
    TuningMixin[ESSKernelState, DefaultTuningInfo],
):
    """
    Elliptical Slice Sampling (ESS) kernel for Gaussian priors.

    ESS is designed for models with Gaussian priors on the parameters being sampled.
    It generates proposals by sampling along an ellipse defined by the prior covariance
    and uses only the likelihood for acceptance decisions. The sampler becomes more
    efficient when the likelihood is less informative relative to the prior.

    The kernel automatically separates the prior contribution of the sampled variable
    from the rest of the model (likelihood + other priors) by computing
    likelihood_contribution = model.log_prob(state) - model.log_prob_vars(
        state, position_keys
    )

    Parameters
    ----------
    position_keys
        Keys for which the kernel handles the transition. Must contain exactly one
        variable that has a Gaussian prior distribution.
    identifier
        Identifier for the kernel. If empty, will be set by EngineBuilder.
    """

    error_book: ClassVar[dict[int, str]] = {
        0: "no errors",
        1: "max iterations reached without acceptance",
    }

    needs_history: ClassVar[bool] = False

    def __init__(
        self,
        position_keys: Sequence[str],
        max_iterations: int = 50,
        identifier: str = "",
    ):
        if len(position_keys) != 1:
            raise ValueError(
                f"ESSKernel requires exactly one position key, got "
                f"{len(position_keys)}: {position_keys}"
            )

        self.position_keys = tuple(position_keys)
        self.identifier = identifier
        self._model = None
        self.max_iterations = max_iterations

    def init_state(self, prng_key: KeyArray, model_state: ModelState) -> ESSKernelState:
        """Creates the initial kernel state."""
        # Validate that the position variable has a supported Gaussian prior
        self._validate_gaussian_prior(model_state)
        return ESSKernelState()

    def _validate_gaussian_prior(self, model_state: ModelState) -> None:
        """Validates that position variable has a supported Gaussian prior."""
        try:
            _, _ = self._extract_gaussian_params(model_state)
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"ESS requires Gaussian prior on '{self.position_keys[0]}'. {e}"
            ) from e

    def _extract_gaussian_params(self, model_state: ModelState) -> tuple[Array, Array]:
        """
        Extracts mean and covariance from the Gaussian prior of the position variable.

        Returns
        -------
        tuple[Array, Array]
            (mean, covariance_matrix) of the Gaussian prior
        """
        # imported here because of circular import issues
        from .. import model
        from .interface import LieselInterface

        var_name = self.position_keys[0]

        if not isinstance(self.model, LieselInterface):
            raise TypeError(
                "ESSKernel currently only supports LieselInterface. "
                "Other interfaces would need manual prior parameter specification."
            )

        # Access the distribution node from the Liesel model
        liesel_model = self.model._model

        try:
            dist_node = liesel_model.vars[var_name].dist_node
            if dist_node is None:
                raise ValueError(f"Variable '{var_name}' does not have a distribution.")
            dist_node_name = dist_node.name
        except KeyError:
            raise KeyError(f"Variable '{var_name}' not found in Liesel model.")

        try:
            dist_node = cast(model.Dist, liesel_model.nodes[dist_node_name])
            # dist_node = liesel_model.nodes[dist_node_name]
        except KeyError:
            raise KeyError(
                f"Distribution node '{dist_node_name}' not found. "
                f"Available nodes: {list(liesel_model.nodes.keys())}"
            )

        # Extract current parameter values from model state
        args = []
        kwargs = {}

        for inp in dist_node.inputs:
            if inp.name in model_state:
                args.append(model_state[inp.name].value)
            else:
                args.append(inp.value)

        for key, inp in dist_node.kwinputs.items():
            if inp.name in model_state:
                kwargs[key] = model_state[inp.name].value
            else:
                kwargs[key] = inp.value

        # Create the distribution with current parameter values
        try:
            distribution = dist_node.distribution(*args, **kwargs)
        except Exception as e:
            raise ValueError(
                f"Failed to create distribution for '{var_name}' with "
                f"args={args}, kwargs={kwargs}. Original error: {e}"
            )

        # Handle different Gaussian distributions
        match distribution:
            case tfd.MultivariateNormalFullCovariance():
                return distribution.loc, distribution.covariance()
            case tfd.MultivariateNormalTriL():
                return distribution.loc, distribution.covariance()
            case tfd.MultivariateNormalDiag():
                return distribution.loc, distribution.covariance()
            case tfd.MultivariateNormalDiagPlusLowRank():
                return distribution.loc, distribution.covariance()
            case tfd.Normal():
                mu = distribution.loc
                sigma_sq = distribution.variance()
                # Broadcast mu and sigma_sq to the same shape
                mu, sigma_sq = jnp.broadcast_arrays(mu, sigma_sq)
                # Convert to appropriate shapes for multivariate operations
                if jnp.ndim(mu) == 0:
                    # Scalar case
                    return jnp.array([mu]), jnp.array([[sigma_sq]])
                else:
                    # Vector case with independent components
                    return mu, jnp.diag(sigma_sq)
            case _:
                raise ValueError(
                    f"ESS requires Gaussian prior on '{var_name}'. "
                    f"Got {type(distribution)}. Supported: "
                    f"MultivariateNormalFullCovariance, MultivariateNormalTriL, "
                    f"MultivariateNormalDiag, MultivariateNormalDiagPlusLowRank, Normal"
                )

    def _standard_transition(
        self,
        prng_key: KeyArray,
        kernel_state: ESSKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[ESSKernelState, DefaultTransitionInfo]:
        """
        Performs an ESS transition outside an adaptation epoch.
        """

        # Extract Gaussian prior parameters
        mu, sigma = self._extract_gaussian_params(model_state)

        # Get current position
        current_position = self.position(model_state)
        current_f = current_position[self.position_keys[0]]

        # Work in deviation space: f_tilde = f - mu
        current_f_tilde = current_f - mu

        # Evaluate current likelihood
        current_log_likelihood = self._evaluate_likelihood(model_state)

        # Generate proposal ellipse direction (in deviation space)
        key1, key2 = random.split(prng_key)
        nu = random.multivariate_normal(key1, jnp.zeros_like(mu), sigma)

        # Perform slice sampling on the angle
        theta_new, accepted = self._slice_sample_angle(
            key2, current_f_tilde, nu, mu, current_log_likelihood, model_state
        )

        # Compute new position in original space
        f_tilde_new = current_f_tilde * jnp.cos(theta_new) + nu * jnp.sin(theta_new)
        f_new = f_tilde_new + mu  # Transform back to original space
        new_position = Position({self.position_keys[0]: f_new})

        # Update model state
        new_model_state = self.model.update_state(new_position, model_state)

        # Create transition info
        # if not accepted, we exceeded the maximum number of iterations
        # record the failure as code 1
        info = DefaultTransitionInfo(
            error_code=jnp.int32(~accepted),
            acceptance_prob=jnp.float32(accepted),
            position_moved=accepted,
        )

        return TransitionOutcome(
            info=info,
            kernel_state=kernel_state,
            model_state=lax.cond(
                accepted, lambda: new_model_state, lambda: model_state
            ),
        )

    def _adaptive_transition(
        self,
        prng_key: KeyArray,
        kernel_state: ESSKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[ESSKernelState, DefaultTransitionInfo]:
        """
        Performs an ESS transition in an adaptation epoch.
        For ESS, this is the same as standard transition since ESS doesn't adapt.
        """
        return self._standard_transition(prng_key, kernel_state, model_state, epoch)

    def _evaluate_likelihood(self, model_state: ModelState) -> float:
        """
        Evaluate likelihood contribution by subtracting prior from total log_prob.

        Returns log(likelihood × other_priors), which is proportional to the
        likelihood for the purpose of ESS since other_priors are constant w.r.t.
        the sampled variable.
        """
        total_log_prob = self.model.log_prob(model_state)
        prior_log_prob = self.model.log_prob_vars(model_state, self.position_keys)
        return total_log_prob - prior_log_prob

    def _slice_sample_angle(
        self,
        prng_key: KeyArray,
        current_f_tilde: Array,
        nu: Array,
        mu: Array,
        current_log_likelihood: float,
        model_state: ModelState,
    ) -> tuple[float, bool]:
        """
        Canonical elliptical slice sampling following Murray et al. (2010).

        Algorithm:
        1. Sample threshold: log_y = log_L(f) + log(u) where u ~ Uniform[0,1]
        2. Sample initial angle: theta ~ Uniform[0, 2pi]
        3. Set bracket: [theta_min, theta_max] = [theta - 2pi, theta]
        4. Shrink bracket until acceptance
        """
        key1, key2, key3 = random.split(prng_key, 3)

        # Step 1: Sample threshold (log_y = log L + log u where u ~ U[0,1])
        # Since log(u) ~ -Exponential(1), we use:
        log_y = current_log_likelihood - random.exponential(key1)

        # Step 2: Sample initial angle theta ~ Uniform[0, 2pi]
        theta_init = random.uniform(key2, minval=0.0, maxval=2 * jnp.pi)

        # Step 3: Set initial bracket [theta - 2pi, theta]
        theta_min_init = theta_init - 2 * jnp.pi
        theta_max_init = theta_init

        # Define likelihood function for angle
        def log_likelihood_at_angle(theta: float) -> float:
            # Compute point on ellipse: f' = f cos theta + nu sin theta
            # in deviation space
            f_tilde_proposal = current_f_tilde * jnp.cos(theta) + nu * jnp.sin(theta)
            # Transform back to original space
            f_proposal = f_tilde_proposal + mu
            proposal_position = Position({self.position_keys[0]: f_proposal})

            # Update model state with proposal
            new_model_state = self.model.update_state(proposal_position, model_state)

            # Return likelihood contribution
            return self._evaluate_likelihood(new_model_state)

        # Step 4: Slice sampling with bracket shrinking using while loop
        def shrink_body(carry):
            theta_min, theta_max, key, accepted, final_theta, iteration = carry

            # Generate new candidate theta ~ Uniform[theta_min, theta_max]
            key, subkey = random.split(key)
            theta_candidate = random.uniform(subkey, minval=theta_min, maxval=theta_max)

            # Evaluate likelihood at candidate
            candidate_ll = log_likelihood_at_angle(theta_candidate)

            # Check if above threshold
            accept_candidate = candidate_ll >= log_y

            # If accepted, update final theta
            new_theta = lax.cond(
                accept_candidate, lambda: theta_candidate, lambda: final_theta
            )
            new_accepted = accepted | accept_candidate

            # If not accepted, shrink bracket according to canonical ESS rules:
            # if theta < 0 then theta_min <- theta else theta_max <- theta
            new_theta_min = lax.cond(
                (~accept_candidate) & (theta_candidate < 0.0),
                lambda: theta_candidate,
                lambda: theta_min,
            )
            new_theta_max = lax.cond(
                (~accept_candidate) & (theta_candidate >= 0.0),
                lambda: theta_candidate,
                lambda: theta_max,
            )

            return (
                new_theta_min,
                new_theta_max,
                key,
                new_accepted,
                new_theta,
                iteration + 1,
            )

        def shrink_cond(carry):
            _, _, _, accepted, _, iteration = carry
            # Continue while not accepted and haven't exceeded max iterations
            return (~accepted) & (iteration < self.max_iterations)

        # Initialize loop state
        init_carry = (theta_min_init, theta_max_init, key3, False, 0.0, 0)
        _, _, _, final_accepted, final_theta, _ = lax.while_loop(
            shrink_cond, shrink_body, init_carry
        )

        return final_theta, final_accepted

    def _tune_fast(
        self,
        prng_key: KeyArray,
        kernel_state: ESSKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None,
    ) -> TuningOutcome[ESSKernelState, DefaultTuningInfo]:
        """ESS doesn't require tuning."""
        info = DefaultTuningInfo(error_code=0, time=epoch.config.duration)
        return TuningOutcome(info=info, kernel_state=kernel_state)

    def _tune_slow(
        self,
        prng_key: KeyArray,
        kernel_state: ESSKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None,
    ) -> TuningOutcome[ESSKernelState, DefaultTuningInfo]:
        """ESS doesn't require tuning."""
        info = DefaultTuningInfo(error_code=0, time=epoch.config.duration)
        return TuningOutcome(info=info, kernel_state=kernel_state)

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: ESSKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> ESSKernelState:
        """Called at the beginning of an epoch."""
        return kernel_state

    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: ESSKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> ESSKernelState:
        """Called at the end of an epoch."""
        return kernel_state

    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_state: ESSKernelState,
        model_state: ModelState,
        tuning_history: DefaultTuningInfo | None,
    ) -> WarmupOutcome[ESSKernelState]:
        """Called at the end of warmup."""
        return WarmupOutcome(error_code=0, kernel_state=kernel_state)
