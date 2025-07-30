import math
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util
import optax
from optax import GradientTransformation, OptState
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.distributions import (
    Distribution as TfpDistribution,
)

from .interface import LieselInterface

tfd = tfp.distributions


class Optimizer:
    """Optimizer for stochastic variational inference.

    This class performs variational inference by optimizing the ELBO using gradient-
    based methods. It initializes variational distributions based on a given model
    interface and latent variable configurations, then runs an optimization loop over a
    specified number of epochs.
    """

    def __init__(
        self,
        seed: int,
        n_epochs: int,
        S: int,
        model_interface: LieselInterface,
        latent_variables: dict[str, dict],
        batch_size: int | None = None,
        patience_tol: float | None = None,
        window_size: int | None = None,
    ) -> None:
        """Initialize the Optimizer.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        n_epochs : int
            Number of epochs to run the optimization.
        S : int
            Number of Monte Carlo samples.
        model_interface : LieselInterface
            Interface to access the model parameters data and quantities.
        latent_variables : Dict[str, Dict]
            Dict of configurations for each latent variable.
        batch_size : int, optional
            Batch size for data subsetting; if None, the full dataset is used.
        patience_tol : float, optional
            Tolerance for early stopping based on ELBO improvements.
        window_size : int, optional
            Number of epochs to wait before early stopping if no improvement occurs.
        """
        self.seed = seed
        self.n_epochs = n_epochs
        self.patience_tol = patience_tol
        self.window_size = window_size
        self.S = S
        self.batch_size = batch_size
        self.model_interface = model_interface
        self.latent_vars_config = latent_variables
        self.rng_key = jax.random.PRNGKey(self.seed)

        self.variational_dists_class = self._init_variational_dists_class()
        self.phi = self._init_phi()
        self.fixed_distribution_params = self._init_fixed_distribution_params()
        self.variational_param_bijectors = self._init_variational_param_bijectors()


        self.opt_state, self.optimizer = self._init_optimizer()
        self.elbo_values: list[float] = []
        self.phi_evolution: list[dict[str, dict[str, Any]]] = []

        try:
            self.dim_data = next(
                var.value.shape[0]
                for var in self.model_interface.model.vars.values()
                if getattr(var, "observed", True)
            )
        except StopIteration:
            raise ValueError("No observed data found in model.")

    def _init_variational_dists_class(self) -> dict[str, type[TfpDistribution]]:
        """Initialize variational distribution classes."""
        variational_dists_class = {
            key: config["dist_class"]
            for key, config in self.latent_vars_config.items()
        }
        return variational_dists_class

    def _init_phi(self) -> dict[str, Any]:
        """Initialize the phi dictionary using unconstrained parameters for optimization.
        The phi dict from builder.py contains constrained parameters; for optimization,
        we store unconstrained parameters by applying the inverse bijector.
        """
        phi = {}
        for key, config in self.latent_vars_config.items():
            phi_constrained = config["phi"]
            parameter_bijectors = config["variational_param_bijectors"]
            
            phi_unconstrained = {}
            for p_name, p_val in phi_constrained.items():
                bij = parameter_bijectors[p_name]
                phi_unconstrained[p_name] = bij.inverse(p_val)
            
            phi[key] = phi_unconstrained
        return phi

    def _init_fixed_distribution_params(self) -> dict[str, dict[str, Any]]:
        """Initialize fixed distribution parameters."""
        fixed_distribution_params = {
            key: (
                config["fixed_distribution_params"]
                if config["fixed_distribution_params"] is not None
                else {}
            )
            for key, config in self.latent_vars_config.items()
        }
        return fixed_distribution_params

    def _init_variational_param_bijectors(self) -> dict[str, dict[str, Any]]:
        """Collect per-parameter bijectors."""
        return {
            key: config["variational_param_bijectors"]
            for key, config in self.latent_vars_config.items()
        }




    def _build_variational_distribution(
        self,
        dist_class: type[TfpDistribution],
        phi: dict[str, Any],
        fixed_distribution_params: dict[str, Any],
        variational_param_bijectors: dict[str, Any],
    ) -> TfpDistribution:
        """Build a TFP distribution with constrained parameters."""
        
        phi_constrained = {}
        for p_name, p_val in phi.items():
            bij = variational_param_bijectors[p_name]
            phi_constrained[p_name] = bij.forward(p_val)
            
        return dist_class(**phi_constrained, **fixed_distribution_params)



    def _init_optimizer(self) -> tuple[OptState, GradientTransformation]:
        """Initialize the optimizer state and transformation."""

        def label_fn(params):
            return {k: k for k in params if k in self.phi}

        # Build optimizer dictionary directly from latent_vars_config
        optim_dict = {
            key: config["optimizer_chain"]
            for key, config in self.latent_vars_config.items()
        }
        tx = optax.multi_transform(optim_dict, label_fn)
        opt_state = tx.init(self.phi)
        return opt_state, tx

    def fit(self) -> None:
        """Run the optimization loop to update variational parameters by maximizing the
        negative ELBO.

        This method iterates over the specified number of epochs. At each epoch, the
        data is partitioned into batches (if a batch_size is provided) and a jitted
        'step' function is executed to compute the ELBO, its gradients, and update the
        variational parameters. Early stopping is implemented based on the relative
        improvement of the ELBO.
        """
        n_epochs = self.n_epochs
        dim_data = self.dim_data
        batch_size = self.batch_size if self.batch_size is not None else dim_data
        patience_tol = self.patience_tol if self.patience_tol is not None else 0.0
        window_size = self.window_size if self.window_size is not None else n_epochs
        S = self.S
        number_batches = (
            dim_data // batch_size
        )  # Assumption: dim_data is divisible by batch_size

        @partial(jax.jit, static_argnames=["batch_size", "S"])
        def step(
            current_phi: dict[str, Any],
            opt_state: OptState,  # Use OptState here
            rng_key: jax.random.PRNGKey,
            dim_data: int,
            batch_size: int,
            batch_indices: jnp.ndarray,
            S: int,
        ) -> tuple[dict[str, Any], OptState, float, jax.random.PRNGKey]:
            """Perform a single optimization step.

            Parameters
            ----------
            current_phi : dict
                Current variational parameters.
            opt_state : object
                Current optimizer state.
            rng_key : jax.random.PRNGKey
                Current random key.
            dim_data : int
                Total number of data points.
            batch_size : int
                Size of the current batch.
            batch_indices : array-like
                Indices for the current batch.
            S : int
                Number of Monte Carlo samples.

            Returns
            -------
            new_phis : dict
                Updated variational parameters.
            new_opt_state : object
                Updated optimizer state.
            loss_val : float
                Computed loss value (negative ELBO) for the current batch.
            new_rng_key : jax.random.PRNGKey
                Updated random key.
            """
            (loss_val, new_rng_key), grads = jax.value_and_grad(
                lambda p, key: self._elbo(
                    p, key, dim_data, batch_size, batch_indices, S
                ),
                has_aux=True,
            )(current_phi, rng_key)
            updates, new_opt_state = self.optimizer.update(
                grads, opt_state, current_phi
            )
            new_phis = optax.apply_updates(current_phi, updates)
            return new_phis, new_opt_state, loss_val, new_rng_key

        def epoch_batches(
            phi: dict[str, Any], opt_state: OptState, rng_key: jax.random.PRNGKey
        ) -> tuple[dict[str, Any], OptState, jax.random.PRNGKey, float]:
            """Process all batches in one epoch and compute the mean ELBO.

            Parameters
            ----------
            phi : dict
                Current variational parameters.
            opt_state : object
                Current optimizer state.
            rng_key : jax.random.PRNGKey
                Current random key.

            Returns
            -------
            phi : dict
                Updated variational parameters.
            opt_state : object
                Updated optimizer state.
            rng_key : jax.random.PRNGKey
                Updated random key.
            current_elbo : float
                Mean ELBO computed over the batches.
            """
            rng_key, perm_key = jax.random.split(rng_key)
            all_indices = jax.random.permutation(perm_key, dim_data)
            epoch_elbos = jnp.zeros((number_batches,))

            def batch_step(i, state):
                phi, opt_state, rng_key, epoch_elbos, all_indices = state
                start = i * batch_size
                batch_indices = jax.lax.dynamic_slice(
                    all_indices, (start,), (batch_size,)
                )
                phi, opt_state, loss_val, rng_key = step(
                    phi, opt_state, rng_key, dim_data, batch_size, batch_indices, S
                )
                epoch_elbos = epoch_elbos.at[i].set(-loss_val)
                return phi, opt_state, rng_key, epoch_elbos, all_indices

            phi, opt_state, rng_key, epoch_elbos, _ = jax.lax.fori_loop(
                0,
                number_batches,
                batch_step,
                (phi, opt_state, rng_key, epoch_elbos, all_indices),
            )
            current_elbo = jnp.mean(epoch_elbos)
            return phi, opt_state, rng_key, current_elbo

        initial_elbo_array = jnp.zeros((n_epochs,))
        initial_state = (
            0,
            self.phi,
            self.opt_state,
            self.rng_key,
            -jnp.inf,
            0,
            initial_elbo_array,
        )

        def epoch_body(state):
            """Update the epoch state and log progress using relative ELBO improvement.

            Parameters
            ----------
            state : tuple
                Contains (epoch, phi, opt_state, rng_key, best_elbo, window_counter,
                elbo_array).

            Returns
            -------
            tuple
                Updated state: (epoch+1, phi, opt_state, rng_key, new_best_elbo,
                new_window_counter, elbo_array).
            """
            epoch, phi, opt_state, rng_key, best_elbo, window_counter, elbo_array = (
                state
            )
            phi, opt_state, rng_key, current_elbo = epoch_batches(
                phi, opt_state, rng_key
            )
            elbo_array = elbo_array.at[epoch].set(current_elbo)
            epsilon = 1e-8
            improvement = jnp.where(
                best_elbo == -jnp.inf,
                1.0,
                (current_elbo - best_elbo) / (jnp.abs(best_elbo) + epsilon),
            )

            new_best_elbo = jax.lax.select(
                improvement > patience_tol, current_elbo, best_elbo
            )
            new_window_counter = jax.lax.select(
                improvement > patience_tol, 0, window_counter + 1
            )

            _ = jax.lax.cond(
                jnp.equal(jnp.mod(epoch + 1, 1000), 0),
                lambda _: (
                    jax.debug.print(
                        "Epoch: {epoch:6d} — ELBO: {elbo:.4f}",
                        epoch=epoch + 1,
                        elbo=current_elbo,
                    ),
                    0,
                )[1],
                lambda _: 0,
                operand=0,
            )

            _ = jax.lax.cond(
                new_window_counter >= window_size,
                lambda _: (
                    jax.debug.print(
                        "Early stopping triggered at epoch: {epoch:6d} with ELBO: "
                        "{elbo:.4f} - Relative Improvement: {imp:.4f}",
                        epoch=epoch + 1,
                        elbo=current_elbo,
                        imp=-improvement,
                    ),
                    0,
                )[1],
                lambda _: 0,
                operand=0,
            )
            return (
                epoch + 1,
                phi,
                opt_state,
                rng_key,
                new_best_elbo,
                new_window_counter,
                elbo_array,
            )

        def loop_cond(state):
            """
            Loop condition: Continue while epoch < n_epochs
            and window_counter < window_size.

            Parameters
            ----------
            state : tuple
                Contains (epoch, phi, opt_state, rng_key, best_elbo, window_counter,
                elbo_array).

            Returns
            -------
            bool
                True if the loop should continue, False otherwise.
            """
            epoch, phi, opt_state, rng_key, best_elbo, window_counter, elbo_array = (
                state
            )
            return (epoch < n_epochs) & (window_counter < window_size)

        final_state = jax.lax.while_loop(loop_cond, epoch_body, initial_state)
        epoch_count, phi, opt_state, rng_key, best_elbo, window_counter, elbo_array = (
            final_state
        )

        self.phi = phi
        self.opt_state = opt_state
        self.rng_key = rng_key
        self.elbo_values = elbo_array[:epoch_count].tolist()
        self.final_variational_distributions = self.get_final_distributions()

    def _elbo(self, phi, rng_key, dim_data, batch_size, batch_indices, S):
        """Compute the negative ELBO (Evidence Lower Bound) for variational inference.

        Parameters
        ----------
        phi : dict
            Current variational parameters.
        rng_key : jax.random.PRNGKey
            Random key for sampling.
        dim_data : int
            Total number of data points.
        batch_size : int
            Batch size used for subsetting the data.
        batch_indices : array-like
            Indices corresponding to the current batch.
        S : int
            Number of Monte Carlo samples.

        Returns
        -------
        float
            Negative ELBO (loss) computed over S Monte Carlo samples.
        jax.random.PRNGKey
            Updated random key.
        """
        rng_key, subkey = jax.random.split(rng_key)
        num_samples = S
        subkeys = jax.random.split(subkey, num_samples)

        @jax.jit
        def _single_sample_elbo(rng_key_sample):
            """Compute the ELBO for a single sample by accessing the model via the
            Interface instance."""
            samples, log_q = self._sample_and_compute_variational_log_prob(phi, rng_key_sample)
            log_prob = self.model_interface.compute_log_prob(
                samples, dim_data, batch_size, batch_indices
            )
            return log_prob - log_q

        elbo_samples = jax.vmap(_single_sample_elbo)(subkeys)
        elbo = jnp.mean(elbo_samples)
        return -elbo, rng_key


    def _sample_and_compute_variational_log_prob(self, phi, rng_key):
        """Sample from the variational distribution for all latent variables and compute log probabilities.

        Parameters
        ----------
        phi : dict
            Dictionary of variational parameters.
        rng_key : jax.random.PRNGKey
            Random key for sampling.

        Returns
        -------
        tuple
            (samples, total_log_q) where samples is a dict of all sampled latent variables
            and total_log_q is the sum of log probabilities under the variational distributions.
        """
        samples = {}
        total_log_q = 0.0

        for key, config in self.latent_vars_config.items():
            pval = phi[key]
            
            # Build the variational distribution
            dist_obj = self._build_variational_distribution(
                self.variational_dists_class[key],
                pval,
                self.fixed_distribution_params[key],
                self.variational_param_bijectors[key],
            )
            
            # Sample from the distribution
            rng_key, subkey = jax.random.split(rng_key)
            z = dist_obj.sample(seed=subkey)
            log_q = dist_obj.log_prob(z)
            
            # For each latent variable name in this configuration
            for pname in config["names"]:
                samples[pname] = z
                total_log_q += log_q

        return samples, total_log_q

    def get_final_distributions(self) -> dict[str, TfpDistribution]:
        """Construct and return the final variational distributions after applying
        specified transformations back into constrained space by applying bijectors.
        DOes differ for univariate and multivariate latent variables by key.

        Returns
        -------
        dict
            Mapping from latent variable names to their final variational distribution
            objects while providing valid TFP distributions.
        """
        final_results = {}

        for key, config in self.latent_vars_config.items():
            names = config["names"]
            dist_class = self.variational_dists_class[key]
            phi_unconstrained = self.phi[key]
            parameter_bijectors = self.variational_param_bijectors[key]

            final_distribution = self._build_variational_distribution(
                dist_class,
                phi_unconstrained,
                self.fixed_distribution_params[key],
                parameter_bijectors,
            )

            final_results.update({name: final_distribution for name in names})
            if len(names) > 1:
                final_results[key] = final_distribution
        return final_results

    def get_results(self, n_samples: int = 10_000, seed: int = 42) -> dict[str, Any]:
        """Generate inference results by sampling from the final variational
        distributions consistently, handling both univariate and multivariate latent
        variables.

        For multivariate latent variables, this method samples once from the composite
        distribution (using the composite key) and splits the flat sample into
        individual components based on the dimensions from the model parameters.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw from each variational distribution.
        seed : int
            Random key for sampling.

        Returns
        -------
        results : dict
            Dictionary containing:
            - "final_variational_distributions": the final variational distribution
              objects.
            - "elbo_values": the ELBO progression.
            - "samples": a dict mapping latent variable names to their samples.
            - "seed": the updated PRNGKey after sampling.
        """
        # results = {}
        samples = {}
        seed = jax.random.PRNGKey(seed)
        keys = jax.random.split(seed, len(self.latent_vars_config) + 1)

        model_params = self.model_interface.get_params()

        for i, config in enumerate(self.latent_vars_config.values()):
            if len(config["names"]) == 1:
                name = config["names"][0]
                dist = self.final_variational_distributions[name]
                samples[name] = dist.sample(n_samples, seed=keys[i + 1])
            else:
                composite_key = config["full_rank_key"]
                dist = self.final_variational_distributions[composite_key]
                composite_sample = dist.sample(n_samples, seed=keys[i + 1])
                dims = []
                for var in config["names"]:
                    dims.append(int(jnp.prod(jnp.array(model_params[var].shape))))
                total_dim = sum(dims)
                flat_sample = jnp.reshape(composite_sample, (n_samples, total_dim))
                cum_dims = []
                running_sum = 0
                for d in dims[:-1]:
                    running_sum += d
                    cum_dims.append(running_sum)
                split_samples = jnp.split(flat_sample, cum_dims, axis=1)
                for j, var in enumerate(config["names"]):
                    var_shape = model_params[var].shape
                    samples[var] = jnp.reshape(
                        split_samples[j], (n_samples,) + var_shape
                    )

        results = {
            "final_variational_distributions": self.final_variational_distributions,
            "elbo_values": self.elbo_values,
            "samples": samples,
            "seed": keys[0],
        }
        # results["final_variational_distributions"] = (
        #     self.final_variational_distributions
        # )
        # results["elbo_values"] = self.elbo_values
        # results["samples"] = samples
        # results["seed"] = keys[0]
        return results
