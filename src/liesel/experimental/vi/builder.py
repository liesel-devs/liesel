from typing import Any

import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax.bijectors as tfb
from jax.typing import ArrayLike
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.distributions import (
    Distribution as TfpDistribution,
)

from .interface import LieselInterface
from .optimizer import Optimizer

tfd = tfp.distributions


class OptimizerBuilder:
    """The :class:`.OptimizerBuilder` is used to construct an Optimizer for stochastic
    variational inference.

    .. rubric:: Workflow

    The general workflow usually looks something like this:

    #. Create a builder with :class:`.OptimizerBuilder`.
    #. Optionally set the optimization parameters (e.g., number of epochs,
    #  sample size, batch size).
    #. Create an instance of the model interface with :class:`.LieselInterface`
       and set it with :meth:`.set_model`.
    #. Add latent variables with :meth:`.add_variational_dist`.
    #. Build an :class:`~.optimizer.Optimizer` with :meth:`.build`.
    #. Run optimization using :meth:`.fit`.

    Optionally, you can also:

    - Specify fixed distribution parameters and optimizer chains
      for each latent variable.

    Parameters
    ----------
    seed : int
        Random seed used for reproducibility and for jittering initial values. Based on
        jax.random.PRNGKey, so it should be a non-negative integer.
    n_epochs : int
        Number of epochs to run the optimization.
    S : int
        Number of samples used in the variational estimation.
    patience_tol : float, optional
        Tolerance for early stopping based on ELBO improvements.
    window_size : int, optional
        Number of epochs to wait before early stopping if no improvement occurs.
    batch_size : int, optional
        Batch size used during optimization for subsetting the data; if None,
        the full dataset is used.

    See Also
    --------
    ~.optimizer.Optimizer : The Optimizer constructed by this builder.
    ~.interface.LieselInterface : Interface for the model.
    ~.optax.GradientTransformation : The optimizer chain transformation.

    Examples
    --------
    **Adding latent variables**

    In this example, we add a univariate latent variable for `sigma_sq` and another for
    `b` using separate calls to :meth:`.add_variational_dist` (Mean-Field).

    >>> import jax.numpy as jnp
    >>> import optax
    >>> import tensorflow_probability.substrates.jax.bijectors as tfb
    >>> from tensorflow_probability.substrates import jax as tfp
    >>> tfd = tfp.distributions
    >>>
    >>> import liesel.model as lsl
    >>> from liesel.experimental.vi import OptimizerBuilder, LieselInterface
    >>>
    >>> key = jax.random.PRNGKey(0)
    >>> X = jax.random.normal(key, (100, 4))
    >>> true_b = jnp.array([1.0, -2.0, 0.5, 3.0])
    >>> y = X @ true_b + jax.random.normal(key, (100,)) * 0.5
    >>>
    >>> # Set up a minimal model.
    >>> X_m = lsl.Var.new_obs(X, name="X_m")
    >>> dist_b = lsl.Dist(tfd.Normal, loc=0.0, scale=10.0)
    >>> b = lsl.Var.new_param(jnp.array([10.0, 10.0, 10.0, 10.0]), dist_b, name="b")
    >>> def linear_model(X, b):
    ...     return X @ b
    >>>
    >>> mu = lsl.Var.new_calc(linear_model, X=X_m, b=b, name="mu")
    >>>
    >>> a = lsl.Var.new_value(0.001, name="a")
    >>> b_var_dist = lsl.Var.new_value(0.001, name="b_var_dist")
    >>>
    >>> sigma_sq_dist = lsl.Dist(tfd.InverseGamma, concentration=a, scale=b_var_dist)
    >>> sigma_sq = lsl.Var.new_param(1.0, sigma_sq_dist, name="sigma_sq")
    >>> log_sigma_sq = sigma_sq.transform()
    >>>
    >>> # sigma = lsl.Var.new_calc(jnp.sqrt, sigma_sq, name="sigma")
    >>>
    >>> y_dist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma_sq)
    >>> y_m = lsl.Var.new_obs(y, y_dist, name="y_m")
    >>>
    >>> gb = lsl.GraphBuilder().add(y_m)
    >>> gb.plot_vars()
    >>>
    >>> model = gb.build_model()
    >>> # Initialize the builder.
    >>> builder = OptimizerBuilder(
    ...     seed=0,
    ...     n_epochs=10000,
    ...     S=32,
    ...     patience_tol=1e-4,
    ...     window_size=500,
    ... )
    >>> # Set up the model interface.
    >>> interface = LieselInterface(model)
    >>> builder.set_model(interface)
    >>>
    >>> # Define optimizer chains.
    >>> optimizer_chain1 = optax.chain(optax.clip(1), optax.adam(learning_rate=0.001))
    >>> optimizer_chain2 = optax.chain(optax.clip(1), optax.adam(learning_rate=0.001))
    >>> # Add a univariate latent variable for sigma_sq.
    >>> builder.add_variational_dist(
    ...     ["sigma_sq_transformed"],
    ...     dist_class=tfd.Normal,
    ...     variational_params={"loc": 1.0, "scale": 0.5},
    ...     optimizer_chain=optimizer_chain1,
    ...     variational_param_bijectors={
    ...         "loc": tfb.Identity(),
    ...         "scale": tfb.Softplus(),
    ...     },
    ... )
    >>> # Add a univariate latent variable for b.
    >>> builder.add_variational_dist(
    ...     ["b"],
    ...     dist_class=tfd.MultivariateNormalDiag,
    ...     variational_params={"loc": jnp.zeros(4), "scale_diag": jnp.ones(4)},
    ...     optimizer_chain=optimizer_chain2,
    ... )
    >>> # Or use joint multivariate latent variable for b and sigma_sq
    >>> builder.add_variational_dist(
    ...     ["b", "sigma_sq_transformed"],
    ...     dist_class=tfd.MultivariateNormalDiag,
    ...     variational_params={"loc": jnp.zeros(5), "scale_diag": jnp.ones(5)},
    ...     optimizer_chain=optimizer_chain2,
    ... )
    >>> # Build and run the optimizer.
    >>> optimizer = builder.build()
    >>> optimizer.fit()
    """

    def __init__(
        self,
        seed: int = 0,
        n_epochs: int = 10_000,
        S: int = 32,
        patience_tol: float | None = None,
        window_size: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.seed = seed
        self.n_epochs = n_epochs
        self.patience_tol = patience_tol
        self.window_size = window_size
        self.batch_size = batch_size
        self.S = S
        self._model_interface: LieselInterface | None = None
        self.latent_variables: list[dict[str, Any]] = []

    def set_model(self, interface: LieselInterface) -> None:
        """Set the model interface for the optimizer.

        Parameters
        ----------
        interface
            An instance that provides access to the model.
        """
        self._model_interface = interface

    def _validate_latent_variable_keys(
        self,
        dist_class: type[TfpDistribution],
        parameter_bijectors: dict[str, tfb.Bijector] | None = None,
        variational_params: dict[str, float] | None = None,
    ) -> None:
        """Validate that custom keys are unique and match the distribution parameters.

        All keys supplied in ``parameter_bijectors`` and ``variational_params`` must be
        unique (within their respective dictionaries) **and** belong to the set of valid
        parameter names returned by
        ``dist_class.parameter_properties().keys()``.  A ``ValueError`` is raised
        if any requirement is violated.

        Parameters
        ----------
        dist_class
            TensorFlow Probability distribution class (e.g., ``tfd.Normal``).
        parameter_bijectors
            Optional user-supplied bijectors. If ``None`` (default), no custom
            bijectors are used.
        variational_params
            Optional initial variational parameters. If ``None`` (default), no
            initial parameters are provided.

        Raises
        ------
        ValueError
            If duplicate keys are found or if a key is not a valid parameter
            of ``dist_class``.
        """
        valid_keys = set(dist_class.parameter_properties().keys())

        def _check(name: str, mapping: dict[str, Any] | None) -> None:
            if mapping is not None:
                invalid = set(mapping) - valid_keys
                if invalid:
                    raise ValueError(
                        f"Invalid key(s) in '{name}': {invalid}. "
                        f"Valid keys are: {valid_keys}."
                    )

        _check("parameter_bijectors", parameter_bijectors)
        _check("variational_params", variational_params)

    def _obtain_parameter_default_bijectors(self, dist_class):
        """Return the default constraining bijectors for all parameters of a
        TensorFlow Probability distribution.

        Parameters
        ----------
        dist_class
            The TensorFlow Probability distribution class (e.g., ``tfd.Normal``).

        Returns
        -------
        dict[str, tfb.Bijector]
            A mapping of parameter names to their default constraining bijectors,
            provided by ``dist_class.parameter_properties()``.
        """
        parameter_properties = dist_class.parameter_properties()
        parameter_names = parameter_properties.keys()
        parameter_default_bijectors = {
            parameter_name: parameter_properties[
                f"{parameter_name}"
            ].default_constraining_bijector_fn()
            for parameter_name in parameter_names
        }
        return parameter_default_bijectors

    def _merge_parameter_bijectors(
        self,
        default_bijectors: dict[str, tfb.Bijector],
        custom_bijectors: dict[str, tfb.Bijector] | None,
    ) -> dict[str, tfb.Bijector]:
        """Merge default and custom parameter bijectors.

        Default bijectors are taken from the distribution’s parameter properties,
        while custom bijectors (if provided) override any defaults for matching
        parameter names.

        Parameters
        ----------
        default_bijectors
            Mapping of parameter names to their default bijectors.
        custom_bijectors
            Optional mapping of parameter names to user-supplied bijectors that
            should replace the defaults.

        Returns
        -------
        dict[str, tfb.Bijector]
            A combined mapping where custom bijectors take precedence.
        """
        merged_dict = default_bijectors | (
            custom_bijectors if custom_bijectors is not None else {}
        )
        return merged_dict

    def _get_latent_var_dims(self, latent_variable_names: list[str]) -> dict[str, int]:
        """Get the dimensionality of each latent variable from the model.

        Parameters
        ----------
        latent_variable_names
            List of latent variable names.

        Returns
        -------
        dict[str, int]
            Mapping from variable names to their dimensionalities.

        Raises
        ------
        KeyError
            If a parameter is not found in the model.
        """
        assert self._model_interface is not None
        model_params = self._model_interface.get_params()

        dims = {}

        for pname in latent_variable_names:
            if pname not in model_params:
                raise KeyError(f"Parameter {pname} not found in model parameters")
            dims[pname] = int(jnp.prod(jnp.array(model_params[pname].shape)))

        return dims

    def _validate_dimensionality(
        self,
        latent_variable_names: list[str],
        event_shape_tensor: ArrayLike,
        variable_dims: dict[str, int],
        batch_shape_tensor: ArrayLike,
        dist_class_name: str,
    ) -> None:
        """Validate that the total dimensions match the event shape.

        Parameters
        ----------
        latent_variable_names
            List of latent variable names.
        event_shape_tensor
            Event shape tensor from distribution.event_shape_tensor().
        variable_dims
            Mapping from variable names to their dimensionalities.
        batch_shape_tensor
            Batch shape tensor from distribution.batch_shape_tensor().
        dist_class_name
            Name of the distribution class for error messages.

        Raises
        ------
        NotImplementedError
            If event shape represents a matrix or higher-dimensional array.
        ValueError
            If dimensions don't match the event shape or if batch shape confusion
            is detected.
        """
        # Convert tensors to tuples for the error messages
        event_shape_tuple = (
            tuple(event_shape_tensor.tolist()) if event_shape_tensor.size > 0 else ()
        )
        batch_shape_tuple = (
            tuple(batch_shape_tensor.tolist()) if batch_shape_tensor.size > 0 else ()
        )

        # Check if event shape represents a matrix
        if len(event_shape_tuple) > 1:
            raise NotImplementedError(
                f"Matrix-valued event shapes are not supported.\n"
                f"The initialization of the {dist_class_name} has "
                f"event_shape={event_shape_tuple}.\n"
                f"Currently, only scalar or vector-valued event shapes are "
                f"supported\n"
                f"(event shapes like () or (n,), not shapes like (n, m))."
            )

        # Calculate flattened event shape
        event_shape = (
            int(jnp.prod(event_shape_tensor)) if event_shape_tensor.size > 0 else 1
        )

        total_dim = sum(variable_dims[name] for name in latent_variable_names)

        # Check if dimensions match event shape (All good case)
        if event_shape == total_dim:
            return

        # Get batch size for additional validation
        batch_size = (
            int(jnp.prod(batch_shape_tensor)) if batch_shape_tensor.size > 0 else 0
        )

        # Check for batch shape confusion (Case 1 mistake)
        if batch_size == total_dim and batch_size > 1:
            if len(latent_variable_names) > 1:
                # Multiple variables - show both options
                var_list = ", ".join([f"'{name}'" for name in latent_variable_names])
                raise ValueError(
                    f"Dimension mismatch:\n"
                    f"Total latent variable dim. ({total_dim}) match batch_shape="
                    f"{batch_shape_tuple}, not event_shape={event_shape_tuple}.\n"
                    f"This suggests you're trying to model the variational dist. "
                    f"with batched distributions.\n\n"
                    f"In Liesel VI, use one of these approaches instead:\n\n"
                    f"1) Separate add_variational_dist calls for the respective set "
                    f"of independent latent variables:\n"
                    f"   [{var_list}]\n\n"
                    f"2) Use a multivariate distribution with the desired "
                    f"independence structure:\n"
                    f"   For example, use tfd.MultivariateNormalDiag instead of "
                    f"batched tfd.Normal\n\n"
                    f"Why: Each add_variational_dist call should represent ONE "
                    f"variational distribution."
                )
            else:
                # Single variable - only show multivariate distribution option
                raise ValueError(
                    f"Dimension mismatch:\n"
                    f"Total latent variable dim. ({total_dim}) match batch_shape="
                    f"{batch_shape_tuple}, not event_shape={event_shape_tuple}.\n"
                    f"This suggests you're trying to model the variational dist. "
                    f"with batched distributions.\n\n"
                    f"In Liesel VI, use a multivariate distribution with the desired "
                    f"independence structure:\n"
                    f"For example, use tfd.MultivariateNormalDiag instead of "
                    f"batched tfd.Normal\n\n"
                    f"Why: Each add_variational_dist call should represent ONE "
                    f"variational distribution."
                )

        # General dimension mismatch (Case 2 mistake)
        raise ValueError(
            f"Dimension mismatch:\n"
            f"Total latent variable dim.: {total_dim}\n"
            f"Distribution event_shape: {event_shape_tuple}\n\n"
            f"The total dimensions of your latent variables must match the "
            f"event shape."
        )

    def add_variational_dist(
        self,
        latent_variable_names: list[str],
        dist_class: type[TfpDistribution],
        *,
        variational_params: dict[str, float],
        fixed_distribution_params: dict[str, float] | None = None,
        optimizer_chain: optax.GradientTransformation,
        variational_param_bijectors: dict[str, tfb.Bijector] | None = None,
    ) -> None:
        """Add a latent variable to the optimizer configuration by adding a variational
        distribution for each latent variable independently (Mean-Field Approach).

        The user passed variational parameters (here named ``variational_params``) are
        expected to be in the to the distribution class according space which is the
        constrained space.

        The ``variational_param_bijectors`` can be used to specify bijectors
        for the variational parameters. The expected behaviour is that if the
        user passes a custom bijector, then the forward should be from unconstrained ->
        constrained and the inverse method from constrained -> unconstrained.

        Parameters
        ----------
        latent_variable_names
            List of parameter names.
        dist_class
            Distribution class (e.g., ``tfd.Normal``).
        variational_params
            Dictionary containing the initial parameters.
        fixed_distribution_params
            Optional fixed parameters for the distribution.
        optimizer_chain
            Optimizer chain for gradient transformations.
        variational_param_bijectors
            Optional overrides that replace the parameter's default
            ``tfp.util.ParameterProperties`` bijector, mapping the parameter from
            unconstrained to a constrained space.
        """

        if isinstance(latent_variable_names, str):
            latent_variable_names = [latent_variable_names]

        self._validate_latent_variable_keys(
            dist_class, variational_param_bijectors, variational_params
        )

        # Only obtain default bijectors for parameters not provided by the user
        parameter_properties = dist_class.parameter_properties()
        parameter_bijectors = {}

        for param_name in parameter_properties.keys():
            if (
                variational_param_bijectors
                and param_name in variational_param_bijectors
            ):
                # Use user-provided bijector
                parameter_bijectors[param_name] = variational_param_bijectors[
                    param_name
                ]
            else:
                # Use default bijector
                parameter_bijectors[param_name] = parameter_properties[
                    param_name
                ].default_constraining_bijector_fn()

        config = {
            "names": latent_variable_names,
            "dist_class": dist_class,
            "variational_params": variational_params,
            "fixed_distribution_params": fixed_distribution_params,
            "optimizer_chain": optimizer_chain,
            "variational_param_bijectors": parameter_bijectors,
        }

        if len(latent_variable_names) > 1:
            config["full_rank_key"] = "_".join(latent_variable_names)
        else:
            config["full_rank_key"] = latent_variable_names[0]

        distribution = self._validate_and_build_distributions(
            config, latent_variable_names
        )

        # Get shape tensors for validation
        event_shape_tensor = distribution.event_shape_tensor()
        batch_shape_tensor = distribution.batch_shape_tensor()
        dist_class_name = dist_class.__name__

        variable_dims = self._get_latent_var_dims(latent_variable_names)
        config["variable_dims"] = variable_dims
        config["dims_list"] = [int(v) for v in variable_dims.values()]
        dims_list = config["dims_list"]
        if len(dims_list) > 1:
            import numpy as np

            split_indices = list(np.cumsum(dims_list)[:-1])
        else:
            split_indices = []
        config["split_indices"] = split_indices

        # Validate dimensionality with shape tensors
        self._validate_dimensionality(
            latent_variable_names,
            event_shape_tensor,
            variable_dims,
            batch_shape_tensor,
            dist_class_name,
        )

        # Calculate flattened event shape for config after validation
        event_shape = (
            int(jnp.prod(event_shape_tensor)) if event_shape_tensor.size > 0 else 1
        )
        config["event_shape"] = event_shape

        self.latent_variables.append(config)

    def _validate_and_build_distributions(
        self, config: dict[str, Any], latent_variable_names: list[str]
    ) -> TfpDistribution:
        dist_class = config["dist_class"]
        variational_params = config["variational_params"]
        fixed_distribution_params = config.get("fixed_distribution_params", {})
        parameter_bijectors = config.get("variational_param_bijectors", None)

        try:
            # Build-time: bijectors + distribution ctor (wrap any failure)
            if parameter_bijectors is not None:
                variational_params_constrained = {}
                for p_name, p_val in variational_params.items():
                    bij = parameter_bijectors.get(p_name)
                    if bij is not None:
                        variational_params_constrained[p_name] = bij.forward(
                            bij.inverse(p_val)
                        )
                    else:
                        variational_params_constrained[p_name] = p_val
            else:
                variational_params_constrained = variational_params

            distribution = dist_class(
                **variational_params_constrained,
                **(fixed_distribution_params or {}),
            )

        except Exception as e:
            names = config.get("names", ["unknown"])
            msg = (
                "Failed to build variational distribution for latent variable(s) "
                f"{names}: {e}"
            )
            raise ValueError(msg) from e

        # Validation-time: preserve the specific error type/message
        if distribution.reparameterization_type != tfd.FULLY_REPARAMETERIZED:
            raise AttributeError(
                "Only fully reparameterized distributions are supported for "
                f"latent variable(s) {latent_variable_names}. "
                f"Got reparameterization type: {distribution.reparameterization_type}"
            )

        return distribution

    def build(self) -> Optimizer:
        """Build and return an Optimizer instance based on the current configuration.

        Returns:
            Optimizer: An optimizer configured with the specified model interface and
            latent variables.
        """
        if self._model_interface is None:
            raise ValueError(
                "Model interface not set. Call builder.set_model(...) first."
            )

        latent_variables_dict = {}
        for config in self.latent_variables:
            names = config["names"]
            if len(names) == 1:
                key = names[0]
            else:
                key = "_".join(names)
            latent_variables_dict[key] = config

        return Optimizer(
            seed=self.seed,
            n_epochs=self.n_epochs,
            S=self.S,
            patience_tol=self.patience_tol,
            window_size=self.window_size,
            batch_size=self.batch_size,
            model_interface=self._model_interface,
            latent_variables=latent_variables_dict,
        )
