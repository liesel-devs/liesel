from collections.abc import Callable
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax.bijectors as tfb
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.distributions import (
    Distribution as TfpDistribution,
)

from liesel.distributions import MultivariateNormalLogCholeskyParametrization

from .interface import LieselInterface
from .optimizer import Optimizer

tfd = tfp.distributions


class Phi_MultivariateNormalLogCholeskyParametrization(TypedDict):
    """A TypedDict to specify the parameters for a multivariate normal distribution
    using a log-Cholesky parametrization."""

    loc: jnp.ndarray
    log_cholesky_parametrization: jnp.ndarray


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
    #. Add latent variables with :meth:`.add_variational_distribution` and/or
       :meth:`.add_multivariate_latent_variable`.
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
    The following examples demonstrate two ways to add latent variables:

    **Example 1: Adding latent variables separately**

    In this example, we add a univariate latent variable for `sigma_sq` and another for
    `b` using separate calls to :meth:`.add_variational_distribution` (Mean-Field).

    >>> import jax.numpy as jnp
    >>> import optax
    >>> import tensorflow_probability.substrates.jax.bijectors as tfb
    >>> from tensorflow_probability.substrates import jax as tfp
    >>> tfd = tfp.distributions
    >>>
    >>> # Set up a minimal model.
    >>> X_m = lsl.Var.new_obs(X, name="X_m")
    >>> dist_b = lsl.Dist(tfd.Normal, loc=0.0, scale=10.0)
    >>> b = lsl.Var.new_param(jnp.array([10.0, 10.0, 10.0, 10.0]), dist_b, name="b")
    >>> def linear_model(X, b):
    >>> return X @ b

    >>> mu = lsl.Var.new_calc(linear_model, X=X_m, b=b, name="mu")

    >>> a = lsl.Var.new_value(0.01, name="a")
    >>> b_var_dist = lsl.new_value(0.01, name="b_var_dist")

    >>> sigma_sq_dist = lsl.Dist(tfd.InverseGamma, concentration=a, scale=b_var_dist)
    >>> sigma_sq = lsl.Var.new_param(1.0, sigma_sq_dist, name="sigma_sq")
    >>> sigma = lsl.Var.new_calc(jnp.sqrt, sigma_sq, name="sigma")


    >>> y_dist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma)
    >>> y_m = lsl.Var.new_obs(y, y_dist, name="y_m")

    >>> gb = lsl.GraphBuilder().add(y_m)
    >>> gb.plot_vars()

    >>> model = gb.build_model()
    >>>
    >>> # Initialize the builder.
    >>> builder = OptimizerBuilder(
    ...     seed=0,
    ...     n_epochs=10000,
    ...     batch_size=64,
    ...     S=32,
    ...     patience_tol=0.001,
    ...     window_size=100,
    ... )
    >>>
    >>> # Set up the model interface.
    >>> interface = LieselInterface(model)
    >>> builder.set_model(interface)
    >>>
    >>> # Define optimizer chains.
    >>> optimizer_chain1 = optax.chain(optax.clip(1), optax.adam(learning_rate=0.001))
    >>> optimizer_chain2 = optax.chain(optax.clip(1), optax.adam(learning_rate=0.001))
    >>>
    >>> # Add a univariate latent variable for sigma_sq.
    >>> builder.add_variational_distribution(
    ...     ["sigma_sq"],
    ...     dist_class=tfd.Normal,
    ...     phi={"loc": 1.0, "scale": 0.5},
    ...     optimizer_chain=optimizer_chain1,
    ... )
    >>>
    >>> # Add a univariate latent variable for b.
    >>> builder.add_variational_distribution(
    ...     ["b"],
    ...     dist_class=tfd.Normal,
    ...     phi={"loc": jnp.zeros(4), "scale": jnp.ones(4)},
    ...     optimizer_chain=optimizer_chain2,
    ... )
    >>>
    >>> # Build and run the optimizer.
    >>> optimizer = builder.build()
    >>> optimizer.fit()

    **Example 2: Adding latent variables jointly as multivariate (Gaussian Full-Rank)**

    In this example, we add both `sigma_sq` and `b` together using a single call to
    :meth:`.add_multivariate_latent_variable`. Assume the same model
    specification as in the previous example.

    >>> # Initialize the builder as before.
    >>> builder = OptimizerBuilder(
    ...     seed=0,
    ...     n_epochs=10000,
    ...     batch_size=64,
    ...     S=32,
    ...     patience_tol=0.001,
    ...     window_size=100,
    ... )
    >>>
    >>> # Set up the model interface.
    >>> builder.set_model(interface)
    >>>
    >>> # Define an optimizer chain.
    >>> optimizer_chain = optax.chain(optax.clip(1), optax.adam(learning_rate=0.001))
    >>>
    >>> # Add a multivariate latent variable for both b and sigma_sq.
    >>> builder.add_multivariate_latent_variable(
    ...     ["b", "sigma_sq"],
    ...     phi={
    ...         "loc": jnp.ones(5),
    ...         "log_cholesky_parametrization": builder.make_log_cholesky_like(5),
    ...     },
    ...     fixed_distribution_params={"validate_args": True, "d": 5},
    ...     optimizer_chain=optimizer_chain,
    ...     transform=None,
    ... )
    >>>
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
        self.latent_variables: list[dict[str, Any]] = []

    def set_model(self, interface: LieselInterface) -> None:
        """Set the model interface for the optimizer.

        Parameters:
            interface (LieselInterface): An instance that provides access to the model.
        """
        self._model_interface = interface

    def _validate_latent_variable_keys(
        self,
        dist_class: type[TfpDistribution],
        parameter_bijectors: dict[str, tfb.Bijector] | None = None,
        phi: dict[str, float] | None = None,
    ) -> None:
        """Validate that custom keys are unique and match the distribution parameters.

        All keys supplied in ``parameter_bijectors`` and ``phi`` must be unique
        (within their respective dictionaries) **and** belong to the set of valid
        parameter names returned by
        ``dist_class.parameter_properties().keys()``.  A ``ValueError`` is raised
        if any requirement is violated.

        Parameters
        ----------
        dist_class : Callable
            TensorFlow Probability distribution class (e.g., ``tfd.Normal``).
        parameter_bijectors : dict[str, tfb.Bijector] | None, default None
            Optional user-supplied bijectors.
        phi : dict[str, float] | None, default None
            Optional initial variational parameters.

        Raises
        ------
        ValueError
            If duplicate keys are found or if a key is not a valid parameter
            of ``dist_class``.
        """
        valid_keys = set(dist_class.parameter_properties().keys())

        def _check(name: str, mapping: dict[str, Any] | None) -> None:
            if mapping is None:
                return
            invalid = set(mapping) - valid_keys
            if invalid:
                raise ValueError(
                    f"Invalid key(s) in '{name}': {invalid}. "
                    f"Valid keys are: {valid_keys}."
                )

        _check("parameter_bijectors", parameter_bijectors)
        _check("phi", phi)

    def _obtain_parameter_default_bijectors(self, dist_class):
        """Return the default constraining bijectors for all parameters of a
        TensorFlow Probability distribution.

        Parameters:
            dist_class : Callable
            The TensorFlow Probability distribution class (e.g., ``tfd.Normal``).

        Returns:
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
        default_bijectors : dict[str, tfb.Bijector]
            Mapping of parameter names to their default bijectors.
        custom_bijectors : dict[str, tfb.Bijector] | None
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

    def add_variational_distribution(
        self,
        latent_variable_names: list[str],
        dist_class: type[TfpDistribution],
        *,
        phi: dict[str, float],
        fixed_distribution_params: dict[str, float] | None = None,
        optimizer_chain: optax.GradientTransformation,
        variational_param_bijectors: dict[str, tfb.Bijector] | None = None,
    ) -> None:
        """Add a latent variable to the optimizer configuration by adding a variational
        distribution for each latent variable independently (Mean-Field Approach).

        The user passed variational parameters (here named ``phi``) are expected
        to be in the to the distribution class according space which is the
        constrained space.

        The ``variational_param_bijectors`` can be used to specify bijectors
        for the variational parameters. The expected behaviour is that if the
        user passes custom, then the forward should be from unconstrained ->
        constrained and the inverse method from constrained -> unconstrained.

        Parameters:
            latent_variable_names (List[str]): List of parameter names.
            dist_class (Callable): Distribution class (e.g., tfd.Normal).
            phi (Dict[str, float]): Dictionary containing the initial parameters.
            fixed_distribution_params (Optional[Dict[str, float]]): Optional fixed
            parameters for the distribution.
            optimizer_chain (optax.GradientTransformation): Optimizer chain for gradient
            transformations.
            variational_param_bijectors (dict[str, tfb.Bijector] | None) – Optional overrides
            that replace the parameter's default tfp.util.ParameterProperties bijector,
            mapping the parameter from unconstrained to a constrained space.
        """
        if isinstance(latent_variable_names, str):
            latent_variable_names = [latent_variable_names]

        self._validate_latent_variable_keys(dist_class, variational_param_bijectors, phi)

        parameter_bijectors_default = self._obtain_parameter_default_bijectors(
            dist_class
        )
        parameter_bijectors = self._merge_parameter_bijectors(
            parameter_bijectors_default, variational_param_bijectors
        )

        self.latent_variables.append(
            {
                "names": latent_variable_names,
                "dist_class": dist_class,
                "phi": phi,
                "fixed_distribution_params": fixed_distribution_params,
                "optimizer_chain": optimizer_chain,
                "variational_param_bijectors": parameter_bijectors,
            }
        )

    def add_multivariate_latent_variable(
        self,
        names: list[str],
        phi: Phi_MultivariateNormalLogCholeskyParametrization | None = None,
        *,
        fixed_distribution_params: dict[str, float] | None = None,
        optimizer_chain: optax.GradientTransformation,
        transform: Callable | tfb.Bijector | None = None,
    ) -> None:
        """Adds a multivariate latent variable to the optimizer configuration (Gaussian
        Full-Rank) with shared covariance of different latent variables.

        If fixed_distribution_params does not include 'd' (the total dimension),
        it is computed from the model's parameter shapes. If phi is
        not provided, a default is generated with ones for the location and a
        computed log-Cholesky vector.

        Parameters:
            names (List[str]): List of parameter names to be grouped as a
            multivariate variable.
            phi (Optional[Phi_MultivariateNormalLogCholeskyParametrization]): Initial
            parameters for the distribution.
            fixed_distribution_params (Optional[Dict[str, float]]): Optional
            fixed parameters for the distribution.
            optimizer_chain (optax.GradientTransformation): Optimizer chain for
            gradient transformations.
            transform (Optional[Union[Callable, tfb.Bijector]]): Transformation to apply
            on the latent variable.
        """
        if fixed_distribution_params is None:
            fixed_distribution_params = {}

        if "d" not in fixed_distribution_params:
            d = sum(
                jnp.array(self._model_interface.model.vars[var].value).size
                for var in names
            )
            fixed_distribution_params["d"] = d

        d = fixed_distribution_params["d"]

        if phi is None:
            phi = {
                "loc": jnp.ones(d),
                "log_cholesky_parametrization": self.make_log_cholesky_like(d),
            }

        self.latent_variables.append(
            {
                "names": names,
                "dist_class": MultivariateNormalLogCholeskyParametrization,
                "phi": phi,
                "fixed_distribution_params": fixed_distribution_params,
                "optimizer_chain": optimizer_chain,
                "transform": transform,
            }
        )

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
            # Create a readable key by joining the latent variable names
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

    def make_log_cholesky_like(self, d: int) -> jnp.ndarray:
        n = d * (d + 1) // 2
        arr = jnp.zeros((n,), dtype=jnp.float32)

        def body_fun(row, arr):
            offset = (row * (row + 1)) // 2
            arr = arr.at[offset + row].set(1.1)
            return arr

        arr = jax.lax.fori_loop(0, d, body_fun, arr)
        return arr
