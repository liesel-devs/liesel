"""
This module provides a :class:`.ModelInterface` implementation for PyMC
models.

To use this module, the pymc package must be installed. To do so, please install
liesel with the optional dependencies pymc:

.. code-block:: bash

    $ pip install liesel[pymc]


Example of an linear model
^^^^^^^^^^^^^^^^^^^^^^^^^^

The model is also used in the test. Please consult the tutorial book for longer
examples::

    RANDOM_SEED = 123
    rng = np.random.RandomState(RANDOM_SEED)

    # set parameter values
    num_obs = 100
    sigma = 1.0
    beta = [1, 1, 2]

    # simulate covariates
    x1 = rng.randn(num_obs)
    x2 = 0.5 * rng.randn(num_obs)

    # simulate outcome variable
    y = beta[0] + beta[1] * x1 + beta[2] * x2 + sigma * rng.normal(size=num_obs)

    basic_model = pm.Model()
    with basic_model:
        # priors
        beta = pm.Normal("beta", mu=0, sigma=10, shape=3)
        sigma = pm.HalfNormal("sigma", sigma=1)
        # sigma is automatically transformed to real (log)
        # the new variable is called sigma_log__

        # predicted value
        mu = beta[0] + beta[1] * x1 + beta[2] * x2

        # track the predicted value of the first obs
        pm.Deterministic("mu[0]", mu[0])

        # distribution of response (likelihood)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    interface = PyMCInterface(basic_model, additional_vars=["sigma", "mu[0]"])
    state = interface.get_initial_state()
    builder = gs.EngineBuilder(1, 2)
    builder.set_initial_values(state)
    builder.set_model(interface)
    builder.set_duration(1000, 2000)

    builder.add_kernel(gs.NUTSKernel(["beta"]))
    builder.add_kernel(gs.NUTSKernel(["sigma_log__"]))

    builder.positions_included = ["sigma", "mu[0]"]

    engine = builder.build()

    engine.sample_all_epochs()
    results = engine.get_results()
    sum = gs.Summary(results)
    sum


Transformations of RVs can be avoided by setting ``transform = None`` in the
distribution argument.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from liesel.goose.types import ModelState, Position

if TYPE_CHECKING:
    try:
        import pymc as pm
    except ImportError as e:
        raise ImportError(
            f"pymc must be installed to use this module. original exception: {e}"
        )


class PyMCInterface:
    """
    An implementation of :class:`~liesel.goose.types.ModelInterface` to be used with a
    PyMC model.

    The initial position can be extraced with :meth:`.get_initial_state`. The model
    state is represented as a :class:`.Position`.


    Parameters
    ----------
    model
        a pymc model
    additional_vars:
        names variables that are by default not but should be available via
        extract_position


    By default, only non-observed random variables are available via extract_position.
    This includes transformed variables but not the untransformed variable. Also,
    `Deterministic` is not available. To make them trackable for
    :class:`~liesel.goose.engine.Engine` these variables must be mentioned in the
    constructor.
    """

    def __init__(self, model: pm.Model, additional_vars: list[str] = []):
        try:
            import pymc as pm
            from pymc.sampling_jax import get_jaxified_graph, get_jaxified_logp
        except ImportError as e:
            raise ImportError(
                f"pymc must be installed to use this module. original exception: {e}"
            )

        self._pymc_model = model
        self._log_prob = get_jaxified_logp(self._pymc_model)
        self._rv_names = [rv.name for rv in model.value_vars]
        self._additional_vars = additional_vars

        # create a function to calculate the additional vars

        all_vars = pm.util.get_default_varnames(
            pm.modelcontext(model).unobserved_value_vars, include_transformed=True
        )
        selected_vars = [var for var in all_vars if var.name in self._additional_vars]
        self._calc_add_vars = get_jaxified_graph(
            inputs=model.value_vars, outputs=selected_vars
        )

    def get_initial_state(self) -> Position:
        """
        Returns the model's initial.
        """
        return Position(self._pymc_model.initial_point())

    def extract_position(
        self, position_keys: Sequence[str], model_state: ModelState
    ) -> Position:
        """Extracts a sub-position specified by position_keys from model_state."""
        # only extend the state if requested
        if self._additional_vars and any(
            key in position_keys for key in self._additional_vars
        ):
            model_state = model_state.copy()
            rv_values = [model_state[rv] for rv in self._rv_names]
            additional_values = self._calc_add_vars(*rv_values)

            for key, val in zip(self._additional_vars, additional_values):
                model_state[key] = val

        return Position({key: model_state[key] for key in position_keys})

    def update_state(self, position: Position, model_state: ModelState) -> ModelState:
        """Updates the model state with position returning the new model state."""
        ms: Position = model_state.copy()  # do not change the input (escaped traces).
        ms.update(position)
        return ms

    def log_prob(self, model_state: ModelState) -> float:
        """Computes the unnormalized log-probability given the model state."""

        rv_values = [model_state[rv] for rv in self._rv_names]
        return self._log_prob(rv_values)
