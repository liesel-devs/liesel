"""
Distributional regression.
"""

from __future__ import annotations

from collections import defaultdict

import jax.numpy as jnp
import jax.random
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel.distributions import MultivariateNormalDegenerate
from liesel.goose import EngineBuilder, GibbsKernel, IWLSKernel, LieselInterface
from liesel.goose.types import JitterFunction
from liesel.option import Option

from .legacy import (
    DesignMatrix,
    Hyperparameter,
    InverseLink,
    Predictor,
    RegressionCoef,
    Response,
    Smooth,
    SmoothingParam,
)
from .model import GraphBuilder, Model
from .nodes import Array, Bijector, Dist, Distribution, Group, NodeState, Var

matrix_rank = np.linalg.matrix_rank

tau2_jitter_fn: JitterFunction = lambda key, val: jax.random.truncated_normal(
    key, 0.0, 1e2, val.shape, val.dtype
)
beta_jitter_fn: JitterFunction = lambda key, val: val + jax.random.uniform(
    key, val.shape, val.dtype, -2.0, 2.0
)


class DistRegBuilder(GraphBuilder):
    """A model builder for distributional regression models."""

    def __init__(self) -> None:
        super().__init__()

        self._smooths: dict[str, list[Var]] = defaultdict(list)
        self._distributional_parameters: dict[str, Var] = {}
        self._predictors: dict[str, Var] = {}
        self._response: Option[Var] = Option(None)

    @property
    def response(self) -> Var:
        """The response node."""
        return self._response.expect(f"No response in {repr(self)}")

    def _smooth_name(self, name: str | None, predictor: str, prefix: str) -> str:
        """Generates a name for a smooth if the ``name`` argument is ``None``."""

        other_smooths = self._smooths[predictor]
        other_names = [node.name for node in other_smooths if node.name]
        prefix = predictor + "_" + prefix
        counter = 0

        while prefix + str(counter) in other_names:
            counter += 1

        if not name:
            name = prefix + str(counter)

        if name in other_names:
            raise RuntimeError(
                f"Smooth {repr(name)} already exists in {repr(self)} "
                f"for predictor {repr(predictor)}"
            )

        return name

    def add_p_smooth(
        self,
        X: Array,
        m: float,
        s: float,
        predictor: str,
        name: str | None = None,
    ) -> DistRegBuilder:
        """
        Adds a parametric smooth to the model builder.

        Parameters
        ----------
        X
            The design matrix.
        m
            The mean of the Gaussian prior.
        s
            The standard deviation of the Gaussian prior.
        predictor
            The name of the predictor to add the smooth to.
        name
            The name of the smooth.
        """

        if predictor not in self._distributional_parameters:
            raise RuntimeError(
                f"No predictor '{predictor}' found. You need to add this predictor to"
                " the builder first."
            )

        name = self._smooth_name(name, predictor, "p")

        X_var = DesignMatrix(X, name=name + "_X")
        m_var = Hyperparameter(m, name=name + "_m")
        s_var = Hyperparameter(s, name=name + "_s")

        beta = np.zeros(np.shape(X)[-1], np.float32)
        beta_distribution = Dist(tfd.Normal, loc=m_var, scale=s_var)
        beta_var = RegressionCoef(beta, beta_distribution, name + "_beta")

        smooth_var = Smooth(X_var, beta_var, name=name)
        self._smooths[predictor].append(smooth_var)

        predictor_var = self._predictors[predictor]
        predictor_var.value_node.add_inputs(smooth_var)

        group = Group(name, smooth=smooth_var, beta=beta_var, X=X_var, m=m_var, s=s_var)
        self.add_groups(group)

        return self

    def add_np_smooth(
        self,
        X: Array,
        K: Array,
        a: float,
        b: float,
        predictor: str,
        name: str | None = None,
    ) -> DistRegBuilder:
        """
        Adds a non-parametric smooth to the model builder.

        Parameters
        ----------
        X
            The design matrix.
        K
            The penalty matrix.
        a
            The a, :math:`\\alpha` or concentration parameter of the inverse gamma
            prior.
        b
            The b, :math:`\\beta` or scale parameter of the inverse gamma prior.
        predictor
            The name of the predictor to add the smooth to.
        name
            The name of the smooth.
        """
        name = self._smooth_name(name, predictor, "np")

        X_var = DesignMatrix(X, name=name + "_X")
        K_var = Hyperparameter(K, name=name + "_K")
        a_var = Hyperparameter(a, name=name + "_a")
        b_var = Hyperparameter(b, name=name + "_b")

        rank_var = Hyperparameter(matrix_rank(K), name=name + "_rank")
        tau2_distribution = Dist(tfd.InverseGamma, concentration=a_var, scale=b_var)
        tau2_var = SmoothingParam(10000.0, tau2_distribution, name + "_tau2")

        beta = np.zeros(np.shape(X)[-1], np.float32)
        beta_distribution = Dist(
            MultivariateNormalDegenerate.from_penalty,
            loc=0.0,
            var=tau2_var,
            pen=K_var,
            rank=rank_var,
        )
        beta_var = RegressionCoef(beta, beta_distribution, name + "_beta")

        smooth_var = Smooth(X_var, beta_var, name=name)
        self._smooths[predictor].append(smooth_var)

        predictor_var = self._predictors[predictor]
        predictor_var.value_node.add_inputs(smooth_var)

        group = Group(
            name,
            smooth=smooth_var,
            beta=beta_var,
            tau2=tau2_var,
            rank=rank_var,
            X=X_var,
            K=K_var,
            a=a_var,
            b=b_var,
        )

        self.add_groups(group)
        return self

    def add_predictor(self, name: str, inverse_link: type[Bijector]) -> DistRegBuilder:
        """
        Adds a predictor to the model builder.

        Parameters
        ----------
        name
            The name of the parameter of the response distribution. Must match the name
            of the parameter of the TFP distribution.
        inverse_link
            The inverse link mapping the regression predictor to the parameter of the
            response distribution. Either a string identifying a TFP bijector, or
            alternatively, a TFP-compatible bijector class. If a class is provided
            instead of a string, the user needs to make sure it uses the right NumPy
            implementation.
        """
        if self.response is None:
            raise RuntimeError("No response found. Add a response first.")
        predictor_var = Predictor(name=name + "_pdt")
        parameter_var = InverseLink(predictor_var, inverse_link, name=name)
        self._predictors[name] = predictor_var
        self._distributional_parameters[name] = parameter_var

        dist_node = self.response.dist_node
        dist_node.set_inputs(**self._distributional_parameters)  # type: ignore

        self.add(predictor_var, parameter_var)
        return self

    def add_response(
        self, response: Array, distribution: type[Distribution]
    ) -> DistRegBuilder:
        """
        Adds the response to the model builder.

        Parameters
        ----------
        response
            The response vector or matrix.
        distribution
            The conditional distribution of the response variable. Either a string
            identifying a TFP distribution, or alternatively, a TFP-compatible
            distribution class. If a class is provided instead of a string, the user
            needs to make sure it uses the right NumPy implementation.
        """

        response_var = Response(response, Dist(distribution), "response")
        self._response = Option(response_var)
        self.add(response_var)

        return self


def tau2_gibbs_kernel(group: Group) -> GibbsKernel:
    """Builds a Gibbs kernel for a smoothing parameter with an inverse gamma prior."""
    position_key = group["tau2"].name

    def transition(prng_key, model_state: dict[str, NodeState]):
        a_prior = group.value_from(model_state, "a")
        rank = group.value_from(model_state, "rank")

        a_gibbs = jnp.squeeze(a_prior + 0.5 * rank)

        b_prior = group.value_from(model_state, "b")
        beta = group.value_from(model_state, "beta")
        K = group.value_from(model_state, "K")

        b_gibbs = jnp.squeeze(b_prior + 0.5 * (beta @ K @ beta))

        draw = b_gibbs / jax.random.gamma(prng_key, a_gibbs)

        return {position_key: draw}

    return GibbsKernel([position_key], transition)


def dist_reg_mcmc(
    model: Model,
    seed: int,
    num_chains: int,
    tau2_jitter_fn: JitterFunction = tau2_jitter_fn,
    beta_jitter_fn: JitterFunction = beta_jitter_fn,
) -> EngineBuilder:
    """
    Configures an :class:`~.goose.EngineBuilder` for a distributional regression model.

    The EngineBuilder uses a Metropolis-in-Gibbs MCMC algorithm with an
    :class:`~.goose.IWLSKernel` for the regression coefficients and a
    :class:`~.goose.GibbsKernel` for the smoothing parameters for a distributional
    regression model.

    Parameters
    ----------
    model
        A model built with a :class:`.DistRegBuilder`.
    seed
        The PRNG seed for the engine builder.
    num_chains
        The number of chains to be sampled.
    tau2_jitter_fn
        Jittering function for the smoothing parameters.
    beta_jitter_fn
        Jittering function for the regression coefficients.
    See Also
    --------
    :meth:`~.goose.EngineBuilder.set_jitter_fns` : Method for setting the jittering
        functions
    """

    builder = EngineBuilder(seed, num_chains)

    builder.set_model(LieselInterface(model))
    builder.set_initial_values(model.state)

    jitter_fns = {}

    for group in model.groups().values():
        if "tau2" in group:
            position_key = group["tau2"].name
            tau2_kernel = tau2_gibbs_kernel(group)  # type: ignore  # only vars
            builder.add_kernel(tau2_kernel)
            jitter_fns[position_key] = tau2_jitter_fn

        if "beta" in group:
            position_key = group["beta"].name
            beta_kernel = IWLSKernel([position_key])
            builder.add_kernel(beta_kernel)
            jitter_fns[position_key] = beta_jitter_fn

    builder.set_jitter_fns(jitter_fns)

    return builder
