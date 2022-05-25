"""
# Distributional regression
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random
import numpy as np

from liesel.goose import EngineBuilder, GibbsKernel, IWLSKernel
from liesel.option import Option

from .goose import GooseModel
from .model import Model, ModelBuilder
from .nodes import (
    PIT,
    ColumnStack,
    DesignMatrix,
    Hyperparameter,
    InverseLink,
    Node,
    NodeDistribution,
    NodeGroup,
    Predictor,
    RegressionCoef,
    Response,
    Smooth,
    SmoothingParam,
)
from .types import Array, TFPBijectorClass, TFPDistributionClass

matrix_rank = np.linalg.matrix_rank


class SmoothGroup(NodeGroup):
    """A node group representing a smooth."""


class PSmoothGroup(SmoothGroup):
    """A node group representing a parametric smooth."""


class NPSmoothGroup(SmoothGroup):
    """A node group representing a non-parametric smooth."""


class DistRegBuilder(ModelBuilder):
    """A model builder for distributional regression models."""

    _howto = "Smooths need to be added first, then the predictors, then the response"

    def __init__(self) -> None:
        super().__init__()

        self._smooths: dict[str, list[Node]] = {}
        self._distributional_parameters: dict[str, Node] = {}
        self._response: Option[Node] = Option(None)

    @property
    def response(self) -> Node:
        return self._response.expect(f"No response in {repr(self)}")

    def _smooth_name(self, name: str | None, predictor: str, prefix: str) -> str:
        """Generates a name for a smooth if the `name` argument is `None`."""

        other_smooths = self._smooths[predictor]
        other_names = [node.name for node in other_smooths if node.has_name]
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
    ) -> PSmoothGroup:
        """Adds a parametric smooth to the model builder.

        ## Parameters

        - `X`: The design matrix.
        - `m`: The mean of the Gaussian prior.
        - `s`: The standard deviation of the Gaussian prior.
        - `predictor`: The name of the predictor to add the smooth to.
        - `name`: The name of the smooth.
        """

        try:
            self._smooths[predictor]
        except KeyError:
            self._smooths[predictor] = []

        name = self._smooth_name(name, predictor, "p")

        X_node = DesignMatrix(X, name=name + "_X")
        m_node = Hyperparameter(m, name=name + "_m")
        s_node = Hyperparameter(s, name=name + "_s")

        beta = np.zeros(np.shape(X)[-1], np.float32)
        beta_distribution = NodeDistribution("Normal", loc=m_node, scale=s_node)
        beta_node = RegressionCoef(beta, beta_distribution, name + "_beta")

        smooth_node = Smooth(X_node, beta_node, name=name)
        self._smooths[predictor].append(smooth_node)

        group = PSmoothGroup(
            smooth=smooth_node,
            beta=beta_node,
            X=X_node,
            m=m_node,
            s=s_node,
        )

        group.name = name
        self.add_groups(group)
        return group

    def add_np_smooth(
        self,
        X: Array,
        K: Array,
        a: float,
        b: float,
        predictor: str,
        name: str | None = None,
    ) -> NPSmoothGroup:
        """
        Adds a non-parametric smooth to the model builder.

        ## Parameters

        - `X`: The design matrix.
        - `K`: The penalty matrix.
        - `a`: The a, α or concentration parameter of the inverse gamma prior.
        - `b`: The b, β or scale parameter of the inverse gamma prior.
        - `predictor`: The name of the predictor to add the smooth to.
        - `name`: The name of the smooth.
        """

        try:
            self._smooths[predictor]
        except KeyError:
            self._smooths[predictor] = []

        name = self._smooth_name(name, predictor, "np")

        X_node = DesignMatrix(X, name=name + "_X")
        K_node = Hyperparameter(K, name=name + "_K")
        a_node = Hyperparameter(a, name=name + "_a")
        b_node = Hyperparameter(b, name=name + "_b")

        rank_node = Hyperparameter(matrix_rank(K), name=name + "_rank")

        tau2_parameters = {"concentration": a_node, "scale": b_node}
        tau2_distribution = NodeDistribution("InverseGamma", **tau2_parameters)
        tau2_node = SmoothingParam(10000.0, tau2_distribution, name + "_tau2")

        beta = np.zeros(np.shape(X)[-1], np.float32)
        beta_parameters = {"tau2": tau2_node, "K": K_node, "rank": rank_node}
        beta_distribution = NodeDistribution("SmoothPrior", **beta_parameters)
        beta_node = RegressionCoef(beta, beta_distribution, name + "_beta")

        smooth_node = Smooth(X_node, beta_node, name=name)
        self._smooths[predictor].append(smooth_node)

        group = NPSmoothGroup(
            smooth=smooth_node,
            beta=beta_node,
            tau2=tau2_node,
            rank=rank_node,
            X=X_node,
            K=K_node,
            a=a_node,
            b=b_node,
        )

        group.name = name
        self.add_groups(group)
        return group

    def add_predictor(
        self, name: str, inverse_link: str | TFPBijectorClass
    ) -> DistRegBuilder:
        """
        Adds a predictor to the model builder.

        ## Parameters

        - `name`:
          The name of the parameter of the response distribution.

          Must match the name of the parameter of the TFP distribution.

        - `inverse_link`:
          The inverse link mapping the regression predictor to the parameter
          of the response distribution.

          Either a string identifying a TFP bijector, or alternatively,
          a TFP-compatible bijector class. If a class is provided instead of a string,
          the user needs to make sure it uses the right NumPy implementation.
        """

        if name not in self._smooths or not self._smooths[name]:
            msg = f"No smooths in {repr(self)} for predictor {repr(name)}. "
            raise RuntimeError(msg + self._howto)

        smooth_nodes = self._smooths[name]
        predictor_node = Predictor(name=name + "_pdt", *smooth_nodes)
        parameter_node = InverseLink(inverse_link, predictor_node, name=name)
        self._distributional_parameters[name] = parameter_node
        self.add_nodes(predictor_node, parameter_node)

        return self

    def add_response(
        self, response: Array, distribution: str | TFPDistributionClass
    ) -> DistRegBuilder:
        """
        Adds the response to the model builder.

        ## Parameters

        - `response`:
          The response vector or matrix.

        - `distribution`:
          The conditional distribution of the response variable.

          Either a string identifying a TFP distribution, or alternatively,
          a TFP-compatible distribution class. If a class is provided instead of
          a string, the user needs to make sure it uses the right NumPy implementation.
        """

        if not self._distributional_parameters:
            raise RuntimeError(f"No predictors in {repr(self)}. {self._howto}")

        response_distribution = NodeDistribution(
            distribution, **self._distributional_parameters
        )

        response_node = Response(response, response_distribution, "response")
        self._response = Option(response_node)
        self.add_nodes(response_node)

        return self


class CopRegBuilder(DistRegBuilder):
    """
    A model builder for copula regression models.

    Remember to add a predictor for the dependence parameter before adding the copula
    and building the model.

    ## Parameters

    - `model0`: The first marginal distributional regression model **builder**.
    - `model1`: The second marginal distributional regression model **builder**.
    - `copula`: The copula of the response variables.

      Either a string identifying a TFP distribution, or alternatively,
      a TFP-compatible distribution class. If a class is provided instead of a string,
      the user needs to make sure it uses the right NumPy implementation.
    """

    def __init__(self, model0: DistRegBuilder, model1: DistRegBuilder) -> None:
        super().__init__()

        arg0_is_mb = isinstance(model0, DistRegBuilder)
        arg1_is_mb = isinstance(model1, DistRegBuilder)

        if not arg0_is_mb or not arg1_is_mb:
            raise RuntimeError(f"Arguments of {repr(self)} must be model builders")

        self.groups = model0.groups + model1.groups
        self.nodes = model0.nodes + model1.nodes

        self._model0 = self._update_names(model0, "m0_")
        self._model1 = self._update_names(model1, "m1_")

    @staticmethod
    def _update_names(model: DistRegBuilder, prefix: str) -> DistRegBuilder:
        for node in model.all_nodes():
            node.name = prefix + node.name

        for group in model.groups:
            group.name = prefix + group.name

        return model

    def add_copula(self, copula: str | TFPDistributionClass) -> CopRegBuilder:
        pit0_node = PIT(self._model0.response, name="m0_pit")
        pit1_node = PIT(self._model1.response, name="m1_pit")

        copula_node = ColumnStack(
            pit0_node,
            pit1_node,
            distribution=NodeDistribution(copula, **self._distributional_parameters),
            name="copula",
        )

        self._response = Option(copula_node)
        self.add_nodes(pit0_node, pit1_node, copula_node)
        return self


def tau2_gibbs_kernel(group: NPSmoothGroup) -> GibbsKernel:
    """Builds a Gibbs kernel for a smoothing parameter with an inverse gamma prior."""

    position_key = group["tau2"].name

    def transition(prng_key, model_state):
        a_prior = model_state[group["a"].name].value
        rank = model_state[group["rank"].name].value

        a_gibbs = jnp.squeeze(a_prior + 0.5 * rank)

        b_prior = model_state[group["b"].name].value
        beta = model_state[group["beta"].name].value
        K = model_state[group["K"].name].value

        b_gibbs = jnp.squeeze(b_prior + 0.5 * (beta @ K @ beta))

        draw = b_gibbs / jax.random.gamma(prng_key, a_gibbs)

        return {position_key: draw}

    return GibbsKernel([position_key], transition)


def dist_reg_mcmc(model: Model, seed: int, num_chains: int) -> EngineBuilder:
    """
    Configures an `EngineBuilder` with a Metropolis-in-Gibbs MCMC algorithm with an
    `IWLSKernel` for the regression coefficients and a `GibbsKernel` for the smoothing
    parameters for a distributional regression model.

    ## Parameters

    - `model`: A model built with a `DistRegBuilder`.
    - `seed`: The PRNG seed for the engine builder.
    - `num_chains`: The number of chains to be sampled.
    """

    builder = EngineBuilder(seed, num_chains)

    builder.set_model(GooseModel(model))
    builder.set_initial_values(model.state)

    for group in model.groups.values():
        if not isinstance(group, SmoothGroup):
            next

        if isinstance(group, NPSmoothGroup):
            tau2_kernel = tau2_gibbs_kernel(group)
            builder.add_kernel(tau2_kernel)

        position_key = group["beta"].name
        beta_kernel = IWLSKernel([position_key])
        builder.add_kernel(beta_kernel)

    return builder
