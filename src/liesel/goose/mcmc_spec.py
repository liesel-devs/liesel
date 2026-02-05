from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, Protocol, assert_never

import tensorflow_probability.substrates.jax.distributions as tfd

from liesel.goose.engine import SamplingResults

from .builder import EngineBuilder
from .interface import LieselInterface
from .types import Array, JitterFunctions, Kernel, KeyArray

if TYPE_CHECKING:
    from liesel.model import Model, Var


logger = logging.getLogger(__name__)


@dataclass
class LieselMCMC:
    """
    Manages the setup of MCMC specifications for a Liesel model.

    Parameters
    ----------
    model
        The Liesel model object containing the variables and their inference \
        specifications.
    which
        A named inference configuration to use. If None, the default inference \
        attached to each variable is used.

    Examples
    --------

    .. rubric:: Liesel Workflow

    For this example, we import ``tensorflow_probability`` as follows:

    >>> import tensorflow_probability.substrates.jax.distributions as tfd

    First, we set up a minimal model:

    >>> mu = lsl.Var.new_param(0.0, name="mu", inference=gs.MCMCSpec(gs.NUTSKernel))
    >>> dist = lsl.Dist(tfd.Normal, loc=mu, scale=1.0)
    >>> y = lsl.Var.new_obs(jnp.array([1.0, 2.0, 3.0]), dist, name="y")
    >>> model = lsl.Model([y])

    Now we run MCMC:

    >>> results = gs.LieselMCMC(model).run_mcmc( # doctest: +SKIP
    ...     seed=1, num_chains=4, adaptation=250, posterior=100 # doctest: +SKIP
    ... ) # doctest: +SKIP

    The function returns a :class:`.SamplingResults` object.

    .. rubric:: More control

    For additional control, we initialize an :class:`.EngineBuilder` and continue
    from there.

    >>> builder = gs.LieselMCMC(model).get_engine_builder(seed=1, num_chains=4)
    >>> builder.add_adaptation(1000)
    >>> builder.add_posterior(1000)
    """

    model: Model
    which: str | None = None

    def get_spec(self, var: Var) -> MCMCSpec | None:
        """
        Retrieve the MCMC specification for a given variable.

        Parameters
        ----------
        var
            The model variable for which to get the MCMC specification.

        Returns
        -------
        The MCMC specification if available, otherwise None.

        Raises
        ------
        ValueError
            If the inference attached to the variable is not of type ``MCMCSpec``.
        """
        inference = var.get_inference(self.which)
        if inference is None:
            return inference

        if not isinstance(inference, MCMCSpec):
            raise ValueError(
                f"Attribute 'inference' of variable {var} is of type"
                f" {type(inference)}, but expected type '{MCMCSpec}'."
            )
        return inference

    def get_kernel_groups(self) -> dict[str, _KernelGroup]:
        """
        Collect and organize model variables into kernel groups.

        Returns
        -------
        A dictionary mapping group names or variable names to their corresponding \
        kernel group specifications.

        Raises
        ------
        ValueError
            If variables in the same kernel group have inconsistent kernels or kernel \
            arguments.
        """
        vars_ = self.model.vars
        kernel_groups: dict[str, _KernelGroup] = {}

        for name, var in vars_.items():
            inference = self.get_spec(var)

            if not inference:
                continue

            group_name = inference.kernel_group

            if group_name is None:
                kernel_groups[name] = _KernelGroup(
                    kernel=inference.kernel,
                    kwargs=inference.kernel_kwargs,
                    position_keys=[name],
                    order=inference.order,
                )

            elif group_name in kernel_groups:
                group = kernel_groups[group_name]
                same_kernel = group.kernel is inference.kernel
                if not same_kernel:
                    raise ValueError(
                        "Found incoherent kernel classes for kernel group"
                        f" {group_name}."
                    )

                if inference.kernel_kwargs is None:
                    pass
                elif not group.kwargs:
                    group.kwargs = inference.kernel_kwargs
                else:
                    if group.kwargs is not inference.kernel_kwargs:
                        raise ValueError(
                            "Found incoherent kernel keyword arguments for "
                            f"kernel group {group_name}. "
                            "When supplying kernel keyword arguments for multiple "
                            "inference objects, they all have to point to the "
                            "same object. "
                            "Alternatively, if you pass the kernel keyword arguments "
                            "to only "
                            "one inference object in the group, they will be applied "
                            "for the whole group."
                        )

                group.position_keys.append(name)

            else:
                kernel_groups[group_name] = _KernelGroup(
                    kernel=inference.kernel,
                    kwargs=inference.kernel_kwargs,
                    position_keys=[name],
                    order=inference.order,
                )

        kernel_groups = dict(
            sorted(kernel_groups.items(), key=lambda item: item[1].order)
        )

        return kernel_groups

    def get_kernel_list(self) -> list[Kernel]:
        """
        Construct the list of MCMC kernels from kernel groups.

        Returns
        -------
        A list of initialized kernel instances ready to be added to the MCMC engine.
        """
        kernel_groups = self.get_kernel_groups()
        kernel_list = [
            g.kernel(g.position_keys, **g.kwargs)  # type: ignore
            for g in kernel_groups.values()
        ]
        return kernel_list

    def get_jitter_functions(self) -> JitterFunctions:
        """
        Collect jitter functions for model variables that define a jitter distribution.

        Returns
        -------
        A dictionary mapping variable names to their jitter application functions.
        """
        jitter_functions: JitterFunctions = {}
        for name, var in self.model.vars.items():
            inference = self.get_spec(var)
            if inference is not None and inference.jitter_dist is not None:
                jitter_functions[name] = inference.apply_jitter

        return jitter_functions

    def get_engine_builder(
        self,
        seed: int,
        num_chains: int,
        apply_jitter: bool = True,
    ) -> EngineBuilder:
        """
        Create and configure an `EngineBuilder` for MCMC sampling.

        Parameters
        ----------
        seed
            Random seed for reproducibility.
        num_chains
            Number of MCMC chains.
        apply_jitter
            Whether to apply jitter to the initial states, by default True. Note that
            initial values for a variable will only jittered if the
            :class:`.MCMCSpec` for this variable was supplied with a ``jitter_dist``.

        Returns
        -------
        EngineBuilder
            A configured ``EngineBuilder`` instance.
        """
        self.validate_inference_specs()

        eb = EngineBuilder(seed=seed, num_chains=num_chains)
        eb.set_model(LieselInterface(self.model))
        eb.set_initial_values(self.model.state)

        for kernel in self.get_kernel_list():
            eb.add_kernel(kernel)

        if apply_jitter:
            eb.set_jitter_fns(self.get_jitter_functions())

        return eb

    def validate_inference_specs(self) -> None:
        """
        Logs a warning, if there are any parameters in the model that have no inference
        specification for MCMC.
        """
        no_inference: list[str] = []
        for name, var in self.model.parameters.items():
            if isinstance(var.inference, MCMCSpec):
                continue
            elif var.inference is None:
                no_inference.append(name)
            elif hasattr(var.inference, "values"):
                specs = list(var.inference.values())
                for spec in specs:
                    if isinstance(spec, MCMCSpec):
                        continue

                # triggers only if None of the specs in the inference dict was an
                # MCMCSpec
                no_inference.append(name)
            else:
                no_inference.append(name)

        for name in no_inference:
            logger.warning(
                f"No inference specification defined for {self.model.vars[name]}. "
                "If you do not add a kernel for this parameter manually to an "
                "EngineBuilder, it will not be"
                " sampled."
            )

    def run_for_epochs(
        self,
        *,
        seed: int,
        num_chains: int,
        adaptation: int,
        posterior: int,
        burnin: int = 0,
        adaptation_thinning: int = 1,
        burnin_thinning: int = 1,
        posterior_thinning: int = 1,
        apply_jitter: bool = True,
    ) -> SamplingResults:
        """
        Shorthand method for quickly running MCMC for a set number of epochs.

        Parameters
        ----------
        seed
            Random seed for reproducibility.
        num_chains
            Number of MCMC chains.
        adaptation, burnin, posterior
            Number of samples to be drawn in the respective epoch.
        adaptation_thinning, burnin_thinning, posterior_thinning
            Thinning to be applied in the respective epoch.
        apply_jitter
            Whether to apply jitter to the initial states, by default True. Note that
            initial values for a variable will only jittered if the
            :class:`.MCMCSpec` for this variable was supplied with a ``jitter_dist``.

        Warnings
        ---------

        This method is *only* appropriate, if your MCMC algorithm is fully specified via
        :class:`.MCMCSpec` objects in the :attr:`.Var.inference` attributes of the
        variables in your model.

        See Also
        ---------
        .get_engine_builder : Method to obtian an :class:`.EngineBuilder` from the
            LieselMCMC object. The :class:`.EngineBuilder` allows for more detailed
            custom configuration; for example you can add MCMC kernels via
            :meth:`.EngineBuilder.add_kernel`.

        Notes
        ------
        The method is euqivalent to the following code::

            eb = LieselMCMC(model).get_engine_builder(
                seed=seed, num_chains=num_chains, apply_jitter=apply_jitter
            )
            if adaptation > 0:
                eb.add_adaptation(adaptation, adaptation_thinning)
            if burnin > 0:
                eb.add_burnin(burnin, burnin_thinning)
            eb.add_posterior(posterior, posterior_thinning)
            engine = eb.build()
            engine.sample_all_epochs()
            engine.get_results()

        """
        eb = self.get_engine_builder(
            seed=seed, num_chains=num_chains, apply_jitter=apply_jitter
        )
        if adaptation > 0:
            eb.add_adaptation(adaptation, adaptation_thinning)
        if burnin > 0:
            eb.add_burnin(burnin, burnin_thinning)
        eb.add_posterior(posterior, posterior_thinning)
        engine = eb.build()
        engine.sample_all_epochs()
        return engine.get_results()


@dataclass
class _KernelGroup:
    kernel: Callable[..., Kernel]
    kwargs: dict[str, Any] = field(default_factory=dict)
    position_keys: list[str] = field(default_factory=list)
    order: int = 99


P = ParamSpec("P")


class KernelFactory(Protocol[P]):
    """Create a kernel instance based on the provided position keys and arguments."""

    def __call__(
        self, position_keys: list[str], *args: P.args, **kwargs: P.kwargs
    ) -> Kernel: ...


@dataclass
class MCMCSpec:
    """
    Specification for the MCMC kernel and optional jitter distribution associated with a
    model variable.

    Parameters
    ----------
    kernel
        A KernelFactory that returns a ``Kernel`` instance when provided with position
        keys and keyword arguments.
    kernel_kwargs
        Additional keyword arguments to be passed to the kernel callable.
    kernel_group
        Name of the kernel group this variable belongs to. Variables in the same group \
        must share the same kernel type and arguments.
    jitter_dist
        A TensorFlow Probability distribution used to apply random jitter to the \
        initial value of the variable.
    jitter_method
        The type of jitter to be applied. This can be one of the following: - `none`: No
        jitter is applied. - `additive`: Additive jitter is applied. - `multiplicative`:
        Multiplicative jitter is applied. - `replacement`: Value is replaced when jitter
        is applied.
    order
        If you want to change the order in which parameter blocks are sampled. Blocks
        will be ordered by default based on the topological order of the graph (from the
        bottom up; i.e. the kernels for sampling parameters closest to the graph's leaf
        nodes/responses come first), which is often a sensible default. After that,
        blocks will be ordered based on the integer provided here. The kernel with the
        smallest ``order`` integer will be used first.


    Examples
    --------

    .. rubric:: Liesel Workflow

    For this example, we import ``tensorflow_probability`` as follows:

    >>> import tensorflow_probability.substrates.jax.distributions as tfd

    First, we set up a minimal model:

    >>> mu = lsl.Var.new_param(0.0, name="mu", inference=gs.MCMCSpec(gs.NUTSKernel))
    >>> dist = lsl.Dist(tfd.Normal, loc=mu, scale=1.0)
    >>> y = lsl.Var.new_obs(jnp.array([1.0, 2.0, 3.0]), dist, name="y")
    >>> model = lsl.Model([y])

    Now we initialize the EngineBuilder and set the desired number of warmup and
    posterior samples:

    >>> builder = gs.LieselMCMC(model).get_engine_builder(seed=1, num_chains=4)
    >>> builder.add_adaptation(1000)
    >>> builder.add_posterior(1000)

    Finally, we build the engine:

    >>> engine = builder.build()

    """

    def __post_init__(self) -> None:
        if self.jitter_method not in self._JITTER_METHODS:
            raise ValueError(
                f"Invalid jitter method: {self.jitter_method}. "
                f"Expected one of {self._JITTER_METHODS}."
            )

    _JITTER_METHODS = ["additive", "multiplicative", "replacement"]

    kernel: KernelFactory
    kernel_kwargs: dict[str, Any] = field(default_factory=dict)
    kernel_group: str | None = None
    jitter_dist: tfd.Distribution | None = None
    jitter_method: Literal["additive", "multiplicative", "replacement"] = "additive"
    order: int = 99

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.kernel}, {self.kernel_group=})"

    def apply_jitter(self, seed: KeyArray, value: Array) -> Array:
        """
        Apply random jitter to a given value using the specified jitter distribution.

        If a jitter distribution is set, a random sample from the distribution is added
        to the original value. If no jitter distribution is set, the original value is
        returned unchanged.

        Parameters
        ----------
        seed
            A PRNG key used for random sampling.
        value
            The value to which jitter should be applied.

        Returns
        -------
        The jittered value with the same shape as the input.
        """
        if self.jitter_dist is None:
            return value

        # check compatibility of shapes
        if (
            self.jitter_dist.batch_shape + self.jitter_dist.event_shape != value.shape
        ) and (
            self.jitter_dist.batch_shape.rank + self.jitter_dist.event_shape.rank > 0
        ):
            raise ValueError(
                f"Jitter distribution shapes "
                f"(batch shape {self.jitter_dist.batch_shape} "
                f"and event shape {self.jitter_dist.event_shape}) "
                f"do not match variable shape {value.shape}."
            )
        sample_shape = (
            value.shape
            if self.jitter_dist.batch_shape + self.jitter_dist.event_shape == ()
            else ()
        )

        jitter = self.jitter_dist.sample(sample_shape=sample_shape, seed=seed)

        match self.jitter_method:
            case "additive":
                value = value + jitter
            case "multiplicative":
                value = value * jitter
            case "replacement":
                value = jitter
            case _:
                assert_never(self.jitter_method)

        return value
