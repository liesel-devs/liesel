from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

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
                )

            elif group_name in kernel_groups:
                group = kernel_groups[group_name]
                same_kernel = group.kernel is inference.kernel
                if not same_kernel:
                    raise ValueError(
                        "Found incoherent kernel classes for kernel group"
                        f" {group_name}."
                    )

                same_kwargs = group.kwargs == inference.kernel_kwargs
                if not same_kwargs:
                    raise ValueError(
                        f"Found incoherent kwargs for kernel group {group_name}."
                    )
                group.position_keys.append(name)

            else:
                kernel_groups[group_name] = _KernelGroup(
                    kernel=inference.kernel,
                    kwargs=inference.kernel_kwargs,
                    position_keys=[name],
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
            Whether to apply jitter to the initial states, by default True.

        Returns
        -------
        EngineBuilder
            A configured ``EngineBuilder`` instance.
        """
        eb = EngineBuilder(seed=seed, num_chains=num_chains)
        eb.set_model(LieselInterface(self.model))
        eb.set_initial_values(self.model.state)

        for kernel in self.get_kernel_list():
            eb.add_kernel(kernel)

        if apply_jitter:
            eb.set_jitter_fns(self.get_jitter_functions())

        return eb


@dataclass
class _KernelGroup:
    kernel: Callable[..., Kernel]
    kwargs: dict[str, Any] = field(default_factory=dict)
    position_keys: list[str] = field(default_factory=list)


@dataclass
class MCMCSpec:
    """
    Specification for the MCMC kernel and optional jitter distribution associated with a
    model variable.

    Parameters
    ----------
    kernel
        A callable that returns a ``Kernel`` instance when provided with position keys \
        and keyword arguments.
    kernel_kwargs
        Additional keyword arguments to be passed to the kernel callable.
    kernel_group
        Name of the kernel group this variable belongs to. Variables in the same group \
        must share the same kernel type and arguments.
    jitter_dist
        A TensorFlow Probability distribution used to apply random jitter to the \
        initial value of the variable.
    """

    kernel: Callable[..., Kernel]
    kernel_kwargs: dict[str, Any] = field(default_factory=dict)
    kernel_group: str | None = None
    jitter_dist: tfd.Distribution | None = None

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

        original_shape = jnp.asarray(value).shape
        value = jnp.atleast_1d(value)

        jitter = self.jitter_dist.sample(sample_shape=jnp.shape(value), seed=seed)

        return jnp.reshape(value + jitter, original_shape)
