"""
Iteratively weighted least squares (IWLS) sampler
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Self, get_args

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpla
from jax import grad, jacfwd
from jax.flatten_util import ravel_pytree

from .da import da_finalize, da_init, da_step
from .epoch import EpochState
from .iwls_utils import mvn_log_prob, mvn_sample, solve
from .kernel import (
    DefaultTransitionInfo,
    DefaultTuningInfo,
    ModelMixin,
    ReprMixin,
    TransitionMixin,
    TransitionOutcome,
    TuningOutcome,
    WarmupOutcome,
)
from .mh import mh_step
from .pytree import register_dataclass_as_pytree
from .types import Array, KeyArray, ModelState, Position


@register_dataclass_as_pytree
@dataclass
class IWLSKernelState:
    """
    A dataclass for the state of a :class:`.IWLSKernel`, implementing the
    :class:`.liesel.goose.da.DAKernelState` protocol.
    """

    step_size: float
    error_sum: float = field(init=False)
    log_avg_step_size: float = field(init=False)
    mu: float = field(init=False)

    def __post_init__(self):
        da_init(self)


IWLSTransitionInfo = DefaultTransitionInfo
IWLSTuningInfo = DefaultTuningInfo


CholInfoFallbackOptions = Literal["identity", "chol_of_modified_info"]


class IWLSKernel(
    ModelMixin, TransitionMixin[IWLSKernelState, IWLSTransitionInfo], ReprMixin
):
    """
    An IWLS kernel with dual averaging and an (optional) user-defined function for
    computing the Cholesky decomposition of the Fisher information matrix, implementing
    the :class:`.liesel.goose.types.Kernel` protocol.

    Parameters
    ----------
    position_keys
        Sequence of position keys (variable names) handled by this kernel.
    chol_info_fn
        A custom function that takes a model state and returns the Cholesky
        decomposition of the information matrix to produce the IWLS proposal. By
        default, this will be the Cholesky decomposition of the observed negative
        hessian at the current values, i.e. the current observed information.
    initial_step_size
        Value at which to start step size tuning.
    da_tune_step_step_size
        Whether to tune the step size using dual averaging.
    da_target_accept
        Target acceptance probability for dual averaging algorithm.
    da_gamma
        The adaptation regularization scale.
    da_kappa
        The adaptation relaxation exponent.
    da_t0
        The adaptation iteration offset.
    identifier
        An string acting as a unique identifier for this kernel.
    fallback_chol_info
        What do do if the Cholesky decomposition of the observed information matrix
        fails. If ``"identity"``, uses an identity matrix as the Cholesky factor. If
        ``"chol_of_modified_info"``, performs an eigendecomposition of the negative
        Hessian and clips the eigenvalues to ``1e-5``. This can be interpreted as
        replacing the observed negative Hessian with a very similar positive definite
        matrix. This is slow, because it performs an eigendecomposition and two cholesky
        factorizations. If ``None``, does nothing.

    Notes
    -----
    For more information on step size tuning via dual averaging,
    see :func:`.da_step` and :class:`.DAKernelState`.
    """

    error_book: ClassVar[dict[int, str]] = {
        0: "no errors",
        1: "indefinite information matrix (no fallback)",
        2: "indefinite information matrix (fallback to identity)",
        3: "indefinite information matrix (fallback to chol_of_modified_info)",
        90: "nan acceptance prob",
        91: "indefinite information matrix (no fallback) + nan acceptance prob",
        92: (
            "indefinite information matrix (fallback to identity) + nan acceptance prob"
        ),
        93: (
            "indefinite information matrix (fallback to chol_of_modified_info) "
            "+ nan acceptance prob"
        ),
    }
    """Dict of error codes and their meaning."""
    needs_history: ClassVar[bool] = False
    """Whether this kernel needs its history for tuning."""
    identifier: str = ""
    """Kernel identifier, set by :class:`~.goose.EngineBuilder`"""
    position_keys: tuple[str, ...]
    """Tuple of position keys handled by this kernel."""

    def __init__(
        self,
        position_keys: Sequence[str],
        chol_info_fn: Callable[[ModelState], Array] | None = None,
        initial_step_size: float = 0.01,
        da_tune_step_size=True,
        da_target_accept: float = 0.8,
        da_gamma: float = 0.05,
        da_kappa: float = 0.75,
        da_t0: int = 10,
        identifier: str = "",
        fallback_chol_info: CholInfoFallbackOptions | None = "identity",
    ):
        self._model = None
        self.position_keys = tuple(position_keys)
        self.chol_info_fn = chol_info_fn

        self.initial_step_size = initial_step_size

        self.da_tune_step_size = da_tune_step_size
        self.da_target_accept = da_target_accept
        self.da_gamma = da_gamma
        self.da_kappa = da_kappa
        self.da_t0 = da_t0
        self.identifier = identifier
        self.fallback_chol_info = fallback_chol_info

    @property
    def fallback_chol_info(self) -> CholInfoFallbackOptions | None:
        return self._fallback_chol_info

    @fallback_chol_info.setter
    def fallback_chol_info(self, value: CholInfoFallbackOptions | None):
        if value is not None and value not in get_args(CholInfoFallbackOptions):
            raise ValueError(
                f"Allowed values for fallback_chol_info: {CholInfoFallbackOptions} "
                "and 'None', "
                f"got {value}"
            )
        self._fallback_chol_info = value

    @classmethod
    def untuned(
        cls,
        position_keys: Sequence[str],
        chol_info_fn: Callable[[ModelState], Array] | None = None,
        fallback_chol_info: CholInfoFallbackOptions | None = "identity",
    ) -> Self:
        """
        Initializes an IWLS kernel that does not conduct step size tuning during warmup.
        Instead, the step size is fixed to 1.
        """
        kernel = cls(
            position_keys=position_keys,
            chol_info_fn=chol_info_fn,
            initial_step_size=1.0,
            da_tune_step_size=False,
            fallback_chol_info=fallback_chol_info,
        )
        return kernel

    def _flat_log_prob_fn(
        self, model_state: ModelState, unravel_fn: Callable[[Array], Position]
    ) -> Callable[[Array], float]:
        """
        Returns a callable which takes a flat position and returns the log-probability
        of the model.
        """

        def flat_log_prob_fn(flat_position: Array) -> float:
            position = unravel_fn(flat_position)
            new_model_state = self.model.update_state(position, model_state)
            return self.model.log_prob(new_model_state)

        return flat_log_prob_fn

    def _score(
        self, model_state: ModelState, flat_score_fn: Callable[[Array], Array]
    ) -> Array:
        """
        Calls :func:`.flat_score_fn` on a flat position.

        The flat position is extracted from the :attr:`.model_state`.
        """

        flat_position, _ = ravel_pytree(self.position(model_state))
        return flat_score_fn(flat_position)

    def _default_chol_info(
        self, model_state: ModelState, flat_hessian_fn: Callable[[Array], Array]
    ) -> Array:
        flat_position, _ = ravel_pytree(self.position(model_state))
        info_matrix = -flat_hessian_fn(flat_position)
        info_matrix += (
            1e-6
            * jnp.mean(jnp.diag(info_matrix))
            * jnp.eye(jnp.shape(flat_position)[-1])
        )
        return jnpla.cholesky(info_matrix)

    def _robust_default_chol_info(
        self, model_state: ModelState, flat_hessian_fn: Callable[[Array], Array]
    ) -> Array:
        flat_position, _ = ravel_pytree(self.position(model_state))
        info_matrix = -flat_hessian_fn(flat_position)
        info_matrix += (
            1e-6
            * jnp.mean(jnp.diag(info_matrix))
            * jnp.eye(jnp.shape(flat_position)[-1])
        )
        eigvals, eigvecs = jnpla.eigh(info_matrix)  # A = U Λ U^T
        eigvals_clipped = jnp.clip(eigvals, min=1e5)  # ensure positivity

        info_matrix = eigvecs @ (eigvals_clipped[..., None, :] * eigvecs.T)

        return jnpla.cholesky(info_matrix)

    def _chol_info(
        self, model_state: ModelState, flat_hessian_fn: Callable[[Array], Array]
    ) -> tuple[Array, int]:
        """
        Computes the Cholesky decomposition of the Fisher information matrix via
        :attr:`.flat_hessian_fn`.

        The flat position is extracted from the :attr:`.model_state`. If the user
        provided a :attr:`.chol_info_fn` when initializing the kernel, this function is
        called instead.
        """

        if self.chol_info_fn is None:
            flat_position, _ = ravel_pytree(self.position(model_state))
            info_matrix = -flat_hessian_fn(flat_position)
            info_matrix += (
                1e-6
                * jnp.mean(jnp.diag(info_matrix))
                * jnp.eye(jnp.shape(flat_position)[-1])
            )
            chol = jnpla.cholesky(info_matrix)
            return self._safe_chol(chol, info_matrix)

        chol = self.chol_info_fn(model_state)
        chol, error_code = self._safe_chol(chol, info_matrix=None)
        return chol, error_code

    def _safe_chol(self, chol, info_matrix) -> tuple[Array, int]:
        """
        Makes sure that the cholesky decomposition does not contain any nan values, if
        the argument ``fallback_chol_info`` was not set to "none".
        """

        def true_branch(info_matrix):
            if self.fallback_chol_info is None:
                return chol, 1

            elif self.fallback_chol_info == "identity":
                # sometimes all you need, always fast.
                return jnp.eye(chol.shape[-1]), 2

            elif self.fallback_chol_info == "chol_of_modified_info":
                if self.chol_info_fn is not None:
                    raise ValueError(
                        "When using a custom 'chol_info_fn', "
                        "fallback_chol_info='chol_of_modified_info' "
                        "is not supported."
                    )

                eigvals, eigvecs = jnpla.eigh(info_matrix)

                # ensure eigenvalue positivity
                eigvals_clipped = jnp.clip(eigvals, min=1e5)
                info_matrix = eigvecs @ (eigvals_clipped[..., None, :] * eigvecs.T)
                return jnpla.cholesky(info_matrix), 3

            else:
                raise ValueError(
                    "Allowed values for fallback_chol_info: "
                    f"{CholInfoFallbackOptions}, "
                    f"got {self.fallback_chol_info}"
                )

        def false_branch(info_matrix):
            return chol, 0

        chol, error_code = jax.lax.cond(
            jnp.any(jnp.isnan(chol)),
            true_branch,
            false_branch,
            info_matrix,
        )
        return chol, error_code

    def init_state(self, prng_key, model_state):
        """
        Initializes the kernel state.
        """

        return IWLSKernelState(self.initial_step_size)

    def _standard_transition(
        self,
        prng_key: KeyArray,
        kernel_state: IWLSKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[IWLSKernelState, IWLSTransitionInfo]:
        """
        Performs an MCMC transition *without* dual averaging.
        """

        key, subkey = jax.random.split(prng_key)
        step_size = kernel_state.step_size

        flat_pos, unravel_fn = ravel_pytree(self.position(model_state))
        flat_log_prob_fn = self._flat_log_prob_fn(model_state, unravel_fn)
        flat_score_fn = grad(flat_log_prob_fn)
        flat_hessian_fn = jacfwd(flat_score_fn)

        # proposal and forward probability

        score_pos = self._score(model_state, flat_score_fn)
        chol_info_pos, error_code_pos = self._chol_info(model_state, flat_hessian_fn)

        mu_pos = flat_pos + ((step_size**2) / 2) * solve(chol_info_pos, score_pos)
        flat_prop = mvn_sample(key, mu_pos, chol_info_pos / step_size)
        proposal = unravel_fn(flat_prop)

        fwd_log_prob = mvn_log_prob(flat_prop, mu_pos, chol_info_pos / step_size)

        # backward probability

        model_state_prop = self.model.update_state(proposal, model_state)

        score_prop = self._score(model_state_prop, flat_score_fn)
        chol_info_prop, _ = self._chol_info(model_state_prop, flat_hessian_fn)
        mu_prop = flat_prop + ((step_size**2) / 2) * solve(chol_info_prop, score_prop)
        bwd_log_prob = mvn_log_prob(flat_pos, mu_prop, chol_info_prop / step_size)

        correction = bwd_log_prob - fwd_log_prob

        info, model_state = mh_step(
            subkey, self.model, proposal, model_state, correction
        )
        info.error_code = info.error_code + error_code_pos

        return TransitionOutcome(info, kernel_state, model_state)

    def _adaptive_transition(
        self,
        prng_key: KeyArray,
        kernel_state: IWLSKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[IWLSKernelState, IWLSTransitionInfo]:
        """
        Performs an MCMC transition *with* dual averaging.
        """

        outcome = self._standard_transition(prng_key, kernel_state, model_state, epoch)

        if self.da_tune_step_size:
            da_step(
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
        prng_key: KeyArray,
        kernel_state: IWLSKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None = None,
    ) -> TuningOutcome[IWLSKernelState, IWLSTuningInfo]:
        """
        Currently does nothing.
        """

        info = IWLSTuningInfo(error_code=0, time=epoch.time)
        return TuningOutcome(info, kernel_state)

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: IWLSKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> IWLSKernelState:
        """
        Resets the state of the dual averaging algorithm.
        """

        da_init(kernel_state)
        return kernel_state

    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: IWLSKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> IWLSKernelState:
        """
        Sets the step size as found by the dual averaging algorithm.
        """

        da_finalize(kernel_state)
        return kernel_state

    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_state: IWLSKernelState,
        model_state: ModelState,
        tuning_history: IWLSTuningInfo | None,
    ) -> WarmupOutcome[IWLSKernelState]:
        """
        Currently does nothing.
        """

        return WarmupOutcome(error_code=0, kernel_state=kernel_state)
