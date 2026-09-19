"""
Iteratively weighted least squares (IWLS) sampler
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal, Self, get_args

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpla
from jax import grad, jacfwd
from jax.flatten_util import ravel_pytree

from .da import DualAvgState
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
from .types import Array, KernelState, KeyArray, ModelState, Position, Scalar


@register_dataclass_as_pytree
@dataclass
class IWLSKernelState:
    """
    Legacy adaptive IWLS state, retained for loading pre-0.6 sampling results.

    Also used by the Langevin kernels, implementing :class:`.DAKernelState`.
    """

    step_size: Scalar
    da_state: DualAvgState | None = None

    def __post_init__(self):
        if self.da_state is None:
            self.da_state = DualAvgState.from_step_size(self.step_size)


IWLSTransitionInfo = DefaultTransitionInfo
IWLSTuningInfo = DefaultTuningInfo


CholInfoFallbackOptions = Literal["identity", "chol_of_modified_info"]


class _GaussianKernel(
    ModelMixin, TransitionMixin[KernelState, IWLSTransitionInfo], ReprMixin
):
    """Shared Gaussian proposal, precision calculation, and MH correction."""

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
        *,
        identifier: str = "",
        fallback_chol_info: CholInfoFallbackOptions | None = "identity",
    ):
        self._model = None
        self.position_keys = tuple(position_keys)
        self.chol_info_fn = chol_info_fn
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

    def _flat_log_prob_fn(
        self, model_state: ModelState, unravel_fn: Callable[[Array], Position]
    ) -> Callable[[Array], Scalar]:
        """
        Returns a callable which takes a flat position and returns the log-probability
        of the model.
        """

        def flat_log_prob_fn(flat_position: Array) -> Scalar:
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

    def _chol_info(
        self, model_state: ModelState, flat_hessian_fn: Callable[[Array], Array]
    ) -> tuple[Array, int]:
        """
        Computes a Cholesky factor of the posterior precision approximation via
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
                eigvals_clipped = jnp.clip(eigvals, min=1e-5)
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

        return {}

    def _proposal_parameters(self, position, score, chol_info, kernel_state):
        return position + solve(chol_info, score), chol_info

    def _standard_transition(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[KernelState, IWLSTransitionInfo]:
        """
        Performs an MCMC transition *without* dual averaging.
        """

        key, subkey = jax.random.split(prng_key)

        flat_pos, unravel_fn = ravel_pytree(self.position(model_state))
        flat_log_prob_fn = self._flat_log_prob_fn(model_state, unravel_fn)
        flat_score_fn = grad(flat_log_prob_fn)
        flat_hessian_fn = jacfwd(flat_score_fn)

        # proposal and forward probability

        score_pos = self._score(model_state, flat_score_fn)
        chol_info_pos, error_code_pos = self._chol_info(model_state, flat_hessian_fn)

        mu_pos, chol_prop_pos = self._proposal_parameters(
            flat_pos, score_pos, chol_info_pos, kernel_state
        )
        flat_prop = mvn_sample(key, mu_pos, chol_prop_pos)
        proposal = unravel_fn(flat_prop)

        fwd_log_prob = mvn_log_prob(flat_prop, mu_pos, chol_prop_pos)

        # backward probability

        model_state_prop = self.model.update_state(proposal, model_state)

        score_prop = self._score(model_state_prop, flat_score_fn)
        chol_info_prop, _ = self._chol_info(model_state_prop, flat_hessian_fn)
        mu_prop, chol_prop_prop = self._proposal_parameters(
            flat_prop, score_prop, chol_info_prop, kernel_state
        )
        bwd_log_prob = mvn_log_prob(flat_pos, mu_prop, chol_prop_prop)

        correction = bwd_log_prob - fwd_log_prob

        info, model_state = mh_step(
            subkey, self.model, proposal, model_state, correction
        )
        info.error_code = info.error_code + error_code_pos

        return TransitionOutcome(info, kernel_state, model_state)

    def _adaptive_transition(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[KernelState, IWLSTransitionInfo]:
        """
        Performs the same transition during adaptation epochs.
        """

        return self._standard_transition(prng_key, kernel_state, model_state, epoch)

    def tune(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None = None,
    ) -> TuningOutcome[KernelState, IWLSTuningInfo]:
        """
        Currently does nothing.
        """

        info = IWLSTuningInfo(error_code=0, time=epoch.time)
        return TuningOutcome(info, kernel_state)

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> KernelState:
        """
        Leaves the kernel state unchanged.
        """

        return kernel_state

    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> KernelState:
        """
        Leaves the kernel state unchanged.
        """

        return kernel_state

    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        tuning_history: IWLSTuningInfo | None,
    ) -> WarmupOutcome[KernelState]:
        """
        Currently does nothing.
        """

        return WarmupOutcome(error_code=0, kernel_state=kernel_state)


class IWLSKernel(_GaussianKernel):
    r"""Gaussian IWLS proposals with Metropolis-Hastings correction.

    For block score g and precision P, the proposal has mean beta + P^{-1} g
    and covariance P^{-1}. This kernel has no step size or adaptation.

    Parameters
    ----------
    position_keys
        Names of the variables sampled together.
    chol_info_fn
        Optional function mapping a model state to the lower Cholesky factor of P.
        By default, use the negative block Hessian of the log posterior, adding
        ``1e-6 * mean(diag(P)) * I`` before factorization. A custom factor bypasses
        this jitter.
    identifier
        Unique identifier, normally assigned by :class:`.EngineBuilder`.
    fallback_chol_info
        On failed factorization, use ``"identity"`` (default), or
        ``"chol_of_modified_info"`` to clip eigenvalues to ``1e-5``. The latter
        is unavailable with a custom factor. With ``None``, reject the invalid
        proposal and report the error in transition diagnostics.

    Notes
    -----
    With the exact precision of a Gaussian full conditional, this is a Gibbs
    update up to numerical error. The default jitter perturbs that conditional;
    retain the MH correction for regularized and non-Gaussian proposals.

    Before 0.6 this class implemented simplified manifold MALA. Use
    :class:`.SMMALAKernel` with an explicit initial step size to preserve that
    behavior (``0.01`` was the old default).
    """

    @classmethod
    def untuned(
        cls,
        position_keys: Sequence[str],
        chol_info_fn: Callable[[ModelState], Array] | None = None,
        fallback_chol_info: CholInfoFallbackOptions | None = "identity",
    ) -> Self:
        """Compatibility constructor; all IWLS kernels are now untuned."""
        return cls(
            position_keys,
            chol_info_fn,
            fallback_chol_info=fallback_chol_info,
        )
