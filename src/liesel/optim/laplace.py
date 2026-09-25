"""Dense Laplace integration of continuous model parameters."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from numbers import Real
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree

from ..goose.pytree import register_dataclass_as_pytree
from ..model import Model
from ._engine_utils import _validate_positive_int
from ._model_utils import continuous_coordinate_nodes
from .approximation import (
    LaplaceApproximation,
    _positive_definite,
    _prepare_approximation,
)
from .loss import LossMixin, SplitConfig, _validate_bool
from .split import PositionSplit
from .state import OptimCarry, OptimResult
from .types import Position


@register_dataclass_as_pytree
@dataclass
class LaplaceState:
    """Conditional mode and its unmodified dense precision factor.

    ``status`` is numeric for JAX: 0 uninitialized, 1 success, 2 iteration limit,
    3 failed backtracking, 4 non-finite evaluation, 5 invalid curvature.
    ``newton_decrement_squared`` is g.T @ solve(H, g); half of it is the
    convergence measure. Names and shapes describe the flattened latent order,
    sorted by parameter name. ``latent_precision_cholesky`` is lower triangular,
    with ``L @ L.T`` equal to the conditional negative-log-density Hessian.
    ``gradient_norm`` is the Euclidean norm of its latent gradient. ``n_iter``
    counts attempted Newton steps; ``n_resolution_steps`` counts accepted steps
    using the resolution safeguard. Only status 1 denotes a valid approximation.
    """

    outer_position: Position
    latent_position: Position
    latent_precision_cholesky: jax.Array
    n_iter: jax.Array
    gradient_norm: jax.Array
    newton_decrement_squared: jax.Array
    status: jax.Array
    n_resolution_steps: jax.Array
    latent_names: tuple[str, ...] = field(metadata={"static": True})
    latent_shapes: tuple[tuple[int, ...], ...] = field(metadata={"static": True})


def _evaluate(joint, theta, z):
    """Compute and retain one point's derivatives and true Cholesky factor."""

    def grad_with_value(x):
        value, gradient = jax.value_and_grad(joint, argnums=1)(theta, x)
        return gradient, (value, gradient)

    hessian, (value, gradient) = jax.jacfwd(grad_with_value, has_aux=True)(z)
    factor = jnp.linalg.cholesky(hessian)
    decrement = 0.5 * gradient @ jsp.linalg.cho_solve((factor, True), gradient)
    return {
        "z": z,
        "value": value,
        "gradient": gradient,
        "hessian": hessian,
        "factor": factor,
        "decrement": decrement,
    }


def _converged(point, tol):
    return (
        jnp.isfinite(point["value"])
        & jnp.isfinite(point["gradient"]).all()
        & jnp.isfinite(point["factor"]).all()
        & (point["decrement"] <= tol)
    )


def _finite(point):
    return (
        jnp.isfinite(point["value"])
        & jnp.isfinite(point["gradient"]).all()
        & jnp.isfinite(point["hessian"]).all()
    )


def _solve(joint, theta, seed, tol, max_iter):
    point = _evaluate(joint, theta, seed)
    state: dict[str, Any] = {
        "point": point,
        "n_iter": jnp.array(0),
        "status": jnp.where(_finite(point), 0, 4),
        "resolution_floor": jnp.zeros_like(point["value"]),
        "min_value": point["value"],
        "n_resolution_steps": jnp.array(0, dtype=jnp.int32),
    }

    def continuing(state):
        return (
            (state["n_iter"] < max_iter)
            & (state["status"] == 0)
            & ~_converged(state["point"], tol)
        )

    def step(state):
        point = state["point"]
        spd = jnp.isfinite(point["factor"]).all()

        def fallback_direction():
            # Scale each eigendirection by its curvature magnitude. This changes
            # only the search direction; the stored Hessian and factor stay exact.
            eigenvalues, eigenvectors = jnp.linalg.eigh(point["hessian"])
            curvature = jnp.abs(eigenvalues)
            floor = jnp.sqrt(jnp.finfo(seed.dtype).eps) * jnp.maximum(
                1.0, jnp.max(curvature)
            )
            return -eigenvectors @ (
                (eigenvectors.T @ point["gradient"]) / jnp.maximum(curvature, floor)
            )

        direction = jax.lax.cond(
            spd,
            lambda: -jsp.linalg.cho_solve((point["factor"], True), point["gradient"]),
            fallback_direction,
        )
        slope = point["gradient"] @ direction
        nominal_resolution = (
            8 * jnp.finfo(seed.dtype).eps * jnp.maximum(1.0, jnp.abs(point["value"]))
        )
        line = (
            jnp.array(0),
            jnp.array(1.0, seed.dtype),
            jnp.array(False),
            point,
            jnp.array(False),
            state["resolution_floor"],
        )

        def try_step(line):
            count, alpha, _, last, _, floor = line
            z = point["z"] + alpha * direction
            value = joint(theta, z)
            predicted = -alpha * slope - 0.5 * alpha**2 * (
                direction @ point["hessian"] @ direction
            )
            tiny_step = jnp.all(
                jnp.abs(z - point["z"])
                <= 32
                * jnp.finfo(seed.dtype).eps
                * jnp.maximum(1.0, jnp.abs(point["z"]))
            )
            observed_jump = jnp.abs(value - point["value"])
            # Cancellation can quantize a small returned objective. Learn its
            # first observed jump only over a few representable coordinates,
            # where the predicted change is below nominal scalar resolution.
            # A genuine equal-energy proposal cannot inflate this floor.
            learn_resolution = (
                spd
                & tiny_step
                & jnp.any(z != point["z"])
                & (predicted <= nominal_resolution)
                & (observed_jump > nominal_resolution)
                & jnp.isfinite(value)
                & (floor == 0)
            )
            floor = jnp.where(learn_resolution, 2 * observed_jump, floor)
            resolution = jnp.maximum(nominal_resolution, floor)
            plateau = (
                (value == point["value"]) & jnp.any(z != point["z"]) & (predicted > 0)
            )
            resolution_mode = spd & ((predicted <= resolution) | plateau)
            armijo = (
                jnp.isfinite(value)
                & (slope < 0)
                & (value <= point["value"] + 1e-4 * alpha * slope)
            )
            candidate = jax.lax.cond(
                armijo | resolution_mode,
                lambda: _evaluate(joint, theta, z),
                lambda: last,
            )
            decreased = jnp.isfinite(candidate["factor"]).all() & (
                (candidate["decrement"] < point["decrement"])
                | (candidate["decrement"] <= tol)
            )
            accepted = (
                _finite(candidate)
                & (candidate["value"] <= state["min_value"] + resolution)
                & jnp.where(resolution_mode, decreased, armijo)
            )
            # Keep the learned floor fixed, and bound total uphill drift by the
            # running minimum rather than allowing another increase each step.
            return count + 1, alpha / 2, accepted, candidate, resolution_mode, floor

        _, _, accepted, candidate, resolution_mode, floor = jax.lax.while_loop(
            lambda line: (line[0] < 24) & ~line[2], try_step, line
        )
        status = jnp.where(accepted, 0, jnp.where(~spd & (slope == 0), 5, 3))
        return state | {
            "point": candidate,
            "n_iter": state["n_iter"] + 1,
            "status": status,
            "resolution_floor": floor,
            "min_value": jnp.minimum(state["min_value"], candidate["value"]),
            "n_resolution_steps": state["n_resolution_steps"]
            + (accepted & resolution_mode).astype(jnp.int32),
        }

    state = jax.lax.while_loop(continuing, step, state)
    point = state["point"]
    status = jnp.where(
        state["status"] != 0,
        state["status"],
        jnp.where(
            _converged(point, tol),
            1,
            jnp.where(jnp.isfinite(point["factor"]).all(), 2, 5),
        ),
    )
    return point | {
        "n_iter": state["n_iter"],
        "n_resolution_steps": state["n_resolution_steps"],
        "status": status,
    }


def _laplace_value(point):
    value = (
        point["value"]
        + jnp.log(jnp.diag(point["factor"])).sum()
        - 0.5 * point["z"].size * jnp.log(2 * jnp.pi)
    )
    return jnp.where(point["status"] == 1, value, jnp.inf)


def _postfit_value(joint, theta, seed, tol, max_iter):
    """Differentiate the implicit mode to higher order only after fitting."""
    root = jax.lax.custom_root(
        lambda z: jax.grad(joint, argnums=1)(theta, z),
        seed,
        lambda equation, z0: _solve(joint, theta, z0, tol, max_iter)["z"],
        lambda linear, b: jnp.linalg.solve(jax.jacfwd(linear)(jnp.zeros_like(b)), b),
    )
    factor = jnp.linalg.cholesky(jax.hessian(joint, argnums=1)(theta, root))
    return (
        joint(theta, root)
        + jnp.log(jnp.diag(factor)).sum()
        - root.size * math.log(2 * math.pi) / 2
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def _fit_value(joint, theta, seed, tol, max_iter):
    point = _solve(joint, theta, seed, tol, max_iter)
    return _laplace_value(point), point


@jax.custom_jvp
def _first_order_only(theta):
    return theta


@_first_order_only.defjvp
def _reject_higher_derivatives(primals, tangents):
    raise TypeError(
        "Laplace fitting supports first derivatives only; "
        "use approximate_joint_posterior for posterior curvature."
    )


def _fit_forward(joint, theta, seed, tol, max_iter):
    theta = _first_order_only(theta)
    point = _solve(joint, theta, seed, tol, max_iter)
    return (_laplace_value(point), point), (theta, point)


def _fit_backward(joint, tol, max_iter, residual, cotangent):
    theta, point = residual
    z, factor = point["z"], point["factor"]
    hessian_cotangent = 0.5 * jsp.linalg.cho_solve(
        (factor, True), jnp.eye(z.size, dtype=z.dtype)
    )

    def value_and_hessian(t, x):
        return joint(t, x), jax.hessian(joint, argnums=1)(t, x)

    # Unused primal outputs are removed by JAX. Only third-derivative
    # contractions survive; the already computed dense factor is reused.
    _, pullback = jax.vjp(value_and_hessian, theta, z)
    partial_theta, partial_z = pullback(
        (jnp.ones_like(point["value"]), hessian_cotangent)
    )
    mode_cotangent = jsp.linalg.cho_solve((factor, True), partial_z)
    _, cross_pullback = jax.vjp(lambda t: jax.grad(joint, argnums=1)(t, z), theta)
    gradient = partial_theta - cross_pullback(mode_cotangent)[0]
    return cotangent[0] * gradient, None


_fit_value.defvjp(_fit_forward, _fit_backward)


class LaplaceLoss(LossMixin):
    """Integrate selected latent parameters out of the model's joint density.

    Use full-data batches and ``loss_monitor="train_full_data"``. The returned
    value is the unscaled negative log Laplace approximation, including priors,
    transformation Jacobians, and normalization constants.

    Parameters
    ----------
    model
        Model supplying the actual joint log density. Its parameter flags and
        current state are left unchanged.
    split
        Training observations to substitute into that density. If omitted, use
        the usual model-derived full-training split. Custom aggregate densities
        may need an explicit :class:`PositionSplit`.
    latent
        Nonempty sequence of writable continuous parameter names to integrate.
        Scalars, vectors, and matrices can be combined. Duplicate aliases, weak
        variables, discrete parameters, and overlap with outer parameters are
        rejected.
    warm_start
        Start each inner solve from the committed latent mode. Defaults to True.
        False always uses the model's original latent values. This is an inner
        warm start, distinct from resuming an optimizer checkpoint.
    inner_max_iter
        Maximum attempted Newton steps per evaluation. Defaults to 100.
    inner_tol
        Positive bound on half the squared Newton decrement. None selects 1e-6
        for float32 or 1e-10 for float64; global JAX precision is never changed.

    Notes
    -----
    The dense solver finds a local conditional mode. Non-concave densities may
    have several modes; warm and cold starts need not choose the same one.
    When curvature is indefinite, a descent direction uses the magnitudes of
    the Hessian eigenvalues with a numerical floor. This modifies only the search
    direction, never the Hessian used for convergence or the approximation.
    True positive-definite curvature and convergence are required for success.
    A failed solve returns an infinite value and an inspectable failed proposal.
    Every evaluation is pure: the engine commits state only at its full-training
    monitor point. No curvature history is retained for every epoch.

    The committed inner seed stays fixed throughout each outer line search. The
    full-training monitor repeats the selected-point solve without calculating
    another outer gradient. Stateful L-BFGS refreshes its gradient at each step
    because a new committed seed may change the loss evaluation.

    Reverse-mode fitting gradients reuse the final latent factor and include the
    latent dependence of its log determinant. Higher fitting derivatives raise;
    use ``approximate_joint_posterior`` after fitting for joint uncertainty.
    """

    default_position_keys: Sequence[str]

    def __init__(
        self,
        model: Model,
        split: SplitConfig | None = None,
        *,
        latent: Sequence[str],
        warm_start: bool = True,
        inner_max_iter: int = 100,
        inner_tol: float | None = None,
    ):
        _validate_bool(warm_start, "warm_start")
        _validate_positive_int(inner_max_iter, "inner_max_iter")
        if inner_tol is not None and (
            isinstance(inner_tol, bool)
            or not isinstance(inner_tol, Real)
            or not math.isfinite(inner_tol)
            or inner_tol <= 0
        ):
            raise ValueError("inner_tol must be a positive finite real number.")
        self.model = model
        self.split = (
            PositionSplit.from_model(model, multi_size="manager", shuffle=False)
            if split is None
            else split
        )
        self._data_nodes = {model._node_for_position_key(k) for k in self.split.train}
        if not latent:
            raise ValueError("latent must contain at least one continuous coordinate.")
        self._latent_nodes = self._coordinate_nodes(latent)
        self.latent_names = tuple(sorted(latent))
        self._initial_latent = model.extract_position(self.latent_names)
        self._seed, self._unravel_latent = ravel_pytree(self._initial_latent)
        self.latent_shapes = tuple(
            jnp.shape(self._initial_latent[k]) for k in self.latent_names
        )
        self.default_position_keys = tuple(
            k
            for k, var in model.parameters.items()
            if var.value_node not in self._latent_nodes
        )
        self.warm_start = warm_start
        self.inner_max_iter = inner_max_iter
        self.inner_tol = (
            (1e-10 if self._seed.dtype == jnp.float64 else 1e-6)
            if inner_tol is None
            else inner_tol
        )

    def position(self, position_keys: Sequence[str]) -> Position:
        """Validate and extract outer parameters, excluding latents and data."""
        nodes = self._coordinate_nodes(position_keys)
        if self._latent_nodes.intersection(nodes):
            raise ValueError("Outer and latent coordinates must not overlap.")
        return self.model.extract_position(position_keys)

    def _coordinate_nodes(self, keys: Sequence[str]) -> set:
        return continuous_coordinate_nodes(self.model, keys, self._data_nodes)

    def init_state(self, params: Position, carry: OptimCarry) -> LaplaceState:
        """Create an uninitialized latent guess with a stable state structure."""
        self.position(tuple(params))
        return LaplaceState(
            outer_position=params,
            latent_position=self._initial_latent,
            latent_precision_cholesky=jnp.zeros(
                (self._seed.size, self._seed.size), dtype=self._seed.dtype
            ),
            n_iter=jnp.array(0),
            gradient_norm=jnp.array(jnp.inf, dtype=self._seed.dtype),
            newton_decrement_squared=jnp.array(jnp.inf, dtype=self._seed.dtype),
            status=jnp.array(0),
            n_resolution_steps=jnp.array(0, dtype=jnp.int32),
            latent_names=self.latent_names,
            latent_shapes=self.latent_shapes,
        )

    def _evaluation_failure(self, value, proposed_state, gradient):
        return jnp.where(proposed_state.status == 1, 0, proposed_state.status)

    def _failure_message(self, reason, failed_state) -> str | None:
        inner = {
            2: "iteration limit",
            3: "backtracking failed",
            4: "non-finite evaluation",
            5: "invalid curvature",
        }.get(reason)
        return None if inner is None else f"Inner Laplace optimization failed: {inner}."

    def _checkpoint_configuration(self) -> tuple:
        return self.latent_names, self.warm_start, self.inner_max_iter, self.inner_tol

    def _joint(self, outer, latent, model_state, fixed_position=None):
        position = Position(outer | latent | self.split.train | (fixed_position or {}))
        return -self.model.update_state(position, model_state)["_model_log_prob"].value

    def approximate_joint_posterior(
        self,
        result: OptimResult,
        *,
        at: str = "min_monitor",
        raise_on_failure: bool = True,
        stationarity_tol: float = 1e-4,
    ) -> LaplaceApproximation:
        """Construct joint uncertainty from marginal and conditional curvature.

        The selected minimum-monitor or final state must belong to this model, loss, and
        training data. Omitted outer parameters retain their model values.
        This optional calculation uses higher implicit derivatives and does no
        work during ordinary fitting.

        Parameters
        ----------
        result
            Fit result with a matching successful conditional state.
        at
            ``"min_monitor"`` (default) selects the smallest monitoring loss;
            ``"final"`` selects the final committed position.
        raise_on_failure
            Raise RuntimeError on an invalid approximation (default True).
            False returns an inspectable object with ``valid=False``. Invalid
            argument values always raise ValueError.
        stationarity_tol
            Positive finite bound on half the outer squared Newton decrement.
            Defaults to 1e-4, on the unscaled marginal objective.

        Notes
        -----
        The approximation combines marginal outer curvature K with conditional
        latent curvature H and the mode sensitivity J. Its latent covariance is
        H^-1 + J K^-1 J.T, and its outer/latent cross covariance is K^-1 J.T.
        Factor solves construct the joint precision without allocating a dense
        covariance. True positive-definite curvature, inner convergence, and
        outer stationarity are required; no jitter or eigenvalue clipping is used.

        Diagnostics include ``reason``, ``inner_state``, ``outer_gradient``,
        ``outer_precision`` (K), ``latent_precision`` (H), and
        ``outer_newton_decrement_squared`` when evaluation reaches those values.
        Successful evaluation also retains ``mode_jacobian`` (J) and
        ``joint_precision``. Model graph and data equality are the caller's
        responsibility; parameter metadata and saved configuration are checked.
        """
        approximation = _prepare_approximation(
            result, at, raise_on_failure, stationarity_tol
        )
        if approximation.diagnostics["reason"] is not None:
            return approximation
        position = approximation.mean

        def failed(reason):
            return approximation._failed(reason, raise_on_failure)

        state = (
            result.loss_state_min_monitor
            if at == "min_monitor"
            else result.loss_state_final
        )
        if not isinstance(state, LaplaceState) or int(state.status) != 1:
            return failed(
                "The selected result has no successful matched Laplace state."
            )
        if (
            result.checkpoint is not None
            and result.checkpoint._loss_configuration
            != self._checkpoint_configuration()
        ):
            return failed("The result's loss configuration is incompatible.")

        def same_layout(values, reference):
            return set(values) == set(reference) and all(
                jnp.shape(values[name]) == jnp.shape(reference[name])
                and jnp.asarray(values[name]).dtype
                == jnp.asarray(reference[name]).dtype
                for name in reference
            )

        try:
            expected_outer = self.position(tuple(position))
        except ValueError as error:
            return failed(f"Incompatible outer coordinates: {error}")
        if (
            state.latent_names != self.latent_names
            or state.latent_shapes != self.latent_shapes
            or not same_layout(state.latent_position, self._initial_latent)
            or not same_layout(position, expected_outer)
            or not same_layout(state.outer_position, position)
            or not all(
                bool(jnp.array_equal(state.outer_position[k], position[k]))
                for k in position
            )
        ):
            return failed(
                "The selected Laplace state has incompatible coordinates or metadata."
            )
        names = tuple(sorted(position)) + state.latent_names
        approximation.mean = Position(position | state.latent_position)
        approximation.names = names
        approximation.shapes = tuple(
            jnp.shape(approximation.mean[name]) for name in names
        )
        theta, unravel_outer = ravel_pytree(position)
        seed, unravel_latent = ravel_pytree(state.latent_position)

        def joint(t, z):
            return self._joint(unravel_outer(t), unravel_latent(z), self.model.state)

        @jax.jit
        def derivatives(theta, seed):
            inner = _evaluate(joint, theta, seed)
            outer = _evaluate(
                lambda _, t: _postfit_value(
                    joint, t, seed, self.inner_tol, self.inner_max_iter
                ),
                None,
                theta,
            )
            cross = jax.jacfwd(jax.grad(joint, argnums=1), argnums=0)(theta, seed)
            return inner, outer, cross

        inner, outer, cross = derivatives(theta, seed)
        approximation.diagnostics.update(
            {
                "inner_state": state,
                "outer_gradient": outer["gradient"],
                "outer_precision": outer["hessian"],
                "latent_precision": inner["hessian"],
                "outer_newton_decrement_squared": 2 * outer["decrement"],
            }
        )
        if not bool(_positive_definite(inner["hessian"], inner["factor"])):
            return failed(
                "Invalid latent curvature: positive definiteness is required."
            )
        if not bool(_converged(inner, self.inner_tol)):
            return failed(
                "The selected conditional mode fails the inner convergence check."
            )
        if not bool(_finite(outer)):
            return failed("Non-finite marginal value or derivatives.")
        if not bool(_positive_definite(outer["hessian"], outer["factor"])):
            return failed("Invalid outer curvature: positive definiteness is required.")
        if not bool(outer["decrement"] <= stationarity_tol):
            return failed(
                "Outer stationarity check failed: half the squared Newton decrement "
                "exceeds stationarity_tol."
            )
        solved_cross = jsp.linalg.cho_solve((inner["factor"], True), cross)
        precision = jnp.block(
            [
                [outer["hessian"] + cross.T @ solved_cross, cross.T],
                [cross, inner["hessian"]],
            ]
        )
        factor = jnp.linalg.cholesky(precision)
        approximation.diagnostics["mode_jacobian"] = -solved_cross
        approximation.diagnostics["joint_precision"] = precision
        if not bool(_positive_definite(precision, factor)):
            return failed("Invalid joint curvature: positive definiteness is required.")
        approximation.precision_cholesky = factor
        approximation.valid = True
        return approximation

    def loss_train_batched(
        self, params: Position, carry: OptimCarry
    ) -> tuple[jax.Array, LaplaceState]:
        """Return the full-training Laplace loss and an uncommitted state."""
        theta, unravel_outer = ravel_pytree(params)
        seed = (
            ravel_pytree(carry.loss_state.latent_position)[0]
            if self.warm_start
            else self._seed
        )

        def joint(t, z):
            return self._joint(
                unravel_outer(t),
                self._unravel_latent(z),
                carry.model_state,
                carry.fixed_position,
            )

        value, point = _fit_value(
            joint, theta, seed, self.inner_tol, self.inner_max_iter
        )
        state = LaplaceState(
            outer_position=Position(carry.position | params),
            latent_position=Position(self._unravel_latent(point["z"])),
            latent_precision_cholesky=point["factor"],
            n_iter=point["n_iter"],
            gradient_norm=jnp.linalg.norm(point["gradient"]),
            newton_decrement_squared=2 * point["decrement"],
            status=point["status"],
            n_resolution_steps=point["n_resolution_steps"],
            latent_names=self.latent_names,
            latent_shapes=self.latent_shapes,
        )
        return value, state

    loss_train = loss_train_batched

    def loss_monitor(self, params: Position, carry: OptimCarry):
        raise ValueError("LaplaceLoss requires loss_monitor='train_full_data'.")
