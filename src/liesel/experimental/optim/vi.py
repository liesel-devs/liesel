from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal, Self

import jax
import jax.flatten_util
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as jb
import tensorflow_probability.substrates.jax.distributions as tfd

from ...docs import usedocs
from ...model import Dist, Model, Var
from ...model.logprob import FlatLogProb
from ...model.model import TemporaryModel
from .loss import LossMixin
from .split import PositionSplit
from .state import OptimCarry
from .types import ModelState, Position


class Elbo(LossMixin):
    def __init__(
        self,
        p: Model,
        q: Model,
        split: PositionSplit | None = None,
        nsamples: int = 10,
        nsamples_validate: int = 50,
        q_to_p: Callable[[Position], Position] = lambda x: x,
        scale: bool = False,
        vdist: VDist | CompositeVDist | None = None,
    ):
        self.p = p
        self.q = q
        self.split = split or PositionSplit.from_model(self.p)
        self.nsamples = nsamples
        self.nsamples_validate = nsamples_validate
        self._q_to_p = q_to_p
        self.scale = scale
        self.scalar = self.split.n_train if self.scale else 1.0
        self.vdist = vdist

    @classmethod
    def from_vdist(
        cls,
        vdist: VDist | CompositeVDist,
        split: PositionSplit,
        nsamples: int = 10,
        nsamples_validate: int = 50,
        scale: bool = False,
    ) -> Elbo:
        assert vdist.q is not None
        return cls(
            vdist.p,
            vdist.q,
            split=split,
            nsamples=nsamples,
            nsamples_validate=nsamples_validate,
            scale=scale,
            q_to_p=vdist.q_to_p,
            vdist=vdist,
        )

    @classmethod
    def mvn_diag(
        cls,
        p: Model,
        split: PositionSplit | None = None,
        nsamples: int = 10,
        nsamples_validate: int = 50,
        scale: bool = False,
    ) -> Self:
        vi_dist = VDist(list(p.parameters), p).mvn_diag().build()
        if vi_dist.q is None:
            raise ValueError

        return cls(
            vi_dist.p,
            vi_dist.q,
            split=split,
            nsamples=nsamples,
            nsamples_validate=nsamples_validate,
            scale=scale,
            q_to_p=vi_dist.q_to_p,
            vdist=vi_dist,
        )

    @classmethod
    def mvn_tril(
        cls,
        p: Model,
        split: PositionSplit | None = None,
        nsamples: int = 10,
        nsamples_validate: int = 50,
        scale: bool = False,
    ) -> Self:
        vi_dist = VDist(list(p.parameters), p).mvn_tril().build()
        if vi_dist.q is None:
            raise ValueError
        return cls(
            vi_dist.p,
            vi_dist.q,
            split=split,
            nsamples=nsamples,
            nsamples_validate=nsamples_validate,
            scale=scale,
            q_to_p=vi_dist.q_to_p,
            vdist=vi_dist,
        )

    @classmethod
    def mvn_blocked(
        cls,
        p: Model,
        split: PositionSplit | None = None,
        nsamples: int = 10,
        nsamples_validate: int = 50,
        scale: bool = False,
    ) -> Self:
        vi_dists = []
        for param_name in p.parameters:
            vi_dist = VDist([param_name], p).mvn_tril()
            vi_dists.append(vi_dist)

        vi_dist = CompositeVDist(*vi_dists).build()
        if vi_dist.q is None:
            raise ValueError
        return cls(
            vi_dist.p,
            vi_dist.q,
            split=split,
            nsamples=nsamples,
            nsamples_validate=nsamples_validate,
            scale=scale,
            q_to_p=vi_dist.q_to_p,
            vdist=vi_dist,
        )

    @property
    def model(self) -> Model:
        return self.p

    def position(self, position_keys: Sequence[str]) -> Position:
        return self.q.extract_position(position_keys)

    def q_to_p(self, q_position: Position) -> Position:
        """
        Maps a position from q to a position accepted by p.
        """
        return self._q_to_p(q_position)

    def evaluate(
        self,
        params: Position,
        key: jax.Array,
        p_state: ModelState,
        q_state: ModelState | None = None,
        obs: Position | None = None,
        scale_log_lik_p_by: float = 1.0,
        nsamples: int | None = None,
    ):
        obs = Position({}) if obs is None else obs
        q_state = self.q.state if q_state is None else q_state

        nsamples = nsamples if nsamples is not None else self.nsamples
        samples = self.q.sample((nsamples,), seed=key, newdata=params)

        @partial(jax.vmap)
        def log_prob_of_p(sample):
            p_state_new = self.p.update_state(self.q_to_p(sample) | obs, p_state)
            log_lik_p = p_state_new["_model_log_lik"].value
            log_prior_p = p_state_new["_model_log_prior"].value
            log_prob_p = scale_log_lik_p_by * log_lik_p + log_prior_p

            return log_prob_p

        @partial(jax.vmap)
        def log_prob_of_q(sample):
            q_state_new = self.q.update_state(sample | params, q_state)
            log_lik_q = q_state_new["_model_log_lik"].value
            log_prior_q = q_state_new["_model_log_prior"].value
            # Here, I subtract the prior from the likelihood, which may be somewhat
            # surprising.
            # The intention here is to allow priors in the variational distribution
            # to be used for regularization.
            # Since the Elbo is maximized, and the variational log prob is subtracted
            # form the Elbo, the variational log prob is minimized. Adding the prior
            # to the log lik of the variational dist would have the opposite of the
            # intended effect, since it would also be minimized. You could say we are
            # treating any priors in the variational model as parts of the main model
            return log_lik_q - log_prior_q

        elbo_samples = log_prob_of_p(samples) - log_prob_of_q(samples)

        return jnp.mean(elbo_samples)

    def loss_train_batched(self, params: Position, carry: OptimCarry) -> jax.Array:
        scale_log_lik_p_by = carry.batches.batch_share
        elbo = self.evaluate(
            Position(params | carry.fixed_position),
            carry.key,
            obs=Position(carry.batch),
            p_state=carry.model_state,
            q_state=self.q.state,
            scale_log_lik_p_by=scale_log_lik_p_by,
            nsamples=self.nsamples,
        )
        return -elbo / self.scalar

    def loss_train(self, params: Position, carry: OptimCarry) -> jax.Array:
        elbo = self.evaluate(
            Position(params | carry.fixed_position),
            carry.key,
            obs=Position(carry.batch),
            p_state=carry.model_state,
            q_state=self.q.state,
            nsamples=self.nsamples,
        )
        return -elbo / self.scalar

    def loss_validate(self, params: Position, carry: OptimCarry) -> jax.Array:
        elbo = self.evaluate(
            Position(params | carry.fixed_position),
            carry.key,
            obs=Position(self.split.validate),
            p_state=carry.model_state,
            q_state=self.q.state,
            scale_log_lik_p_by=self.scale_validate,
            nsamples=self.nsamples_validate,
        )
        return -elbo / self.scalar

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name}(nsamples={self.nsamples})"


class VDist:
    r"""
    Represents a variational distribution.

    Parameters
    -----------
    position_keys
        Sequence / list of strings, giving the names of the parameters in the model
        ``p`` whose posterior is to be approximated by this :class:`.VDist`.
    p
        The :class:`.Model` whose posterior is to be approximated by this
        :class:`.VDist`.

    See Also
    --------

    .VDist : Represents a single variational distribution.
    .CompositeVDist : Represents a composite variational distribution constructed
        from independent blocks, where each block is given by a :class:`.VDist`.

    Examples
    --------
    Take the model :math:`y \sim N(\mu, \sigma^2)`, where the posterior distribution
    of :math:`(\mu, \ln(\sigma))^\top` is modeled by a two independent Gaussian
    distributions, i.e. we define the following variational distributions:

    .. math::
        \mu & \sim N(\phi_1, \phi_2^2) \\
        \ln(\sigma) & \sim N(\phi_3, \phi_4^2)

    This variational distribution can be defined by a single
    :class:`.VDist` using the :meth:`.VDist.mvn_diag` method like this:

    >>> import jax.numpy as jnp
    >>> import liesel.model as lsl
    >>> import liesel.experimental.optim as opt
    >>> import tensorflow_probability.substrates.jax as tfp

    >>> loc = lsl.Var.new_param(jnp.array(0.0), name="mu")
    >>> scale = lsl.Var.new_param(1.0, name="sigma", bijector=tfp.bijectors.Exp())
    >>> y = lsl.Var.new_obs(
    ...     jnp.linspace(-2, 2, 50),
    ...     lsl.Dist(tfp.distributions.Normal, loc=loc, scale=scale),
    ...     name="y",
    ... )
    >>> p = lsl.Model([y])

    >>> vdist = opt.VDist(["mu", "h(sigma)"], p).mvn_diag().build()


    If the variational distribution is intended to capture correlation between the
    parameters, the correlated parameters should be governed jointly by a single
    :class:`.VDist`. For example, to use a multivariate Gaussian with a dense
    covariance matrix in this case, you can use :meth:`.VDist.mvn_tril`:

    >>> vdist = opt.VDist(["mu", "h(sigma)"], p).mvn_tril().build()

    This would lead to the variational distribution

    .. math::

        \begin{bmatrix}
        \mu \\ \ln(\sigma)
        \end{bmatrix}
        \sim N(\boldsymbol{\phi}, \boldsymbol{\Lambda}),

    where :math:`\boldsymbol{\Lambda}` is a :math:`2 \times 2` covariance matrix.

    ..rubric:: Custom variational distributions

    You can use any fully reparameterized tensorflow distribution of fitting
    event shape, wrapped in a :class:`.Dist`. For example, you can define the
    model with diagonal covariance matrix from above like this:

    >>> q_loc = lsl.Var.new_param(jnp.zeros(2), name="q_loc")
    >>> q_scale = lsl.Var.new_param(jnp.ones(2), name="q_scale")
    >>> dist = lsl.Dist(tfd.MultivariateNormalDiag, loc=q_loc, scale_diag=q_scale)
    >>> vdist = opt.VDist(["mu", "h(sigma)"], p).init(dist).build()

    .. note::
        If you use :meth:`.VDist.init`, make sure that the parameters of your
        variational distribution are :class:`.Var` objects with :attr:`.Var.parameter`
        set to ``True``.

    """

    def __init__(self, position_keys: Sequence[str], p: Model):
        self.position_keys = position_keys
        self._p = p

        pos = self.p.extract_position(self.position_keys)
        flat_pos, unflatten = jax.flatten_util.ravel_pytree(pos)
        self._unflatten = unflatten
        self._flat_pos = flat_pos
        self._flat_pos_name = "(" + "|".join(self.position_keys) + ")"
        self.var: Var | None = None
        self.q: Model | None = None

        self._to_float32 = p.to_float32

    @property
    def p(self) -> Model:
        """
        The :class:`.Model` whose posterior is approximated by this :class:`.VDist`.
        """
        return self._p

    def q_to_p(self, pos: Position) -> Position:
        """
        Turns flat q representation of parameters back into a position dictionary for p.

        In more words: Turns the flat representation of the parameters of p that are
        used as pseudo- observed variables in the variational approximation q into a
        position dictionary that fits the original representation in p.
        """
        return self._unflatten(pos[self._flat_pos_name])

    def p_to_q_array(self, pos: Position) -> jax.Array:
        """
        Turns the values of a position dictionary in p-representation into a flat
        q-representation array.

        Potentially useful for initializing with pre-fitted values.
        """
        if not list(pos) == self.position_keys:
            raise ValueError("list(pos) must be equal to self.position_keys.")

        return jax.flatten_util.ravel_pytree(pos)[0]

    @property
    def parameters(self) -> list[str]:
        """
        List of the names of the variational parameters in q.
        """
        if self.var is None:
            return []

        if self.var.model is not None:
            model = self.var.model.parental_submodel(self.var)
            params = list(model.parameters)
            return params

        with TemporaryModel(self.var) as model:
            params = list(model.parameters)

        return params

    def _validate(self, dist: Dist): ...

    def _reparameterize(self, dist: Dist): ...

    def init(self, dist: Dist) -> Self:
        """
        Initializies the pseudo-response variable with a variational distribution.

        Populates the :attr:`.var` attribute with an observed :class:`.Var`. This
        variable represents the flattened position governed by this :class:`.VDist`.

        Parameters
        ----------
        dist
            A :class:`.Dist`, representing the joint variational distribution for
            the flattened position governed by this :class:`.VDist`.

        Notes
        -----
        The docstring of :class:`.VDist` includes an example using this method.
        """
        self._validate(dist)
        self._reparameterize(dist)

        flat_pos_var = Var.new_obs(
            self._flat_pos,
            distribution=dist,
            name=self._flat_pos_name,
        )

        self.var = flat_pos_var
        return self

    def normal(
        self,
        loc: jax.typing.ArrayLike | None = None,
        scale: Literal["laplace"] | jax.typing.ArrayLike = 0.01,
        scale_bijector: type[jb.Bijector]
        | jb.Bijector
        | None
        | Literal["auto"] = "auto",
        *bijector_args,
        **bijector_kwargs,
    ) -> Self:
        if loc is None:
            loc_value = jnp.asarray(self._flat_pos)
        else:
            loc_value = jnp.asarray(loc)

        if scale == "laplace":
            info_matrix = -FlatLogProb(self.p, self.position_keys).hessian(loc_value)
            info_matrix += (
                1e-6
                * jnp.mean(jnp.diag(info_matrix))
                * jnp.eye(jnp.shape(info_matrix)[-1])
            )
            eigvals, eigvecs = jnp.linalg.eigh(info_matrix)

            # ensure eigenvalue positivity
            inv_eigvals_clipped = 1 / jnp.clip(eigvals, min=1e-5)
            cov_matrix = eigvecs @ (inv_eigvals_clipped[..., None, :] * eigvecs.T)
            scale = jnp.diag(cov_matrix)
        else:
            scale_arr = jnp.asarray(scale)
            if scale_arr.size == 1:
                scale_value = scale_arr * jnp.ones_like(self._flat_pos)
            else:
                scale_value = scale_arr

        loc_var = Var.new_param(loc_value, name=self._flat_pos_name + "_loc")
        scale_var = Var.new_param(scale_value, name=self._flat_pos_name + "_scale")

        dist = Dist(tfd.Normal, loc=loc_var, scale=scale_var)

        if scale_bijector is None:
            pass
        elif scale_bijector == "auto":
            dist.biject_parameters({"scale": "auto"})
        else:
            scale_var.transform(scale_bijector, *bijector_args, **bijector_kwargs)

        return self.init(dist)

    def mvn_diag(
        self,
        loc: jax.typing.ArrayLike | None = None,
        scale_diag: Literal["laplace"] | jax.typing.ArrayLike = 0.01,
        scale_diag_bijector: type[jb.Bijector]
        | jb.Bijector
        | None
        | Literal["auto"] = "auto",
        *bijector_args,
        **bijector_kwargs,
    ) -> Self:
        """
        Initializes a multivariate normal distribution with diagonal covariance matrix
        as the variational distribution q.

        Internally calls :meth:`.init`.

        Parameters
        ----------
        loc
            Initial value for the location of the variational distribution. If None,
            initialized to zero.
        scale_diag
            Initial value for the square roots of the diagonal elements of the
            variational distribution's covariance matrix. In other words: The marginal
            standard deviations/scales. If None, gets initialized to ``0.01``.

        Notes
        -----
        The docstring of :class:`.VDist` includes an example using this method.
        """
        if loc is None:
            loc_value = jnp.asarray(self._flat_pos)
        else:
            loc_value = loc

        if scale_diag == "laplace":
            info_matrix = -FlatLogProb(self.p, self.position_keys).hessian(loc_value)
            info_matrix += (
                1e-6
                * jnp.mean(jnp.diag(info_matrix))
                * jnp.eye(jnp.shape(info_matrix)[-1])
            )
            eigvals, eigvecs = jnp.linalg.eigh(info_matrix)

            # ensure eigenvalue positivity
            inv_eigvals_clipped = 1 / jnp.clip(eigvals, min=1e-5)
            cov_matrix = eigvecs @ (inv_eigvals_clipped[..., None, :] * eigvecs.T)
            scale_diag_value = jnp.diag(cov_matrix)
        else:
            scale_diag_arr = jnp.asarray(scale_diag)
            if scale_diag_arr.size == 1:
                scale_diag_value = scale_diag_arr * jnp.ones_like(self._flat_pos)
            else:
                scale_diag_value = scale_diag_arr

        loc_var = Var.new_param(loc_value, name=self._flat_pos_name + "_loc")
        scale_diag_var = Var.new_param(
            scale_diag_value, name=self._flat_pos_name + "_scale"
        )

        dist = Dist(tfd.MultivariateNormalDiag, loc=loc_var, scale_diag=scale_diag_var)

        if scale_diag_bijector is None:
            pass
        elif scale_diag_bijector == "auto":
            dist.biject_parameters({"scale_diag": "auto"})
        else:
            scale_diag_var.transform(
                scale_diag_bijector, *bijector_args, **bijector_kwargs
            )

        return self.init(dist)

    def mvn_tril(
        self,
        loc: jax.typing.ArrayLike | None = None,
        scale_tril: Literal["laplace"] | jax.typing.ArrayLike = 0.01,
        scale_tril_bijector: type[jb.Bijector]
        | jb.Bijector
        | None
        | Literal["auto"] = "auto",
        *bijector_args,
        **bijector_kwargs,
    ) -> Self:
        """
        Initializes a multivariate normal distribution with dense covariance matrix
        as the variational distribution q.

        The covariance matrix is parameterized by a lower Cholesky factor, where
        the covariance matrix is given by ``S = scale_tril @ scale_tril.T``.

        Parameters
        ----------
        loc
            Initial value for the location of the variational distribution. If None,
            initialized to zero.
        scale_tril
            Initial value for the lower Cholesky factor, must have non-zero diagonal
            elements.

        Notes
        -----
        The bijector :class:`tfp.bijectors.FillScaleTril` will be automatically applied
        to ``scale_tril`` to map its elements to the real line.

        The docstring of :class:`.VDist` includes an example using this method.
        """
        if loc is None:
            loc_value = jnp.asarray(self._flat_pos)
        else:
            loc_value = loc

        if scale_tril == "laplace":
            info_matrix = -FlatLogProb(self.p, self.position_keys).hessian(loc_value)
            info_matrix += (
                1e-6
                * jnp.mean(jnp.diag(info_matrix))
                * jnp.eye(jnp.shape(info_matrix)[-1])
            )
            eigvals, eigvecs = jnp.linalg.eigh(info_matrix)

            # ensure eigenvalue positivity
            inv_eigvals_clipped = 1 / jnp.clip(eigvals, min=1e-5)
            cov_matrix = eigvecs @ (inv_eigvals_clipped[..., None, :] * eigvecs.T)
            scale_tril_value = jnp.linalg.cholesky(cov_matrix)
        else:
            scale_tril_value_arr = jnp.asarray(scale_tril)
            if scale_tril_value_arr.size == 1:
                n = self._flat_pos.size
                scale_tril_value = scale_tril_value_arr * jnp.eye(n)
            else:
                scale_tril_value = scale_tril_value_arr

        loc_var = Var.new_param(loc_value, name=self._flat_pos_name + "_loc")
        scale_tril_var = Var.new_param(
            scale_tril_value, name=self._flat_pos_name + "_scale_tril"
        )

        dist = Dist(tfd.MultivariateNormalTriL, loc=loc_var, scale_tril=scale_tril_var)

        if scale_tril_bijector is None:
            pass
        elif scale_tril_bijector == "auto":
            dist.biject_parameters({"scale_tril": "auto"})
        else:
            scale_tril_var.transform(
                scale_tril_bijector, *bijector_args, **bijector_kwargs
            )

        return self.init(dist)

    def build(self) -> Self:
        """
        Builds the :class:`.Model` for the variational distribution, populates
        :attr:`.q`.

        Raises
        ------
        ValueError
            If :attr:`.var` has not been populated yet. See :meth:`.init` to populate
            :attr:`.var`.
        """
        if self.var is None:
            raise ValueError("The .var attribute must be set, but is currently None.")
        self.q = Model([self.var], to_float32=self.p.to_float32)
        return self

    def sample(
        self,
        seed: jax.Array,
        sample_shape: Sequence[int] = (),
        at_position: Position | None = None,
    ) -> Position:
        """
        Draws samples from the variational approximation to the posterior.

        Parameters
        ----------
        seed
            A jax key array, the seed for pseudo-random number generation.
        sample_shape
            Desired sample shape.
        at_position
            Position dictionary holding parameter values (position) of the variational
            distribution q to use for sampling. No leading batching dimensions are
            supported for this position.

        Returns
        -------
        A position dictionary holding samples for the parameters governed by this
        :class:`.VDist` in the representation of p.
        """
        if self.q is None:
            raise ValueError("The object has no model.")

        if at_position is not None:
            at_position = jax.tree.map(
                lambda x: jnp.expand_dims(x, (0, 1)), at_position
            )

        q_samples = self.q.sample(
            shape=sample_shape, seed=seed, posterior_samples=at_position
        )
        if at_position is not None:
            q_samples = jax.tree.map(
                lambda x: jnp.squeeze(
                    x, (len(sample_shape) + 0, len(sample_shape) + 1)
                ),
                q_samples,
            )

        return vmap_batched(Position(q_samples), self.q_to_p, batch_shape=sample_shape)

    def __repr__(self) -> str:
        name = type(self).__name__
        if self.var is not None:
            if self.var.dist_node is not None:
                dist = self.var.dist_node.distribution.__name__
            else:
                dist = None
        else:
            dist = None
        return f"{name}({self.position_keys}, dist={dist})"


def flatten_leading_batch(pytree, batch_ndim: int):
    """Flattens the first `batch_ndim` leading dims of every leaf into one dim."""

    def _f(x):
        x = jnp.asarray(x)
        b = int(jnp.prod(jnp.array(x.shape[:batch_ndim])))
        return x.reshape((b,) + x.shape[batch_ndim:])

    return jax.tree.map(_f, pytree)


def unflatten_leading_batch(pytree, batch_shape):
    """Inverse of flatten_leading_batch given the original batch_shape."""
    batch_shape = tuple(batch_shape)

    def _f(x):
        x = jnp.asarray(x)
        return x.reshape(batch_shape + x.shape[1:])

    return jax.tree.map(_f, pytree)


def vmap_batched(
    pos: Position, fun: Callable[[Position], Position], batch_shape: Sequence[int]
):
    batch_ndim = len(batch_shape)
    if batch_ndim == 0:
        return fun(pos)

    flat_pos = flatten_leading_batch(pos, batch_ndim=batch_ndim)
    flat_out_pos = jax.vmap(fun)(flat_pos)
    out_pos = unflatten_leading_batch(flat_out_pos, batch_shape)
    return out_pos


class CompositeVDist:
    r"""
    Composes several :class:`.VDist` instances into a :class:`.Model` that
    represents a variational distribution.

    Parameters
    ----------
    *vdists
        Variable numbers of :class:`.VDist` objects.

    See Also
    --------

    .VDist : Represents a single variational distribution.
    .CompositeVDist : Represents a composite variational distribution constructed
        from independent blocks, where each block is given by a :class:`.VDist`.

    Notes
    -----
    The :class:`.CompositeVDist` class assumes that each individual parameter in the
    underlying model p is governed by not more than one :class:`.VDist`.

    The :class:`.CompositeVDist` assumes independence between the parameters
    governed by its individual :class:`.VDist`s.

    Examples
    --------
    Take the model :math:`y \sim N(\mu, \sigma^2)`, where the posterior distribution
    of :math:`(\mu, \ln(\sigma))^\top` is modeled by a two independent Gaussian
    distributions, i.e. we define the following variational distributions:

    .. math::
        \mu & \sim N(\phi_1, \phi_2^2) \\
        \ln(\sigma) & \sim N(\phi_3, \phi_4^2)

    This variational distribution can be composed by defining separate :class:`.Vdist`
    objects for :math:`\mu` and :math:`\ln(\sigma)`, and combining them in a
    :class:`.CompositeVDist`:

    >>> import jax.numpy as jnp
    >>> import liesel.model as lsl
    >>> import liesel.experimental.optim as opt
    >>> import tensorflow_probability.substrates.jax as tfp

    >>> loc = lsl.Var.new_param(jnp.array(0.0), name="mu")
    >>> scale = lsl.Var.new_param(1.0, name="sigma", bijector=tfp.bijectors.Exp())
    >>> y = lsl.Var.new_obs(
    ...     jnp.linspace(-2, 2, 50),
    ...     lsl.Dist(tfp.distributions.Normal, loc=loc, scale=scale),
    ...     name="y",
    ... )
    >>> p = lsl.Model([y])

    >>> q1 = opt.VDist(["mu"], p).mvn_diag()
    >>> q2 = opt.VDist(["h(sigma)"], p).mvn_diag()
    >>> vdist = opt.CompositeVDist(q1, q2).build()

    In this case, the variational model is equivalent to defining a single
    :class:`.VDist` using the :meth:`.VDist.mvn_diag` method like this:

    >>> vdist = opt.VDist(["mu", "h(sigma)"], p).mvn_diag()

    If the variational distribution is intended to capture correlation between the
    parameters, the correlated parameters should be governed jointly by a single
    :class:`.VDist`. For example, to use a multivariate Gaussian with a dense
    covariance matrix in this case, you can use :meth:`.VDist.mvn_tril`:

    >>> vdist = opt.VDist(["mu", "h(sigma)"], p).mvn_tril()

    This would lead to the variational distribution

    .. math::

        \begin{bmatrix}
        \mu \\ \ln(\sigma)
        \end{bmatrix}
        \sim N(\boldsymbol{\phi}, \boldsymbol{\Lambda}),

    where :math:`\boldsymbol{\Lambda}` is a :math:`2 \times 2` covariance matrix.

    """

    def __init__(self, *vdists: VDist):
        self.vi_dists = vdists
        self.q: Model | None = None

    @property
    @usedocs(VDist.parameters)
    def parameters(self) -> list[str]:
        if self.q is None:
            return []
        return list(self.q.parameters)

    @property
    @usedocs(VDist.p)
    def p(self) -> Model:
        return self.vi_dists[0].p

    def _to_float32(self) -> bool:
        """
        Whether the values of the distributions' nodes will be converted from float64 to
        float32.
        """
        f32 = [dist.p.to_float32 for dist in self.vi_dists]
        if len(set(f32)) > 1:
            raise ValueError(
                "Some variational distributions seem to have to_float32=True, "
                "others have to_float32=False. The setting must be consistent."
            )
        return f32[0]

    @usedocs(VDist.build)
    def build(self) -> Self:
        vars_ = []
        for dist in self.vi_dists:
            if dist.var is None:
                raise ValueError(f".var attribute of {dist} must be set, but is None.")
            vars_.append(dist.var)
        q = Model(vars_, to_float32=self._to_float32())
        self.q = q
        return self

    @usedocs(VDist.q_to_p)
    def q_to_p(self, pos: Position) -> Position:
        positions = [dist.q_to_p(pos) for dist in self.vi_dists]
        combined = Position({k: v for d in positions for k, v in d.items()})
        return combined

    @usedocs(VDist.sample)
    def sample(
        self,
        seed: jax.Array,
        sample_shape: Sequence[int] = (),
        at_position: Position | None = None,
    ) -> Position:
        if self.q is None:
            raise ValueError("The object has no model.")

        if at_position is not None:
            at_position = jax.tree.map(
                lambda x: jnp.expand_dims(x, (0, 1)), at_position
            )

        q_samples = self.q.sample(
            shape=sample_shape, seed=seed, posterior_samples=at_position
        )
        if at_position is not None:
            q_samples = jax.tree.map(
                lambda x: jnp.squeeze(
                    x, (len(sample_shape) + 0, len(sample_shape) + 1)
                ),
                q_samples,
            )

        return vmap_batched(Position(q_samples), self.q_to_p, batch_shape=sample_shape)

    def __repr__(self) -> str:
        name = type(self).__name__
        if self.q is not None:
            built = "built"
        else:
            built = "not built"
        return f"{name}(n={len(self.vi_dists)}, {built})"
