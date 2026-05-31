"""Variational inference losses and variational distribution builders.

This module provides the pieces used by :class:`.LieselVI` and by custom
variational workflows:

``Elbo``
    A :class:`.LossMixin` implementation that evaluates a Monte Carlo estimate of
    the evidence lower bound (ELBO).
``VDist``
    A builder for one variational block. The block governs one or more parameters
    from a target Liesel model and turns them into a flattened observed variable in
    a variational model.
``CompositeVDist``
    A builder that combines several independent ``VDist`` blocks into one
    variational model.

Examples
--------
Build a diagonal multivariate normal variational distribution and wrap it in an
ELBO loss:

>>> import jax.numpy as jnp
>>> import liesel.model as lsl
>>> import liesel.experimental.optim as opt
>>> import tensorflow_probability.substrates.jax as tfp
>>> loc = lsl.Var.new_param(jnp.array(0.0), name="mu")
>>> y = lsl.Var.new_obs(
...     jnp.array([0.1, -0.2]),
...     lsl.Dist(tfp.distributions.Normal, loc=loc, scale=1.0),
...     name="y",
... )
>>> p = lsl.Model([y])
>>> vdist = opt.VDist(["mu"], p).mvn_diag().build()
>>> elbo = opt.Elbo.from_vdist(vdist, opt.PositionSplit.from_model(p), nsamples=2)
>>> repr(elbo)
'Elbo(nsamples=2)'
>>> elbo.position(vdist.parameters).keys()
dict_keys(['(mu)_loc', 'h((mu)_scale)'])
"""

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
from .split import PositionSplit, PositionSplitManager
from .state import OptimCarry
from .types import ModelState, Position

SplitConfig = PositionSplit | PositionSplitManager


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, but got {value!r}.")


def _is_laplace_init(value, name: str) -> bool:
    if not isinstance(value, str):
        return False

    if value != "laplace":
        raise ValueError(f"{name} must be 'laplace' or an array-like value.")

    return True


def _laplace_covariance(model: Model, position_keys: Sequence[str], loc: jax.Array):
    info_matrix = -FlatLogProb(model, position_keys).hessian(loc)
    diag = jnp.diag(info_matrix)
    ridge = 1e-6 * jnp.maximum(jnp.mean(jnp.abs(diag)), 1.0)
    info_matrix += ridge * jnp.eye(jnp.shape(info_matrix)[-1])

    eigvals, eigvecs = jnp.linalg.eigh(info_matrix)
    inv_eigvals_clipped = 1 / jnp.clip(eigvals, min=1e-5)
    cov_matrix = eigvecs @ (inv_eigvals_clipped[..., None, :] * eigvecs.T)
    return cov_matrix


def _distribution_sample_shape(distribution, value_shape: tuple[int, ...]):
    event_shape = tuple(distribution.event_shape)
    batch_shape = tuple(distribution.batch_shape)
    distribution_shape = batch_shape + event_shape
    n_distribution_dims = len(distribution_shape)

    if len(value_shape) < n_distribution_dims:
        raise ValueError(
            "The variational distribution's event and batch shape "
            f"{distribution_shape} is incompatible with flattened position shape "
            f"{value_shape}."
        )

    if n_distribution_dims and value_shape[-n_distribution_dims:] != distribution_shape:
        raise ValueError(
            "The variational distribution's event and batch shape "
            f"{distribution_shape} does not match the trailing dimensions of the "
            f"flattened position shape {value_shape}."
        )

    return value_shape[: len(value_shape) - n_distribution_dims]


class Elbo(LossMixin):
    """
    Monte Carlo evidence lower bound loss.

    ``Elbo`` connects a target model ``p`` and a variational model ``q``. The
    variational model must be able to sample parameter positions, and ``q_to_p`` must
    map those sampled positions into the parameter names expected by ``p``. The loss
    is minimized by the experimental optimization engine, so the public loss methods
    return the negative ELBO.

    Parameters
    ----------
    p
        Target Liesel model whose posterior is approximated.
    q
        Variational Liesel model. Its observed variables are sampled as
        reparameterized variational draws.
    split
        Train/validation/test split for observed data in ``p``. If omitted,
        :meth:`.PositionSplit.from_model` is used.
    nsamples
        Number of Monte Carlo samples used for training losses.
    nsamples_validate
        Number of Monte Carlo samples used for validation losses.
    q_to_p
        Function mapping a sampled position from ``q`` to a position accepted by
        ``p``. Builders such as :class:`VDist` provide this mapping automatically.
    scale
        If ``True``, divide losses by ``split.n_train``. This requires a common
        scalar training sample size and raises for multi-branch splits with unequal
        train sizes.
    vdist
        Optional variational distribution builder that created ``q``. Stored for
        introspection and convenience; it is not required for evaluating the loss.
    regularize_q_prior
        Whether priors in ``q`` should be added to the ELBO as regularization terms.
        The default preserves the historical behavior of this class. Set to
        ``False`` to subtract only the variational likelihood term.

    Attributes
    ----------
    p
        Target model.
    q
        Variational model.
    split
        Data split used for training and validation observations.
    scalar
        Normalization constant used when ``scale=True``.

    Examples
    --------
    The convenience constructor :meth:`mvn_diag` builds a diagonal multivariate
    normal variational distribution over all parameters of ``p``:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import liesel.model as lsl
    >>> import liesel.experimental.optim as opt
    >>> import tensorflow_probability.substrates.jax as tfp
    >>> loc = lsl.Var.new_param(jnp.array(0.0), name="mu")
    >>> y = lsl.Var.new_obs(
    ...     jnp.array([0.1, -0.2]),
    ...     lsl.Dist(tfp.distributions.Normal, loc=loc, scale=1.0),
    ...     name="y",
    ... )
    >>> p = lsl.Model([y])
    >>> elbo = opt.Elbo.mvn_diag(p, nsamples=2, nsamples_validate=3)
    >>> repr(elbo)
    'Elbo(nsamples=2)'
    >>> sorted(elbo.position(elbo.vdist.parameters))
    ['(mu)_loc', 'h((mu)_scale)']
    >>> value = elbo.evaluate(
    ...     elbo.position(elbo.vdist.parameters),
    ...     jax.random.key(1),
    ...     p.state,
    ...     nsamples=2,
    ... )
    >>> value.shape
    ()
    """

    def __init__(
        self,
        p: Model,
        q: Model,
        split: SplitConfig | None = None,
        nsamples: int = 10,
        nsamples_validate: int = 50,
        q_to_p: Callable[[Position], Position] = lambda x: x,
        scale: bool = False,
        vdist: VDist | CompositeVDist | None = None,
        regularize_q_prior: bool = True,
    ):
        _validate_positive_int(nsamples, "nsamples")
        _validate_positive_int(nsamples_validate, "nsamples_validate")
        self.p = p
        self.q = q
        self.split = split or PositionSplit.from_model(self.p)
        self.nsamples = nsamples
        self.nsamples_validate = nsamples_validate
        self._q_to_p = q_to_p
        self.scale = scale
        try:
            self.scalar = self.split.n_train if self.scale else 1.0
        except ValueError as error:
            raise ValueError(
                "scale=True requires a common training sample size. For "
                "multi-branch splits with unequal training sizes, use scale=False "
                "or a custom normalized ELBO."
            ) from error
        self.vdist = vdist
        self.regularize_q_prior = regularize_q_prior

    @classmethod
    def from_vdist(
        cls,
        vdist: VDist | CompositeVDist,
        split: SplitConfig,
        nsamples: int = 10,
        nsamples_validate: int = 50,
        scale: bool = False,
        regularize_q_prior: bool = True,
    ) -> Elbo:
        """
        Constructs an ELBO loss from a built variational distribution.

        Parameters
        ----------
        vdist
            Built :class:`VDist` or :class:`CompositeVDist`. Its :attr:`q` model must
            already be available, usually by calling :meth:`VDist.build` or
            :meth:`CompositeVDist.build`.
        split
            Data split for the target model.
        nsamples
            Number of Monte Carlo samples used for training losses.
        nsamples_validate
            Number of Monte Carlo samples used for validation losses.
        scale
            Whether to normalize losses by the common training sample size.
        regularize_q_prior
            Whether priors in ``vdist.q`` should be added to the ELBO as
            regularization terms.

        Returns
        -------
        Elbo
            Loss object using ``vdist.q`` and ``vdist.q_to_p``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> import liesel.experimental.optim as opt
        >>> import tensorflow_probability.substrates.jax as tfp
        >>> loc = lsl.Var.new_param(jnp.array(0.0), name="mu")
        >>> y = lsl.Var.new_obs(
        ...     jnp.array([0.0, 1.0]),
        ...     lsl.Dist(tfp.distributions.Normal, loc=loc, scale=1.0),
        ...     name="y",
        ... )
        >>> p = lsl.Model([y])
        >>> split = opt.PositionSplit.from_model(p)
        >>> vdist = opt.VDist(["mu"], p).mvn_diag().build()
        >>> opt.Elbo.from_vdist(vdist, split).vdist is vdist
        True
        """
        if vdist.q is None:
            raise ValueError(
                "vdist.q is None. Call .build() on the variational distribution "
                "before constructing an Elbo."
            )

        return cls(
            vdist.p,
            vdist.q,
            split=split,
            nsamples=nsamples,
            nsamples_validate=nsamples_validate,
            scale=scale,
            q_to_p=vdist.q_to_p,
            vdist=vdist,
            regularize_q_prior=regularize_q_prior,
        )

    @classmethod
    def mvn_diag(
        cls,
        p: Model,
        split: SplitConfig | None = None,
        nsamples: int = 10,
        nsamples_validate: int = 50,
        scale: bool = False,
        regularize_q_prior: bool = True,
    ) -> Self:
        """
        Builds a diagonal multivariate normal ELBO over all parameters of ``p``.

        Each unconstrained parameter in ``p.parameters`` is included in one joint
        :class:`VDist` with a diagonal covariance matrix. Use :meth:`mvn_tril` when
        the variational approximation should model posterior correlations.

        Parameters
        ----------
        p
            Target model.
        split
            Optional data split. If omitted, :meth:`PositionSplit.from_model` is
            used by :class:`Elbo`.
        nsamples
            Number of Monte Carlo samples used for training losses.
        nsamples_validate
            Number of Monte Carlo samples used for validation losses.
        scale
            Whether to normalize losses by the common training sample size.
        regularize_q_prior
            Whether priors in the variational model should be added to the ELBO as
            regularization terms.

        Returns
        -------
        Elbo
            ELBO loss with a built diagonal multivariate normal variational
            distribution.
        """
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
            regularize_q_prior=regularize_q_prior,
        )

    @classmethod
    def mvn_tril(
        cls,
        p: Model,
        split: SplitConfig | None = None,
        nsamples: int = 10,
        nsamples_validate: int = 50,
        scale: bool = False,
        regularize_q_prior: bool = True,
    ) -> Self:
        """
        Builds a dense multivariate normal ELBO over all parameters of ``p``.

        The variational distribution uses a single :class:`VDist` block with a full
        lower-triangular scale matrix. This can represent correlations among all
        optimized parameters.

        Parameters
        ----------
        p
            Target model.
        split
            Optional data split. If omitted, :meth:`PositionSplit.from_model` is
            used by :class:`Elbo`.
        nsamples
            Number of Monte Carlo samples used for training losses.
        nsamples_validate
            Number of Monte Carlo samples used for validation losses.
        scale
            Whether to normalize losses by the common training sample size.
        regularize_q_prior
            Whether priors in the variational model should be added to the ELBO as
            regularization terms.

        Returns
        -------
        Elbo
            ELBO loss with a built dense multivariate normal variational
            distribution.
        """
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
            regularize_q_prior=regularize_q_prior,
        )

    @classmethod
    def mvn_blocked(
        cls,
        p: Model,
        split: SplitConfig | None = None,
        nsamples: int = 10,
        nsamples_validate: int = 50,
        scale: bool = False,
        regularize_q_prior: bool = True,
    ) -> Self:
        """
        Builds an ELBO with one dense normal variational block per parameter.

        The resulting :class:`CompositeVDist` treats parameter blocks as independent,
        but each individual parameter can have an internal dense covariance structure
        when it is vector-valued.

        Parameters
        ----------
        p
            Target model.
        split
            Optional data split. If omitted, :meth:`PositionSplit.from_model` is
            used by :class:`Elbo`.
        nsamples
            Number of Monte Carlo samples used for training losses.
        nsamples_validate
            Number of Monte Carlo samples used for validation losses.
        scale
            Whether to normalize losses by the common training sample size.
        regularize_q_prior
            Whether priors in the variational model should be added to the ELBO as
            regularization terms.

        Returns
        -------
        Elbo
            ELBO loss with a built blocked variational distribution.
        """
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
            regularize_q_prior=regularize_q_prior,
        )

    @property
    def model(self) -> Model:
        """Target model evaluated by this loss."""
        return self.p

    def position(self, position_keys: Sequence[str]) -> Position:
        """
        Extracts an initial optimizer position from the variational model.

        Parameters
        ----------
        position_keys
            Names of variational parameters in ``q``.

        Returns
        -------
        Position
            Current ``q`` position restricted to ``position_keys``.
        """
        return self.q.extract_position(position_keys)

    def q_to_p(self, q_position: Position) -> Position:
        """
        Maps a variational position to a target-model position.

        Parameters
        ----------
        q_position
            Sampled position from ``q``.

        Returns
        -------
        Position
            Position accepted by ``p``.
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
        split: SplitConfig | None = None,
        batches=None,
        nsamples: int | None = None,
    ) -> jax.Array:
        """
        Estimates the ELBO at a variational parameter position.

        The method draws ``nsamples`` samples from ``q`` at ``params`` and computes
        ``E_q[log p(theta, y) - log q(theta)]``. Mini-batch training passes
        ``batches`` so observed log-likelihood terms can be scaled by the active
        batch configuration. Validation passes ``split`` so validation likelihoods
        can be scaled branch by branch.

        Parameters
        ----------
        params
            Variational parameter position at which to evaluate ``q``.
        key
            JAX pseudo-random key used for sampling from ``q``.
        p_state
            Current state of the target model ``p``.
        q_state
            Optional state of the variational model ``q``. Defaults to
            ``self.q.state``.
        obs
            Observed data position used to update ``p`` before evaluating
            likelihood terms.
        scale_log_lik_p_by
            Scalar multiplier for ``p``'s log-likelihood when neither ``split`` nor
            ``batches`` is supplied.
        split
            Optional split object used to compute validation-scaled log likelihoods.
        batches
            Optional batch object used to compute mini-batch-scaled log likelihoods.
        nsamples
            Number of Monte Carlo samples. Defaults to ``self.nsamples``.

        Returns
        -------
        jax.Array
            Scalar Monte Carlo estimate of the ELBO.
        """
        obs = Position({}) if obs is None else obs
        q_state = self.q.state if q_state is None else q_state

        nsamples = nsamples if nsamples is not None else self.nsamples
        samples = self.q.sample((nsamples,), seed=key, newdata=params)

        @partial(jax.vmap)
        def log_prob_of_p(sample):
            p_state_new = self.p.update_state(self.q_to_p(sample) | obs, p_state)
            if batches is None:
                if split is None:
                    log_lik_p = scale_log_lik_p_by * p_state_new["_model_log_lik"].value
                else:
                    log_lik_p = split.scaled_log_lik(
                        self.p, p_state_new, part="validate"
                    )
            else:
                log_lik_p = batches.scaled_log_lik(self.p, p_state_new)
            log_prior_p = p_state_new["_model_log_prior"].value
            log_prob_p = log_lik_p + log_prior_p

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
            # from the Elbo, the variational log prob is minimized. Adding the prior
            # to the log lik of the variational dist would have the opposite of the
            # intended effect, since it would also be minimized. You could say we are
            # treating any priors in the variational model as parts of the main model
            if self.regularize_q_prior:
                return log_lik_q - log_prior_q

            return log_lik_q

        elbo_samples = log_prob_of_p(samples) - log_prob_of_q(samples)

        return jnp.mean(elbo_samples)

    def loss_train_batched(self, params: Position, carry: OptimCarry) -> jax.Array:
        """
        Computes the negative mini-batch ELBO used by optimizer updates.

        ``carry.batch`` supplies observed mini-batch values, and ``carry.batches``
        supplies the corresponding likelihood scaling, including per-branch scaling
        for :class:`.BatchManager`.
        """
        elbo = self.evaluate(
            Position(params | carry.fixed_position),
            carry.key,
            obs=Position(carry.batch),
            p_state=carry.model_state,
            q_state=self.q.state,
            batches=carry.batches,
            nsamples=self.nsamples,
        )
        return -elbo / self.scalar

    def loss_train(self, params: Position, carry: OptimCarry) -> jax.Array:
        """
        Computes the negative full-training-data ELBO.

        This method uses :attr:`split.train` as observed data and ignores the
        current mini-batch in ``carry.batch``. It is useful for diagnostics or
        full-data optimization.
        """
        elbo = self.evaluate(
            Position(params | carry.fixed_position),
            carry.key,
            obs=Position(self.split.train),
            p_state=carry.model_state,
            q_state=self.q.state,
            nsamples=self.nsamples,
        )
        return -elbo / self.scalar

    def loss_validate(self, params: Position, carry: OptimCarry) -> jax.Array:
        """
        Computes the negative validation ELBO.

        If the split has a validation part, validation observations and
        validation-scaled likelihoods are used. If no validation part exists,
        :class:`.LossMixin` falls back to the training observations.
        """
        elbo = self.evaluate(
            Position(params | carry.fixed_position),
            carry.key,
            obs=Position(self.obs_validate),
            p_state=carry.model_state,
            q_state=self.q.state,
            split=self.split,
            nsamples=self.nsamples_validate,
        )
        return -elbo / self.scalar

    def __repr__(self) -> str:
        """Returns a compact representation showing the training Monte Carlo count."""
        name = type(self).__name__
        return f"{name}(nsamples={self.nsamples})"


class VDist:
    r"""
    Represents a variational distribution.

    Parameters
    ----------
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
    of :math:`(\mu, \ln(\sigma))^\top` is modeled by two independent Gaussian
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
        self._flat_pos_name = "(" + "|".join(sorted(self.position_keys)) + ")"
        self.var: Var | None = None
        self.q: Model | None = None

        self._to_float32 = p.to_float32

    @property
    def p(self) -> Model:
        """
        Target model whose posterior is approximated by this variational block.

        Returns
        -------
        Model
            Model passed to :class:`VDist`.
        """
        return self._p

    def q_to_p(self, pos: Position) -> Position:
        """
        Maps a flat variational position back to the target-model representation.

        ``VDist`` represents the governed target parameters as one flattened
        pseudo-observed variable in ``q``. This method unflattens that variable into
        the original target-model position.

        Parameters
        ----------
        pos
            Position in the variational model representation. It must contain this
            block's flattened pseudo-observed variable.

        Returns
        -------
        Position
            Position in the representation expected by :attr:`p`.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> import liesel.experimental.optim as opt
        >>> theta = lsl.Var.new_param(jnp.array([1.0, 2.0]), name="theta")
        >>> p = lsl.Model([theta])
        >>> vdist = opt.VDist(["theta"], p)
        >>> vdist.q_to_p({"(theta)": jnp.array([3.0, 4.0])})["theta"].tolist()
        [3.0, 4.0]
        """
        return self._unflatten(pos[self._flat_pos_name])

    def p_to_q_array(self, pos: Position) -> jax.Array:
        """
        Flattens target-model position values into this block's variational array.

        Parameters
        ----------
        pos
            Position in the target-model representation. Its keys must match
            :attr:`position_keys` in the same order.

        Returns
        -------
        jax.Array
            Flattened array used as this block's pseudo-observed value in ``q``.

        Notes
        -----
        This is useful for initializing a variational distribution around pre-fitted
        parameter values.
        """
        if not list(pos) == self.position_keys:
            raise ValueError("list(pos) must be equal to self.position_keys.")

        return jax.flatten_util.ravel_pytree(pos)[0]

    @property
    def parameters(self) -> list[str]:
        """
        Names of the variational parameters in ``q``.

        Returns an empty list before the variational distribution has been
        initialized with :meth:`init`, :meth:`normal`, :meth:`mvn_diag`, or
        :meth:`mvn_tril`.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> import liesel.experimental.optim as opt
        >>> theta = lsl.Var.new_param(jnp.array(0.0), name="theta")
        >>> p = lsl.Model([theta])
        >>> vdist = opt.VDist(["theta"], p)
        >>> vdist.parameters
        []
        >>> vdist.mvn_diag().parameters
        ['(theta)_loc', 'h((theta)_scale)']
        """
        if self.var is None:
            return []

        if self.var.model is not None:
            model = self.var.model.parental_submodel(self.var)
            params = list(model.parameters)
            return params

        with TemporaryModel(self.var, to_float32=self._to_float32) as model:
            params = list(model.parameters)

        return params

    def _validate(self, dist: Dist) -> None:
        distribution = dist.init_dist()
        if distribution.reparameterization_type != tfd.FULLY_REPARAMETERIZED:
            raise ValueError(
                "Variational distributions must be fully reparameterized, but "
                f"{distribution!r} has "
                f"{distribution.reparameterization_type!r}."
            )

        value_shape = tuple(jnp.shape(self._flat_pos))
        try:
            sample_shape = _distribution_sample_shape(distribution, value_shape)
            sample = distribution.sample(sample_shape, seed=jax.random.key(0))
            if tuple(jnp.shape(sample)) != value_shape:
                raise ValueError(
                    "Sampling from the variational distribution with inferred "
                    f"{sample_shape=} returned shape {jnp.shape(sample)}, but the "
                    f"flattened position has shape {value_shape}."
                )
            distribution.log_prob(self._flat_pos)
        except Exception as error:
            if isinstance(error, ValueError):
                raise
            raise ValueError(
                "The variational distribution is incompatible with the flattened "
                f"position shape {value_shape}."
            ) from error

    def init(self, dist: Dist) -> Self:
        """
        Initializes this block with a custom variational distribution.

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

        Returns
        -------
        Self
            This ``VDist`` instance, allowing chained calls to :meth:`build`.
        """
        self._validate(dist)

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
        """
        Initializes independent univariate normal variational factors.

        The governed target-model position is flattened, and each flat component is
        assigned a ``tfd.Normal`` variational distribution. For vector-valued
        governed positions, use :meth:`mvn_diag` if you prefer one multivariate
        distribution with diagonal covariance.

        Parameters
        ----------
        loc
            Initial location. If ``None``, the current flattened target position is
            used.
        scale
            Initial scale. A scalar is broadcast to all flat components. The special
            value ``"laplace"`` initializes the scale from the diagonal of a
            Laplace-approximation covariance.
        scale_bijector
            Bijector applied to the scale parameter. ``"auto"`` delegates the choice
            to :meth:`liesel.model.Dist.biject_parameters`; ``None`` leaves the scale
            parameter untransformed.
        *bijector_args
            Positional arguments passed to ``scale_bijector`` when a bijector class
            is supplied.
        **bijector_kwargs
            Keyword arguments passed to ``scale_bijector`` when a bijector class is
            supplied.

        Returns
        -------
        Self
            This ``VDist`` instance, initialized with a normal variational
            distribution.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> import liesel.experimental.optim as opt
        >>> theta = lsl.Var.new_param(jnp.array(0.0), name="theta")
        >>> p = lsl.Model([theta])
        >>> opt.VDist(["theta"], p).normal(scale=0.5)
        VDist(['theta'], dist=Normal)
        """
        if loc is None:
            loc_value = jnp.asarray(self._flat_pos)
        else:
            loc_value = jnp.asarray(loc)

        if _is_laplace_init(scale, "scale"):
            cov_matrix = _laplace_covariance(self.p, self.position_keys, loc_value)
            scale_value = jnp.sqrt(jnp.clip(jnp.diag(cov_matrix), min=1e-12))
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
            Initial value for the location of the variational distribution. If
            ``None``, the current flattened target position is used.
        scale_diag
            Initial value for the square roots of the diagonal elements of the
            variational distribution's covariance matrix. In other words: The marginal
            standard deviations/scales. A scalar is broadcast to all flat components.
            The special value ``"laplace"`` initializes the diagonal scale from a
            Laplace-approximation covariance.
        scale_diag_bijector
            Bijector applied to the diagonal scale parameter. ``"auto"`` delegates
            to :meth:`liesel.model.Dist.biject_parameters`; ``None`` leaves the
            parameter untransformed.
        *bijector_args
            Positional arguments passed to ``scale_diag_bijector`` when a bijector
            class is supplied.
        **bijector_kwargs
            Keyword arguments passed to ``scale_diag_bijector`` when a bijector class
            is supplied.

        Returns
        -------
        Self
            This ``VDist`` instance, initialized with a diagonal multivariate normal
            variational distribution.

        Notes
        -----
        The docstring of :class:`.VDist` includes an example using this method.
        """
        if loc is None:
            loc_value = jnp.asarray(self._flat_pos)
        else:
            loc_value = loc

        if _is_laplace_init(scale_diag, "scale_diag"):
            cov_matrix = _laplace_covariance(self.p, self.position_keys, loc_value)
            scale_diag_value = jnp.sqrt(jnp.clip(jnp.diag(cov_matrix), min=1e-12))
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
            Initial value for the location of the variational distribution. If
            ``None``, the current flattened target position is used.
        scale_tril
            Initial value for the lower Cholesky factor, must have non-zero diagonal
            elements. A scalar is interpreted as a multiple of the identity matrix.
            The special value ``"laplace"`` initializes the lower Cholesky factor
            from a Laplace-approximation covariance.
        scale_tril_bijector
            Bijector applied to the lower-triangular scale parameter. ``"auto"``
            delegates to :meth:`liesel.model.Dist.biject_parameters`; ``None`` leaves
            the parameter untransformed.
        *bijector_args
            Positional arguments passed to ``scale_tril_bijector`` when a bijector
            class is supplied.
        **bijector_kwargs
            Keyword arguments passed to ``scale_tril_bijector`` when a bijector class
            is supplied.

        Returns
        -------
        Self
            This ``VDist`` instance, initialized with a dense multivariate normal
            variational distribution.

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

        if _is_laplace_init(scale_tril, "scale_tril"):
            cov_matrix = _laplace_covariance(self.p, self.position_keys, loc_value)
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

        Returns
        -------
        Self
            This ``VDist`` instance with :attr:`q` populated.

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
        Position
            Samples for the parameters governed by this :class:`.VDist` in the
            representation of :attr:`p`.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> import liesel.experimental.optim as opt
        >>> theta = lsl.Var.new_param(jnp.array([0.0]), name="theta")
        >>> p = lsl.Model([theta])
        >>> vdist = opt.VDist(["theta"], p).mvn_diag().build()
        >>> samples = vdist.sample(jax.random.key(1), sample_shape=(3,))
        >>> samples["theta"].shape
        (3, 1)
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
        """Returns a compact representation showing governed keys and distribution."""
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
    """
    Flattens leading batch dimensions of every pytree leaf.

    Parameters
    ----------
    pytree
        Pytree whose leaves have at least ``batch_ndim`` leading dimensions.
    batch_ndim
        Number of leading dimensions to flatten into one dimension.

    Returns
    -------
    pytree
        Pytree with the same structure and flattened leading dimensions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim.vi import flatten_leading_batch
    >>> out = flatten_leading_batch({"x": jnp.zeros((2, 3, 4))}, batch_ndim=2)
    >>> out["x"].shape
    (6, 4)
    """

    def _f(x):
        x = jnp.asarray(x)
        b = int(jnp.prod(jnp.array(x.shape[:batch_ndim])))
        return x.reshape((b,) + x.shape[batch_ndim:])

    return jax.tree.map(_f, pytree)


def unflatten_leading_batch(pytree, batch_shape):
    """
    Restores previously flattened leading batch dimensions.

    Parameters
    ----------
    pytree
        Pytree whose leaves have one flattened leading batch dimension.
    batch_shape
        Original leading batch shape.

    Returns
    -------
    pytree
        Pytree with the same structure and restored leading batch dimensions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim.vi import unflatten_leading_batch
    >>> out = unflatten_leading_batch({"x": jnp.zeros((6, 4))}, batch_shape=(2, 3))
    >>> out["x"].shape
    (2, 3, 4)
    """
    batch_shape = tuple(batch_shape)

    def _f(x):
        x = jnp.asarray(x)
        return x.reshape(batch_shape + x.shape[1:])

    return jax.tree.map(_f, pytree)


def vmap_batched(
    pos: Position, fun: Callable[[Position], Position], batch_shape: Sequence[int]
):
    """
    Applies a position transformation across leading batch dimensions.

    ``jax.vmap`` maps over one leading axis. This helper flattens an arbitrary
    ``batch_shape`` first, applies ``fun`` once with ``vmap``, and then restores the
    original batch shape. With an empty ``batch_shape``, ``fun`` is called directly.

    Parameters
    ----------
    pos
        Batched position passed to ``fun``.
    fun
        Function mapping one unbatched :class:`Position` to another.
    batch_shape
        Leading batch shape in every leaf of ``pos``.

    Returns
    -------
    Position
        Transformed position with the same leading batch shape.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim.types import Position
    >>> from liesel.experimental.optim.vi import vmap_batched
    >>> pos = Position({"x": jnp.arange(6).reshape(2, 3)})
    >>> out = vmap_batched(pos, lambda p: Position({"y": p["x"] + 1}), (2,))
    >>> out["y"].tolist()
    [[1, 2, 3], [4, 5, 6]]
    """
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
    of :math:`(\mu, \ln(\sigma))^\top` is modeled by two independent Gaussian
    distributions, i.e. we define the following variational distributions:

    .. math::
        \mu & \sim N(\phi_1, \phi_2^2) \\
        \ln(\sigma) & \sim N(\phi_3, \phi_4^2)

    This variational distribution can be composed by defining separate :class:`.VDist`
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
        if not vdists:
            raise ValueError("CompositeVDist requires at least one VDist.")

        first_model = vdists[0].p
        if any(vdist.p is not first_model for vdist in vdists):
            raise ValueError("All VDist objects in a CompositeVDist must share one p.")

        all_position_keys = [
            position_key for vdist in vdists for position_key in vdist.position_keys
        ]
        duplicate_position_keys = sorted(
            {
                position_key
                for position_key in all_position_keys
                if all_position_keys.count(position_key) > 1
            }
        )
        if duplicate_position_keys:
            raise ValueError(
                "Each target position key can be governed by at most one VDist. "
                f"Got duplicates: {duplicate_position_keys}."
            )

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
        Whether variational model values should be converted to ``float32``.

        All component distributions must agree on their target model's
        ``to_float32`` setting.
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
        """Returns a compact representation showing the number of blocks."""
        name = type(self).__name__
        if self.q is not None:
            built = "built"
        else:
            built = "not built"
        return f"{name}(n={len(self.vi_dists)}, {built})"
