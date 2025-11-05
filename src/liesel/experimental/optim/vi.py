from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Self

import jax
import jax.flatten_util
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from ...model import Dist, Model, Var
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
        split: PositionSplit,
        nsamples: int = 10,
        nsamples_validate: int = 50,
        q_to_p: Callable[[Position], Position] = lambda x: x,
        scale: bool = False,
    ):
        self.p = p
        self.q = q
        self.split = split
        self.nsamples = nsamples
        self.nsamples_validate = nsamples_validate
        self._q_to_p = q_to_p
        self.scale = scale
        self.scalar = self.split.n_train if self.scale else 1.0

    @classmethod
    def from_vdist(
        cls,
        vi_dist: VDist | CompositeVDist,
        split: PositionSplit,
        nsamples: int = 10,
        nsamples_validate: int = 50,
        scale: bool = False,
    ) -> Elbo:
        assert vi_dist.q is not None
        return cls(
            vi_dist.p,
            vi_dist.q,
            split=split,
            nsamples=nsamples,
            nsamples_validate=nsamples_validate,
            scale=scale,
            q_to_p=vi_dist.q_to_p,
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
            log_prob_q = q_state_new["_model_log_prob"].value
            return log_prob_q

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
    def __init__(self, position_keys: Sequence[str], p: Model):
        self.position_keys = position_keys
        self.p = p

        pos = self.p.extract_position(self.position_keys)
        flat_pos, unflatten = jax.flatten_util.ravel_pytree(pos)
        self._unflatten = unflatten
        self._flat_pos = flat_pos
        self._flat_pos_name = "(" + "|".join(self.position_keys) + ")"
        self.var: Var | None = None
        self.q: Model | None = None

    def q_to_p(self, pos: Position) -> Position:
        return self._unflatten(pos[self._flat_pos_name])

    def p_to_q_array(self, pos: Position) -> jax.Array:
        """Potentially useful for initializing with pre-fitted values."""
        if not list(pos) == self.position_keys:
            raise ValueError("list(pos) must be equal to self.position_keys.")

        return jax.flatten_util.ravel_pytree(pos)[0]

    @property
    def parameters(self) -> list[str]:
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
        self._validate(dist)
        self._reparameterize(dist)

        flat_pos_var = Var.new_obs(
            self._flat_pos,
            distribution=dist,
            name=self._flat_pos_name,
        )

        self.var = flat_pos_var
        return self

    def mvn_diag(
        self,
        loc: jax.typing.ArrayLike | None = None,
        scale_diag: jax.typing.ArrayLike | None = None,
    ) -> Self:
        if loc is None:
            loc_value = jnp.asarray(self._flat_pos)
        else:
            loc_value = loc

        if scale_diag is None:
            scale_diag_value = 0.01 * jnp.ones_like(self._flat_pos)
        else:
            scale_diag_value = scale_diag

        locs = Var.new_param(loc_value, name=self._flat_pos_name + "_loc")
        scales = Var.new_param(scale_diag_value, name=self._flat_pos_name + "_scale")
        scales.transform(tfb.Softplus())

        dist = Dist(tfd.MultivariateNormalDiag, loc=locs, scale_diag=scales)

        return self.init(dist)

    def build(self) -> Self:
        if self.var is None:
            raise ValueError("The .var attribute must be set, but is currently None.")
        self.q = Model([self.var])
        return self

    def sample_p(
        self,
        n: int,
        seed: jax.Array,
        prepend_axis: bool = True,
        at_position: Position | None = None,
    ) -> Position:
        if self.q is None:
            raise ValueError("The object has no built model.")
        if at_position is not None:
            at_position = jax.tree.map(
                lambda x: jnp.expand_dims(x, (0, 1)), at_position
            )
        q_samples = self.q.sample(shape=(n,), seed=seed, posterior_samples=at_position)
        p_samples = jax.vmap(self.q_to_p)(q_samples)
        if prepend_axis:
            p_samples = jax.tree.map(lambda x: jnp.expand_dims(x, 0), p_samples)
        return p_samples

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


class CompositeVDist:
    def __init__(self, *vi_dists: VDist):
        self.vi_dists = vi_dists
        self.q: Model | None = None

    @property
    def parameters(self) -> list[str]:
        if self.q is None:
            return []
        return list(self.q.parameters)

    @property
    def p(self) -> Model:
        return self.vi_dists[0].p

    def build(self) -> Self:
        vars_ = []
        for dist in self.vi_dists:
            if dist.var is None:
                raise ValueError(f".var attribute of {dist} must be set, but is None.")
            vars_.append(dist.var)
        q = Model(vars_)
        self.q = q
        return self

    def q_to_p(self, pos: Position) -> Position:
        positions = [dist.q_to_p(pos) for dist in self.vi_dists]
        combined = Position({k: v for d in positions for k, v in d.items()})
        return combined

    def sample_p(
        self,
        n: int,
        seed: jax.Array,
        prepend_axis: bool = True,
        at_position: Position | None = None,
    ) -> Position:
        if self.q is None:
            raise ValueError("The object has no built model.")
        if at_position is not None:
            at_position = jax.tree.map(
                lambda x: jnp.expand_dims(x, (0, 1)), at_position
            )
        q_samples = self.q.sample(shape=(n,), seed=seed, posterior_samples=at_position)
        p_samples = jax.vmap(self.q_to_p)(q_samples)
        if prepend_axis:
            p_samples = jax.tree.map(lambda x: jnp.expand_dims(x, 0), p_samples)
        return p_samples

    def __repr__(self) -> str:
        name = type(self).__name__
        if self.q is not None:
            built = "built"
        else:
            built = "not built"
        return f"{name}(n={len(self.vi_dists)}, {built})"
