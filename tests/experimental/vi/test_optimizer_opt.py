"""Unit tests for the Optimizer class optimization related functions."""

from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import pytest

from liesel.experimental.vi import LieselInterface
from liesel.experimental.vi.optimizer import Optimizer

# -- Fakes and Fixtures --


class FakeBijector:
    """Simple identity-like bijector used to avoid TFP deps."""

    def forward(self, x):
        return x

    def inverse(self, x):
        return x


class FakeDistribution:
    """Tiny stand-in for a TFP distribution.

    - Requires a `loc` parameter.
    - Uses the shape of `loc` to define the event shape.
    - Ignores extra kwargs (e.g., covariance_matrix, scale).
    - sample() returns a array (arange over event size).
    - log_prob(z) returns a scalar constant.
    """

    def __init__(self, *, loc, **kwargs):
        self.loc = jnp.asarray(loc)
        self._shape = tuple(self.loc.shape)
        size = int(jnp.prod(jnp.array(self._shape)))
        self._vec = jnp.arange(size, dtype=jnp.float32).reshape(self._shape)

    def sample(self, seed=None):
        return self._vec

    def log_prob(self, z):
        return jnp.array(1.5, dtype=jnp.float32)


@pytest.fixture
def mock_model_interface():
    """Create a mock LieselInterface exposing only latents a and b."""
    interface = Mock(spec=LieselInterface)

    # Model params used by Optimizer for reshaping/splitting
    interface.get_params.return_value = {
        "a": jnp.zeros((2,)),
        "b": jnp.zeros((1,)),
    }

    # Observed variable so Optimizer can infer dim_data
    mock_observed_var = Mock()
    mock_observed_var.observed = True
    mock_observed_var.value = Mock()
    mock_observed_var.value.shape = (4,)  # dim_data = 4

    # Optional unobserved var (not required, but mirrors structure)
    mock_unobserved_var = Mock()
    mock_unobserved_var.observed = False
    mock_unobserved_var.value = Mock()
    mock_unobserved_var.value.shape = (2,)

    mock_model = Mock()
    mock_model.vars = {
        "y_obs": mock_observed_var,
        "u": mock_unobserved_var,
    }
    interface.model = mock_model

    # Keep ELBO arithmetic simple and deterministic
    def _compute_log_prob(samples, dim_data, batch_size, batch_indices):
        return jnp.array(2.5, dtype=jnp.float32)

    interface.compute_log_prob.side_effect = _compute_log_prob

    return interface


@pytest.fixture
def common_latent_configs():
    """Two patterns for latents a, b:
    - 'composite': one variational distribution for both a and b (key 'a_b')
    - 'mean_field': separate variational distributions for 'a' and 'b'
    Returned shape matches what Optimizer expects post-Builder.build(): a dict.
    """
    id_bij = FakeBijector()

    # (1) Composite: one variational distribution for ['a','b'] → key 'a_b'
    composite_cfg = {
        "names": ["a", "b"],
        "dist_class": FakeDistribution,  # stand-in for MVN full-cov
        "variational_params": {
            "loc": jnp.zeros(3),
            "covariance_matrix": jnp.eye(3),  # mirrors MVN usage
        },
        "fixed_distribution_params": None,  # builder passes None
        "optimizer_chain": "no_optimizer",
        "variational_param_bijectors": {
            "loc": id_bij,
            "covariance_matrix": id_bij,
        },
        "variable_dims": {"a": 2, "b": 1},
        "dims_list": [2, 1],
        "split_indices": [2],  # split 3-vector into [0:2], [2:]
        "event_shape": 3,
    }
    composite_dict = {"a_b": composite_cfg}

    # (2) Mean-field: each latent has its own variational distribution → keys 'a', 'b'
    a_cfg = {
        "names": ["a"],
        "dist_class": FakeDistribution,
        "variational_params": {"loc": jnp.zeros(2), "scale": jnp.ones(2)},
        "fixed_distribution_params": None,
        "optimizer_chain": "no_optimizer",
        "variational_param_bijectors": {"loc": id_bij, "scale": id_bij},
        "variable_dims": {"a": 2},
        "dims_list": [2],
        "split_indices": [],
        "event_shape": 2,
    }
    b_cfg = {
        "names": ["b"],
        "dist_class": FakeDistribution,
        "variational_params": {"loc": jnp.zeros(1), "scale": jnp.ones(1)},
        "fixed_distribution_params": None,
        "optimizer_chain": "no_optimizer",
        "variational_param_bijectors": {"loc": id_bij, "scale": id_bij},
        "variable_dims": {"b": 1},
        "dims_list": [1],
        "split_indices": [],
        "event_shape": 1,
    }
    mean_field_dict = {"a": a_cfg, "b": b_cfg}

    return {
        "composite": composite_dict,
        "mean_field": mean_field_dict,
    }


@pytest.fixture
def patch_init_optimizer(monkeypatch):
    """Avoids building real Optax chains; return a optimizer and static state."""

    def fake_init_optimizer(self):
        # update(grads, opt_state, params) -> (updates, new_state)
        noop = SimpleNamespace(update=lambda g, s, p: ({}, s))
        return "fake_opt_state", noop

    monkeypatch.setattr(Optimizer, "_init_optimizer", fake_init_optimizer)


# -- Unit Tests - _sample_and_compute_variational_log_prob --


class TestSampleAndLogProbCompute:
    def test_mean_field_sampling_and_logprob(
        self, mock_model_interface, common_latent_configs, patch_init_optimizer
    ):
        mean_field_cfg = common_latent_configs["mean_field"]

        opt = Optimizer(
            seed=0,
            n_epochs=1,
            S=1,
            model_interface=mock_model_interface,
            latent_variables=mean_field_cfg,
            batch_size=None,
            patience_tol=None,
            window_size=None,
        )

        rng = jax.random.PRNGKey(0)
        samples, total_log_q = opt._sample_and_compute_variational_log_prob(
            opt.variational_params, rng
        )

        # sanity check we're really in the mean-field case
        assert set(opt.latent_vars_config.keys()) == {"a", "b"}

        # FakeDistribution.sample for 'a' (event dim=2) -> arange(2) -> [0., 1.]
        assert "a" in samples
        assert jnp.array_equal(samples["a"], jnp.array([0.0, 1.0], dtype=jnp.float32))

        # FakeDistribution.sample for 'b' (event dim=1) -> arange(1) -> [0.]
        assert "b" in samples
        assert jnp.array_equal(samples["b"], jnp.array([0.0], dtype=jnp.float32))

        # One variational distribution -> one log_q contribution
        assert total_log_q == 3.0

    def test_composite_sampling_and_logprob(
        self, mock_model_interface, common_latent_configs, patch_init_optimizer
    ):
        # Composite configuration over ['a','b'] (key 'a_b')
        composite_cfg = common_latent_configs["composite"]

        # Ensure model param shapes so reshaping/splitting is well-defined
        mock_model_interface.get_params.return_value = {
            "a": jnp.zeros((2,)),
            "b": jnp.zeros((1,)),
        }

        opt = Optimizer(
            seed=0,
            n_epochs=1,
            S=1,
            model_interface=mock_model_interface,
            latent_variables=composite_cfg,  # {'a_b': {...}}
            batch_size=None,
            patience_tol=None,
            window_size=None,
        )

        rng = jax.random.PRNGKey(0)
        samples, total_log_q = opt._sample_and_compute_variational_log_prob(
            opt.variational_params, rng
        )

        # FakeDistribution.sample uses arange(3) -> [0., 1., 2.]
        # split_indices = [2] -> a gets [0.,1.], b gets [2.]
        assert jnp.array_equal(samples["a"], jnp.array([0.0, 1.0]))
        assert jnp.array_equal(samples["b"], jnp.array([2.0]))
        assert jnp.array_equal(total_log_q, 1.5)


# -- Unit Tests - _elbo --


class TestElboComputation:
    @pytest.mark.parametrize("cfg_key", ["mean_field", "composite"])
    def test_elbo(
        self,
        cfg_key,
        mock_model_interface,
        common_latent_configs,
        patch_init_optimizer,
        monkeypatch,
    ):
        latent_cfg = common_latent_configs[cfg_key]

        opt = Optimizer(
            seed=0,
            n_epochs=1,
            S=2,
            model_interface=mock_model_interface,
            latent_variables=latent_cfg,
            batch_size=None,
            patience_tol=None,
            window_size=None,
        )

        def fake_sample_and_logq(params, key):
            samples = {
                "a": jnp.array([0.0, 1.0], dtype=jnp.float32),
                "b": jnp.array([0.0], dtype=jnp.float32),
            }
            total_log_q = jnp.array(0.5, dtype=jnp.float32)
            return samples, total_log_q

        monkeypatch.setattr(
            opt, "_sample_and_compute_variational_log_prob", fake_sample_and_logq
        )

        dim_data = 4  # not relevant
        batch_size = 4  # not relevant
        batch_indices = jnp.arange(batch_size)  # not relevant
        rng = jax.random.PRNGKey(0)

        loss, _ = opt._elbo(
            opt.variational_params, rng, dim_data, batch_size, batch_indices, opt.S
        )
        # -(2*(2.5-0.5)/2)=-2.0
        assert jnp.array_equal(loss, -2.0)


# -- Unit Tests - get_final_distributions --


class TestGetFinalDistributions:
    def test_mapping_includes_each_name_and_composite_for_multivariate(
        self,
        mock_model_interface,
        common_latent_configs,
        patch_init_optimizer,
        monkeypatch,
    ):
        # Use the composite configuration specifically
        composite_cfg = common_latent_configs["composite"]

        opt = Optimizer(
            seed=0,
            n_epochs=1,
            S=1,
            model_interface=mock_model_interface,
            latent_variables=composite_cfg,
            batch_size=None,
            patience_tol=None,
            window_size=None,
        )

        # Spies on _build_variational_distribution to ensure it was called per key
        build_calls = []

        def fake_build(dist_class, vparams, fixed, bijectors):
            # Track the call (by set of parameter keys) and return a tagged object
            build_calls.append(
                {"dist_class": dist_class, "keys": sorted(vparams.keys())}
            )
            return SimpleNamespace(tag="built")

        monkeypatch.setattr(opt, "_build_variational_distribution", fake_build)

        results = opt.get_final_distributions()

        # For composite config: should include 'a', 'b' and composite key 'a_b'
        assert set(results.keys()) == {"a", "b", "a_b"}

        # The same distribution instance should be assigned to all names of a multi-var
        assert results["a"] is results["b"] is results["a_b"]

        # Ensure we built for the composite latent variable entry (just one call)
        assert len(build_calls) == 1
        # The dist_class should be our FakeDistribution
        assert build_calls[0]["dist_class"] is FakeDistribution

    def test_mapping_for_mean_field_separate_distributions(
        self,
        mock_model_interface,
        common_latent_configs,
        patch_init_optimizer,
        monkeypatch,
    ):
        # Use the mean_field configuration specifically
        mean_field_cfg = common_latent_configs["mean_field"]

        opt = Optimizer(
            seed=0,
            n_epochs=1,
            S=1,
            model_interface=mock_model_interface,
            latent_variables=mean_field_cfg,  # Use mean_field config
            batch_size=None,
            patience_tol=None,
            window_size=None,
        )

        # Spy on _build_variational_distribution
        build_calls = []

        def fake_build(dist_class, vparams, fixed, bijectors):
            build_calls.append(
                {"dist_class": dist_class, "keys": sorted(vparams.keys())}
            )
            return SimpleNamespace(tag=f"built_{len(build_calls)}")

        monkeypatch.setattr(opt, "_build_variational_distribution", fake_build)

        results = opt.get_final_distributions()

        # For mean_field config: should include individual names 'a', 'b' only
        assert set(results.keys()) == {"a", "b"}

        # Different distribution instances for each variable
        assert results["a"] is not results["b"]

        # Ensure we built for both latent variable entries (two calls)
        assert len(build_calls) == 2
        # Both should use FakeDistribution
        assert all(c["dist_class"] is FakeDistribution for c in build_calls)


# -- Unit Tests - fit --


class TestFitLoop:
    @pytest.mark.parametrize("cfg_key", ["mean_field", "composite"])
    def test_fit_runs_with_noop_optimizer_and_records_elbo_and_final_dists(
        self,
        cfg_key,
        mock_model_interface,
        common_latent_configs,
        patch_init_optimizer,
        monkeypatch,
    ):
        latent_cfg = common_latent_configs[cfg_key]

        opt = Optimizer(
            seed=0,
            n_epochs=2,
            S=1,
            model_interface=mock_model_interface,
            latent_variables=latent_cfg,
            batch_size=None,
            patience_tol=0.0,
            window_size=2,  # run until n_epochs=2 anyway
        )

        # -------- monkeypatch JAX control flow & transforms to pure-Python --------

        # jit: return function unchanged
        monkeypatch.setattr(
            jax, "jit", lambda f=None, **kw: (lambda g: g) if f is None else f
        )

        # value_and_grad: compute loss/aux by calling f, return zero-like grads
        def fake_vag(f, has_aux=False):
            def wrapped(*args, **kwargs):
                out = f(*args, **kwargs)
                if has_aux:
                    (loss, aux) = out
                else:
                    loss, aux = out, None
                # params tree is at args[0] in our usage (see optimizer.step)
                params = args[0]
                grads = jax.tree.map(lambda x: jnp.zeros_like(x), params)
                return (loss, aux), grads

            return wrapped

        monkeypatch.setattr(jax, "value_and_grad", fake_vag)

        # lax.fori_loop: basic Python for-loop
        def fake_fori_loop(lower, upper, body_fun, init_val):
            state = init_val
            for i in range(lower, upper):
                state = body_fun(i, state)
            return state

        # lax.while_loop: basic Python while-loop with guard
        def fake_while_loop(cond_fun, body_fun, init_val):
            state = init_val
            guard = 0
            while bool(cond_fun(state)):
                state = body_fun(state)
                guard += 1
                if guard > 10000:  # safety
                    break
            return state

        # lax.cond: evaluate predicate and pick a branch
        def fake_cond(pred, true_fun, false_fun, operand):
            pred_bool = bool(jnp.asarray(pred))
            return true_fun(operand) if pred_bool else false_fun(operand)

        monkeypatch.setattr(jax.lax, "fori_loop", fake_fori_loop)
        monkeypatch.setattr(jax.lax, "while_loop", fake_while_loop)
        monkeypatch.setattr(jax.lax, "cond", fake_cond)

        # optax.apply_updates: params unchanged (no-op)
        import optax as _optax  # local import to patch the real module

        monkeypatch.setattr(_optax, "apply_updates", lambda params, updates: params)

        # Stub _elbo to return a constant loss and propagate the rng key unchanged.
        # Track calls to verify per-epoch invocation.
        elbo_calls = []

        def fake_elbo(p, key, dim_data, batch_size, batch_indices, S):
            elbo_calls.append((tuple(sorted(p.keys())), dim_data, batch_size, int(S)))
            # loss = 1.0 -> epoch ELBO stored as -loss = -1.0
            return (jnp.array(1.0, dtype=jnp.float32), key)

        monkeypatch.setattr(opt, "_elbo", fake_elbo)

        # Also spy get_final_distributions so we can assert it was called at the end
        final_dist = {"fake": "dist"}
        monkeypatch.setattr(opt, "get_final_distributions", lambda: final_dist)

        # ------------------ run fit ------------------
        opt.fit()

        # ------------------ assertions ------------------
        # With n_epochs=2 and 1 batch/epoch, _elbo should be called twice
        assert len(elbo_calls) == 2

        # elbo_values should have one entry per finished epoch, equal to -1.0
        assert len(opt.elbo_values) == 2
        assert all(jnp.allclose(v, -1.0) for v in opt.elbo_values)

        # final variational distributions captured
        assert opt.final_variational_distributions is final_dist
