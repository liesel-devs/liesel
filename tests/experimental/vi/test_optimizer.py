"""Unit and Integration tests for the Optimizer class."""

from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import optax
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

from liesel.experimental.vi import LieselInterface
from liesel.experimental.vi.optimizer import Optimizer


# --- Fakes / helpers ----------------------------------------------------------


class FakeBijector:
    def forward(self, x):
        return x

    def inverse(self, x):
        return x

    def __call__(self, x):
        return self.forward(x)


class FakeDistribution:
    """Accepts loc, optional scale or covariance_matrix; returns constant log_prob."""

    def __init__(self, *, loc, scale=None, covariance_matrix=None, **kwargs):
        self.loc = jnp.asarray(loc)
        self.scale = None if scale is None else jnp.asarray(scale)
        self.covariance_matrix = (
            None if covariance_matrix is None else jnp.asarray(covariance_matrix)
        )
        self._shape = tuple(self.loc.shape)
        size = int(jnp.prod(jnp.array(self._shape))) if self._shape else 1
        self._vec = (
            jnp.arange(size, dtype=jnp.float32).reshape(self._shape)
            if self._shape
            else jnp.array(0.0, dtype=jnp.float32)
        )

    def sample(self, seed=None):
        return self._vec

    def log_prob(self, z):
        return jnp.array(1.5, dtype=jnp.float32)


class DummyVar:
    def __init__(self, value, observed=True):
        self.value = jnp.asarray(value)
        self.observed = observed


class DummyModel:
    def __init__(self, obs_n=12):
        self.vars = {"y": DummyVar(jnp.zeros((obs_n,)), observed=True)}


class DummyInterface:
    def __init__(self, params, obs_n=12):
        self._params = {k: jnp.asarray(v) for k, v in params.items()}
        self.model = DummyModel(obs_n=obs_n)

    def get_params(self):
        return self._params

    def compute_log_prob(self, samples, dim_data, batch_size, batch_indices):
        return jnp.array(0.0)


# --- Fixtures --------------------------------------------------------------------


@pytest.fixture
def make_latent_config():
    def _make():
        return {
            "z": {
                "names": ["z"],
                "dist_class": FakeDistribution,
                "variational_params": {"loc": jnp.array(0.0), "scale": jnp.array(1.0)},
                "fixed_distribution_params": None,
                "optimizer_chain": optax.adam(1e-2),
                "variational_param_bijectors": {
                    "loc": FakeBijector(),
                    "scale": FakeBijector(),
                },
                "split_indices": [],
            }
        }

    return _make


@pytest.fixture
def mock_model_interface():
    interface = Mock(spec=LieselInterface)

    interface.get_params.return_value = {"a": jnp.zeros((2,)), "b": jnp.zeros((1,))}

    obs = Mock()
    obs.observed = True
    obs.value = Mock()
    obs.value.shape = (4,)
    unobs = Mock()
    unobs.observed = False
    unobs.value = Mock()
    unobs.value.shape = (2,)

    model = Mock()
    model.vars = {"y_obs": obs, "u": unobs}
    interface.model = model

    def _compute_log_prob(samples, dim_data, batch_size, batch_indices):
        return jnp.array(2.5, dtype=jnp.float32)

    interface.compute_log_prob.side_effect = _compute_log_prob
    return interface


@pytest.fixture
def mean_field_cfg():
    id_bij = FakeBijector()
    return {
        "a": {
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
        },
        "b": {
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
        },
    }


@pytest.fixture
def composite_cfg():
    id_bij = FakeBijector()
    return {
        "a_b": {
            "names": ["a", "b"],
            "dist_class": FakeDistribution,
            "variational_params": {
                "loc": jnp.zeros(3),
                "covariance_matrix": jnp.eye(3),
            },
            "fixed_distribution_params": None,
            "optimizer_chain": "no_optimizer",
            "variational_param_bijectors": {"loc": id_bij, "covariance_matrix": id_bij},
            "variable_dims": {"a": 2, "b": 1},
            "dims_list": [2, 1],
            "split_indices": [2],
            "event_shape": 3,
        }
    }


@pytest.fixture(params=["mean_field", "composite"], ids=["mean_field", "composite"])
def latent_cfg(request, mean_field_cfg, composite_cfg):
    return mean_field_cfg if request.param == "mean_field" else composite_cfg


@pytest.fixture
def disable_jit_and_zero_grads(monkeypatch):
    import jax

    monkeypatch.setattr(
        jax, "jit", lambda f=None, **kw: (lambda g: g) if f is None else f
    )

    def fake_vag(f, has_aux=False):
        def wrapped(*args, **kwargs):
            out = f(*args, **kwargs)
            (loss, aux) = out if has_aux else (out, None)
            params = args[0]
            grads = jax.tree.map(lambda x: jnp.zeros_like(x), params)
            return (loss, aux), grads

        return wrapped

    monkeypatch.setattr(jax, "value_and_grad", fake_vag)


@pytest.fixture
def lax_loops_as_python(monkeypatch):
    import jax

    def fake_fori_loop(lower, upper, body_fun, init_val):
        state = init_val
        for i in range(lower, upper):
            state = body_fun(i, state)
        return state

    def fake_while_loop(cond_fun, body_fun, init_val):
        state = init_val
        guard = 0
        while bool(cond_fun(state)):
            state = body_fun(state)
            guard += 1
            if guard > 10000:
                break
        return state

    def fake_cond(pred, true_fun, false_fun, operand):
        pred_bool = bool(jnp.asarray(pred))
        return true_fun(operand) if pred_bool else false_fun(operand)

    monkeypatch.setattr(jax.lax, "fori_loop", fake_fori_loop)
    monkeypatch.setattr(jax.lax, "while_loop", fake_while_loop)
    monkeypatch.setattr(jax.lax, "cond", fake_cond)


@pytest.fixture
def optax_noop_apply_updates(monkeypatch):
    import optax as _optax

    monkeypatch.setattr(_optax, "apply_updates", lambda params, updates: params)


@pytest.fixture
def patch_init_optimizer(monkeypatch):
    def fake_init_optimizer(self):
        noop = SimpleNamespace(update=lambda g, s, p: ({}, s))
        return "fake_opt_state", noop

    monkeypatch.setattr(Optimizer, "_init_optimizer", fake_init_optimizer)

# --- Tests --------------------------------------------------------------------

def test_accepts_tfp_distribution_class(make_latent_config):
    latent = make_latent_config()
    latent["z"]["dist_class"] = tfd.Normal
    opt = Optimizer(
        seed=0,
        n_epochs=1,
        S=2,
        model_interface=DummyInterface({"z": jnp.array(0.0)}),
        latent_variables=latent,
    )
    assert opt.variational_dists_class["z"] is tfd.Normal


@pytest.mark.xfail(reason="Should fail: dist_class is an instance, not a class")
def test_rejects_distribution_instance_not_class(make_latent_config):
    latent = make_latent_config()
    latent["z"]["dist_class"] = tfd.Normal(loc=0.0, scale=1.0)
    with pytest.raises(ValueError, match="dist_class.*class|TFP|Distribution"):
        Optimizer(
            seed=0,
            n_epochs=1,
            S=2,
            model_interface=DummyInterface({"z": jnp.array(0.0)}),
            latent_variables=latent,
        )


@pytest.mark.xfail(reason="Should fail: dist_class is not a TFP Distribution class")
def test_rejects_non_tfp_distribution_class(make_latent_config):
    latent = make_latent_config()
    latent["z"]["dist_class"] = object
    with pytest.raises(ValueError, match="TFP|Distribution|dist_class"):
        Optimizer(
            seed=0,
            n_epochs=1,
            S=2,
            model_interface=DummyInterface({"z": jnp.array(0.0)}),
            latent_variables=latent,
        )


def test_init_variational_params_inverse_bijectors(make_latent_config):
    latent = make_latent_config()
    opt = Optimizer(
        seed=0,
        n_epochs=1,
        S=2,
        model_interface=DummyInterface({"z": jnp.array(0.0)}),
        latent_variables=latent,
    )
    unconstrained = opt.variational_params["z"]
    assert set(unconstrained.keys()) == {"loc", "scale"}
    assert jnp.allclose(unconstrained["loc"], jnp.array(0.0))


def test_init_variational_params_missing_bijector_key_raises(make_latent_config):
    latent = make_latent_config()
    latent["z"]["variational_param_bijectors"].pop("scale")
    with pytest.raises(KeyError):
        _ = Optimizer(
            seed=0,
            n_epochs=1,
            S=2,
            model_interface=DummyInterface({"z": jnp.array(0.0)}),
            latent_variables=latent,
        )


def test_init_fixed_params_validate_args_separate_from_variational(make_latent_config):
    latent = make_latent_config()
    latent["z"]["fixed_distribution_params"] = {"validate_args": True}
    opt = Optimizer(
        seed=0,
        n_epochs=1,
        S=2,
        model_interface=DummyInterface({"z": jnp.array(0.0)}),
        latent_variables=latent,
    )
    assert "validate_args" not in opt.variational_params["z"]
    fixed = opt.fixed_distribution_params["z"]
    assert "validate_args" in fixed
    assert bool(fixed["validate_args"]) is True


def test_init_variational_params_inverse_bijectors_scale_softplus_roundtrip(
    make_latent_config,
):
    for s in [0.1, 1.0, 3.0]:
        latent = make_latent_config()
        latent["z"]["variational_param_bijectors"]["scale"] = tfb.Softplus()
        latent["z"]["variational_params"]["scale"] = jnp.array(s)
        opt = Optimizer(
            seed=0,
            n_epochs=1,
            S=2,
            model_interface=DummyInterface({"z": jnp.array(0.0)}),
            latent_variables=latent,
        )
        uncon = opt.variational_params["z"]
        expected_uncon = tfb.Softplus().inverse(jnp.array(s))
        assert jnp.allclose(uncon["scale"], expected_uncon)
        back = tfb.Softplus()(uncon["scale"])
        assert jnp.allclose(back, jnp.array(s), rtol=1e-5, atol=1e-6)


def test_unconstrained_values_are_finite_after_init(make_latent_config):
    latent = make_latent_config()
    opt = Optimizer(
        seed=0,
        n_epochs=1,
        S=2,
        model_interface=DummyInterface({"z": jnp.array(0.0)}),
        latent_variables=latent,
    )
    for block in opt.variational_params.values():
        for v in block.values():
            assert jnp.all(jnp.isfinite(v))


def test_init_fixed_distribution_params_handles_none(make_latent_config):
    latent = make_latent_config()
    opt = Optimizer(
        seed=0,
        n_epochs=1,
        S=2,
        model_interface=DummyInterface({"z": jnp.array(0.0)}),
        latent_variables=latent,
    )
    assert opt.fixed_distribution_params["z"] == {}


def test_init_variational_param_bijectors_mapping(make_latent_config):
    latent = make_latent_config()
    opt = Optimizer(
        seed=0,
        n_epochs=1,
        S=2,
        model_interface=DummyInterface({"z": jnp.array(0.0)}),
        latent_variables=latent,
    )
    assert "loc" in opt.variational_param_bijectors["z"]
    assert "scale" in opt.variational_param_bijectors["z"]


def test_init_optimizer_and_update_step_tree_shapes(make_latent_config):
    latent = make_latent_config()
    opt = Optimizer(
        seed=0,
        n_epochs=1,
        S=2,
        model_interface=DummyInterface({"z": jnp.array(0.0)}),
        latent_variables=latent,
    )
    grads = {"z": {"loc": jnp.array(0.0), "scale": jnp.array(0.0)}}
    updates, _ = opt.optimizer.update(grads, opt.opt_state, opt.variational_params)
    assert set(updates["z"].keys()) == set(opt.variational_params["z"].keys())


def test_build_applies_bijectors_and_merges_fixed(make_latent_config):
    latent = make_latent_config()
    latent["z"]["fixed_distribution_params"] = {"validate_args": True}
    opt = Optimizer(0, 1, 2, DummyInterface({"z": 0.0}), latent)
    dist = opt._build_variational_distribution(
        opt.variational_dists_class["z"],
        opt.variational_params["z"],
        opt.fixed_distribution_params["z"],
        opt.variational_param_bijectors["z"],
    )
    assert jnp.allclose(dist.loc, 0.0)
    assert jnp.allclose(dist.scale, 1.0)


def test_build_raises_on_invalid_scale_with_validate_args(make_latent_config):
    latent = make_latent_config()
    latent["z"]["dist_class"] = tfd.Normal
    latent["z"]["fixed_distribution_params"] = {"validate_args": True}
    latent["z"]["variational_params"]["scale"] = jnp.array(0.0)
    latent["z"]["variational_param_bijectors"]["scale"] = tfb.Identity()
    opt = Optimizer(0, 1, 2, DummyInterface({"z": 0.0}), latent)
    with pytest.raises(Exception):
        _ = opt._build_variational_distribution(
            opt.variational_dists_class["z"],
            opt.variational_params["z"],
            opt.fixed_distribution_params["z"],
            opt.variational_param_bijectors["z"],
        )


def test_build_mvn_diag_shapes():
    params = {"beta": jnp.zeros(4)}
    latent = {
        "beta": {
            "names": ["beta"],
            "dist_class": tfd.MultivariateNormalDiag,
            "variational_params": {"loc": jnp.zeros(4), "scale_diag": jnp.ones(4)},
            "fixed_distribution_params": {},
            "optimizer_chain": optax.adam(1e-2),
            "variational_param_bijectors": {
                "loc": tfb.Identity(),
                "scale_diag": tfb.Softplus(),
            },
            "split_indices": [],
        }
    }
    opt = Optimizer(0, 1, 2, DummyInterface(params), latent)
    dist = opt._build_variational_distribution(
        opt.variational_dists_class["beta"],
        opt.variational_params["beta"],
        opt.fixed_distribution_params["beta"],
        opt.variational_param_bijectors["beta"],
    )
    assert dist.event_shape == (4,)
    assert jnp.allclose(dist.loc, jnp.zeros(4))
    assert jnp.allclose(jnp.diag(dist.covariance()), jnp.ones(4))


def test_init_optimizer_assigns_different_transforms_per_block():
    latent = {
        "z": {
            "names": ["z"],
            "dist_class": tfd.Normal,
            "variational_params": {"loc": jnp.array(0.0), "scale": jnp.array(1.0)},
            "fixed_distribution_params": {},
            "optimizer_chain": optax.adam(1e-3),
            "variational_param_bijectors": {
                "loc": tfb.Identity(),
                "scale": tfb.Softplus(),
            },
            "split_indices": [],
        },
        "w": {
            "names": ["w"],
            "dist_class": tfd.Normal,
            "variational_params": {"loc": jnp.array(0.5), "scale": jnp.array(1.5)},
            "fixed_distribution_params": {},
            "optimizer_chain": optax.sgd(1e-1),
            "variational_param_bijectors": {
                "loc": tfb.Identity(),
                "scale": tfb.Softplus(),
            },
            "split_indices": [],
        },
    }
    opt = Optimizer(0, 1, 2, DummyInterface({"z": 0.0, "w": 0.0}), latent)
    grads = {
        "z": {"loc": jnp.array(1.0), "scale": jnp.array(1.0)},
        "w": {"loc": jnp.array(1.0), "scale": jnp.array(1.0)},
    }
    updates, _ = opt.optimizer.update(grads, opt.opt_state, opt.variational_params)
    assert not jnp.allclose(updates["z"]["loc"], updates["w"]["loc"])
    assert not jnp.allclose(updates["z"]["scale"], updates["w"]["scale"])


def test_zero_gradients_produce_zero_updates_initially(make_latent_config):
    latent = make_latent_config()
    opt = Optimizer(0, 1, 2, DummyInterface({"z": 0.0}), latent)
    grads = {"z": {"loc": jnp.array(0.0), "scale": jnp.array(0.0)}}
    updates, _ = opt.optimizer.update(grads, opt.opt_state, opt.variational_params)
    assert jnp.allclose(updates["z"]["loc"], 0.0)
    assert jnp.allclose(updates["z"]["scale"], 0.0)


def test_init_optimizer_errors_if_optimizer_chain_missing():
    latent = {
        "z": {
            "names": ["z"],
            "dist_class": tfd.Normal,
            "variational_params": {"loc": jnp.array(0.0), "scale": jnp.array(1.0)},
            "fixed_distribution_params": {},
            "optimizer_chain": None,
            "variational_param_bijectors": {
                "loc": tfb.Identity(),
                "scale": tfb.Softplus(),
            },
            "split_indices": [],
        }
    }
    with pytest.raises(Exception):
        _ = Optimizer(0, 1, 2, DummyInterface({"z": 0.0}), latent)


def test_init_structure_key_coherence_only():
    params = {"z": jnp.array(0.0), "beta": jnp.zeros(3)}
    latent = {
        "z": {
            "names": ["z"],
            "dist_class": tfd.Normal,
            "variational_params": {"loc": jnp.array(0.0), "scale": jnp.array(1.0)},
            "fixed_distribution_params": {"validate_args": True},
            "optimizer_chain": optax.adam(1e-3),
            "variational_param_bijectors": {
                "loc": tfb.Identity(),
                "scale": tfb.Softplus(),
            },
            "split_indices": [],
        },
        "beta": {
            "names": ["beta"],
            "dist_class": tfd.MultivariateNormalDiag,
            "variational_params": {"loc": jnp.zeros(3), "scale_diag": jnp.ones(3)},
            "fixed_distribution_params": {},
            "optimizer_chain": optax.sgd(1e-1),
            "variational_param_bijectors": {
                "loc": tfb.Identity(),
                "scale_diag": tfb.Softplus(),
            },
            "split_indices": [],
        },
    }
    opt = Optimizer(
        seed=0,
        n_epochs=1,
        S=2,
        model_interface=DummyInterface(params),
        latent_variables=latent,
    )
    expected = set(latent.keys())
    assert expected == set(opt.variational_params.keys())
    assert expected == set(opt.fixed_distribution_params.keys())
    assert expected == set(opt.variational_param_bijectors.keys())
    assert expected == set(opt.variational_dists_class.keys())
    for k in expected:
        vp_keys = set(opt.variational_params[k].keys())
        vb_keys = set(opt.variational_param_bijectors[k].keys())
        assert vp_keys == vb_keys


def test_build_variational_distribution_does_not_mutate_inputs(make_latent_config):
    latent = make_latent_config()
    opt = Optimizer(0, 1, 2, DummyInterface({"z": 0.0}), latent)
    vp_before = {
        k: v.copy() if hasattr(v, "copy") else v
        for k, v in opt.variational_params["z"].items()
    }
    fd_before = dict(opt.fixed_distribution_params["z"])
    vb_before = dict(opt.variational_param_bijectors["z"])
    _ = opt._build_variational_distribution(
        opt.variational_dists_class["z"],
        opt.variational_params["z"],
        opt.fixed_distribution_params["z"],
        opt.variational_param_bijectors["z"],
    )
    assert jnp.allclose(opt.variational_params["z"]["loc"], vp_before["loc"])
    assert jnp.allclose(opt.variational_params["z"]["scale"], vp_before["scale"])
    assert fd_before == opt.fixed_distribution_params["z"]
    assert set(vb_before.keys()) == set(opt.variational_param_bijectors["z"].keys())


def test_multi_block_update_tree_matches_each_block():
    params = {"z": jnp.array(0.0), "beta": jnp.zeros(3)}
    latent = {
        "z": {
            "names": ["z"],
            "dist_class": tfd.Normal,
            "variational_params": {"loc": jnp.array(0.0), "scale": jnp.array(1.0)},
            "fixed_distribution_params": {},
            "optimizer_chain": optax.adam(1e-3),
            "variational_param_bijectors": {
                "loc": tfb.Identity(),
                "scale": tfb.Softplus(),
            },
            "split_indices": [],
        },
        "beta": {
            "names": ["beta"],
            "dist_class": tfd.MultivariateNormalDiag,
            "variational_params": {"loc": jnp.zeros(3), "scale_diag": jnp.ones(3)},
            "fixed_distribution_params": {},
            "optimizer_chain": optax.sgd(1e-1),
            "variational_param_bijectors": {
                "loc": tfb.Identity(),
                "scale_diag": tfb.Softplus(),
            },
            "split_indices": [],
        },
    }
    opt = Optimizer(0, 1, 2, DummyInterface(params), latent)
    grads = {
        "z": {"loc": jnp.array(0.1), "scale": jnp.array(-0.2)},
        "beta": {"loc": jnp.ones(3) * 0.3, "scale_diag": -jnp.ones(3) * 0.4},
    }
    updates, _ = opt.optimizer.update(grads, opt.opt_state, opt.variational_params)
    assert set(updates.keys()) == {"z", "beta"}
    assert set(updates["z"].keys()) == set(opt.variational_params["z"].keys())
    assert set(updates["beta"].keys()) == set(opt.variational_params["beta"].keys())
    for k in updates["beta"]:
        assert updates["beta"][k].shape == opt.variational_params["beta"][k].shape


class TestSampleAndLogProbCompute:
    def test_mean_field_sampling_and_logprob(
        self, mock_model_interface, mean_field_cfg, patch_init_optimizer
    ):
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
        assert set(opt.latent_vars_config.keys()) == {"a", "b"}
        assert jnp.array_equal(samples["a"], jnp.array([0.0, 1.0], dtype=jnp.float32))
        assert jnp.array_equal(samples["b"], jnp.array([0.0], dtype=jnp.float32))
        assert total_log_q == 3.0

    def test_composite_sampling_and_logprob(
        self, mock_model_interface, composite_cfg, patch_init_optimizer
    ):
        mock_model_interface.get_params.return_value = {
            "a": jnp.zeros((2,)),
            "b": jnp.zeros((1,)),
        }
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
        rng = jax.random.PRNGKey(0)
        samples, total_log_q = opt._sample_and_compute_variational_log_prob(
            opt.variational_params, rng
        )
        assert jnp.array_equal(samples["a"], jnp.array([0.0, 1.0]))
        assert jnp.array_equal(samples["b"], jnp.array([2.0]))
        assert jnp.array_equal(total_log_q, 1.5)


class TestElboComputation:
    def test_elbo(
        self,
        latent_cfg,
        mock_model_interface,
        patch_init_optimizer,
        monkeypatch,
    ):
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

        dim_data = 4
        batch_size = 4
        batch_indices = jnp.arange(batch_size)
        rng = jax.random.PRNGKey(0)

        loss, _ = opt._elbo(
            opt.variational_params, rng, dim_data, batch_size, batch_indices, opt.S
        )
        assert jnp.array_equal(loss, -2.0)


class TestGetFinalDistributions:
    def test_mapping_includes_each_name_and_composite_for_multivariate(
        self,
        mock_model_interface,
        composite_cfg,
        patch_init_optimizer,
        monkeypatch,
    ):
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

        calls = []

        def fake_build(dist_class, vparams, fixed, bijectors):
            calls.append({"dist_class": dist_class, "keys": sorted(vparams.keys())})
            return SimpleNamespace(tag="built")

        monkeypatch.setattr(opt, "_build_variational_distribution", fake_build)

        results = opt.get_final_distributions()
        assert set(results.keys()) == {"a", "b", "a_b"}
        assert results["a"] is results["b"] is results["a_b"]
        assert len(calls) == 1

    def test_mapping_for_mean_field_separate_distributions(
        self,
        mock_model_interface,
        mean_field_cfg,
        patch_init_optimizer,
        monkeypatch,
    ):
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

        calls = []

        def fake_build(dist_class, vparams, fixed, bijectors):
            calls.append({"dist_class": dist_class, "keys": sorted(vparams.keys())})
            return SimpleNamespace(tag=f"built_{len(calls)}")

        monkeypatch.setattr(opt, "_build_variational_distribution", fake_build)

        results = opt.get_final_distributions()
        assert set(results.keys()) == {"a", "b"}
        assert results["a"] is not results["b"]
        assert len(calls) == 2


class TestFitLoop:
    def test_fit_runs_with_noop_optimizer_and_records_elbo_and_final_dists(
        self,
        latent_cfg,
        mock_model_interface,
        patch_init_optimizer,
        monkeypatch,
        disable_jit_and_zero_grads,
        lax_loops_as_python,
        optax_noop_apply_updates,
    ):
        opt = Optimizer(
            seed=0,
            n_epochs=2,
            S=1,
            model_interface=mock_model_interface,
            latent_variables=latent_cfg,
            batch_size=None,
            patience_tol=0.0,
            window_size=2,
        )

        elbo_calls = []

        def fake_elbo(p, key, dim_data, batch_size, batch_indices, S):
            elbo_calls.append((tuple(sorted(p.keys())), dim_data, batch_size, int(S)))
            return (jnp.array(1.0, dtype=jnp.float32), key)

        monkeypatch.setattr(opt, "_elbo", fake_elbo)

        final_dist = {"fake": "dist"}
        monkeypatch.setattr(opt, "get_final_distributions", lambda: final_dist)

        opt.fit()
        assert len(elbo_calls) == 2
        assert len(opt.elbo_values) == 2
        assert all(jnp.allclose(v, -1.0) for v in opt.elbo_values)
        assert opt.final_variational_distributions is final_dist
