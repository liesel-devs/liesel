import jax.numpy as jnp
import optax
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
from tensorflow_probability.substrates import jax as tfp

from liesel.experimental.vi.optimizer import Optimizer

tfd = tfp.distributions


# --- Minimal stubs / fixtures -------------------------------------------------


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


def make_latent_config():
    dist_class = tfd.Normal
    variational_params = {"loc": jnp.array(0.0), "scale": jnp.array(1.0)}
    bijectors = {"loc": tfb.Identity(), "scale": tfb.Softplus()}
    fixed = None
    opt_chain = optax.adam(1e-2)
    return {
        "z": {
            "names": ["z"],
            "dist_class": dist_class,
            "variational_params": variational_params,
            "fixed_distribution_params": fixed,
            "optimizer_chain": opt_chain,
            "variational_param_bijectors": bijectors,
            "split_indices": [],
        }
    }


# --- _init_variational_dists_class -------------------------------------------


def test_accepts_tfp_distribution_class():
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
def test_rejects_distribution_instance_not_class():
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
def test_rejects_non_tfp_distribution_class():
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


# --- _init_variational_params -------------------------------------------------


def test_init_variational_params_inverse_bijectors():
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


def test_init_variational_params_missing_bijector_key_raises():
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


def test_init_fixed_params_validate_args_separate_from_variational():
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


def test_init_variational_params_inverse_bijectors_scale_softplus_roundtrip():
    for s in [0.1, 1.0, 3.0]:
        latent = make_latent_config()
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
        back_to_constrained = tfb.Softplus()(uncon["scale"])
        assert jnp.allclose(back_to_constrained, jnp.array(s), rtol=1e-5, atol=1e-6)


def test_unconstrained_values_are_finite_after_init():
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


def test_init_fixed_distribution_params_handles_none():
    latent = make_latent_config()
    opt = Optimizer(
        seed=0,
        n_epochs=1,
        S=2,
        model_interface=DummyInterface({"z": jnp.array(0.0)}),
        latent_variables=latent,
    )
    assert opt.fixed_distribution_params["z"] == {}


def test_init_variational_param_bijectors_mapping():
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


def test_init_optimizer_and_update_step_tree_shapes():
    latent = make_latent_config()
    opt = Optimizer(
        seed=0,
        n_epochs=1,
        S=2,
        model_interface=DummyInterface({"z": jnp.array(0.0)}),
        latent_variables=latent,
    )
    grads = {"z": {"loc": jnp.array(0.0), "scale": jnp.array(0.0)}}
    updates, new_state = opt.optimizer.update(
        grads, opt.opt_state, opt.variational_params
    )
    assert set(updates["z"].keys()) == set(opt.variational_params["z"].keys())


# --- _build_variational_distribution -----------------------------------------


def test_build_applies_bijectors_and_merges_fixed():
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


def test_build_raises_on_invalid_scale_with_validate_args():
    latent = make_latent_config()
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


# --- _init_optimizer (multi_transform) ---------------------------------------


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


def test_zero_gradients_produce_zero_updates_initially():
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


# --- General structure / coherence -------------------------------------------


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
    # Top-level keys must match across all internal dictionaries
    expected = set(latent.keys())
    assert expected == set(opt.variational_params.keys())
    assert expected == set(opt.fixed_distribution_params.keys())
    assert expected == set(opt.variational_param_bijectors.keys())
    assert expected == set(opt.variational_dists_class.keys())
    # Per-block: variational param keys == bijector keys
    for k in expected:
        vp_keys = set(opt.variational_params[k].keys())
        vb_keys = set(opt.variational_param_bijectors[k].keys())
        assert vp_keys == vb_keys


def test_build_variational_distribution_does_not_mutate_inputs():
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

    # Ensure no side effects on the stored (unconstrained) params / maps
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
