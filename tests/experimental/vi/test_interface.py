"""Unit tests for the VI Interface class."""

import copy

import jax
import jax.numpy as jnp
import pytest

from liesel.experimental.vi import LieselInterface

# --- Dummy model & helper to create interface -------------------------------
class DummyVar:
    def __init__(self, value, observed=True):
        self.value = jnp.asarray(value)
        self.observed = observed


class DummyModel:
    def __init__(self, y, theta=0.0, with_unobserved=True):
        self.vars = {
            "y": DummyVar(y, observed=True),
            "theta": DummyVar(jnp.asarray(theta), observed=False),
        }
        if with_unobserved:
            self.vars["u"] = DummyVar(jnp.array([1.0, 2.0, 3.0]), observed=False)
        self.auto_update = True
        self.log_lik = None
        self.log_prior = None
        self.log_prob = None

    def update(self):
        y = self.vars["y"].value
        theta = self.vars["theta"].value
        self.log_lik = -0.5 * jnp.sum((y - theta) ** 2)
        self.log_prior = -0.5 * jnp.sum(theta**2)
        self.log_prob = self.log_lik + self.log_prior


# --- Fixtures -----------------------------------------------------------------
def make_interface(y=jnp.arange(5.0), theta=1.0, with_unobserved=True):
    model = DummyModel(y=y, theta=theta, with_unobserved=with_unobserved)
    return LieselInterface(model), model


# --- get_params --------------------------------------------------------------



def test_get_params_returns_dict():
    interface, _ = make_interface()
    params = interface.get_params()
    assert isinstance(params, dict)
    assert all(isinstance(v, jnp.ndarray) for v in params.values())


def test_get_params_returns_arrays_and_contains_all_vars():
    interface, model = make_interface()
    params = interface.get_params()
    assert set(params.keys()) == set(model.vars.keys())
    assert params["y"].shape == model.vars["y"].value.shape
    assert params["theta"].shape == model.vars["theta"].value.shape


# --- compute_log_prob (full data) -------------------------------------------



def test_compute_log_prob_full_equals_model_update():
    interface, model = make_interface(y=jnp.array([0.0, 1.0, 2.0, 3.0]), theta=1.5)
    model_ref = copy.deepcopy(model)
    model_ref.update()
    ref = model_ref.log_prob

    out = interface.compute_log_prob(
        param_values={"theta": jnp.array(1.5)},
        dim_data=len(model.vars["y"].value),
    )
    assert isinstance(out, jnp.ndarray)
    assert jnp.allclose(out, ref)


def test_compute_log_prob_raises_on_unknown_param():
    interface, _ = make_interface()
    with pytest.raises(KeyError, match="not part of the model"):
        interface.compute_log_prob({"does_not_exist": jnp.array(0.0)}, dim_data=5)


# --- compute_log_prob (batching semantics) ----------------------------------



def test_compute_log_prob_batch_scales_likelihood_only():
    y = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    theta = 1.0
    N = len(y)
    batch_idx = jnp.array([1, 3, 4])

    interface, _ = make_interface(y=y, theta=theta)
    got = interface.compute_log_prob(
        param_values={"theta": jnp.array(theta)},
        dim_data=N,
        batch_size=len(batch_idx),
        batch_indices=batch_idx,
    )

    y_b = y[batch_idx]
    ll_batch = -0.5 * jnp.sum((y_b - theta) ** 2)
    expected = (N / len(batch_idx)) * ll_batch + (-0.5 * theta**2)
    assert jnp.allclose(got, expected)


def test_subset_applies_only_to_observed_vars_not_params():
    y = jnp.arange(10.0)
    theta = 0.0
    interface, model = make_interface(y=y, theta=theta, with_unobserved=True)
    model_before = copy.deepcopy(model)

    _ = interface.compute_log_prob(
        param_values={"theta": jnp.array(theta)},
        dim_data=len(y),
        batch_size=4,
        batch_indices=[2, 5, 6, 7],
    )

    assert jnp.allclose(model.vars["y"].value, model_before.vars["y"].value)
    assert jnp.allclose(model.vars["theta"].value, model_before.vars["theta"].value)
    if "u" in model.vars:
        assert jnp.allclose(model.vars["u"].value, model_before.vars["u"].value)


def test_compute_log_prob_updates_given_params_only():
    interface, _ = make_interface(y=jnp.array([1.0, 2.0, 3.0]), theta=0.0)
    out0 = interface.compute_log_prob({"theta": jnp.array(0.0)}, dim_data=3)
    out1 = interface.compute_log_prob({"theta": jnp.array(1.0)}, dim_data=3)
    assert out0 != out1


def test_batch_path_respects_auto_update_flag_on_copy():
    interface, model = make_interface()
    _ = interface.compute_log_prob(
        {"theta": model.vars["theta"].value},
        dim_data=len(model.vars["y"].value),
        batch_size=2,
        batch_indices=[0, 1],
    )
    assert model.auto_update is True


def test_batch_indices_accept_flat_list():
    y = jnp.arange(6.0)
    interface, _ = make_interface(y=y, theta=0.0)
    idx = [0, 2, 5]
    out = interface.compute_log_prob(
        {"theta": jnp.array(0.0)},
        dim_data=len(y),
        batch_size=len(idx),
        batch_indices=idx,
    )
    assert isinstance(out, jnp.ndarray)


# --- _subset_data unit tests -------------------------------------------------



def test_subset_data_flat_model_2d_observed_rows_in_place():
    y2d = jnp.arange(5 * 3, dtype=float).reshape(5, 3)
    interface, model = make_interface(y=y2d, theta=0.0, with_unobserved=True)

    y_before = model.vars["y"].value.copy()
    theta_before = model.vars["theta"].value.copy()
    u_before = model.vars["u"].value.copy()

    idx = [0, 4, 2]
    result = interface._subset_data(model, batch_indices=idx)

    assert result is model
    assert model.vars["y"].value.shape == (len(idx), 3)
    assert jnp.allclose(model.vars["y"].value, y_before[jnp.array(idx), :])
    assert jnp.allclose(model.vars["theta"].value, theta_before)
    assert jnp.allclose(model.vars["u"].value, u_before)


def test_subset_data_empty_indices_yields_empty_first_axis():
    interface, model = make_interface(
        y=jnp.arange(6.0), theta=0.0, with_unobserved=True
    )
    theta_before = model.vars["theta"].value.copy()
    u_before = model.vars["u"].value.copy()

    interface._subset_data(model, batch_indices=[])

    assert model.vars["y"].value.shape == (0,)
    assert jnp.allclose(model.vars["theta"].value, theta_before)
    assert jnp.allclose(model.vars["u"].value, u_before)


def test_subset_data_flat_model_1d_2d_observed_only_in_place():
    interface, model = make_interface(
        y=jnp.array([10.0, 20.0, 30.0]), theta=1.5, with_unobserved=True
    )
    model.vars["theta_test"] = DummyVar(
        jnp.array([[1, 9, 1], [1, 9, 1], [1, 9, 1]]), observed=True
    )

    y_before = model.vars["y"].value.copy()
    theta_before = model.vars["theta"].value.copy()
    u_before = model.vars["u"].value.copy()
    theta_before_test = model.vars["theta_test"].value.copy()

    result = interface._subset_data(model, batch_indices=[0, 2])

    assert result is model
    assert jnp.allclose(model.vars["y"].value, y_before[jnp.array([0, 2])])
    assert jnp.allclose(
        model.vars["theta_test"].value, theta_before_test[jnp.array([0, 2])]
    )
    assert jnp.allclose(model.vars["theta"].value, theta_before)
    assert jnp.allclose(model.vars["u"].value, u_before)


def test_subset_data_accepts_tensor_observed_with_leading_batch_axis():
    y = jnp.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3)
    interface, model = make_interface(y=y, theta=0.0, with_unobserved=False)

    y_before = model.vars["y"].value.copy()
    idx = [0, 4, 2]

    result = interface._subset_data(model, batch_indices=idx)

    assert result is model
    assert model.vars["y"].value.shape == (len(idx), 2, 3)
    expected = y_before[jnp.array(idx), ...]
    assert jnp.allclose(model.vars["y"].value, expected)


# --- Helper to mirror batch_step index windows ------------------------------



def collect_batch_indices(dim_data: int, batch_size: int, key: jax.Array):
    _, perm_key = jax.random.split(key)
    all_indices = jax.random.permutation(perm_key, dim_data)
    number_batches = dim_data // batch_size
    batches = []
    for i in range(number_batches):
        start = i * batch_size
        idx = jax.lax.dynamic_slice(all_indices, (start,), (batch_size,))
        batches.append(idx)
    return all_indices, batches


# --- Applying batch_step indices to flat models -----------------------------



def test_batch_step_indices_work_with_flat_1d_2d_and_higherD_observed():
    N, B = 9, 3
    key = jax.random.PRNGKey(2)
    _, batches = collect_batch_indices(N, B, key)

    for idx in batches:
        interface, m1 = make_interface(
            y=jnp.arange(float(N)), theta=0.0, with_unobserved=False
        )
        y_before = m1.vars["y"].value.copy()
        interface._subset_data(m1, idx)
        assert m1.vars["y"].value.shape == (idx.size,)
        assert jnp.allclose(m1.vars["y"].value, y_before[idx, ...])

        interface, m2 = make_interface(
            y=jnp.arange(float(N * 3)).reshape(N, 3), theta=0.0, with_unobserved=False
        )
        Y2_before = m2.vars["y"].value.copy()
        interface._subset_data(m2, idx)
        assert m2.vars["y"].value.shape == (idx.size, 3)
        assert jnp.allclose(m2.vars["y"].value, Y2_before[idx, ...])

        y3d = jnp.arange(float(N * 2 * 4)).reshape(N, 2, 4)
        interface, m3 = make_interface(y=y3d, theta=0.0, with_unobserved=False)
        Y3_before = m3.vars["y"].value.copy()
        interface._subset_data(m3, idx)
        assert m3.vars["y"].value.shape == (idx.size, 2, 4)
        assert jnp.allclose(m3.vars["y"].value, Y3_before[idx, ...])


def test_subset_data_slices_all_observed_consistently():
    N = 6
    y = jnp.arange(float(N * 2)).reshape(N, 2)
    interface, model = make_interface(y=y, theta=0.0, with_unobserved=False)
    model.vars["x"] = DummyVar(jnp.arange(float(N * 3)).reshape(N, 3), observed=True)

    y0 = model.vars["y"].value.copy()
    x0 = model.vars["x"].value.copy()
    idx = jnp.array([1, 4, 5])

    interface._subset_data(model, batch_indices=idx)

    assert model.vars["y"].value.shape == (idx.size, 2)
    assert model.vars["x"].value.shape == (idx.size, 3)
    assert jnp.allclose(model.vars["y"].value, y0[idx, :])
    assert jnp.allclose(model.vars["x"].value, x0[idx, :])
