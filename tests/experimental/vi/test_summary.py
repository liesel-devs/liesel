import os
import tempfile

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel.experimental.vi import Summary


# --- Fixtures & helpers ----------------------------------------------


class DummyResults:
    def __init__(self):
        seed = jax.random.PRNGKey(0)
        key, subkey1, subkey2 = jax.random.split(seed, 3)

        dummy_dist = tfd.MultivariateNormalFullCovariance(
            loc=jnp.ones(4),
            covariance_matrix=jnp.diag(jnp.ones(4)),
        )

        dummy_dist2 = tfd.InverseGamma(concentration=1.0, scale=1.0)

        self.final_variational_distributions = {"b": dummy_dist, "sigma_sq": dummy_dist2}

        noise = tfd.Uniform(low=-1, high=1).sample(1000, seed=subkey2)
        self.elbo_values = -jnp.square(jnp.arange(11, 1, -0.01)) + noise

        self.samples = {
            "b": dummy_dist.sample(seed=key, sample_shape=(1000,)),
            "sigma_sq": dummy_dist2.sample(seed=subkey1, sample_shape=(1000,)),
        }

    def to_dict(self):
        return {
            "final_variational_distributions": self.final_variational_distributions,
            "elbo_values": self.elbo_values,
            "samples": self.samples,
        }


@pytest.fixture(autouse=True)
def no_plot(monkeypatch):
    """Global fixture to suppress plt.show() during tests."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


@pytest.fixture
def summary():
    """Return a Summary instance built from DummyResults."""
    results = DummyResults().to_dict()
    return Summary(results)


def sample_cov(X):
    """Compute sample covariance matrix (unbiased) for X with shape [n_samples, n_features]."""
    X_centered = X - jnp.mean(X, axis=0)
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    return cov


# --- Tests compute_posterior_summary ---------------------------------


def test_summary_stats_match_theoretical_distributions(summary):
    df = summary.compute_posterior_summary(hdi_prob=0.9)

    mvn = summary.final_variational_distributions["b"]
    mu = jnp.array(mvn.loc)
    var = jnp.diag(jnp.array(mvn.covariance()))

    normal_dist = tfd.Normal(loc=1.0, scale=1.0)

    for idx in range(len(mu)):
        row = df[df["variable"] == f"b[{idx}]"].iloc[0]

        assert jnp.isclose(row["mean"], mu[idx], rtol=0.1)
        assert jnp.isclose(row["variance"], var[idx], rtol=0.2)

        q025 = normal_dist.quantile(0.025)
        q975 = normal_dist.quantile(0.975)
        assert jnp.isclose(row["2.5%"], jnp.array(q025), rtol=0.2)
        assert jnp.isclose(row["97.5%"], jnp.array(q975), rtol=0.2)

    invgamma = summary.final_variational_distributions["sigma_sq"]
    row = df[df["variable"] == "sigma_sq"].iloc[0]

    q025 = invgamma.quantile(0.025)
    q975 = invgamma.quantile(0.975)

    assert jnp.isclose(row["2.5%"], jnp.array(q025), rtol=0.2, atol=0.2)
    assert jnp.isclose(row["97.5%"], jnp.array(q975), rtol=0.2, atol=0.2)

    median = invgamma.quantile(0.5)
    assert row["hdi_low"] < median < row["hdi_high"]


def test_posterior_summary_structure(summary):
    df = summary.compute_posterior_summary()
    expected_cols = ["variable", "mean", "variance", "2.5%", "97.5%", "hdi_low", "hdi_high"]
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in expected_cols)
    assert len(df) > 0


def test_posterior_summary_scalar_and_vector(summary):
    df = summary.compute_posterior_summary()
    sigma_rows = df[df["variable"] == "sigma_sq"]
    assert len(sigma_rows) == 1
    b_rows = df[df["variable"].str.startswith("b[")]
    assert len(b_rows) == 4


def test_posterior_summary_invalid_shape(summary):
    summary.samples["bad"] = jnp.ones((100, 2, 2))
    with pytest.raises(ValueError):
        summary.compute_posterior_summary()


def test_posterior_variance_matches_sample_cov(summary):
    """Compare the variance column from the summary to the empirical covariance diagonal."""
    df = summary.compute_posterior_summary()
    samples = summary.samples["b"]  
    cov_emp = sample_cov(samples)
    var_from_summary = df[df["variable"].str.startswith("b[")]["variance"].values
    diag_cov = jnp.diag(jnp.asarray(cov_emp))
    assert jnp.allclose(var_from_summary, diag_cov, rtol=0.2)


# --- Tests plot_elbo -------------------------------------------------


def test_plot_elbo_runs(summary):
    summary.plot_elbo()


def test_plot_elbo_saves_file(summary):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "elbo.png")
        summary.plot_elbo(save_path=path)
        assert os.path.exists(path)


# --- Tests plot_density ----------------------------------------------


def test_plot_density_scalar(summary):
    summary.plot_density("sigma_sq")


def test_plot_density_vector(summary):
    summary.plot_density("b")


def test_plot_density_invalid_variable(summary):
    with pytest.raises(ValueError):
        summary.plot_density("not_a_var")


def test_plot_density_invalid_dim(summary):
    summary.samples["bad"] = jnp.ones((10, 2, 2, 2))
    with pytest.raises(ValueError):
        summary.plot_density("bad")


def test_plot_density_saves_file(summary):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "density.png")
        summary.plot_density("b", save_path=path)
        assert os.path.exists(path)


# --- Tests plot_pairwise ---------------------------------------------


def test_plot_pairwise_vector(summary):
    summary.plot_pairwise("b")


def test_plot_pairwise_scalar(summary):
    summary.plot_pairwise("sigma_sq")


def test_plot_pairwise_invalid_dim(summary):
    summary.samples["bad"] = jnp.ones((10, 2, 2))
    with pytest.raises(ValueError):
        summary.plot_pairwise("bad")


def test_plot_pairwise_saves_file(summary):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "pairwise.png")
        summary.plot_pairwise("b", save_path=path)
        assert os.path.exists(path)


# --- Tests: string representations -----------------------------------


def test_str_and_repr(summary):
    s = str(summary)
    r = repr(summary)
    assert "variable" in s
    assert "variable" in r
