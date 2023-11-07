import tempfile

import jax.numpy as jnp
import jax.random as rnd
import matplotlib
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel.model.model import Model
from liesel.model.nodes import Calc, Dist, Var
from liesel.model.viz import plot_nodes, plot_vars

matplotlib.use("template")
key = rnd.PRNGKey(13)

n = 500
true_beta = jnp.array([1.0, 2.0])
true_sigma = 1.0

key_x, key_y = rnd.split(key, 2)
x0 = tfd.Uniform().sample(seed=key_x, sample_shape=n)
x = jnp.column_stack([jnp.ones(n), x0])

y = tfd.Normal(loc=x @ true_beta, scale=true_sigma).sample(seed=key_y)

beta_loc = Var(0.0, name="beta_loc")
beta_scale = Var(100.0, name="beta_scale")
beta_prior = Dist(tfd.Normal, loc=beta_loc, scale=beta_scale)
beta = Var(jnp.array([0.0, 0.0]), distribution=beta_prior, name="beta")

sigma_concentration = Var(0.01, name="sigma_concentration")
sigma_scale = Var(0.01, name="sigma_scale")

sigma_prior = Dist(
    tfd.InverseGamma,
    concentration=sigma_concentration,
    scale=sigma_scale,
)

sigma = Var(10.0, distribution=sigma_prior, name="sigma")

x = Var(x, name="x")
y_loc = Var(Calc(lambda x, beta: x @ beta, x, beta), name="mu")
likelihood = Dist(tfd.Normal, loc=y_loc, scale=sigma)
y = Var(y, distribution=likelihood, name="y")

model = Model([y])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test plot_nodes() ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes() -> None:
    fp = tempfile.TemporaryFile()
    plot_nodes(model, save_path=fp)
    matplotlib.pyplot.close()
    fp.close()


def test_plot_nodes_negative_width() -> None:
    fp = tempfile.TemporaryFile()

    with pytest.raises(ValueError):
        plot_nodes(model, width=-1, save_path=fp)

    matplotlib.pyplot.close()
    fp.close()


def test_plot_nodes_negative_height() -> None:
    fp = tempfile.TemporaryFile()

    with pytest.raises(ValueError):
        plot_nodes(model, height=-1, save_path=fp)

    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_circo_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_nodes(model, save_path=fp, prog="circo")
    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_dot_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_nodes(model, save_path=fp, prog="dot")
    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_fdp_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_nodes(model, save_path=fp, prog="fdp")
    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_neato_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_nodes(model, save_path=fp, prog="neato")
    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_osage_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_nodes(model, save_path=fp, prog="osage")
    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_patchwork_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_nodes(model, save_path=fp, prog="patchwork")
    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_twopi_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_nodes(model, save_path=fp, prog="twopi")
    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_sfdp_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_nodes(model, save_path=fp, prog="sfdp")
    matplotlib.pyplot.close()
    fp.close()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test plot_vars() ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars() -> None:
    fp = tempfile.TemporaryFile()
    plot_vars(model, save_path=fp)
    matplotlib.pyplot.close()
    fp.close()


def test_plot_vars_negative_width() -> None:
    fp = tempfile.TemporaryFile()

    with pytest.raises(ValueError):
        plot_vars(model, width=-1, save_path=fp)

    matplotlib.pyplot.close()
    fp.close()


def test_plot_vars_negative_height() -> None:
    fp = tempfile.TemporaryFile()

    with pytest.raises(ValueError):
        plot_vars(model, height=-1, save_path=fp)

    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_circo_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_vars(model, save_path=fp, prog="circo")
    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_dot_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_vars(model, save_path=fp, prog="dot")
    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_fdp_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_vars(model, save_path=fp, prog="fdp")
    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_neato_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_vars(model, save_path=fp, prog="neato")
    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_osage_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_vars(model, save_path=fp, prog="osage")
    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_patchwork_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_vars(model, save_path=fp, prog="patchwork")
    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_twopi_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_vars(model, save_path=fp, prog="twopi")
    matplotlib.pyplot.close()
    fp.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_sfdp_prog() -> None:
    fp = tempfile.TemporaryFile()
    plot_vars(model, save_path=fp, prog="sfdp")
    matplotlib.pyplot.close()
    fp.close()
