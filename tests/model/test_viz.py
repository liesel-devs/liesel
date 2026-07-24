import tempfile
from collections.abc import Generator
from typing import BinaryIO

import jax.numpy as jnp
import jax.random as rnd
import matplotlib
import matplotlib.pyplot as plt
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel.model.model import Model
from liesel.model.nodes import Dist, Var
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
y_loc = Var.new_calc(lambda x, beta: x @ beta, x, beta, name="mu")
likelihood = Dist(tfd.Normal, loc=y_loc, scale=sigma)
y = Var(y, distribution=likelihood, name="y")

model = Model([y])


@pytest.fixture
def temp_file() -> Generator[BinaryIO]:
    with tempfile.TemporaryFile() as fp:
        yield fp


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test plot_nodes() ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes(temp_file: BinaryIO) -> None:
    plot_nodes(model, save_path=temp_file)
    plt.close()


def test_plot_nodes_negative_width(temp_file: BinaryIO) -> None:
    with pytest.raises(ValueError):
        plot_nodes(model, width=-1, save_path=temp_file)

    plt.close()


def test_plot_nodes_negative_height(temp_file: BinaryIO) -> None:
    with pytest.raises(ValueError):
        plot_nodes(model, height=-1, save_path=temp_file)

    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_circo_prog(temp_file: BinaryIO) -> None:
    plot_nodes(model, save_path=temp_file, prog="circo")
    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_dot_prog(temp_file: BinaryIO) -> None:
    plot_nodes(model, save_path=temp_file, prog="dot")
    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_fdp_prog(temp_file: BinaryIO) -> None:
    plot_nodes(model, save_path=temp_file, prog="fdp")
    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_neato_prog(temp_file: BinaryIO) -> None:
    plot_nodes(model, save_path=temp_file, prog="neato")
    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_osage_prog(temp_file: BinaryIO) -> None:
    plot_nodes(model, save_path=temp_file, prog="osage")
    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_patchwork_prog(temp_file: BinaryIO) -> None:
    plot_nodes(model, save_path=temp_file, prog="patchwork")
    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_twopi_prog(temp_file: BinaryIO) -> None:
    plot_nodes(model, save_path=temp_file, prog="twopi")
    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_nodes_sfdp_prog(temp_file: BinaryIO) -> None:
    plot_nodes(model, save_path=temp_file, prog="sfdp")
    plt.close()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test plot_vars() ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars(temp_file: BinaryIO) -> None:
    plot_vars(model, save_path=temp_file)
    plt.close()


def test_plot_vars_negative_width(temp_file: BinaryIO) -> None:
    with pytest.raises(ValueError):
        plot_vars(model, width=-1, save_path=temp_file)

    plt.close()


def test_plot_vars_negative_height(temp_file: BinaryIO) -> None:
    with pytest.raises(ValueError):
        plot_vars(model, height=-1, save_path=temp_file)

    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_circo_prog(temp_file: BinaryIO) -> None:
    plot_vars(model, save_path=temp_file, prog="circo")
    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_dot_prog(temp_file: BinaryIO) -> None:
    plot_vars(model, save_path=temp_file, prog="dot")
    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_fdp_prog(temp_file: BinaryIO) -> None:
    plot_vars(model, save_path=temp_file, prog="fdp")
    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_neato_prog(temp_file: BinaryIO) -> None:
    plot_vars(model, save_path=temp_file, prog="neato")
    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_osage_prog(temp_file: BinaryIO) -> None:
    plot_vars(model, save_path=temp_file, prog="osage")
    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_patchwork_prog(temp_file: BinaryIO) -> None:
    plot_vars(model, save_path=temp_file, prog="patchwork")
    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_twopi_prog(temp_file: BinaryIO) -> None:
    plot_vars(model, save_path=temp_file, prog="twopi")
    plt.close()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_vars_sfdp_prog(temp_file: BinaryIO) -> None:
    plot_vars(model, save_path=temp_file, prog="sfdp")
    plt.close()
