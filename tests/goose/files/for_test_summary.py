import os.path

import jax
import jax.numpy as jnp
import numpy as np

from liesel.goose.engine import Engine, SamplingResults
from liesel.goose.interface import DictInterface
from liesel.goose.kernel_sequence import KernelSequence
from liesel.goose.pytree import concatenate_leaves, slice_leaves
from liesel.goose.rw import RWKernel
from liesel.goose.warmup import stan_epochs


def log_prob_log_sigma(model_state):
    y = model_state["y"]
    mu = model_state["X"] @ model_state["beta"]
    sigma = jnp.exp(model_state["log_sigma"])

    log_probs = jax.scipy.stats.norm.logpdf(y, mu, sigma)

    return jnp.sum(log_probs)


def setup_tests(beta_dim: int, num_chains: int = 3) -> SamplingResults:
    rng = np.random.default_rng(1337)

    n = 30
    beta = np.ones(beta_dim)
    sigma = 0.1
    X = np.column_stack([np.ones(n), rng.uniform(size=[n, beta_dim - 1])])
    y = rng.normal(X @ beta, sigma, size=n)

    model_state = {"y": y, "X": X, "beta": beta, "log_sigma": np.log(sigma)}

    model = DictInterface(log_prob_log_sigma)
    kernel = RWKernel(["beta", "log_sigma"])
    kernel.set_model(model)
    kernel.identifier = "kernel_01"
    kernels = KernelSequence([kernel])
    epochs = stan_epochs()

    num_chains = num_chains

    model_states = _stack_for_multi([model_state for _ in range(num_chains)])
    seeds = jax.random.split(jax.random.PRNGKey(0), num_chains)

    engine = Engine(
        seeds=seeds,
        model_states=model_states,
        kernel_sequence=kernels,
        epoch_configs=epochs,
        jitted_sample_duration=25,
        model=model,
        position_keys=["beta", "log_sigma"],
    )

    engine.sample_all_epochs()
    return engine.get_results()


# save results in module folder
results_long = setup_tests(beta_dim=5, num_chains=15)
path3 = os.path.join(os.path.dirname(__file__), "summary_viz_res.pkl")
results_long.pkl_save(path3)


## helper functions
def _stack_for_multi(chunks: list):
    chunks = slice_leaves(chunks, jnp.s_[jnp.newaxis, ...])
    return concatenate_leaves(chunks, axis=0)
