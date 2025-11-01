import jax.numpy as jnp

from ...model import Model


def guess_n(model: Model, axis: int = 0) -> int:
    obs = list(model.observed.values())
    obs_ndim = [jnp.asarray(o.value).ndim for o in obs]
    min_ndim = min(obs_ndim)

    obs_shapes = [jnp.shape(o.value) for o in obs]

    dims = []
    for j in range(min_ndim):
        dims.append([s[j] for s in obs_shapes])

    potential_ns = []
    for dim in dims:
        if len(set(dim)) == 1:
            n = dim[0]
            potential_ns.append(n)
            return n

    if not potential_ns:
        raise RuntimeError("Failed to guess sample size.")

    return potential_ns[axis]
