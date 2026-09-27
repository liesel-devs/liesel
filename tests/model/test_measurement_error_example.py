"""Check the example's Gibbs conditionals against its actual joint density."""

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def test_measurement_error_conditionals():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/source/tutorials/md/07-error-correction.md"
    )
    blocks = re.findall(
        r"```\{code-cell\} ipython3\n(?:---\n.*?\n---\n)?(.*?)\n```",
        path.read_text(),
        re.DOTALL,
    )
    namespace = {}
    for block in blocks:
        if ".plot(" in block or ").show()" in block:
            continue
        exec(block, namespace)  # noqa: S102 - Execute trusted, checked-in docs.
        if "draw_tau2_x" in namespace:
            break

    model = namespace["model"]
    interface = namespace["interface"]
    assert set(model.observed) == {"y", "w"}
    np.testing.assert_allclose(
        model.log_prob, model.log_lik + model.log_prior, rtol=1e-6
    )
    assert set(model.parameters) == {
        "beta",
        "x",
        "mu_x",
        "tau2_x",
        "log_sigma2_y",
        "log_sigma2_u",
    }

    # Changing x and the conditioning hyperparameter must change the update.
    # Density differences cancel normalization constants independent of the target.
    state = model.update_state(
        {
            "x": model.vars["x"].value + 0.6,
            "mu_x": jnp.asarray(0.2),
            "tau2_x": jnp.asarray(1.7),
        }
    )
    for name, values in [("mu_x", [-0.25, 0.35]), ("tau2_x", [0.5, 1.4])]:
        conditional = namespace[f"{name}_conditional"](state)
        joint = [
            interface.log_prob(interface.update_state({name: jnp.asarray(v)}, state))
            for v in values
        ]
        np.testing.assert_allclose(
            joint[1] - joint[0],
            conditional.log_prob(values[1]) - conditional.log_prob(values[0]),
            rtol=1e-5,
            atol=2e-4,
        )

        draw = jax.jit(namespace[f"draw_{name}"])(jax.random.key(6), state)
        assert set(draw) == {name}
        assert draw[name].shape == ()
        np.testing.assert_allclose(
            draw[name], conditional.sample(seed=jax.random.key(6)), rtol=1e-6
        )
        if name == "tau2_x":
            assert draw[name] > 0
