"""Stable latent indices must survive shuffled data splits and batches."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt


def test_missing_covariate_batches_preserve_rows_and_objective():
    raw = np.array([0.2, np.nan, -0.4, np.nan, 0.7, np.nan, 0.1, np.nan])
    response = jnp.arange(8.0) / 4
    splitter = opt.Split(axis_size=8, validate_axis_size=2, seed=42)
    ids = np.asarray(splitter.indices_train)
    missing = np.sort(ids[np.isnan(raw[ids])])
    base = jnp.asarray(np.where(np.isnan(raw), 0.0, raw))
    lookup = np.full(8, -1, dtype=np.int32)
    lookup[missing] = np.arange(len(missing))
    split = splitter.split_position(
        {
            "row_ids": jnp.arange(8),
            "x_observed": base,
            "missing_index": jnp.asarray(lookup),
            "y": response,
        }
    )
    latent = lsl.Var.new_param(
        jnp.linspace(-0.7, 0.8, len(missing)),
        lsl.Dist(tfd.Normal, 0.0, 1.0),
        name="latent",
    )
    beta = lsl.Var.new_param(1.3, lsl.Dist(tfd.Normal, 0.0, 5.0), name="beta")
    row_ids = lsl.Var.new_obs(split.train["row_ids"], name="row_ids")
    x_observed = lsl.Var.new_obs(split.train["x_observed"], name="x_observed")
    missing_index = lsl.Var.new_obs(split.train["missing_index"], name="missing_index")
    x = lsl.Var.new_calc(
        lambda observed, index, z: jnp.where(
            index >= 0, z[jnp.maximum(index, 0)], observed
        ),
        x_observed,
        missing_index,
        latent,
        name="x",
    )
    mu = lsl.Var.new_calc(lambda b, x: b * x, beta, x)
    y = lsl.Var.new_obs(split.train["y"], lsl.Dist(tfd.Normal, mu, 0.5), name="y")
    model = lsl.Model([y, row_ids])
    params = model.extract_position(["beta", "latent"])
    calc = x.value_node
    assert isinstance(calc, lsl.Calc)
    traced = jax.make_jaxpr(calc.function)(
        split.train["x_observed"], split.train["missing_index"], params["latent"]
    )
    assert not traced.consts  # The batch calculation must not capture full data.
    loss = opt.NegLogProbLoss(model, split, scale=False)
    batches = opt.Batches.from_split(split, batch_size=2)

    def reference(p):
        complete = base.at[missing].set(p["latent"])
        log_lik = tfd.Normal(p["beta"] * complete[ids], 0.5).log_prob(response[ids])
        prior = tfd.Normal(0.0, 1.0).log_prob(p["latent"]).sum()
        prior += tfd.Normal(0.0, 5.0).log_prob(p["beta"])
        return -log_lik.sum() - prior

    expected_loss, expected_grad = jax.value_and_grad(reference)(params)
    for seed in (101, 102):
        batches.start_epoch(jax.random.key(seed))
        values, gradients, seen = [], [], []
        for i in range(len(batches.batch_indices)):
            batch = batches.get_batched_position(split.train, i)
            rows = batch["row_ids"]
            state = model.update_state(params | batch)
            actual_x = model.extract_position(["x"], state)["x"]
            np.testing.assert_allclose(
                actual_x, base.at[missing].set(latent.value)[rows]
            )
            np.testing.assert_array_equal(batch["y"], response[rows])
            seen.extend(np.asarray(rows).tolist())
            carry = SimpleNamespace(
                batch=batch,
                fixed_position={},
                model_state=model.state,
                batches=batches,
                i_batch=i,
            )
            value, grad = jax.value_and_grad(loss.loss_train_batched)(params, carry)
            values.append(value)
            gradients.append(grad)

        assert sorted(seen) == sorted(ids.tolist())
        assert not np.intersect1d(missing, split.validate["row_ids"]).size
        np.testing.assert_allclose(
            jnp.mean(jnp.stack(values)), expected_loss, rtol=2e-5
        )
        for key in params:
            mean_grad = jnp.mean(jnp.stack([g[key] for g in gradients]), axis=0)
            np.testing.assert_allclose(
                mean_grad, expected_grad[key], rtol=2e-5, atol=2e-5
            )
