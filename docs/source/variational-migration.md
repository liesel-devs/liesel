---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
mystnb:
  execution_mode: force
  execution_raise_on_error: true
---

# Update VI code

The unreleased VI API now makes the following defaults and calling conventions
explicit. These changes can affect optimization paths and fitted uncertainty.

| Setting | Earlier behavior | Current behavior |
| --- | --- | --- |
| Gaussian initial SD | `0.01` | `0.1`; set the scale explicitly to retain the old initialization |
| Implicit VI stopper | 1,000 epochs, patience 10 | Fixed 1,000-epoch budget; pass a stopper to enable early stopping |
| Fitted approximation | Minimum-monitor position | Final position; use `at="min_monitor"` to select the old position |
| Multiple observation sizes | Automatic grouping in `LieselVI` | Explicit split required |
| `estimate_elbo` target state | Required `p_state` | Uses the target model's current state when omitted |
| Builder sampling | `vdist.sample(key, (100,))` | `vdist.sample(100, seed=key)` |
| Priors on q parameters | Included as penalties | Excluded unless `regularize_q_prior=True` |

The initial SD is a heuristic in model units, not a universal improvement over
`0.01`. See {ref}`vi-initial-scale`. The default ordinary ELBO still includes
**target-model priors**; the changed flag concerns priors on fixed optimization
parameters in the variational model. See {ref}`vi-q-prior-penalties` for an
executable comparison and instructions for opting in.

## Set initialization explicitly

Use the same model and scale when reproducing an earlier fit. Here the explicit
`scale_diag=0.01` retains the previous Gaussian initialization; omitting it now
uses `0.1`. `scale_tril=0.01` does the same for dense builders, and `scale=0.01`
for {meth}`~liesel.optim.VDist.normal`.

```{code-cell} python
import jax
import jax.numpy as jnp
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt

loc = lsl.Var.new_param(0.0, name="loc")
y = lsl.Var.new_obs(jnp.array([1.0, 2.0]), lsl.Dist(tfd.Normal, loc, 1.0), name="y")
model = lsl.Model(y)
vdist = opt.VDist(["loc"], model).mvn_diag(scale_diag=0.01).build()
loss = opt.NegElboLoss.from_vdist(vdist)
```

## Update sampling calls

Both {class}`~liesel.optim.VDist` and {class}`~liesel.optim.CompositeVDist` now
use the fitted approximation's convention: sample shape first, required keyword
`seed`. Integer counts and tuples work; omitting the shape returns one draw in
the original parameter shapes. `at_position` is also keyword-only.

```{code-cell} python
key = jax.random.key(42)
draws = vdist.sample(100, seed=key)
```

```{code-cell} python
pd.DataFrame(draws).agg(["mean", "std"]).round(4)
```

The old positional seed calls raise `TypeError`; there is no ambiguous alias.
When sampling fitted values directly, pass `at_position=result.position_final`,
or bind them with `loss.approximate_joint_posterior(result)`.

## Choose grouping and stopping

For multiple observation sizes, construct
`split=opt.PositionSplit.from_model(model, multi_size="manager")` explicitly
after checking row alignment. Use `split_axes={key: None}` for shared values and
nested `position_keys` for explicit groups. See {doc}`optimizer-splitting`.

To retain the earlier early-stopping rule, pass
`stopper=opt.Stopper(epochs=1000, patience=10, rtol=1e-6)` to `LieselVI`.
Monte Carlo noise can trigger early stopping or a low monitoring minimum before
the family stabilizes. The new fixed budget and final-iterate selection avoid
those automatic decisions, but do not certify convergence.
