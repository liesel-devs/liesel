# Migrating IWLS and Langevin kernels in 0.6

**`IWLSKernel` changes its proposal in 0.6.0.** Existing calls may still run, but
seeded trajectories, acceptance rates, and sampling efficiency can change. This is
a direct breaking release under Liesel's effort-based versioning policy; there is
no intermediate deprecation period or proposal-mode switch.

## Choose the intended proposal

Let $g$ be the block gradient of the log posterior, $P$ a local positive-definite
precision, and $s$ the Langevin step size.

| Kernel | Proposal mean | Proposal covariance | Defaults |
|---|---|---|---|
| `IWLSKernel` | $\beta + P^{-1}g$ | $P^{-1}$ | No scaling or adaptation |
| `MALAKernel` | $\beta + \tfrac{s^2}{2}g$ | $s^2 I$ | Automatic initial step, adaptation, target 0.574 |
| `SMMALAKernel` | $\beta + \tfrac{s^2}{2}P^{-1}g$ | $s^2 P^{-1}$ | Automatic initial step, adaptation, target 0.8 |

All three use Metropolis-Hastings correction, evaluating the actual forward and
reverse proposal densities. `SMMALAKernel` implements **simplified** manifold MALA,
without metric-derivative drift terms. Full manifold MALA/PMALA is not included.

For IWLS and SMMALA, the default precision is the negative block Hessian of the
log posterior, including prior contributions. It is not generally expected Fisher
information. The existing jitter, `1e-6 * mean(diag(P)) * I`, and fallback choices
are retained. Identity remains the default fallback; transition diagnostics report
its use at the current position. Eigenvalue clipping and no fallback remain options.
With no fallback, an invalid proposal is rejected and reported in diagnostics.

For a Gaussian full conditional, IWLS with its **exact precision** produces the
conditional draw up to numerical error. Default jitter perturbs that precision;
it does not guarantee exact Gibbs draws or acceptance one. Even a small jitter
coefficient can materially change poorly conditioned directions. A custom `chol_info_fn`
can supply the exact lower Cholesky factor without added jitter. Keep the MH
correction for regularized or non-Gaussian proposals.

## Preserve the previous algorithm

Replace the class and explicitly preserve the previous initial step:

```python
import liesel.goose as gs

# Before 0.6:
# kernel = gs.IWLSKernel(["beta"])

kernel = gs.SMMALAKernel(["beta"], initial_step_size=0.01)
```

If you supplied a different initial step, a custom precision, a fallback, or tuning
settings, keep those values. With the same settings and software environment,
SMMALA preserves the old seeded transitions and adaptation. Its constructor retains
the old positional argument order. Automatic initialization is intentionally the
new default, so a class-name replacement alone does not reproduce the old run.

For the previous `.untuned()` proposal:

```python
kernel = gs.SMMALAKernel.untuned(["beta"])
```

This fixes the step at one and disables both initial-step search and adaptation.

The equivalent `MCMCSpec` migration is:

```python
inference = gs.MCMCSpec(gs.SMMALAKernel, kernel_kwargs={"initial_step_size": 0.01})
# For the old fixed unit-step proposal:
untuned_inference = gs.MCMCSpec(gs.SMMALAKernel.untuned)
```

Both classes can also be imported from `liesel.goose.mala`.

## Adopt corrected IWLS

Keep `gs.IWLSKernel` and remove any step-size or dual-averaging arguments:

```python
kernel = gs.IWLSKernel(["beta"])
inference = gs.MCMCSpec(gs.IWLSKernel)
```

`IWLSKernel.untuned()` remains a compatibility convenience and now produces the same
proposal as the ordinary constructor. Neither constructor adapts during warmup.
Removed arguments such as `initial_step_size`, `da_tune_step_size`, and
`da_target_accept` raise `TypeError`; they are not silently ignored. Arguments after
`position_keys` and `chol_info_fn` are keyword-only.

Liesel-GAM defaults that select `IWLSKernel`, including `.untuned()`, automatically
adopt corrected IWLS. The same applies to Liesel's `distreg` defaults. Review
sampling diagnostics for your models after upgrading; the proposal change does not
imply that posterior estimates from the old, MH-corrected sampler were invalid.

## Initial steps and adaptation

For MALA and SMMALA, `initial_step_size=None` searches for a starting step using
the actual kernel proposal and its acceptance target. Search evaluates proposals
from the initial position without advancing the chain. A positive numeric value
bypasses search. `da_tune_step_size=False` keeps the selected initial scale fixed;
otherwise dual averaging adapts it during warmup.

The public step size is a standard-deviation multiplier: covariance is $s^2P^{-1}$
(or $s^2I$ for MALA). MALA's 0.574 acceptance target is a heuristic from the
[asymptotic scaling result](https://doi.org/10.1111/1467-9868.00123), not a universal
optimum. SMMALA retains its previous target of 0.8. Both targets can be overridden.

MALA uses only gradients and identity precision. For a fixed preconditioner, pass
a constant `chol_info_fn` to SMMALA.

## Saved results

Previously saved `SamplingResults` can still be loaded and summarized, including
legacy kernel states and error codes. Historical metadata naming `IWLSKernel`
describes the pre-0.6 Langevin algorithm, even though that class name now selects
corrected IWLS. This compatibility does not cover resuming serialized engines
across versions.
