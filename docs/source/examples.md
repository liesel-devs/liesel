# Example library

Use these examples to follow a statistical problem from its assumptions to
posterior interpretation. For the software basics, start with
{doc}`model-building`, {doc}`sampling`, or {doc}`optimization`.

```{toctree}
:maxdepth: 1

GEV regression <tutorials/md/03-gev>
Compare samplers <tutorials/md/04-mcycle>
Variable selection with PyMC <tutorials/md/06-pymc>
Correct measurement error <tutorials/md/07-error-correction>
```

| Example | Main question | Additional requirements |
| --- | --- | --- |
| GEV regression | How can location, scale, and shape vary with covariates while respecting response support? | Liesel-GAM |
| Compare samplers | How do IWLS/Gibbs and NUTS/Gibbs behave on the same motorcycle regression model? | Liesel-GAM; bundled data with provenance |
| Variable selection with PyMC | How can Goose combine discrete indicator updates with continuous posterior sampling? | PyMC and its experimental Liesel interface |
| Correct measurement error | How can repeated noisy covariate measurements inform a latent-covariate regression? | Core Liesel |

Each page contains its own setup and executes during the documentation build.
The GEV and motorcycle examples are maintained here; follow their Liesel-GAM
links for details of constructing additive terms. Reuse the canonical examples
instead of maintaining a second copy in another package.

For a reproducible development environment and build command, see the
{download}`documentation build instructions <../README.md>`.
