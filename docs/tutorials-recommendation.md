# Recommendation for the existing tutorials

Status: proposal for review, 26 September 2026. The model and Goose guide edits
are integrated on `docs/model-goose-guides`. The restructuring below has not
been implemented.

This recommendation combines inspection of the current sources with discussion
between the **Liesel Model Guides**, **Liesel Goose Guides**, and
**Plan and improve liesel-GAM guides** sessions. The GAM session contributed
experience migrating notebooks, separating tests from teaching examples, and
checking independently executed documentation.

## Recommended structure

Keep a short learning route, task guides, and a small **Example library**:

- **Tutorials** teach a complete workflow: build a model, sample a posterior,
  combine NUTS and Gibbs, fit and predict, and optimize parameters.
- **Task guides** answer specific questions: transform a parameter, inspect
  results, reproduce a run, or implement a custom kernel.
- **Examples** answer substantive statistical questions using those tools.
  Start with GEV regression and the motorcycle sampler comparison. Add PyMC
  variable selection and measurement-error correction after correctness review.
- **API reference** remains the home for exact contracts and defaults.

An example library should be an ordinary Sphinx index linking to canonical
executable pages. It needs no gallery framework, second implementation, or
separate Python example package. Organize it by statistical problem rather than
legacy numbering. Each page should explain its model, consequential choices,
diagnostics, and interpretation, linking to guides for routine API details.

Keep GEV and motorcycle in Liesel initially. Their main lessons are
support-sensitive modeling and sampler comparison, even though they use GAM.
Liesel should own their complete execution; the GAM documentation owner should
review term, prior, and scale choices and link to the same pages. Relocate a
recipe only if its main lesson becomes GAM-specific construction. This resolves
the Goose session's initial suggestion to move these applications to GAM in
favor of the GAM session's recommendation to retain one Liesel-owned source.

## Page-by-page disposition

Paths below are relative to `docs/source/tutorials`. A decision to retire a
page means replacing useful incoming URLs with short routing pages, after its
unique content has a verified home.

| Existing page | Proposed home and action | Content that must survive |
| --- | --- | --- |
| `md/01a-lin-reg` | Consolidate into `notebooks/11-model-building` and Goose's `md/01c-transform`; retire the duplicate introduction. | Explain that a parameter without a kernel stays fixed. Its variance currently has a prior but is not sampled; avoid silently teaching this as full posterior inference. Keep this distinction in the kernel guide. |
| `md/01b-model` | Keep the existing short routing page to model building; no second full tutorial. | Variable-wise density arrays versus summed likelihood/prior/total, zero contribution without a distribution, variable roles, strong/weak status, and reactive updates are covered in the new model material. |
| `md/01c-transform` | Keep **Sample your first posterior** as the Goose entry tutorial. | Variance prior, bijection, joint NUTS, original-scale results, diagnostics, prediction; preserve its historical transformation anchor. |
| `md/01d-gibbs-sampling` | Keep **Combine NUTS and Gibbs** as the second Goose tutorial. | Derive the full conditional, use current state in the transition, and compare with joint NUTS. Preserve its historical Gibbs anchor. |
| `md/02-ls-reg` | Transfer the independent mean/scale design-matrix lesson, then consolidate into `notebooks/12-model-predictions`. | The old example uses different covariates in `X` and `Z`; the new tutorial uses the same `x`. Add a concise explanation or small recipe, not a second complete fit. Log-scale modeling and separate NUTS blocks already have homes. |
| `md/03-gev` | Migrate to the Example library after execution and support/initialization review. | Three distributional predictors, a smooth effect, parameter-dependent support, and initialization away from the numerically delicate zero-shape case. |
| `md/04-mcycle` | Migrate to the Example library as a carefully framed sampler comparison. | Real data; IWLS/Gibbs versus term-blocked NUTS on the same observations, priors, and constraints. Report diagnostics and define any efficiency metric; do not infer a universal ranking from one run. |
| `md/05-reproducibility` | Keep one reproducibility task guide under Goose, preserving its URL. | Seeds, environment and backend information, and limitations of reproducibility. Verify external claims when revising; execute its seed example with complete setup. |
| `md/06-pymc` | Retain the integration use case, but repair and validate before promoting it as a maintained example. | Spike-and-slab selection, discrete/continuous updates, and an alternate model interface. See the algorithm issue below. |
| `md/07-error-correction` | Retain the scientific use case, but audit the model and Gibbs conditionals before promotion. | Latent covariates, replicated noisy measurements, two observation processes, and posterior correction. See the parameterization issue below. |
| `md/08-custom-kernel` | Keep **Define a custom kernel** as the canonical advanced Goose guide. | Simple MH proposal first, full state/adaptation/Kernel protocol second. Keep the instruction to use built-in kernels for ordinary work. |
| `notebooks/09-liesel-optim-basic`, `10-liesel-optim-advanced` | Keep optimizer-owned tutorials. | First fit and fitting two data groups. Link to them where useful; do not duplicate them in the library. |
| `notebooks/11-model-building`, `12-model-predictions` | Keep model-owned entry tutorials. | Model construction, interpreted diagnostics, fitted quantities and predictive uncertainty. Transfer the independent-covariate lesson from 02 before retiring it. |

When transferring the independent-design example from 02, show that new `X`
and new `Z` describe the same prediction rows. For a plotted slice, state which
other covariate is held fixed. This preserves the prediction consequence as
well as the construction recipe.

The model and Goose tutorials use different data, priors, and variable names.
Their links must make clear when a reader starts a fresh example. In particular,
Goose's short-guide prerequisites use its `X`, `beta`, and `sigma_sq` regression;
they cannot silently reuse the model tutorial's objects.

## Correctness issues found in the old applications

These are source-review findings, not results of freshly executing the legacy
applications. They make automatic format conversion insufficient:

- In `qmd/06-pymc.qmd`, `delta_transition_fn` proposes independent
  Bernoulli draws with probability `theta` and calls `mh_step` without a proposal
  correction. The implementation of `mh_step` defaults `log_correction` to zero.
  For this generally asymmetric independence proposal, the backward/forward
  proposal-density ratio is required. Review the transition against the stated
  target and use an explicit MH implementation or a derived full conditional.
  Also validate the experimental PyMC interface against the chosen environment.
- In `qmd/07-error-correction.qmd`, `tau2_x` and `tau2_mu` enter normal priors
  directly as `scale`, while the displayed Gibbs calculations treat them as
  variances. Reconcile the statistical model and every conditional, rather than
  mechanically adding square roots. The page also uses old `distribution=`,
  transformation, and plotting APIs; review observation roles and intentionally
  fixed hyperparameters during the rewrite.

Do not call these pages maintained simply because Sphinx can display their
committed Markdown and images. If retained during migration, label their
historical status and exclude them from the recommended learning route until
their review is complete. Add a brief note on each affected page naming its
unresolved transition or parameterization issue and linking to the review or
replacement status, so readers arriving through old direct URLs also see it.

## Source, execution, and compatibility

Use executable MyST Markdown for migrated pages, following
[the writing guide](writing-guides.md): native outputs, one inspection or plot
per cell, readable layout, `dist=`, built-in plotting helpers, and `remove-cell`
for build-only prerequisites with visible prose contracts.

Preserve docnames and useful anchors where practical. Remove the corresponding
QMD source when a generated Markdown page becomes maintained MyST; never leave
two render pipelines owning the same page. The combined branch already preserves
the Goose conversion of 01c, 01d, and 08 and their QMD deletions. It also removes
the obsolete Read the Docs artifact overlay: the current workflows have no
matching artifact producer, and extraction into `tutorials/md` could overwrite
native sources. Remaining QMD pages still use the existing render/update-PR
workflow until they are migrated. No dependency or lockfile change is needed.

For each promoted example, record the tested package versions or revisions,
optional dependencies, data provenance, and execution command. Motorcycle
currently requires R/MASS through `ryp`: either maintain and test that setup or
choose a documented Python-accessible dataset with verified provenance and
redistribution terms. GEV and motorcycle require a tested GAM version; PyMC
requires its optional environment. A local `PYTHONPATH` workaround alone is not
evidence that a hosted build can reproduce the example.

Execute maintained examples from independent kernels and fail on errors. Keep
core examples small enough for routine builds. If heavier optional examples need
a separate job, make it an explicit required execution job for their changes,
with a reproducible environment and a visible validation record. Do not blanket
exclude `.ipynb`: 09–12 are maintained executable tutorials today. Retire the
Quarto machinery only after accounting for its last remaining source.

Check scientific outputs as well as exit codes: meaningful posterior behavior,
diagnostics, appropriate scales and uncertainty. Avoid hard-coded exact R-hat,
ESS, or posterior estimates as unit assertions. Move genuine deterministic
contracts to tests only after checking existing coverage; relevant starting
points include `tests/experimental/test_pymc.py`, model distribution and
transformation tests, and Goose transition/kernel tests.

## Suggested implementation order

1. Approve this classification and canonical ownership. The combined guide
   branch is independently useful before any tutorial restructuring.
2. Transfer the small unique lessons from 01a and 02. Replace redundant pages
   with routing stubs, preserve relevant anchors, and update prerequisites and
   navigation.
3. Migrate and review GEV and motorcycle as the first curated examples. Establish
   their optional environments and provenance before advertising them as current.
4. Repair and independently validate PyMC and measurement-error examples, with
   focused deterministic checks for the corrected transition/model contracts.
5. Remove obsolete generated figures and source/build machinery only when no
   surviving page uses them. Build into a clean output directory and audit both
   article and sidebar links, including generated API/viewcode pages.

The GAM migration showed why a clean output build matters: deleting a source did
not remove its old HTML or cached sidebar links. It also showed that successful
execution can reveal poor mixing rather than validate an example's teaching
value. These are separate checks, and both belong in the migration review.
