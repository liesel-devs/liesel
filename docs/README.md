# Build and review the documentation

The tutorials, task guides, and example library execute during Sphinx builds.
Use the repository's Python version and lockfile, including the optional PyMC
integration and documentation dependencies:

```sh
uv sync --locked --group dev --group pymc
uv run python -m sphinx -E -b html -j 1 -W --keep-going docs/source docs/build/html
```

Install Graphviz on the system (`dot -V` should work). Python dependencies come
from `uv.lock`; do not substitute a separately installed Liesel or an unpublished
GAM checkout. The development dependency requires released Liesel-GAM 0.2.4 or
later; the lockfile pins the exact tested GAM and smoothcon versions. The PyMC
example uses the locked PyMC/PyTensor pair and explicitly enables JAX 64-bit mode
in its own kernel. Other pages use the default precision unless stated.

The motorcycle data are bundled with provenance and license in
`source/tutorials/md/data/`. Builds require neither R nor a runtime dataset
download. All other application data are simulated with fixed seeds. See the
example pages for assumptions and the statistical quantities being estimated.

Use a new output directory when removing or renaming pages, for example
`docs/build/review-html`: `-E` rebuilds Sphinx's environment but does not delete
stale HTML from an old output directory. Check both article and sidebar links,
including generated API/source pages. Do not set `MPLBACKEND=Agg` when executing
notebooks; the notebook inline backend is needed for native figure outputs.

MyST sources are authoritative. The old Quarto sources, generated Markdown
refresh workflow, and Read the Docs artifact overlay have been retired. For
`.ipynb` tutorials, source and saved output are in the notebook; Sphinx executes
it again. Do not hand-edit saved numerical output or figures.

The `tutorials` CI workflow runs a strict fresh build for documentation, source,
and dependency changes, on its weekly schedule, and on manual dispatch. Read the
Docs uses the same locked dependency groups. Execution errors fail the build;
no examples are silently excluded or replaced with saved results.

Read [writing-guides.md](writing-guides.md) before editing. Review both code and
statistical outputs: a successful build alone does not establish useful mixing,
correct interpretation, or a valid sampler comparison. The focused example tests
check the repaired Gibbs conditionals against their joint model densities:

```sh
uv run pytest tests/model/test_measurement_error_example.py tests/experimental/test_pymc_spike_slab_example.py
```

Record whether a review build was fresh or incremental and whether each example
was executed. Source changes that affect numerical contracts should receive
focused tests; avoid assertions about exact stochastic diagnostic values.
