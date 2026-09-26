# Writing short, useful guides

Use these notes for Liesel guides and tutorial notebooks, including model
building, optimization, and MCMC sampling.

## Give each topic a home

Read the current docs and implementation, identify the reader's task, and choose
where the explanation belongs:

- **Landing page:** explain the purpose, show a small working example, and link
  to the next steps.
- **Tutorial:** walk through a complete, realistic task, introducing one idea at
  a time and showing the results.
- **Task guide:** answer a specific question, such as choosing priors,
  configuring a sampler, or inspecting results.
- **API reference:** document arguments, defaults, exact rules, and edge cases.

Give each explanation one main home and link to it elsewhere. Keep prerequisites
and essential constraints beside the example they affect. Explain choices that
change the model, algorithm, or reported results; brevity must not hide them.
When retiring duplicate notebooks, preserve useful examples in tutorials and
genuine checks in the test suite.

Put `Overview <self>` first in the first toctree of each guide landing page.
Use Sphinx's special `self` entry, not the page's filename, so Overview links to
the landing page without nested children. Preserve the remaining entries and
their order. Check the sidebar on both the landing page and its child pages.

## Lead with the main workflow

Introduce a feature's general purpose before relating it to special cases. Show
the main workflow and a useful result first. Put tuning, performance details,
and extended diagnostics later, or link to a separate task guide or reference.

Give each section a practical heading, a short introduction, the relevant code,
and an interpretation of the result. Name the actual operation, such as “Define
the model” or “Choose the source.” Avoid vague headings and explanations of every
visible line of code.

Keep page and section headings on one line in both rendered sidebars at desktop
widths. Use a short toctree label when a longer tutorial title is useful. Shorten
the wording without losing its meaning; do not hide or clip wrapping text.

## Keep the language human

Use short sentences, familiar verbs, and concrete names. Address the reader
directly and explain technical terms when they become necessary. Preserve exact
API names and distinctions that affect the result.

Call model parameters “parameters,” not “coordinates.” Where transformations
matter, use “transformed parameters” or specify the parameter scale.

Prefer “This saves memory” to “This configuration facilitates reduced memory
consumption.” Remove repeated introductions, promotional claims, and closing
summaries that restate the section.

## Make examples easy to use

Use executable MyST Markdown for guides with code examples. Put runnable
Python in `{code-cell}` blocks without interactive prompts (`>>>` or `...`).

- State prerequisites, such as an existing `model`, before a snippet. Make
  complete tutorials runnable from top to bottom. Tag build-only setup cells
  with `:tags: [remove-cell]`: they execute but show no code, output, or expandable
  box. Keep prerequisites and links to relevant tutorials visible in the prose.
  Imports stay visible in a code cell at the top of each guide. Hide only
  non-import build setup, such as logging configuration and fixture models or data.
- Prefer existing public Liesel helpers over manual calculations that repeat model
  definitions or library functionality. For example, use
  `model.predict(samples, predict=[...], newdata=...)` to evaluate model quantities
  at posterior draws and new inputs instead of repeating design-matrix products
  and transformations. Keep necessary postprocessing explicit: identify sample
  axes and state whether transformations happen before averaging. Retain manual
  calculations when they teach a distinct concept or no suitable helper exists.
  Execute revised examples and verify that their statistical meaning and results
  are preserved.
- Format for visual readability, not just line length. Group code into meaningful
  stages, with blank lines between stages and substantial independent definitions.
  Keep short, closely related statements together. Use brief comments to label
  conceptual groups when the surrounding prose does not make them clear.
- Apply this judgment to all functions and constructors: wrap dense calls so
  functions or lambdas, inputs, nested expressions, and named options are easy to
  distinguish. Use trailing commas so Ruff preserves the layout; keep simple
  calls compact and follow the repository formatter.
- Keep related setup and modifications together in one cell. Give each inspection
  expression its own `{code-cell}`, with its native output immediately below.
- Pass distributions to `lsl.Var` and its factory methods using `dist=`.
- Pass a single model root directly, as in `lsl.Model(y)`. Choose `to_float32`
  for the needs of the example, independently of this calling style.
- Use realistic data, fixed seeds, and only the settings needed for the task.
- Prefer expressions over `print()`. Use native tables for related results.
  Select useful fields and round numbers for readability. Remove unnecessary
  inspection calls instead of leaving them without output.
- Keep verification assertions in tests. Investigate awkward API behavior before
  adding repeated defensive checks to examples.

Migration guides should show the old and new code and explain meaningful
behavior changes. Write them only for changes to released APIs. Don't document
the history of unreleased APIs, such as earlier defaults, renamed arguments, or
before-and-after tables; describe the current behavior instead.

## Show useful visuals

Use built-in Liesel/Goose plotting helpers where available, and plotnine for
plots without a suitable helper. Put each plotting call, such as `model.plot()`
or `gs.plot_trace()`, in its own code cell directly above its rendered figure.
Keep model construction and fitting separate. Explain what readers should look
for and what the plot cannot establish. Use enough contrast,
distinct shapes, or small positional offsets to keep overlapping marks visible.

In model walkthroughs, include `model.plot()` after constructing the model.
The Read the Docs build installs Graphviz for layout. Give every figure
descriptive alt text. For cell outputs, use `mystnb.image.alt` cell metadata
and check that it appears on the rendered image.

Embed interactive explanations beside the relevant text. A separate-page link
can supplement the embed. Keep essential explanations readable without
interacting with the visual.

Static assets can illustrate concepts outside the executable example. Record
their source and refresh them when the explanation changes.

## Document public attributes compactly

Document public class attributes in the API reference's compact Attributes table,
with one row per attribute and a concise description. Follow the `LossMixin` and
`NegElboLoss` pattern: use class-level type annotations with adjacent attribute
docstrings so the existing autosummary template generates the table. Keep
property documentation in the property's docstring. Avoid separate standalone
attribute headings or duplicating these descriptions in a class-level
`Attributes` section. Preserve runtime behavior when adding annotations,
including dataclass fields and constructor signatures. Keep generated attribute
pages available as cross-reference targets.

Build affected API pages and inspect their rendered tables for completeness,
readability, and duplication.

## Link documented API objects

Use Sphinx cross-references consistently when mentioning documented classes,
methods, functions, or attributes, including in parameter descriptions, return
descriptions, notes, and attribute tables. Use MyST roles in Markdown and
reStructuredText roles in docstrings, with fully qualified targets and short,
readable labels. For example, write
``{meth}`model.predict <liesel.model.Model.predict>` `` in MyST,
and ``:meth:`NegElboLoss.from_vdist <liesel.optim.NegElboLoss.from_vdist>` ``
in a docstring. Instance-style wording can remain the label.

Keep executable examples unchanged when adding cross-references. Use plain inline
code for local variables, argument values, and expressions without a documentation
target. Build affected pages, resolve broken references, and inspect the rendered
links. Links intended for use outside the docs must work from that context.

## Check the finished result

Execute guide examples during the docs build and fail the build on execution
errors. Let execution produce the displayed results; do not maintain copied
output blocks by hand.

Read the guide from top to bottom for flow, missing prerequisites, repetition,
and unnecessary detours. Match verification to the change: execute changed
examples, refresh affected saved outputs, and inspect the rendered code and plots.
Check that build-only setup is absent, both sidebars remain readable, and links
resolve. Run the relevant hooks on edited files.

Report validation accurately,
including whether builds were fresh or incremental and notebooks were executed,
cached, or skipped.
