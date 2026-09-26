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

- Prefer executable MyST Markdown for task guides with code and results. Use
  native notebook cells and outputs instead of generated RST output blocks.
  Add `file_format: mystnb` and a Python `kernelspec` in YAML front matter,
  then use `{code-cell} ipython3` cells. Keep one source per page; remove the
  corresponding Quarto source when converting generated Markdown to MyST.
- State prerequisites, such as an existing `model`, before a snippet. Make
  complete guides and tutorials runnable from top to bottom. Build-only setup
  cells may load non-import fixture code for a linked prerequisite tutorial; use
  `remove-cell` to omit their source and outputs from the rendered page while
  still executing them.
  Imports stay visible in a code cell at the top of each guide. Hide only
  non-import build setup, such as logging configuration and fixture models or data.
  Keep reader prerequisites visible in the prose and instructional setup visible
  in the code. Shared setup can also use MyST-NB's native `:load:` option.
- Use existing public helpers instead of manual setup or calculations that they
  already handle. Include a lower-level recipe or alternative only when it serves
  a distinct task or helps the reader make a meaningful choice.
- Make the structure of the example visible through its layout. Group imports
  and organize code into meaningful stages, such as parameters, inputs, mean,
  scale, response, fitting, and inspection. Use blank lines between stages and
  substantial independent definitions, including consecutive multiline calls.
  Keep short, closely related statements together; do not separate every line.
- Wrap visually dense calls, not only calls that exceed the line-length limit.
  Put arguments on separate lines when that helps distinguish a function or
  lambda, its inputs, nested expressions, and named options. Use trailing commas
  so Ruff preserves the chosen layout. Keep short, easily scanned calls on one
  line. Apply this judgment to all functions and constructors.
- Use short comments to label conceptual groups when the surrounding prose does
  not already make them clear. Explain the role of a group, such as “Scale
  predictor,” rather than narrating individual assignments.
- Pass distributions to `lsl.Var` and its factory methods with the explicit
  `dist=` keyword, rather than as a positional argument.
- Use code cells without interactive prompts (`>>>` or `...`).
- Pass a single model root directly, as in `lsl.Model(y)`. Use a sequence for
  multiple roots. Choose `to_float32` for the needs of the example, independently
  of this calling style.
- Use realistic data, fixed seeds, and only the settings needed for the task.
- Keep setup and modifications together, then put each inspection expression
  in its own cell so its native output appears immediately below it. Prefer
  expressions to `print()` for inspecting values, and native tables for related
  results. Reserve `print()` for messages. Round numbers and select useful fields;
  remove unnecessary inspection calls instead of leaving them without output.
- Execute examples when building the docs and fail the build on execution errors.
  Keep image alt text in cell metadata and inspect the rendered outputs.
- Keep verification assertions in tests. Investigate awkward API behavior before
  adding repeated defensive checks to examples.

Migration guides should show the old and new code and explain meaningful
behavior changes. Write them only for changes to released APIs. Don't document
the history of unreleased APIs, such as earlier defaults, renamed arguments, or
before-and-after tables; describe the current behavior instead.

## Show useful visuals

Prefer built-in Liesel and Goose plotting helpers when they cover the task,
for example `model.plot()`, `gs.plot_trace()`, and `result.plot_loss_overview()`.
Use plotnine for custom statistical plots without a suitable built-in helper.
Show the rendered plot with its code. Explain what readers should look for and
what the plot cannot establish. Use enough contrast, distinct shapes, or small
positional offsets to keep overlapping marks visible.

Put each plotting call in its own code cell, with the rendered figure directly
below it. Keep plot preparation, construction, and fitting code in separate cells.

In model walkthroughs, include `model.plot()` after constructing the model and
show the graph beside its code. Generate it in an executed tutorial cell where
possible; the Read the Docs build installs Graphviz for layout. Give figures
descriptive alt text. For notebooks, use `mystnb.image.alt` cell metadata and
check that it appears on the rendered image.

Embed interactive explanations beside the relevant text. A separate-page link
can supplement the embed. Keep essential explanations readable without
interacting with the visual.

Static assets can illustrate concepts outside the executable example. Record
their source and refresh them when the explanation changes.

## Check the finished result

Read the guide from top to bottom for flow, missing prerequisites, repetition,
and unnecessary detours. Match verification to the change: execute changed
examples, refresh affected saved outputs, and inspect plots. Build the docs when
changing rendering or navigation; check both sidebars and that links resolve.
Execute notebook examples as part of the Sphinx build. Keep
`nb_execution_mode = "force"`, `nb_execution_allow_errors = False`, and
`nb_execution_raise_on_error = True` so execution errors fail the build.
Sphinx's incremental build only executes pages it reads; use `-E` for a fresh
execution of all notebook sources. Also use `-E` after changing a shared
`:load:` file, since MyST-NB does not track that file as an incremental-build
dependency. Run the relevant hooks on edited files.

Use Sphinx cross-references for internal pages and API objects. Links intended
for use outside the docs must work from that context. Report validation accurately,
including whether builds were fresh or incremental and notebooks were executed,
cached, or skipped.
