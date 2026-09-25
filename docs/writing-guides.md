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

- State prerequisites, such as an existing `model`, before a snippet. Make
  complete tutorials runnable from top to bottom.
- Use existing public helpers instead of manual setup or calculations that they
  already handle. Include a lower-level recipe or alternative only when it serves
  a distinct task or helps the reader make a meaningful choice.
- Keep code consistent with the repository formatter. Group imports, leave blank
  lines between steps, and separate model construction, fitting, and inspection.
  Lay out data and long calls so they are easy to scan.
- Use plain Python blocks without interactive prompts (`>>>` or `...`).
- Pass a single model root directly, as in `lsl.Model(y)`. Use a sequence for
  multiple roots.
- Use realistic data, fixed seeds, and only the settings needed for the task.
- Pair print statements and inspection expressions with actual executed output.
  Use notebook outputs or separate text blocks in text guides. Round numbers and
  select useful fields; remove unnecessary inspection calls instead of leaving
  them without output.
- Keep verification assertions in tests. Investigate awkward API behavior before
  adding repeated defensive checks to examples.

Migration guides should show the old and new code and explain meaningful
behavior changes.

## Show useful visuals

Use plotnine for statistical plots and show the rendered plot with its code.
Explain what readers should look for and what the plot cannot establish. Use
enough contrast, distinct shapes, or small positional offsets to keep overlapping
marks visible.

In model walkthroughs, include `model.plot()` after constructing the model and
show the graph beside its code. Generate it in an executed tutorial cell where
possible; the Read the Docs build installs Graphviz for layout. Give figures
descriptive alt text. For notebooks, use `mystnb.image.alt` cell metadata and
check that it appears on the rendered image.

Embed interactive explanations beside the relevant text. A separate-page link
can supplement the embed. Keep essential explanations readable without
interacting with the visual.

Use static images for guides that do not execute code during the build or when
generation is unreliable. Record the source command and refresh the image when
the example changes.

## Check the finished result

Read the guide from top to bottom for flow, missing prerequisites, repetition,
and unnecessary detours. Match verification to the change: execute changed
examples, refresh affected saved outputs, and inspect plots. Build the docs when
changing rendering or navigation; check both sidebars and that links resolve.
Run the relevant hooks on edited files.

Use Sphinx cross-references for internal pages and API objects. Links intended
for use outside the docs must work from that context. Report validation accurately,
including whether builds were fresh or incremental and notebooks were executed,
cached, or skipped.
