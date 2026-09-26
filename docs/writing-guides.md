# Writing short, useful guides

Use these notes when writing or reorganizing Liesel guides and tutorial
notebooks, including model building, optimization, and MCMC sampling.

## Give readers a clear route

Start by reading the current docs and implementation. Identify the tasks readers
need to complete, then give each piece of information one main home:

- **Landing page:** explain the purpose, show a small working example, and link
  to the next steps.
- **Tutorial:** walk through a complete, realistic task. Introduce one idea at a
  time and show the results.
- **Task guide:** answer a specific question, such as choosing priors,
  configuring a sampler, or inspecting results.
- **API reference:** document arguments, defaults, exact rules, and edge cases.

Keep essential constraints beside the example they affect. Link to deeper
details instead of repeating them. When retiring duplicate notebooks, preserve
useful examples in the tutorials and genuine checks in the test suite.

## Write each section around an action

Use a practical heading, a short introduction, the code, and an explanation of
what the code changes. Include a link when there is a useful next step.

Prefer headings such as “Define the model” or “Inspect the chains” to vague
headings such as “Advanced usage.” After the example, explain how to interpret
the output or how the choice affects the next step. Avoid explaining every
visible line of code.

## Keep the language human

Use short sentences, familiar verbs, and concrete names. Address the reader
directly. Explain a technical term where it first becomes necessary. Keep exact
API names and distinctions that affect the result.

Call model parameters “parameters,” not “coordinates.” Where transformations
matter, use “transformed parameters” or specify the parameter scale.

Prefer “This saves memory” to “This configuration facilitates reduced memory
consumption.” Remove repeated introductions, promotional claims, and closing
summaries that restate the section.

Brevity means removing repetition, not hiding prerequisites or meaningful
choices. Explain when a setting changes the model, the algorithm, or the
reported results. Preserve deliberate API choices instead of silently choosing
for the reader.

## Make examples easy to use

Imports stay visible in a code cell at the top of each guide. Hide only
non-import build setup, such as logging configuration and fixture models or data.

- State prerequisites, such as an existing `model`, before a snippet. Make
  complete tutorials runnable from top to bottom.
- Keep code blocks orderly and consistent with the repository formatter. Group
  imports, use blank lines between logical steps, and separate model construction,
  fitting, and inspection into focused blocks. Lay out data and long calls so they
  are easy to scan.
- Use plain Python code blocks without interactive prompts (`>>>` or `...`).
  Keep code easy to copy and run.
- Use realistic data, fixed seeds, and only the settings needed to teach the
  task. Show a useful result, rather than a wall of diagnostic output.
- Pair print statements and inspection expressions with their actual output.
  Use executed notebook outputs or separate text output blocks in text guides.
  Capture results by running the examples; round displayed numbers and select
  useful diagnostic fields to keep the output readable.
- Do not use `assert` statements in documentation. Keep verification in tests.
  When a type diagnostic exposes awkward or unsafe API behavior, investigate
  the cause before adding checks to every example.
- Choose outputs that help readers judge the result: a model graph, a summary,
  or a diagnostic plot. Explain what to look for and what the output cannot
  establish on its own.

## Put visuals where they help

Use plotnine for statistical plots, and show the rendered plot with its code.

Embed interactive explanations beside the relevant text. A separate-page link
can supplement the embed, but should not be the only way readers discover it.

In model walkthroughs, include `model.plot()` after constructing the model and
show its rendered graph beside the code. Generate graphs in executed tutorial
cells where possible. The Read the Docs build installs Graphviz for their layout.
Give figures descriptive alt text; for notebooks, use the cell metadata at
`mystnb.image.alt` and check that it appears on the rendered image.

Use static images in guides that do not execute code during the build, or when a
figure cannot be generated reliably there. Record the source command and refresh
the image when the example changes. Keep essential explanations readable without
interacting with a visual.

## Check the finished result

Execute changed examples and notebooks, refresh affected saved outputs, and
inspect the plots. Build the docs when changing rendering or navigation, and
check that links resolve. Use Sphinx cross-references for internal pages and API
objects. Links intended for use outside the docs must work from that context.
Run the relevant hooks on the edited files.

Match descriptions to actual behavior. A migration guide should show the old
and new code and identify meaningful behavior changes. Verify notebook build
settings before claiming that execution is skipped or cached.
