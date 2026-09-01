# AI-assisted development

Liesel permits the responsible use of generative-AI tools for code,
tests, documentation, issues, and other project work. AI may assist human
judgment, but it does not replace human responsibility.

## Expectations

Anyone submitting AI-assisted work must:

- review, understand, and take responsibility for everything submitted;
- be able to explain and modify the contribution;
- verify generated code, claims, references, and test results;
- run the relevant tests and follow the project's existing conventions;
- ensure that the output complies with applicable copyright and licenses.
- ensure that AI-generated text is precise and brief.

Unreviewed or autonomously submitted AI output is not accepted. A human must
remain involved throughout the contribution and review process.

## Disclosure

Please disclose substantial use of generative AI in a pull request. This applies
when generated code, tests, or documentation are included in the contribution,
or when AI materially shaped its implementation.

A brief statement is sufficient:

> AI assistance: [tool] helped with [task]. I reviewed and tested the result.

Disclosure is not required for routine autocomplete, search, formatting,
spelling or grammar correction, or occasional explanatory questions. Prompt
logs, usage percentages, and line-by-line attribution are not required.

For unknown contributors, equipping individual commits that were mainly AI-generated
with a brief `Assisted-by: <harness>:<model>` in addition to the general disclosure in
PR descriptions is appreciated, but optional.

Disclosure is optional for maintainers and known, trusted contributors.

## Communication and sensitive information

Issues, pull-request descriptions, and review responses must reflect the
contributor's own understanding. AI-assisted text must be checked for accuracy
and edited as needed.

Do not provide AI services with secrets, personal or confidential data, private
security reports, or other information that you are not authorized to share.

## Maintainers and enforcement

Maintainers follow the same standards when using AI for project work or review, but
disclosure is optional for maintainers and known, trusted contributors.
Final technical, governance, security, and release decisions remain human
decisions.

Maintainers may request clarification or additional verification and may close
contributions that are undisclosed, unreviewed, substantially autonomous, or
create unreasonable work for reviewers.
