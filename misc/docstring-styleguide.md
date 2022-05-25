## Styleguide

- Keep the docstrings short and concise. Do not (usually) mention types in the docstrings, because pdoc shows them in the function signatures.
- Start each file with a header (a first-level `#` headline).
- Parameters, attributes and return values are second-level headlines (`## Parameters`, `## Attributes`, `## Returns`).
- Parameters and attributes are described in simple Markdown lists (``- `parameter`: Description, blah, blah.``).
- Enclose object names with `` `backticks` ``.
- All sentences and list items should start with a capital letter and end with a full stop.
- Start each docstring with a minimal description of what the object does. Any additional information should go in a new paragraph (separated by two line breaks).
- The line length of the docstrings should be limited to 88 characters. For convenience, the lines can be rewrapped automatically, for example with the VS Code extension [rewrap](https://marketplace.visualstudio.com/items?itemName=stkb.rewrap).
- If a docstring does not fit on a single line, put the opening and closing triple quotes on separate lines.

## Examples

### File header

``` python
"""
# Nodes for building probabilistic graphical models (PGMs)
"""
```

### Method docstring

``` python
def initialize(self) -> Parameter:
    """Initializes the value of the parameter with its prior mean."""

    ...
```

### Class docstrings

``` python
class SmoothCalculator(NodeCalculator):
    """
    Calculates a smooth `x @ beta`.

    A smooth is the matrix-vector product of a design matrix `x` and a vector of
    regression coefficients `beta`. Must have just two inputs with the labels `x`
    and `beta`.
    """

    ...
```

``` python
class Node(Generic[TNodeCalculator]):
    """
    A node, strong or weak, with or without a probability distribution,
    which can be used to build probabilistic graphical models (PGMs).

    ## Attributes

    - `log_prob`:
      The log-probability of the node.

    - `outdated`:
      Whether the node is outdated.

      A node is outdated if its value or the value of one of its inputs has changed.
      The value and the log-probability of an outdated node need to be recomputed.
    """

    ...
```

## Preview

```
pdoc -t ./misc/pdoc-template liesel
```
