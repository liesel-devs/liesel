"""
# Type aliases
"""

from typing import Any, NamedTuple

from liesel.goose import Position

__all__ = [
    "Array",
    "ModelState",
    "NodeState",
    "Position",
    "TFPBijector",
    "TFPBijectorClass",
    "TFPDistribution",
    "TFPDistributionClass",
]

Array = Any


class NodeState(NamedTuple):
    """The state of a node as a named tuple."""

    value: Array
    """The value of the node."""

    log_prob: Array
    """The log-probability of the node."""


ModelState = dict[str, NodeState]

TFPBijector = Any
TFPBijectorClass = Any

TFPDistribution = Any
TFPDistributionClass = Any
