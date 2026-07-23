"""
Compatibility layer for :mod:`liesel.optim`.

The optimization API now lives in :mod:`liesel.optim`. This module keeps the old
``liesel.experimental.optim`` import path working for existing code.
"""

import sys
from importlib import import_module

from liesel.optim import *  # noqa: F403
from liesel.optim import __all__ as __all__

_SUBMODULES = (
    "_engine_utils",
    "_log_lik",
    "_model_utils",
    "batch",
    "engine",
    "liesel_optim",
    "liesel_vi",
    "loss",
    "optimizer",
    "split",
    "state",
    "stop",
    "types",
    "util",
    "vi",
)

for _name in _SUBMODULES:
    sys.modules[f"{__name__}.{_name}"] = import_module(f"liesel.optim.{_name}")

del import_module, sys
