"""
# Goose MCMC framework

This module defines the public API of Goose. Expect breakages if using anything
beyond this module (and maybe also if using this module, as this is early-stage
research software).
"""

from .builder import EngineBuilder as EngineBuilder
from .engine import Engine as Engine
from .epoch import EpochConfig, EpochType
from .gibbs import GibbsKernel
from .hmc import HMCKernel
from .iwls import IWLSKernel
from .models import DictModel
from .nuts import NUTSKernel
from .rw import RWKernel
from .summary_m import Summary, summary
from .summary_viz import (
    plot_cor,
    plot_density,
    plot_pairs,
    plot_param,
    plot_scatter,
    plot_trace,
)
from .types import ModelInterface, Position
from .warmup import stan_epochs

__all__ = [
    "DictModel",
    "Engine",
    "EngineBuilder",
    "EpochConfig",
    "EpochType",
    "GibbsKernel",
    "HMCKernel",
    "IWLSKernel",
    "ModelInterface",
    "NUTSKernel",
    "Position",
    "RWKernel",
    "Summary",
    "plot_cor",
    "plot_density",
    "plot_param",
    "plot_trace",
    "plot_scatter",
    "plot_pairs",
    "stan_epochs",
    "summary",
]
