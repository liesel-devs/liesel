"""
Goose MCMC framework.
"""

from .builder import EngineBuilder
from .engine import Engine, SamplingResults
from .epoch import EpochConfig, EpochState, EpochType
from .gibbs import GibbsKernel
from .hmc import HMCKernel
from .interface import (
    DataclassInterface,
    DictInterface,
    LieselInterface,
    NamedTupleInterface,
)
from .interface_log_prob import FlatInterfaceLogProb, InterfaceLogProb
from .iwls import IWLSKernel
from .kernel import (
    DefaultTransitionInfo,
    DefaultTuningInfo,
    ModelMixin,
    TransitionMixin,
    TransitionOutcome,
    TuningOutcome,
    WarmupOutcome,
)
from .mcmc_spec import LieselMCMC, MCMCSpec
from .mh import mh_step
from .mh_kernel import MHKernel, MHProposal
from .nuts import NUTSKernel
from .optim import OptimResult, Stopper, history_to_df, optim_flat
from .rw import RWKernel
from .summary_m import SamplesSummary, Summary, loo
from .summary_viz import (
    plot_cor,
    plot_density,
    plot_pairs,
    plot_param,
    plot_scatter,
    plot_trace,
)
from .types import (
    Kernel,
    KernelState,
    ModelInterface,
    ModelState,
    Position,
    TransitionInfo,
    TuningInfo,
)
from .warmup import stan_epochs

__all__ = [
    "Engine",
    "EngineBuilder",
    "mh_step",
    "EpochState",
    "KernelState",
    "DefaultTransitionInfo",
    "DefaultTuningInfo",
    "ModelMixin",
    "TransitionMixin",
    "TransitionOutcome",
    "TuningOutcome",
    "WarmupOutcome",
    "DictInterface",
    "DataclassInterface",
    "DictModel",
    "DataClassModel",
    "LieselInterface",
    "Engine",
    "EngineBuilder",
    "MCMCSpec",
    "LieselMCMC",
    "EpochConfig",
    "EpochType",
    "GibbsKernel",
    "HMCKernel",
    "IWLSKernel",
    "Kernel",
    "MHKernel",
    "MHProposal",
    "ModelInterface",
    "ModelState",
    "NamedTupleInterface",
    "NUTSKernel",
    "Position",
    "history_to_df",
    "Stopper",
    "TransitionInfo",
    "TuningInfo",
    "RWKernel",
    "Summary",
    "SamplesSummary",
    "SamplingResults",
    "loo",
    "plot_cor",
    "plot_density",
    "plot_param",
    "plot_trace",
    "plot_scatter",
    "plot_pairs",
    "stan_epochs",
    "optim_flat",
    "OptimResult",
    "InterfaceLogProb",
    "FlatInterfaceLogProb",
]
