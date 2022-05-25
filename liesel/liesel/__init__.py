"""
# Liesel modeling framework
"""

from .distreg import CopRegBuilder, DistRegBuilder, dist_reg_mcmc, tau2_gibbs_kernel
from .goose import GooseModel
from .model import Model, ModelBuilder, load_model, save_model
from .nodes import (
    PIT,
    Addition,
    Bijector,
    ColumnStack,
    DesignMatrix,
    Hyperparameter,
    InverseLink,
    Node,
    NodeCalculator,
    NodeDistribution,
    NodeGroup,
    NodeState,
    Parameter,
    Predictor,
    RegressionCoef,
    Response,
    Smooth,
    SmoothingParam,
    transform_parameter,
)
from .types import ModelState, Position
from .viz import plot_model

__all__ = [
    "Addition",
    "Bijector",
    "ColumnStack",
    "CopRegBuilder",
    "DesignMatrix",
    "DistRegBuilder",
    "GooseModel",
    "Hyperparameter",
    "InverseLink",
    "Model",
    "ModelBuilder",
    "ModelState",
    "Node",
    "NodeCalculator",
    "NodeDistribution",
    "NodeGroup",
    "NodeState",
    "PIT",
    "Parameter",
    "Position",
    "Predictor",
    "RegressionCoef",
    "Response",
    "Smooth",
    "SmoothingParam",
    "dist_reg_mcmc",
    "load_model",
    "plot_model",
    "save_model",
    "tau2_gibbs_kernel",
    "transform_parameter",
]

ModelState = ModelState
"""The state of a model as a dictionary of node names and states."""

Position = Position
"""The position of a model as a dictionary of node names and values."""
