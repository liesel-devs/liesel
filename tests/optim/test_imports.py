from importlib import import_module

import pytest

from liesel import optim
from liesel.optim.state import OptimResult
from liesel.optim.vi import NegElboLoss


def test_canonical_optim_imports():
    assert import_module("liesel.optim.liesel_optim").LieselOptim is optim.LieselOptim
    assert import_module("liesel.optim.liesel_vi").LieselVI is optim.LieselVI
    assert import_module("liesel.optim.state").OptimResult is OptimResult
    assert optim.NegElboLoss is NegElboLoss


@pytest.mark.parametrize("module", ["liesel.optim", "liesel.optim.vi"])
def test_elbo_alias_is_unavailable(module):
    assert not hasattr(import_module(module), "Elbo")


def test_evaluate_alias_is_unavailable():
    assert not hasattr(NegElboLoss, "evaluate")
    assert callable(NegElboLoss.estimate_elbo)


@pytest.mark.parametrize(
    "module",
    [
        "liesel.experimental.optim",
        "liesel.experimental.optim.state",
        "liesel.experimental.optim.vi",
    ],
)
def test_experimental_optim_import_is_unavailable(module):
    with pytest.raises(ModuleNotFoundError, match="liesel.experimental.optim"):
        import_module(module)
