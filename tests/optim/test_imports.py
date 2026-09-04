from importlib import import_module

import liesel.experimental.optim as experimental_optim
from liesel import optim
from liesel.optim.state import OptimResult


def test_experimental_optim_aliases_optim():
    experimental_state = import_module("liesel.experimental.optim.state")

    assert experimental_optim.LieselOptim is optim.LieselOptim
    assert experimental_optim.LieselVI is optim.LieselVI
    assert experimental_state.OptimResult is OptimResult
