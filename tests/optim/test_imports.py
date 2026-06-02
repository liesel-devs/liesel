from liesel.experimental.optim.state import OptimResult as ExperimentalOptimResult

import liesel.experimental.optim as experimental_optim
import liesel.optim as optim
from liesel.optim.state import OptimResult


def test_experimental_optim_aliases_optim():
    assert experimental_optim.LieselOptim is optim.LieselOptim
    assert experimental_optim.LieselVI is optim.LieselVI
    assert ExperimentalOptimResult is OptimResult
