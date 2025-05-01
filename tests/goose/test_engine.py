"""
some tests for the engine
"""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel.goose.builder import EngineBuilder
from liesel.goose.chain import EpochChainManager
from liesel.goose.engine import (
    Engine,
    SamplingResults,
    _add_time_dimension,
    # stack_for_multi,
)
from liesel.goose.epoch import EpochConfig, EpochState, EpochType
from liesel.goose.interface import DictInterface
from liesel.goose.kernel import DefaultTransitionInfo
from liesel.goose.kernel_sequence import KernelSequence
from liesel.goose.pytree import (
    concatenate_leaves,
    register_dataclass_as_pytree,
    slice_leaves,
)
from liesel.goose.types import Array, KeyArray, ModelInterface, ModelState
from liesel.model import Model, Var
from liesel.option import Option

from .deterministic_kernels import DetCountingKernel, DetCountingKernelState


@register_dataclass_as_pytree
@dataclass
class FooQuant:
    error_code: int
    result: tuple[Array, Array]


class FooQauntGen:
    error_book: ClassVar[dict[int, str]] = {0: "no errors"}

    def __init__(self, identifier):
        self.identifier = identifier

    def set_model(self, model: ModelInterface):
        pass

    def has_model(self) -> bool:
        return False

    def generate(
        self, prng_key: KeyArray, model_state: ModelState, epoch: EpochState
    ) -> FooQuant:
        u = jax.random.normal(prng_key)
        return FooQuant(0, (u, model_state["x"]))


def test_add_time_dimension():
    def get_dims(t):
        return [t[0].shape, t[1][0].shape, t[1][1]["f"].shape]

    tree0 = jax.jit(lambda x: x)(
        (jnp.array([1.0, 2.0]), [jnp.zeros((3, 3)), {"f": jnp.array([1.0, 2.0])}])
    )
    tree3 = _add_time_dimension(tree0)
    dims3 = [
        (2, 1),
        (3, 1, 3),
        (2, 1),
    ]
    assert dims3 == get_dims(tree3)


def test_error_log():
    errs: np.ndarray = np.array([0, 0, 1, 0, 0, 0, 1, 1]).reshape((2, -1))
    ti = DefaultTransitionInfo(errs, np.zeros((2, 4)), np.zeros((2, 4), np.int8))
    tis = {"kern0": ti}

    em = EpochChainManager()
    em.advance_epoch(EpochConfig(EpochType.POSTERIOR, 4, 1, None))
    em.append(tis)
    em.combine_all()

    sr = SamplingResults(
        EpochChainManager(),
        em,
        Option.none(),
        Option.none(),
        Option.none(),
        Option.none(),
        Option.none(),
        Option.none(),
    )

    error_log = sr.get_error_log().unwrap()
    kel = error_log["kern0"]
    assert kel.kernel_ident == "kern0"
    assert np.array_equal(kel.transition, np.array([2, 3]))
    assert np.array_equal(kel.error_codes, np.array([[1, 0], [1, 1]]))


def t_test_engine():
    num_chains = 4
    epoch_configs = [
        EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
        EpochConfig(EpochType.FAST_ADAPTATION, 50, 1, None),
        EpochConfig(EpochType.BURNIN, 50, 1, None),
        EpochConfig(EpochType.POSTERIOR, 100, 1, None),
    ]

    ms = {"x": jnp.array(1), "y": jnp.array(-1)}
    mss = _stack_for_multi([ms for _ in range(num_chains)])
    con = DictInterface(lambda ms: -0.5 * ms["x"] ** 2 - 0.5 * ms["y"])
    ker0 = DetCountingKernel(["x"], DetCountingKernelState.default())
    ker1 = DetCountingKernel(["y"], DetCountingKernelState.default())
    ker0.set_model(con)
    ker1.set_model(con)

    ks = KernelSequence([ker0, ker1])

    seeds = jax.random.split(jax.random.PRNGKey(0), num_chains)

    engine = Engine(
        seeds,
        mss,
        ks,
        epoch_configs,
        25,
        con,
        ["x"],
        minimize_transition_infos=False,
        store_kernel_states=True,
        quantity_generators=[FooQauntGen("foo"), FooQauntGen("bar")],
    )

    engine.sample_all_epochs()

    results: SamplingResults = engine.get_results()

    print(results.positions.combine_all())
    print(results.transition_infos.combine_all())
    print(results.kernel_states.combine_all())

    print(results.get_posterior_samples())
    print(results.get_tuning_times())

    print(results.generated_quantities.unwrap().combine_all().unwrap())


def test_liesel_model_in_engine_builder() -> None:
    builder = EngineBuilder(seed=1, num_chains=4)
    y = Var.new_obs(1.0, name="y")
    model = Model([y])

    with pytest.raises(TypeError):
        builder.set_model(model)  # type: ignore


def t_test_engine_builder() -> None:
    builder = EngineBuilder(seed=1, num_chains=4)

    builder.set_epochs(
        [
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.FAST_ADAPTATION, 50, 1, None),
            EpochConfig(EpochType.BURNIN, 55, 10, None),
            EpochConfig(EpochType.POSTERIOR, 100, 10, None),
        ]
    )
    ms = {"x": jnp.array(1), "y": jnp.array(-1)}
    builder.set_initial_values(ms, multiple_chains=False)
    builder.set_jitter_fns(
        {
            "x": lambda key, cv: cv + tfd.Uniform(-1.0, 1.0).sample(cv.shape, key),
            "y": lambda key, cv: cv + tfd.Uniform(-1.0, 1.0).sample(cv.shape, key),
        }
    )
    con = DictInterface(lambda ms: -0.5 * ms["x"] ** 2 - 0.5 * ms["y"])
    builder.set_model(con)
    builder.add_kernel(DetCountingKernel(["x"], DetCountingKernelState.default()))
    builder.add_kernel(DetCountingKernel(["y"], DetCountingKernelState.default()))
    builder.add_quantity_generator(FooQauntGen("foo"))
    builder.add_quantity_generator(FooQauntGen("bar"))
    builder.positions_excluded = ["y"]
    engine = builder.build()

    engine.sample_all_epochs()
    results: SamplingResults = engine.get_results()

    # print(results.get_posterior_samples())
    # print(results.get_tuning_times())

    # print(results.generated_quantities.unwrap().combine_all().unwrap())
    # print(results.transition_infos.combine_all().unwrap())

    # test thinning worked
    assert results.get_posterior_samples()["x"].shape == (4, 10)
    assert results.get_samples()["x"].shape == (4, 66)
    assert results.generated_quantities.unwrap().combine_all().unwrap()["foo"].result[
        0
    ].shape == (4, 66)

    # test thinning is not applied to TIs
    assert results.transition_infos.combine_all().unwrap()[
        "kernel_01"
    ].error_code.shape == (4, 205)


if __name__ == "__main__":
    t_test_engine_builder()


## helper functions
def _stack_for_multi(chunks: list):
    chunks = slice_leaves(chunks, jnp.s_[jnp.newaxis, ...])
    return concatenate_leaves(chunks, axis=0)
