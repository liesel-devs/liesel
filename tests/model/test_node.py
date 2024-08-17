import logging

import numpy as np
import pytest
import tensorflow_probability.substrates.numpy.distributions as tfd

from liesel.model.model import Model
from liesel.model.nodes import (
    Calc,
    Data,
    Dist,
    NodeState,
    TransientCalc,
    TransientDist,
    Var,
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test Data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_data_var() -> None:
    x = Data(1.0, _name="x")
    v1 = Var(value=x)

    assert x.var is v1


def test_data_init() -> None:
    x = Data(0.0, _name="node")

    assert x.value == pytest.approx(0.0)
    assert x.name == "node"
    assert isinstance(x.value, float)
    assert not x.model
    assert x.state.value == pytest.approx(0.0)
    assert not x.state.outdated


def test_data_clear_state() -> None:
    x = Data(1.0, _name="x")
    x.clear_state()

    assert not x.outdated
    assert x.value is None


def test_data_state_manipulation() -> None:
    x = Data(0.0, _name="x")
    x.state = NodeState(1.0, True)
    x.name = "n"
    x.update()

    assert x.name == "n"
    assert x.state.value == pytest.approx(1.0)
    assert not x.state.outdated


def test_data() -> None:
    x = Data(13.0, _name="x")

    assert x.value == pytest.approx(13.0)
    assert not x.outdated

    x.value = 14.0

    assert x.value == pytest.approx(14.0)
    assert not x.outdated


def test_frozen_data_name_manipulation() -> None:
    x = Data(2.0, _name="x")
    x.update()

    with pytest.raises(RuntimeError):
        _ = Model([x])
        x.name = "a"


def test_frozen_data_needs_seed_manipulation() -> None:
    x = Data(2.0, _name="x")
    x.update()

    with pytest.raises(RuntimeError):
        _ = Model([x])
        x.needs_seed = True


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test Calc, TransientCalc ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_calculator_var(Calc) -> None:
    x = Data(2.0, _name="x")
    calc = Calc(np.exp, x)
    v1 = Var(value=calc)

    assert calc.var is v1


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_calculator_init(Calc) -> None:
    x = Data(2.0, _name="x")
    calc = Calc(np.exp, x)
    id_all_inputs = {id(i) for i in calc.all_input_nodes()}

    assert calc.function is np.exp
    assert id_all_inputs == {id(x)}


def test_calculator_clear_state() -> None:
    x = Data(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.clear_state()

    assert calc.outdated
    assert calc.value is None


def test_transient_calculator_clear_state() -> None:
    x = Data(2.0, _name="x")
    calc = TransientCalc(np.exp, x)
    calc.clear_state()

    assert calc.outdated
    assert calc.value == pytest.approx(np.exp(2.0))


def test_calculator_state_manipulation() -> None:
    x = Data(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.state = NodeState(1.0, True)

    assert calc.state.value == pytest.approx(1.0)

    calc.update()

    assert calc.state.value != 1.0
    assert calc.state.outdated


def test_transient_calculator_state_manipulation() -> None:
    x = Data(2.0, _name="x")
    calc = TransientCalc(np.exp, x)
    calc.state = NodeState(1.0, True)

    assert calc.state.value is None
    assert calc.state.extra is None

    calc.update()

    assert calc.state.value is None
    assert calc.state.extra is None


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_calculator_arg_input_nodes(Calc) -> None:
    x = Data(2.0, _name="x")
    calc = Calc(np.exp, x)
    id_all_inputs = {id(i) for i in calc.all_input_nodes()}

    assert id_all_inputs == {id(x)}


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_calculator_kwarg_input_nodes(Calc) -> None:
    x = Data(2.0, _name="x")
    calc = Calc(np.exp, x=x)
    id_all_inputs = {id(i) for i in calc.all_input_nodes()}

    assert id_all_inputs == {id(x)}


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_manipulated_calculator_input_nodes(Calc) -> None:
    x = Data(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.update()

    y = Data(3.0, _name="y")
    calc.set_inputs(y)
    id_all_inputs = {id(i) for i in calc.all_input_nodes()}

    assert id_all_inputs == {id(y)}


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_univariate_calculator(Calc) -> None:
    x = Data(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.update()

    assert calc.value == pytest.approx(np.exp(2.0))


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_frozen_calc_name_manipulation(Calc) -> None:
    x = Data(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.update()

    with pytest.raises(RuntimeError):
        _ = Model([calc])
        calc.name = "a"


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_frozen_calc_needs_seed_manipulation(Calc) -> None:
    x = Data(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.update()

    with pytest.raises(RuntimeError):
        _ = Model([calc])
        calc.needs_seed = True


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_frozen_calculator_function_manipulation(Calc) -> None:
    x = Data(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.update()

    with pytest.raises(RuntimeError):
        _ = Model([calc])
        calc.function = np.log


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_frozen_calculator_kwinputs_manipulation(Calc) -> None:
    x = Data(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.update()

    with pytest.raises(RuntimeError):
        _ = Model([calc])
        calc.set_inputs()

    with pytest.raises(RuntimeError):
        calc.set_inputs(y=Data(1.0))


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_calculator_input_manipulation(Calc) -> None:
    x = Data(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.update()

    y = Data(1.0, _name="y")
    calc.set_inputs(y)
    calc.update()

    assert calc.value == pytest.approx(np.exp(1.0))


def test_calculator_kwinput_manipulation() -> None:
    x = Data(2, _name="x")
    calc = Calc(lambda x: x, x=x)
    calc.update()
    assert calc.value == 2

    calc.set_inputs(y=Data(1))

    with pytest.raises(RuntimeError):
        calc.update()


def test_transient_calculator_kwinput_manipulation() -> None:
    x = Data(2, _name="x")
    calc = TransientCalc(lambda x: x, x=x)
    calc.update()
    assert calc.value == 2

    calc.set_inputs(y=Data(1))

    with pytest.raises(RuntimeError):
        calc.value


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_n_to_1_calculator(Calc) -> None:
    x_vec = np.array([1.0, 2.0])
    x = Data(x_vec, _name="x")
    calc = Calc(np.linalg.norm, x)
    calc.update()

    assert np.allclose(calc.value, np.linalg.norm(x_vec))


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_n_to_n_calculator(Calc) -> None:
    x_vec = np.array([[0, 1, 2], [3, 4, 5], [7, 8, 9]])
    x = Data(x_vec, _name="x")
    calc = Calc(np.transpose, x)
    calc.update()

    assert np.allclose(calc.value, np.transpose(x_vec))


def test_calculator_error_in_update() -> None:
    x = Data(2.0, _name="x")

    def update_fn(x):
        raise ValueError("Testing error message.")

    calc = Calc(update_fn, x=x)
    with pytest.raises(RuntimeError):
        calc.update()

    calc = Calc(lambda x: x / 0, x=x)
    with pytest.raises(RuntimeError):
        calc.update()


def test_transient_calculator_error_in_update() -> None:
    x = Data(2.0, _name="x")

    def update_fn(x):
        raise ValueError("Testing error message.")

    calc = TransientCalc(update_fn, x=x)
    with pytest.raises(RuntimeError):
        calc.value

    calc = TransientCalc(lambda x: x / 0, x=x)
    with pytest.raises(RuntimeError):
        calc.value


def test_calculator_update_on_init_error(local_caplog) -> None:
    def _raise_error(a):
        raise RuntimeError("Testing Error")

    a = Data(1.0)
    with local_caplog() as caplog:
        Calc(_raise_error, a)
        assert "was not updated during initialization" in caplog.records[0].message
        assert "Calc" in caplog.records[0].message
        assert "RuntimeError" in caplog.records[0].message

    with local_caplog(level=logging.DEBUG) as caplog:
        Calc(_raise_error, a)
        assert "was not updated during initialization" in caplog.records[1].message
        assert "Calc" in caplog.records[1].message
        assert "RuntimeError" not in caplog.records[1].message
        assert caplog.records[1].exc_info is not None
        assert caplog.records[1].exc_text is not None


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_calculator_update_on_init(Calc, local_caplog) -> None:
    a = Data(0.0)
    b = Calc(np.exp, a)
    assert b.value == pytest.approx(1.0)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test Dist, TransientDist ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_distribution_var(Dist) -> None:
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    v1 = Var(value=dist)

    assert dist.var is v1


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_distribution_init(Dist) -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    id_all_inputs = {id(i) for i in dist.all_input_nodes()}

    assert dist.distribution is tfd.Normal
    assert id(dist.at) == id(x)
    assert id_all_inputs == {id(x), id(loc), id(scale)}


def test_distribution_clear_state() -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.clear_state()

    assert dist.outdated
    assert dist.value is None


def test_transient_distribution_clear_state() -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = TransientDist(tfd.Normal, loc, scale)
    dist.at = x
    dist.clear_state()

    assert dist.outdated
    assert dist.value == pytest.approx(tfd.Normal(0.0, 1.0).log_prob(1.0))


def test_distribution_state_manipulation() -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.state = NodeState(1.0, True)

    assert dist.state.value == pytest.approx(1.0)

    dist.update()

    assert dist.state.value != 1.0
    assert dist.state.outdated


def test_transient_distribution_state_manipulation() -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = TransientDist(tfd.Normal, loc, scale)
    dist.at = x
    dist.state = NodeState(1.0, True)

    assert dist.state.value is None
    assert dist.state.extra is None

    dist.update()

    assert dist.state.value is None
    assert dist.state.extra is None


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_distribution_arg_all_input_nodes(Dist) -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    id_all_inputs = {id(i) for i in dist.all_input_nodes()}

    assert id_all_inputs == {id(x), id(loc), id(scale)}


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_distribution_kwarg_all_input_nodes(Dist) -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc=loc, scale=scale)
    dist.at = x
    id_all_inputs = {id(i) for i in dist.all_input_nodes()}

    assert id_all_inputs == {id(x), id(loc), id(scale)}


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_manipulated_distribution_all_input_nodes(Dist) -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc=loc, scale=scale)
    dist.at = x
    dist.update()

    y = Data(3.0, _name="y")
    dist.at = y
    id_all_inputs = {id(i) for i in dist.all_input_nodes()}

    assert id_all_inputs == {id(y), id(loc), id(scale)}


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_univariate_distribution(Dist) -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    assert dist.value == pytest.approx(tfd.Normal(0.0, 1.0).log_prob(1.0))
    assert dist.log_prob == pytest.approx(tfd.Normal(0.0, 1.0).log_prob(1.0))


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_frozen_distribution_name_manipulation(Dist) -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    with pytest.raises(RuntimeError):
        _ = Model([dist])
        dist.name = "normal"


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_frozen_distribution_needs_seed_manipulation(Dist) -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    with pytest.raises(RuntimeError):
        _ = Model([dist])
        dist.needs_seed = True


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_frozen_distribution_distribution_manipulation(Dist) -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    with pytest.raises(RuntimeError):
        _ = Model([dist])
        dist.distribution = tfd.Exponential


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_frozen_distribution_inputs_manipulation(Dist) -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    rate = Data(1.0, _name="rate")

    with pytest.raises(RuntimeError):
        _ = Model([dist])
        dist.set_inputs(rate)


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_frozen_distribution_kwinputs_manipulation(Dist) -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    with pytest.raises(RuntimeError):
        _ = Model([dist])
        dist.set_inputs()

    with pytest.raises(RuntimeError):
        dist.set_inputs(rate=Data(1.0))


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_distribution_input_manipulation(Dist) -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    rate = Data(1.0, _name="rate")
    dist.distribution = tfd.Exponential
    dist.set_inputs(rate)
    x = Data(2.0, _name="x")
    dist.at = x
    dist.update()

    assert id(dist.at) == id(x)
    assert dist.distribution is tfd.Exponential
    assert dist.value == pytest.approx(tfd.Exponential(1.0).log_prob(2.0))


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_distribution_kwinput_manipulation(Dist) -> None:
    x = Data(1.0, _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    dist.distribution = tfd.Exponential
    dist.set_inputs(rate=Data(1.0))
    x = Data(2.0, _name="x")
    dist.at = x
    dist.update()

    assert id(dist.at) == id(x)
    assert dist.distribution is tfd.Exponential
    assert dist.value == pytest.approx(tfd.Exponential(1.0).log_prob(2.0))


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_univariate_batched_distribution(Dist) -> None:
    x = Data([1.0, 2.0], _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.per_obs = True
    dist.update()

    assert np.allclose(dist.value, tfd.Normal(0.0, 1.0).log_prob([1.0, 2.0]))


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_univariate_batched_distribution_per_obs_false(Dist) -> None:
    x = Data([1.0, 2.0], _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.per_obs = True
    dist.update()

    assert dist.per_obs
    assert np.allclose(dist.value, tfd.Normal(0.0, 1.0).log_prob([1.0, 2.0]))


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_univariate_batched_distribution_per_obs_default(Dist) -> None:
    x = Data([1.0, 2.0], _name="x")
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    assert dist.per_obs
    assert np.allclose(dist.value, tfd.Normal(0.0, 1.0).log_prob([1.0, 2.0]))


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_multivariate_distribution(Dist) -> None:
    x_vec = np.array([1.0, 0.5])
    loc_vec = np.array([0.0, 0.5])
    scale_m = np.identity(2)
    x = Data(x_vec, _name="x")
    loc = Data(loc_vec, _name="loc")
    scale = Data(scale_m, _name="scale")
    dist = Dist(tfd.MultivariateNormalFullCovariance, loc, scale)
    dist.at = x
    dist.update()

    assert id(dist.at) == id(x)
    assert dist.distribution is tfd.MultivariateNormalFullCovariance
    assert dist.value == pytest.approx(
        tfd.MultivariateNormalFullCovariance(loc_vec, scale_m).log_prob(x_vec)
    )


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_multivariate_batched_distribution(Dist) -> None:
    x_vec = np.array([[1.0, 0.5], [0.3, 0.8]])
    x = Data(x_vec, _name="x")
    loc_vec = np.array([0.0, 0.5])
    scale_m = np.identity(2)
    loc = Data(loc_vec, _name="loc")
    scale = Data(scale_m, _name="scale")
    dist = Dist(tfd.MultivariateNormalFullCovariance, loc, scale)
    dist.at = x
    dist.per_obs = True
    dist.update()

    assert np.allclose(
        dist.value,
        tfd.MultivariateNormalFullCovariance(loc_vec, scale_m).log_prob(x_vec),
    )


class TestDistGetitem:
    def test_anonymous_values(self):
        x = Dist(tfd.Normal, loc=0.0, scale=1.0)

        assert x[0] is x.all_input_nodes()[0]
        assert x[1] is x.all_input_nodes()[1]

        x = Dist(tfd.Normal, loc=Data(0.0), scale=Data(1.0))

        assert x[0] is x.all_input_nodes()[0]
        assert x[1] is x.all_input_nodes()[1]

    def test_anonymous_values_positional(self):
        x = Dist(tfd.Normal, 0.0, 1.0)

        assert x[0] is x.all_input_nodes()[0]
        assert x[1] is x.all_input_nodes()[1]

    def test_vars_named_kw(self):
        loc = Var(0.0, name="loc")
        scale = Var(1.0, name="scale")
        x = Dist(tfd.Normal, loc=loc, scale=scale)

        assert x[0] is loc
        assert x[1] is scale

        assert x["loc"] is loc
        assert x["scale"] is scale

    def test_vars_named_positional(self):
        loc = Var(0.0, name="loc")
        scale = Var(1.0, name="scale")
        x = Dist(tfd.Normal, loc, scale)

        assert x[0] is loc
        assert x[1] is scale

        with pytest.raises(KeyError):
            x["loc"]

        with pytest.raises(KeyError):
            x["scale"]

    def test_vars_unnamed(self):
        loc = Var(0.0)
        scale = Var(1.0)
        x = Dist(tfd.Normal, loc=loc, scale=scale)

        assert x[0] is loc
        assert x[1] is scale

        assert x["loc"] is loc
        assert x["scale"] is scale


class TestCalcGetitem:
    def test_anonymous_values(self):
        x = Calc(lambda loc, scale: loc * scale, loc=0.0, scale=1.0)

        assert x[0] is x.all_input_nodes()[0]
        assert x[1] is x.all_input_nodes()[1]

    def test_anonymous_value_nodes(self):
        x = Calc(lambda loc, scale: loc * scale, loc=Data(0.0), scale=Data(1.0))

        assert x[0] is x.all_input_nodes()[0]
        assert x[1] is x.all_input_nodes()[1]

        loc = Data(0.0)
        scale = Data(0.0)
        x = Calc(lambda loc, scale: loc * scale, loc=loc, scale=scale)

        assert x[0] is loc
        assert x[1] is scale

    def test_anonymous_values_positional(self):
        x = Calc(lambda loc, scale: loc * scale, 0.0, 1.0)

        assert x[0] is x.all_input_nodes()[0]
        assert x[1] is x.all_input_nodes()[1]

    def test_vars_named_kw(self):
        loc = Var(0.0, name="loc")
        scale = Var(1.0, name="scale")
        x = Calc(lambda loc, scale: loc * scale, loc=loc, scale=scale)

        assert x[0] is loc
        assert x[1] is scale

        assert x["loc"] is loc
        assert x["scale"] is scale

    def test_vars_named_positional(self):
        loc = Var(0.0, name="loc")
        scale = Var(1.0, name="scale")
        x = Calc(lambda loc, scale: loc * scale, loc, scale)

        assert x[0] is loc
        assert x[1] is scale

        with pytest.raises(KeyError):
            x["loc"]

        with pytest.raises(KeyError):
            x["scale"]

    def test_vars_unnamed(self):
        loc = Var(0.0)
        scale = Var(1.0)
        x = Calc(lambda loc, scale: loc * scale, loc=loc, scale=scale)

        assert x[0] is loc
        assert x[1] is scale

        assert x["loc"] is loc
        assert x["scale"] is scale


class TestVarGetitem:
    def test_calc_with_var_input(self):
        xvar = Var(2.0, name="xvar")
        y = Var(Calc(lambda xarg: xarg + 1.0, xarg=xvar))

        assert y[0] is xvar
        assert y["xarg"] is xvar

    def test_calc_with_positional_var_input(self):
        xvar = Var(2.0, name="xvar")
        y = Var(Calc(lambda xarg: xarg + 1.0, xvar))

        assert y[0] is xvar
        with pytest.raises(KeyError):
            y["xarg"]

    def test_calc_with_multiple_var_inputs(self):
        xvar = Var(2.0, name="xvar")
        zvar = Var(3.0, name="zvar")
        y = Var(Calc(lambda xarg, zarg: xarg + zarg, xarg=xvar, zarg=zvar))

        assert y[0] is xvar
        assert y["xarg"] is xvar

        assert y[1] is zvar
        assert y["zarg"] is zvar

    def test_calc_with_node_input(self):
        xnode = Data(2.0, _name="xnode")
        y = Var(Calc(lambda xarg: xarg + 1.0, xarg=xnode))

        assert y[0] is xnode
        assert y["xarg"] is xnode

    def test_calc_with_float_input(self):
        xfloat = 2.0
        y = Var(Calc(lambda xarg: xarg + 1.0, xarg=2.0))

        assert y[0] is not xfloat
        assert y["xarg"] is not xfloat

        assert y[0] is y.value_node.all_input_nodes()[0]
        assert y["xarg"] is y.value_node.all_input_nodes()[0]

    def test_dist_with_var_input(self):
        xvar = Var(2.0, name="xvar")
        y = Var(1.0, Dist(tfd.Normal, loc=xvar, scale=1.0))

        assert y.dist_node[0] is xvar
        assert y.dist_node["loc"] is xvar
