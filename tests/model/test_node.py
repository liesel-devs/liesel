import logging

import numpy as np
import pytest
import tensorflow_probability.substrates.numpy.distributions as tfd

from liesel.model.model import Model
from liesel.model.nodes import (
    Calc,
    Dist,
    NodeState,
    TransientCalc,
    TransientDist,
    Value,
    Var,
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test Data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_data_var() -> None:
    x = Value(1.0, _name="x")
    v1 = Var(value=x)

    assert x.var is v1


def test_data_init() -> None:
    x = Value(0.0, _name="node")

    assert x.value == pytest.approx(0.0)
    assert x.name == "node"
    assert isinstance(x.value, float)
    assert not x.model
    assert x.state.value == pytest.approx(0.0)
    assert not x.state.outdated


def test_data_clear_state() -> None:
    x = Value(1.0, _name="x")
    x.clear_state()

    assert not x.outdated
    assert x.value is None


def test_data_state_manipulation() -> None:
    x = Value(0.0, _name="x")
    x.state = NodeState(1.0, True)
    x.name = "n"
    x.update()

    assert x.name == "n"
    assert x.state.value == pytest.approx(1.0)
    assert not x.state.outdated


def test_data() -> None:
    x = Value(13.0, _name="x")

    assert x.value == pytest.approx(13.0)
    assert not x.outdated

    x.value = 14.0

    assert x.value == pytest.approx(14.0)
    assert not x.outdated


def test_frozen_data_name_manipulation() -> None:
    x = Value(2.0, _name="x")
    x.update()

    with pytest.raises(RuntimeError):
        _ = Model([x])
        x.name = "a"


def test_frozen_data_needs_seed_manipulation() -> None:
    x = Value(2.0, _name="x")
    x.update()

    with pytest.raises(RuntimeError):
        _ = Model([x])
        x.needs_seed = True


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test Calc, TransientCalc ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_calculator_var(Calc) -> None:
    x = Value(2.0, _name="x")
    calc = Calc(np.exp, x)
    v1 = Var(value=calc)

    assert calc.var is v1


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_calculator_init(Calc) -> None:
    x = Value(2.0, _name="x")
    calc = Calc(np.exp, x)
    id_all_inputs = {id(i) for i in calc.all_input_nodes()}

    assert calc.function is np.exp
    assert id_all_inputs == {id(x)}


def test_calculator_clear_state() -> None:
    x = Value(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.clear_state()

    assert calc.outdated
    assert calc.value is None


def test_transient_calculator_clear_state() -> None:
    x = Value(2.0, _name="x")
    calc = TransientCalc(np.exp, x)
    calc.clear_state()

    assert calc.outdated
    assert calc.value == pytest.approx(np.exp(2.0))


def test_calculator_state_manipulation() -> None:
    x = Value(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.state = NodeState(1.0, True)

    assert calc.state.value == pytest.approx(1.0)

    calc.update()

    assert calc.state.value != 1.0
    assert calc.state.outdated


def test_transient_calculator_state_manipulation() -> None:
    x = Value(2.0, _name="x")
    calc = TransientCalc(np.exp, x)
    calc.state = NodeState(1.0, True)

    assert calc.state.value is None
    assert calc.state.extra is None

    calc.update()

    assert calc.state.value is None
    assert calc.state.extra is None


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_calculator_arg_input_nodes(Calc) -> None:
    x = Value(2.0, _name="x")
    calc = Calc(np.exp, x)
    id_all_inputs = {id(i) for i in calc.all_input_nodes()}

    assert id_all_inputs == {id(x)}


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_calculator_kwarg_input_nodes(Calc) -> None:
    x = Value(2.0, _name="x")
    calc = Calc(np.exp, x=x)
    id_all_inputs = {id(i) for i in calc.all_input_nodes()}

    assert id_all_inputs == {id(x)}


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_manipulated_calculator_input_nodes(Calc) -> None:
    x = Value(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.update()

    y = Value(3.0, _name="y")
    calc.set_inputs(y)
    id_all_inputs = {id(i) for i in calc.all_input_nodes()}

    assert id_all_inputs == {id(y)}


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_univariate_calculator(Calc) -> None:
    x = Value(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.update()

    assert calc.value == pytest.approx(np.exp(2.0))


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_frozen_calc_name_manipulation(Calc) -> None:
    x = Value(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.update()

    with pytest.raises(RuntimeError):
        _ = Model([calc])
        calc.name = "a"


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_frozen_calc_needs_seed_manipulation(Calc) -> None:
    x = Value(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.update()

    with pytest.raises(RuntimeError):
        _ = Model([calc])
        calc.needs_seed = True


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_frozen_calculator_function_manipulation(Calc) -> None:
    x = Value(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.update()

    with pytest.raises(RuntimeError):
        _ = Model([calc])
        calc.function = np.log


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_frozen_calculator_kwinputs_manipulation(Calc) -> None:
    x = Value(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.update()

    with pytest.raises(RuntimeError):
        _ = Model([calc])
        calc.set_inputs()

    with pytest.raises(RuntimeError):
        calc.set_inputs(y=Value(1.0))


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_calculator_input_manipulation(Calc) -> None:
    x = Value(2.0, _name="x")
    calc = Calc(np.exp, x)
    calc.update()

    y = Value(1.0, _name="y")
    calc.set_inputs(y)
    calc.update()

    assert calc.value == pytest.approx(np.exp(1.0))


def test_calculator_kwinput_manipulation() -> None:
    x = Value(2, _name="x")
    calc = Calc(lambda x: x, x=x)
    calc.update()
    assert calc.value == 2

    calc.set_inputs(y=Value(1))

    with pytest.raises(RuntimeError):
        calc.update()


def test_transient_calculator_kwinput_manipulation() -> None:
    x = Value(2, _name="x")
    calc = TransientCalc(lambda x: x, x=x)
    calc.update()
    assert calc.value == 2

    calc.set_inputs(y=Value(1))

    with pytest.raises(RuntimeError):
        calc.value


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_n_to_1_calculator(Calc) -> None:
    x_vec = np.array([1.0, 2.0])
    x = Value(x_vec, _name="x")
    calc = Calc(np.linalg.norm, x)
    calc.update()

    assert np.allclose(calc.value, np.linalg.norm(x_vec))


@pytest.mark.parametrize("Calc", [Calc, TransientCalc])
def test_n_to_n_calculator(Calc) -> None:
    x_vec = np.array([[0, 1, 2], [3, 4, 5], [7, 8, 9]])
    x = Value(x_vec, _name="x")
    calc = Calc(np.transpose, x)
    calc.update()

    assert np.allclose(calc.value, np.transpose(x_vec))


def test_calculator_error_in_update() -> None:
    x = Value(2.0, _name="x")

    def update_fn(x):
        raise ValueError("Testing error message.")

    calc = Calc(update_fn, x=x)
    with pytest.raises(RuntimeError):
        calc.update()

    calc = Calc(lambda x: x / 0, x=x)
    with pytest.raises(RuntimeError):
        calc.update()


def test_transient_calculator_error_in_update() -> None:
    x = Value(2.0, _name="x")

    def update_fn(x):
        raise ValueError("Testing error message.")

    calc = TransientCalc(update_fn, x=x)
    with pytest.raises(RuntimeError):
        calc.value

    calc = TransientCalc(lambda x: x / 0, x=x)
    with pytest.raises(RuntimeError):
        calc.value


def test_calculator__update_on_init_error(local_caplog) -> None:
    def _raise_error(a):
        raise RuntimeError("Testing Error")

    a = Value(1.0)
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
def test_calculator__update_on_init(Calc, local_caplog) -> None:
    a = Value(0.0)
    b = Calc(np.exp, a)
    assert b.value == pytest.approx(1.0)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test Dist, TransientDist ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_distribution_var(Dist) -> None:
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    v1 = Var(value=dist)

    assert dist.var is v1


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_distribution_init(Dist) -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    id_all_inputs = {id(i) for i in dist.all_input_nodes()}

    assert dist.distribution is tfd.Normal
    assert id(dist.at) == id(x)
    assert id_all_inputs == {id(x), id(loc), id(scale)}


def test_distribution_clear_state() -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.clear_state()

    assert dist.outdated
    assert dist.value is None


def test_transient_distribution_clear_state() -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = TransientDist(tfd.Normal, loc, scale)
    dist.at = x
    dist.clear_state()

    assert dist.outdated
    assert dist.value == pytest.approx(tfd.Normal(0.0, 1.0).log_prob(1.0))


def test_distribution_state_manipulation() -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.state = NodeState(1.0, True)

    assert dist.state.value == pytest.approx(1.0)

    dist.update()

    assert dist.state.value != 1.0
    assert dist.state.outdated


def test_transient_distribution_state_manipulation() -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
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
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    id_all_inputs = {id(i) for i in dist.all_input_nodes()}

    assert id_all_inputs == {id(x), id(loc), id(scale)}


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_distribution_kwarg_all_input_nodes(Dist) -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc=loc, scale=scale)
    dist.at = x
    id_all_inputs = {id(i) for i in dist.all_input_nodes()}

    assert id_all_inputs == {id(x), id(loc), id(scale)}


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_manipulated_distribution_all_input_nodes(Dist) -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc=loc, scale=scale)
    dist.at = x
    dist.update()

    y = Value(3.0, _name="y")
    dist.at = y
    id_all_inputs = {id(i) for i in dist.all_input_nodes()}

    assert id_all_inputs == {id(y), id(loc), id(scale)}


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_univariate_distribution(Dist) -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    assert dist.value == pytest.approx(tfd.Normal(0.0, 1.0).log_prob(1.0))
    assert dist.log_prob == pytest.approx(tfd.Normal(0.0, 1.0).log_prob(1.0))


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_frozen_distribution_name_manipulation(Dist) -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    with pytest.raises(RuntimeError):
        _ = Model([dist])
        dist.name = "normal"


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_frozen_distribution_needs_seed_manipulation(Dist) -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    with pytest.raises(RuntimeError):
        _ = Model([dist])
        dist.needs_seed = True


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_frozen_distribution_distribution_manipulation(Dist) -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    with pytest.raises(RuntimeError):
        _ = Model([dist])
        dist.distribution = tfd.Exponential


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_frozen_distribution_inputs_manipulation(Dist) -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    rate = Value(1.0, _name="rate")

    with pytest.raises(RuntimeError):
        _ = Model([dist])
        dist.set_inputs(rate)


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_frozen_distribution_kwinputs_manipulation(Dist) -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    with pytest.raises(RuntimeError):
        _ = Model([dist])
        dist.set_inputs()

    with pytest.raises(RuntimeError):
        dist.set_inputs(rate=Value(1.0))


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_distribution_input_manipulation(Dist) -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    rate = Value(1.0, _name="rate")
    dist.distribution = tfd.Exponential
    dist.set_inputs(rate)
    x = Value(2.0, _name="x")
    dist.at = x
    dist.update()

    assert id(dist.at) == id(x)
    assert dist.distribution is tfd.Exponential
    assert dist.value == pytest.approx(tfd.Exponential(1.0).log_prob(2.0))


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_distribution_kwinput_manipulation(Dist) -> None:
    x = Value(1.0, _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.update()

    dist.distribution = tfd.Exponential
    dist.set_inputs(rate=Value(1.0))
    x = Value(2.0, _name="x")
    dist.at = x
    dist.update()

    assert id(dist.at) == id(x)
    assert dist.distribution is tfd.Exponential
    assert dist.value == pytest.approx(tfd.Exponential(1.0).log_prob(2.0))


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_univariate_batched_distribution(Dist) -> None:
    x = Value([1.0, 2.0], _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.per_obs = True
    dist.update()

    assert np.allclose(dist.value, tfd.Normal(0.0, 1.0).log_prob([1.0, 2.0]))


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_univariate_batched_distribution_per_obs_false(Dist) -> None:
    x = Value([1.0, 2.0], _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = x
    dist.per_obs = True
    dist.update()

    assert dist.per_obs
    assert np.allclose(dist.value, tfd.Normal(0.0, 1.0).log_prob([1.0, 2.0]))


@pytest.mark.parametrize("Dist", [Dist, TransientDist])
def test_univariate_batched_distribution_per_obs_default(Dist) -> None:
    x = Value([1.0, 2.0], _name="x")
    loc = Value(0.0, _name="loc")
    scale = Value(1.0, _name="scale")
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
    x = Value(x_vec, _name="x")
    loc = Value(loc_vec, _name="loc")
    scale = Value(scale_m, _name="scale")
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
    x = Value(x_vec, _name="x")
    loc_vec = np.array([0.0, 0.5])
    scale_m = np.identity(2)
    loc = Value(loc_vec, _name="loc")
    scale = Value(scale_m, _name="scale")
    dist = Dist(tfd.MultivariateNormalFullCovariance, loc, scale)
    dist.at = x
    dist.per_obs = True
    dist.update()

    assert np.allclose(
        dist.value,
        tfd.MultivariateNormalFullCovariance(loc_vec, scale_m).log_prob(x_vec),
    )


class TestDistGetitem:
    def test_positional_access_works(self):
        x = Dist(tfd.Normal, 0.0, 1.0)

        assert x[0] is x.inputs[0]
        assert x[1] is x.inputs[1]

    def test_keyword_access_to_positional_args_raises_error(self):
        loc = Var(0.0, name="loc")
        scale = Var(1.0, name="scale")
        x = Dist(tfd.Normal, loc, scale)

        assert x[0] is loc
        assert x[1] is scale

        with pytest.raises(KeyError):
            x["loc"]

        with pytest.raises(KeyError):
            x["scale"]

    def test_positional_access_to_keyword_args_raises_error(self):
        loc = Var(0.0)
        scale = Var(1.0)
        x = Dist(tfd.Normal, loc=loc, scale=scale)

        with pytest.raises(IndexError):
            x[0]

        with pytest.raises(IndexError):
            x[1]

    def test_keyword_access_works(self):
        loc = Var(0.0)
        scale = Var(1.0)
        x = Dist(tfd.Normal, loc=loc, scale=scale)

        assert x["loc"] is loc
        assert x["scale"] is scale


class TestDistSetitem:
    def test_replacing_input_with_self_breaks_model(self):
        a = Dist(tfd.Normal, loc=0.0, scale=1.0)
        b = Var(2.0, a)
        a["loc"] = a

        with pytest.raises(Exception):
            Model([b])

    def test_replacing_keyword_input_leads_to_correct_log_prob(self):
        x = Var(0.0, Dist(tfd.Normal, loc=0.0, scale=1.0)).update()

        assert x.log_prob == pytest.approx(
            tfd.Normal(loc=0.0, scale=1.0).log_prob(x.value)
        )

        x.dist_node["loc"] = 2.0
        x.update()

        assert x.log_prob == pytest.approx(
            tfd.Normal(loc=2.0, scale=1.0).log_prob(x.value)
        )
        assert len(x.dist_node._loc) == 2

        x.dist_node["scale"] = 2.0
        x.update()

        assert x.log_prob == pytest.approx(
            tfd.Normal(loc=2.0, scale=2.0).log_prob(x.value)
        )

    def test_cannot_replace_dist_at(self):
        x = Var(0.0, Dist(tfd.Normal, loc=0.0, scale=1.0)).update()
        with pytest.raises(IndexError):
            x.dist_node[2] = Var(1.0)

    def test_replacing_positional_input_leads_to_correct_log_prob(self):
        x = Var(0.0, Dist(tfd.Normal, 0.0, 1.0)).update()

        assert x.log_prob == pytest.approx(
            tfd.Normal(loc=0.0, scale=1.0).log_prob(x.value)
        )

        x.dist_node[0] = 2.0
        x.update()

        assert x.log_prob == pytest.approx(
            tfd.Normal(loc=2.0, scale=1.0).log_prob(x.value)
        )

    def test_positional_input_cannot_be_replaced_via_keyword(self):
        x = Var(0.0, Dist(tfd.Normal, 0.0, 1.0)).update()

        with pytest.raises(KeyError):
            x.dist_node["loc"] = 2.0

    def test_replacing_an_input_node_with_a_var_works(self):
        x = Var(0.0, Dist(tfd.Normal, 0.0, 1.0)).update()

        assert x.log_prob == pytest.approx(
            tfd.Normal(loc=0.0, scale=1.0).log_prob(x.value)
        )

        x.dist_node[0] = Var(2.0, name="loc")
        x.update()

        assert x.log_prob == pytest.approx(
            tfd.Normal(loc=2.0, scale=1.0).log_prob(x.value)
        )

    def test_del_raises_attribute_error(self):
        x = Var(
            0.0,
            Dist(tfd.Normal, loc=Var(0.0, name="loc"), scale=Var(1.0, name="scale")),
        ).update()

        with pytest.raises(AttributeError):
            del x.dist_node["loc"]

        x = Var(
            0.0, Dist(tfd.Normal, Var(0.0, name="loc"), scale=Var(1.0, name="scale"))
        ).update()

        with pytest.raises(AttributeError):
            del x.dist_node[0]

        x = Var(0.0, Dist(tfd.Normal, 0.0, scale=Var(1.0, name="scale"))).update()

        with pytest.raises(AttributeError):
            del x.dist_node[0]

    def test_assign_none_works(self):
        x = Var(
            0.0,
            Dist(tfd.Normal, loc=Var(0.0, name="loc"), scale=Var(1.0, name="scale")),
        ).update()

        x.dist_node["loc"] = None

        assert isinstance(x.dist_node["loc"], Value)
        assert x.dist_node["loc"].value is None


class TestCalcGetitem:
    def test_positional_access_to_keyword_args_raises_error(self):
        x = Calc(lambda loc, scale: loc * scale, loc=0.0, scale=1.0)

        with pytest.raises(IndexError):
            x[0]

        with pytest.raises(IndexError):
            x[1]

    def test_positional_access_works(self):
        x = Calc(lambda loc, scale: loc * scale, 0.0, 1.0)

        assert x[0] is x.inputs[0]
        assert x[1] is x.inputs[1]

    def test_keyword_access_works(self):
        loc = Var(0.0, name="loc")
        scale = Var(1.0, name="scale")
        x = Calc(lambda loc, scale: loc * scale, loc=loc, scale=scale)

        assert x["loc"] is loc
        assert x["scale"] is scale

    def test_keyword_access_to_positional_args_raises_error(self):
        loc = Var(0.0, name="loc")
        scale = Var(1.0, name="scale")
        x = Calc(lambda loc, scale: loc * scale, loc, scale)

        with pytest.raises(KeyError):
            x["loc"]

        with pytest.raises(KeyError):
            x["scale"]

    def test_keyword_access_works_with_unnamed_vars(self):
        """Because keyword access is based on argument name, not variable name."""
        loc = Var(0.0)
        scale = Var(1.0)
        x = Calc(lambda loc, scale: loc * scale, loc=loc, scale=scale)

        with pytest.raises(IndexError):
            x[0]

        with pytest.raises(IndexError):
            x[1]

        assert x["loc"] is loc
        assert x["scale"] is scale

    def test_informative_key_error(self):
        loc = Var(0.0, name="locx")
        scale = Var(1.0, name="scalex")
        x = Calc(lambda loc, scale: loc * scale, loc=loc, scale=scale)

        with pytest.raises(KeyError, match="Available index-variable pairs:"):
            x["locx"]

    def test_informative_index_error(self):
        loc = Var(0.0, name="locx")
        scale = Var(1.0, name="scalex")
        x = Calc(lambda loc, scale: loc * scale, loc, scale)

        with pytest.raises(IndexError, match="Available index-variable pairs:"):
            x[3]


class TestCalcSetItem:
    def test_replacing_input_with_self_breaks_model(self):
        def sum_(*args):
            return sum(args)

        x = Calc(sum_, 1.0, 2.0)

        assert x.value == pytest.approx(3.0)

        x[0] = x

        with pytest.raises(Exception):
            Model([x])

    def test_variable_length_args_cannot_be_extended_by_setitem(self):
        def sum_(*args):
            return sum(args)

        x = Calc(sum_, 1.0, 2.0)

        assert x.value == pytest.approx(3.0)

        x[0] = 2.0
        x.update()
        assert x.value == pytest.approx(4.0)

        with pytest.raises(IndexError):
            x[2] = 3.0

    def test_variable_length_kwargs_cannot_be_extended_by_setitem(self):
        def sum_(**kwargs):
            return sum(kwargs.values())

        x = Calc(sum_, a=1.0, b=2.0)

        assert x.value == pytest.approx(3.0)

        x["a"] = 2.0
        x.update()
        assert x.value == pytest.approx(4.0)

        with pytest.raises(KeyError):
            x["c"] = 3.0

    def test_argument_defaults_cannot_be_changed_by_setitem_after_init(self):
        def sum_(a, b=3.0):
            return a + b

        x = Calc(sum_, 1.0)
        assert x.value == pytest.approx(4.0)

        x[0] = 2.0
        x.update()
        assert x.value == pytest.approx(5.0)

        with pytest.raises(IndexError):
            x[1] = 0.0

        with pytest.raises(KeyError):
            x["b"] = 0.0
