import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.numpy.distributions as tfd

import liesel.model as lsl


def test_create_subset_log_prob_calc():
    # create 3 variables: 2 params with normal dist (requiring reduction), 1 sum calc
    param1 = lsl.Var.new_param(
        jnp.array([1.0, 2.0]), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="param1"
    )
    param2 = lsl.Var.new_param(
        jnp.array([1.0, 2.0]), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="param2"
    )

    # sum of the two params
    sum_var = lsl.Var.new_calc(lambda x, y: x + y, param1, param2, name="sum_var")

    # create subset with only param1 distribution
    subset_calc = lsl.create_subset_log_prob_calc([param1], "_subset_param1")

    # build model with subset calc included
    gb = lsl.GraphBuilder().add(param1).add(param2).add(sum_var).add(subset_calc)
    model = gb.build_model()

    # get full model log prob (should include both param1 and param2 distributions)
    full_log_prob = model.log_prob

    # get subset log prob
    subset_log_prob = model.nodes["_subset_param1"].value

    # test: since both params have same values and same distribution,
    # subset should be half of full model log prob
    expected_half = full_log_prob / 2.0

    # verify they're approximately equal
    assert subset_log_prob == pytest.approx(expected_half)


def test_create_subset_log_prob_calc_empty_subset():
    # create a simple variable
    param1 = lsl.Var.new_param(
        1.0, lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="param1"
    )

    # test with empty list
    with pytest.raises(ValueError):
        lsl.create_subset_log_prob_calc([], "_empty_subset")

    # test with non-dist variables
    calc_var = lsl.Var.new_calc(lambda x: x + 1, param1, name="calc_var")
    with pytest.raises(ValueError):
        lsl.create_subset_log_prob_calc([calc_var], "_invalid_subset")
