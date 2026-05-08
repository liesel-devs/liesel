import jax.numpy as jnp
import jax.random as jrd
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl


class TestBasicModifyModel:
    def test_biject_variable(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(1.0, name="scale")

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert scale.name in model.parameters

        scale.biject(tfb.Exp())

        assert scale.bijected_var.name in model.vars
        assert scale.bijected_var.name in model.parameters
        assert scale.name not in model.parameters

    def test_rename_variable(self):
        """
        When renaming a var, its corresponding value_node is renamed accordingly if
        it is either unnamed or follows the {var_name}_value pattern.
        """
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0, lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="scale"
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        scale.name = "renamed"

        assert "renamed" in model.vars
        assert scale.value_node.name == "renamed_value"
        assert scale.dist_node.name == "renamed_log_prob"

        assert "scale" not in model.vars
        assert "scale_value" not in model.nodes
        assert "scale_log_prob" not in model.nodes

        # case 2: Input is a variable
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            x,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        scale.name = "renamed"

        assert "renamed" in model.vars
        assert scale.value_node.name == "renamed_value"
        assert scale.dist_node.name == "renamed_log_prob"

        assert "scale" not in model.vars
        assert "scale_value" not in model.nodes
        assert "scale_log_prob" not in model.nodes

        # case 3: Input is a named node
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            lsl.Calc(lambda x: x, x, _name="named_node"),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        scale.name = "renamed"

        assert "renamed" in model.vars
        assert scale.value_node.name == "named_node"
        assert scale.dist_node.name == "renamed_log_prob"

        assert "scale" not in model.vars
        assert "renamed_value" not in model.nodes
        assert "scale_log_prob" not in model.nodes

    def test_remove_dist(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        dist_node = scale.dist_node

        scale.dist_node = None

        assert dist_node.name not in model.nodes

    def test_change_per_obs_of_a_dist(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert y.dist_node.per_obs
        assert y.log_prob.size == 10

        y.dist_node.per_obs = False
        assert not y.dist_node.per_obs
        assert y.log_prob.size == 1

    def test_change_obs_flag_of_a_var(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        prior_before = model.log_prior
        lik_before = model.log_lik
        prob_before = model.log_prob

        assert y.observed
        with pytest.raises(ValueError):
            y.parameter = True

        y.observed, y.parameter = False, True

        model.log_prior == pytest.approx(prior_before + lik_before)
        model.log_lik == pytest.approx(0.0)
        model.log_prob == pytest.approx(prob_before)

    def test_change_parameter_flag_of_a_var(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        prior_before = model.log_prior
        lik_before = model.log_lik
        prob_before = model.log_prob

        assert y.observed
        with pytest.raises(ValueError):
            scale.observed = True

        scale.parameter, scale.observed = False, True

        model.log_prior == pytest.approx(0.0)
        model.log_lik == pytest.approx(prior_before + lik_before)
        model.log_prob == pytest.approx(prob_before)

    def test_node_add_inputs(self):
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")

        loc = lsl.Var.new_calc(lambda *args: sum(args), x1, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value)

        loc.value_node.add_inputs(x2)
        assert jnp.allclose(loc.value, x1.value + x2.value)

        assert x2.name in model.vars

    def test_node_set_inputs(self):
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")

        loc = lsl.Var.new_calc(lambda *args: sum(args), x1, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value)

        loc.value_node.set_inputs(x2)
        assert jnp.allclose(loc.value, x2.value)

        assert x2.name in model.vars
        assert x1.name in model.vars  # now singleton node, not dropped

    def test_modify_names(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0, lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="scale"
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        model.modify_names(lambda x: x + ".m1")

        for var in model.vars.values():
            assert var.name[-3:] == ".m1"

        for node in model.nodes.values():
            if node.name.startswith("_model"):
                continue
            assert node.name[-3:] == ".m1"

        for nv in model.vars | model.nodes:
            if nv.startswith("_model"):
                continue
            assert nv[-3:] == ".m1"

    def test_prefix_names(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0, lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="scale"
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        model.prefix_names("m1.")

        for var in model.vars.values():
            assert var.name[:3] == "m1."

        for node in model.nodes.values():
            if node.name.startswith("_model"):
                continue
            assert node.name[:3] == "m1."

        for nv in model.vars | model.nodes:
            if nv.startswith("_model"):
                continue
            assert nv[:3] == "m1."


class TestBracketReplace:
    def test_bracket_replace_value_input_var_with_var(self):
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")
        x3 = lsl.Var.new_obs(jrd.normal(jrd.key(3), (10,)), name="x3")

        loc = lsl.Var.new_calc(lambda *args: sum(args), x1, x2, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value + x2.value)

        loc.value_node[0] = x3

        assert jnp.allclose(loc.value, x2.value + x3.value)

        assert x3.name in model.vars
        assert x2.name in model.vars
        assert x1.name in model.vars  # now singleton node, not dropped

        # Second case, keyword lookup
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")
        x3 = lsl.Var.new_obs(jrd.normal(jrd.key(3), (10,)), name="x3")

        loc = lsl.Var.new_calc(lambda a, b: sum((a, b)), a=x1, b=x2, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value + x2.value)

        loc.value_node["a"] = x3

        assert jnp.allclose(loc.value, x2.value + x3.value)

        assert x3.name in model.vars
        assert x2.name in model.vars
        assert x1.name in model.vars  # now singleton node, not dropped

    def test_bracket_replace_value_input_var_with_var_that_has_model(self):
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")
        x3 = lsl.Var.new_obs(jrd.normal(jrd.key(3), (10,)), name="x3")
        _ = lsl.Model([x3])

        loc = lsl.Var.new_calc(lambda *args: sum(args), x1, x2, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value + x2.value)

        with pytest.raises(RuntimeError, match="can only be part of one model"):
            loc.value_node[0] = x3

    def test_bracket_replace_value_input_var_with_var_in_same_model(self):
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")
        x3 = lsl.Var.new_obs(jrd.normal(jrd.key(3), (10,)), name="x3")

        loc = lsl.Var.new_calc(lambda *args: sum(args), x1, x2, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y, x3])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value + x2.value)

        loc.value_node[0] = x3

        assert jnp.allclose(loc.value, x2.value + x3.value)

        assert x3.name in model.vars
        assert x2.name in model.vars
        assert x1.name in model.vars  # now singleton node, not dropped

    def test_bracket_replace_value_input_var_with_node(self):
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")
        x3 = lsl.Value(jrd.normal(jrd.key(3), (10,)), _name="x3")

        loc = lsl.Var.new_calc(lambda *args: sum(args), x1, x2, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value + x2.value)

        loc.value_node[0] = x3

        assert jnp.allclose(loc.value, x2.value + x3.value)

        assert x3.name in model.nodes
        assert x2.name in model.vars
        assert x1.name in model.vars  # now singleton node, not dropped

    def test_bracket_replace_value_input_var_with_array(self):
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")
        x3 = lsl.Value(jrd.normal(jrd.key(3), (10,)))

        loc = lsl.Var.new_calc(lambda *args: sum(args), x1, x2, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value + x2.value)

        nodes_before = set(list(model.nodes))

        loc.value_node[0] = x3

        nodes_after = set(list(model.nodes))

        assert len(nodes_after - nodes_before) == 1
        assert x3.name in nodes_after - nodes_before

        assert jnp.allclose(loc.value, x2.value + x3.value)

        assert x3.name in model.nodes
        assert x2.name in model.vars
        assert x1.name in model.vars  # now singleton node, not dropped

    def test_bracket_replace_log_prob_input_var_with_var(self):
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")
        x3 = lsl.Var.new_obs(jrd.normal(jrd.key(3), (10,)), name="x3")

        loc = lsl.Var.new_calc(lambda *args: sum(args), x1, x2, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value + x2.value)

        y.dist_node["loc"] = x3

        assert jnp.allclose(loc.value, x1.value + x2.value)
        assert y.dist_node["loc"] is x3

        assert loc.name in model.vars
        assert x3.name in model.vars
        assert x2.name in model.vars
        assert x1.name in model.vars  # now singleton node, not dropped

    def test_bracket_replace_log_prob_input_var_with_node(self):
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")
        x3 = lsl.Value(jrd.normal(jrd.key(3), (10,)), _name="x3")

        loc = lsl.Var.new_calc(lambda *args: sum(args), x1, x2, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value + x2.value)

        y.dist_node["loc"] = x3

        assert jnp.allclose(loc.value, x1.value + x2.value)
        assert y.dist_node["loc"] is x3

        assert loc.name in model.vars
        assert x3.name in model.nodes
        assert x2.name in model.vars
        assert x1.name in model.vars  # now singleton node, not dropped

    def test_bracket_replace_log_prob_input_var_with_array(self):
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")
        x3 = lsl.Value(jrd.normal(jrd.key(3), (10,)))

        loc = lsl.Var.new_calc(lambda *args: sum(args), x1, x2, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value + x2.value)

        y.dist_node["loc"] = x3

        assert jnp.allclose(loc.value, x1.value + x2.value)
        assert jnp.allclose(y.dist_node["loc"].value, x3.value)

        assert loc.name in model.vars
        assert x3.name in model.nodes
        assert x2.name in model.vars
        assert x1.name in model.vars  # now singleton node, not dropped


class TestReplace:
    def test_replace_var_with_var(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.InverseGamma,
                concentration=lsl.Var.new_param(1.0, name="a"),
                scale=lsl.Var.new_param(1.0, name="b"),
            ),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        scale2 = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.Weibull,
                concentration=0.0,
                scale=lsl.Var.new_param(1.0, name="c"),
            ),
            name="scale2",
        )

        model.replace(scale, scale2)

        assert y.dist_node["scale"] is scale2
        assert scale.name not in model.vars
        assert "scale_value" not in model.nodes
        assert "scale_log_prob" not in model.nodes
        assert "scale_var_value" not in model.nodes

        assert not scale.model

    def test_replace_var_with_var_by_name(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.InverseGamma,
                concentration=lsl.Var.new_param(1.0, name="a"),
                scale=lsl.Var.new_param(1.0, name="b"),
            ),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        scale2 = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.Weibull,
                concentration=0.0,
                scale=lsl.Var.new_param(1.0, name="c"),
            ),
            name="scale2",
        )

        model.replace("scale", scale2)

        assert y.dist_node["scale"] is scale2
        assert scale.name not in model.vars
        assert "scale_value" not in model.nodes
        assert "scale_log_prob" not in model.nodes
        assert "scale_var_value" not in model.nodes

    def test_replace_var_with_var_that_has_model(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.InverseGamma,
                concentration=lsl.Var.new_param(1.0, name="a"),
                scale=lsl.Var.new_param(1.0, name="b"),
            ),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        scale2 = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.Weibull,
                concentration=0.0,
                scale=lsl.Var.new_param(1.0, name="c"),
            ),
            name="scale2",
        )
        _ = lsl.Model([scale2])

        with pytest.raises(RuntimeError, match="can only be part of one model"):
            model.replace(scale, scale2)

        assert y.dist_node["scale"] is scale
        assert scale2.name not in model.vars
        assert scale.name in model.vars
        assert "scale_value" in model.nodes
        assert "scale_log_prob" in model.nodes
        assert "scale_var_value" in model.nodes

    def test_replace_var_with_var_in_same_model(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.InverseGamma,
                concentration=lsl.Var.new_param(1.0, name="a"),
                scale=lsl.Var.new_param(1.0, name="b"),
            ),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        scale2 = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.Weibull,
                concentration=0.0,
                scale=lsl.Var.new_param(1.0, name="c"),
            ),
            name="scale2",
        )

        model = lsl.Model([y, scale2])
        model.locked = False

        model.replace(scale, scale2)

        assert y.dist_node["scale"] is scale2
        assert scale2.name in model.vars
        assert scale.name not in model.vars
        assert "scale_value" not in model.nodes
        assert "scale_log_prob" not in model.nodes
        assert "scale_var_value" not in model.nodes

    def test_replace_seed_var_with_var_in_same_model(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.InverseGamma,
                concentration=lsl.Var.new_param(1.0, name="a"),
                scale=lsl.Var.new_param(1.0, name="b"),
            ),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        scale2 = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.Weibull,
                concentration=0.0,
                scale=lsl.Var.new_param(1.0, name="c"),
            ),
            name="scale2",
        )

        model = lsl.Model([y, scale, scale2])
        model.locked = False

        model.replace(scale, scale2)

        assert y.dist_node["scale"] is scale2
        assert scale2.name in model.vars
        assert scale.name not in model.vars
        assert "scale_value" not in model.nodes
        assert "scale_log_prob" not in model.nodes
        assert "scale_var_value" not in model.nodes

        assert scale2 in model.seed_nodes_and_vars
        assert scale not in model.seed_nodes_and_vars

    def test_replace_var_with_node(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.InverseGamma,
                concentration=lsl.Var.new_param(1.0, name="a"),
                scale=lsl.Var.new_param(1.0, name="b"),
            ),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        scale2 = lsl.Value(1.0, _name="scale2")

        model.replace(scale, scale2)

        assert y.dist_node["scale"] is scale2
        assert scale2.name in model.nodes
        assert scale.name not in model.vars
        assert "scale_value" not in model.nodes
        assert "scale_log_prob" not in model.nodes
        assert "scale_var_value" not in model.nodes

    def test_replace_var_with_array(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.InverseGamma,
                concentration=lsl.Var.new_param(1.0, name="a"),
                scale=lsl.Var.new_param(1.0, name="b"),
            ),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        auto_names_nodes = [
            name
            for name in model.nodes
            if name.startswith("_") and not name.startswith("_model")
        ]
        assert not auto_names_nodes
        model.replace(scale, 10.0)

        assert y.dist_node["scale"].value == pytest.approx(10.0)

        auto_names_nodes = [
            name
            for name in model.nodes
            if name.startswith("_") and not name.startswith("_model")
        ]
        assert not auto_names_nodes
        assert model.vars[scale.name] is not scale
        assert 10.0 == pytest.approx(model.vars[scale.name].value)
        assert scale.name in model.vars
        assert "scale_value" in model.nodes
        assert "scale_var_value" in model.nodes
        assert "scale_log_prob" not in model.nodes

    def test_disconnected_parents_of_replaced_var_are_removed(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.InverseGamma,
                concentration=lsl.Var.new_param(1.0, name="a"),
                scale=lsl.Var.new_param(1.0, name="b"),
            ),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        scale2 = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.Weibull,
                concentration=0.0,
                scale=lsl.Var.new_param(1.0, name="c"),
            ),
            name="scale2",
        )

        model.replace(scale, scale2)

        assert y.dist_node["scale"] is scale2
        assert scale.name not in model.vars
        assert "a" not in model.vars
        assert "b" not in model.vars

    def test_connected_parents_of_replaced_var_are_kept(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.InverseGamma,
                concentration=lsl.Var.new_param(1.0, name="a"),
                scale=lsl.Var.new_param(1.0, name="b"),
            ),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        scale2 = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.Weibull,
                concentration=scale.dist_node["concentration"],
                scale=lsl.Var.new_param(1.0, name="c"),
            ),
            name="scale2",
        )

        model.replace(scale, scale2)

        assert y.dist_node["scale"] is scale2
        assert scale.name not in model.vars
        assert "a" in model.vars
        assert "b" not in model.vars

    def test_replace_var_with_itself(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.InverseGamma,
                concentration=lsl.Var.new_param(1.0, name="a"),
                scale=lsl.Var.new_param(1.0, name="b"),
            ),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        model.replace(scale, scale)

        assert y.dist_node["scale"] is scale
        assert scale.name in model.vars
        assert "scale_value" in model.nodes
        assert "scale_log_prob" in model.nodes
        assert "scale_var_value" in model.nodes

        assert scale.model

        model.replace(y, y)

        assert y.name in model.vars
        assert x.name in model.vars
        assert scale.name in model.vars

        assert y in model.seed_nodes_and_vars

    def test_replace_var_with_other_var_of_same_name(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.InverseGamma,
                concentration=lsl.Var.new_param(1.0, name="a"),
                scale=lsl.Var.new_param(1.0, name="b"),
            ),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        newx = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x")

        model = lsl.Model([y])
        model.locked = False

        model.replace(x, newx)

        assert newx.name in model.vars
        assert model.vars["x"] is newx
        assert model.vars["x"] is not x
        assert newx.value_node.name == "x_value"
        assert newx.var_value_node.name == "x_var_value"

        assert newx not in model.seed_nodes_and_vars
        assert x not in model.seed_nodes_and_vars

    def test_replace_var_with_other_var_of_different_shape(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.InverseGamma,
                concentration=lsl.Var.new_param(1.0, name="a"),
                scale=lsl.Var.new_param(1.0, name="b"),
            ),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        newx = lsl.Var.new_obs(jrd.normal(jrd.key(2), (11,)), name="x")

        model = lsl.Model([y])
        model.locked = False

        with pytest.raises(TypeError):
            model.replace(x, newx)

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(
                tfd.InverseGamma,
                concentration=lsl.Var.new_param(1.0, name="a"),
                scale=lsl.Var.new_param(1.0, name="b"),
            ),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=scale),
            name="y",
        )

        newy = lsl.Var.new_obs(jrd.normal(jrd.key(2), (11,)), name="y")

        model = lsl.Model([y])
        model.locked = False

        model.replace(y, newy)


class TestDropSingletons:
    def test_singleton_var_is_not_dropped(self):
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")

        loc = lsl.Var.new_calc(lambda *args: sum(args), x1, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value)

        loc.value_node.set_inputs(x2)
        assert jnp.allclose(loc.value, x2.value)

        assert x2.name in model.vars
        assert x1.name in model.vars  # now singleton node, not dropped

    def test_singleton_var_is_explicitly_dropped(self):
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")

        loc = lsl.Var.new_calc(lambda *args: sum(args), x1, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value)

        loc.value_node.set_inputs(x2)
        assert jnp.allclose(loc.value, x2.value)
        assert x2.name in model.vars

        model.drop_singletons()
        assert x1.name not in model.vars  # now singleton node, dropped explicitly


class TestModelAdd:
    def test_add_one_var(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(1.0, name="scale")

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        x2 = lsl.Var.new_obs(
            jrd.normal(jrd.key(1), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x2",
        )
        model.add(x2)

        assert x2 in model.seed_nodes_and_vars
        assert x2.name in model.vars
        assert x2.name in model.observed
        assert x2.log_prob is not None
        assert model.log_lik == pytest.approx((y.log_prob + x2.log_prob).sum())

    def test_add_multiple_vars(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(1.0, name="scale")

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        x2 = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x2",
        )

        x3 = lsl.Var.new_obs(
            jrd.normal(jrd.key(3), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x3",
        )
        model.add(x2, x3)

        assert x2.name in model.vars
        assert x2.name in model.observed
        assert x2.log_prob is not None

        assert x3.name in model.vars
        assert x3.name in model.observed
        assert x3.log_prob is not None

        assert model.log_lik == pytest.approx(
            (y.log_prob + x2.log_prob + x3.log_prob).sum()
        )

    def test_add_one_node(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(1.0, name="scale")

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        x2 = lsl.Value(
            jrd.normal(jrd.key(1), (10,)),
            _name="x2",
        )
        model.add(x2)

        assert x2.name in model.nodes

    def test_add_one_model(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(1.0, name="scale")

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        x2 = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x2",
        )

        model2 = lsl.Model([x2])
        model.add(model2)

        assert x not in model.seed_nodes_and_vars
        assert y in model.seed_nodes_and_vars
        assert x2 in model.seed_nodes_and_vars
        assert x2.dist_node.name in model.nodes

    def test_add_one_model_copy(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(1.0, name="scale")

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        x2 = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x2",
        )

        model2 = lsl.Model([x2])
        model.add(model2, copy=True)

        assert x not in model.seed_nodes_and_vars
        assert y in model.seed_nodes_and_vars
        assert x2 not in model.seed_nodes_and_vars

        assert x2.name in model.vars
        assert x2.dist_node.name in model.nodes

        seed_node_names = [n.name for n in model.seed_nodes_and_vars]
        assert x2.name in seed_node_names

    def test_add_name_clash(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(1.0, name="scale")

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        x2 = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x",
        )

        model2 = lsl.Model([x2])
        with pytest.raises(RuntimeError):
            model.add(model2)

        assert "x" in model2.vars
        assert x not in model.seed_nodes_and_vars
        assert y in model.seed_nodes_and_vars
        assert x2 not in model.seed_nodes_and_vars

    def test_add_two_models(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(1.0, name="scale")

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        x2 = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x2",
        )

        x3 = lsl.Var.new_obs(
            jrd.normal(jrd.key(3), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x3",
        )
        model2 = lsl.Model([x2])
        model3 = lsl.Model([x3])

        model.add(model2, model3)

        assert not model2.vars
        assert x2.name not in model2.vars

        assert not model3.vars
        assert x3.name not in model3.vars

        assert model.vars["x2"] is x2
        assert model.vars["x3"] is x3

        assert x2.name in model.vars
        assert x2.name in model.observed
        assert x2.log_prob is not None

        assert x3.name in model.vars
        assert x3.name in model.observed
        assert x3.log_prob is not None

        assert model.log_lik == pytest.approx(
            (y.log_prob + x2.log_prob + x3.log_prob).sum()
        )


class TestModelJoin:
    def test_join_without_overlapping_names(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(1.0, name="scale")

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        x2 = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x2",
        )

        x3 = lsl.Var.new_obs(
            jrd.normal(jrd.key(3), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x3",
        )
        model2 = lsl.Model([x2, x3])

        model.join(model2)

        assert not model2.vars
        assert x2.name not in model2.vars
        assert x3.name not in model2.vars

        assert model.vars["x2"] is x2
        assert model.vars["x3"] is x3

        assert x2.name in model.vars
        assert x2.name in model.observed
        assert x2.log_prob is not None

        assert x3.name in model.vars
        assert x3.name in model.observed
        assert x3.log_prob is not None

        assert model.log_lik == pytest.approx(
            (y.log_prob + x2.log_prob + x3.log_prob).sum()
        )

    def test_join_with_copy(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(1.0, name="scale")

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        x2 = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x",
        )

        x3 = lsl.Var.new_obs(
            jrd.normal(jrd.key(3), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x3",
        )
        model2 = lsl.Model([x2, x3])

        model.join(model2, copy=True)

        assert model2.vars
        assert x2.name in model2.vars
        assert x3.name in model2.vars

        assert model2.vars["x"] is x2
        assert model2.vars["x3"] is x3

        assert "x" not in model.vars
        assert model.vars["x.x"] is x
        assert model.vars["x.y"] is not x2
        assert model.vars["x3"] is not x3

    def test_join_with_renaming(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(1.0, name="scale")

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        x2 = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x",
        )

        x3 = lsl.Var.new_obs(
            jrd.normal(jrd.key(3), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x3",
        )
        model2 = lsl.Model([x2, x3])

        model.join(model2)

        assert not model2.vars

        assert "x" not in model.vars
        assert model.vars["x.x"] is x
        assert model.vars["x.y"] is x2
        assert model.vars["x3"] is x3

        assert model.nodes["x.x_value"] is x.value_node
        assert model.nodes["x.x_var_value"] is x.var_value_node

        assert model.nodes["x.y_value"] is x2.value_node
        assert model.nodes["x.y_var_value"] is x2.var_value_node

        assert model.nodes["x3_value"] is x3.value_node
        assert model.nodes["x3_var_value"] is x3.var_value_node

    def test_join(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(1.0, name="scale")

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        x2 = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x",
        )

        x3 = lsl.Var.new_obs(
            jrd.normal(jrd.key(3), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x3",
        )
        model2 = lsl.Model([x2, x3])

        model.join(model2, by=["x"])

        assert not model2.vars

        assert not x2.model

        assert "x" in model.vars
        assert model.vars["x"] is x
        assert model.vars["x3"] is x3

        assert model.nodes["x_value"] is x.value_node
        assert model.nodes["x_var_value"] is x.var_value_node

        assert x2.name == "x.y"
        assert x2.value_node.name == "x.y_value"
        assert x2.var_value_node.name == "x.y_var_value"

        assert model.nodes["x3_value"] is x3.value_node
        assert model.nodes["x3_var_value"] is x3.var_value_node

    def test_join_by_all(self):
        x = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x")
        scale = lsl.Var.new_param(1.0, name="scale")

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=x, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        x2 = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x",
        )

        y2 = lsl.Var.new_obs(
            jrd.normal(jrd.key(3), (10,)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="y",
        )
        model2 = lsl.Model([x2, y2])

        model.join_by_all(model2)

        assert not model2.vars

        assert not x2.model
        assert not y2.model

        assert "x" in model.vars
        assert model.vars["x"] is x
        assert model.vars["y"] is y

        assert model.nodes["x_value"] is x.value_node
        assert model.nodes["x_var_value"] is x.var_value_node

        assert model.nodes["y_value"] is y.value_node
        assert model.nodes["y_var_value"] is y.var_value_node

        assert x2.name == "x.y"
        assert x2.value_node.name == "x.y_value"
        assert x2.var_value_node.name == "x.y_var_value"

        assert y2.name == "y.y"
        assert y2.value_node.name == "y.y_value"
        assert y2.var_value_node.name == "y.y_var_value"

    def test_join_seed_nodes_behavior(self):
        x1 = lsl.Var.new_obs(1.0, name="x")
        x2 = lsl.Var.new_obs(1.0, name="x")
        y = lsl.Var.new_calc(lambda x: x, x2, name="y")

        m1 = lsl.Model(x1)
        m2 = lsl.Model(x2, y)

        m1.join(m2, by=["x"])
        list(m1.vars)
        list(m2.vars)

        y.value_node[0] is x1
        assert x1 in m1.seed_nodes_and_vars
        assert y in m1.seed_nodes_and_vars
        assert x2 not in m1.seed_nodes_and_vars
        assert len(m1.seed_nodes_and_vars) == 2


class TestRebuildModel:
    def test_singleton_var_is_dropped(self):
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")

        loc = lsl.Var.new_calc(lambda *args: sum(args), x1, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value)

        loc.value_node.set_inputs(x2)
        assert jnp.allclose(loc.value, x2.value)

        model.rebuild_graph()

        assert x2.name in model.vars
        assert x1.name not in model.vars

    def test_rebuild_after_dropping_singletons(self):
        x1 = lsl.Var.new_obs(1.0, name="x1")
        x2 = lsl.Var.new_obs(1.0, name="x2")

        m = lsl.Model(x1, x2)
        m.drop_singletons()
        assert not list(m.vars)
        assert len(list(m.nodes)) == 3

        assert x1 in m.seed_nodes_and_vars
        assert x2 in m.seed_nodes_and_vars

        m.rebuild_graph()
        assert len(list(m.vars)) == 2
        assert len(list(m.nodes)) == 7

    def test_rebuild_from_x2(self):
        x1 = lsl.Var.new_obs(jrd.normal(jrd.key(1), (10,)), name="x1")
        x2 = lsl.Var.new_obs(jrd.normal(jrd.key(2), (10,)), name="x2")

        loc = lsl.Var.new_calc(lambda *args: sum(args), x1, name="loc")

        scale = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="scale",
        )

        y = lsl.Var.new_obs(
            jrd.normal(jrd.key(2), (10,)),
            lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="y",
        )

        model = lsl.Model([y])
        model.locked = False

        assert jnp.allclose(loc.value, x1.value)

        loc.value_node.set_inputs(x2)
        assert jnp.allclose(loc.value, x2.value)

        model.rebuild_graph(x2)

        assert x2.name in model.vars
        assert x1.name not in model.vars
        assert y.name not in model.vars

        assert x1.model is None
        assert y.model is None
