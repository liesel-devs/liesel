"""Tests for Dist.biject_parameters validation and behavior."""

import logging

import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl


class TestBijectParametersValidation:
    """Test validation in Dist.biject_parameters."""

    def test_mixed_positional_keyword_inputs_raises(self):
        """Auto bijectors should reject mixed positional and keyword inputs."""
        scale = lsl.Var.new_param(1.0, name="scale")
        loc = lsl.Var.new_param(0.0, name="loc")

        # Mixed: one positional, one keyword
        dist = lsl.Dist(tfd.Normal, loc, scale=scale)

        with pytest.raises(ValueError, match="mixed positional and keyword inputs"):
            dist.biject_parameters(bijectors="auto")

    def test_too_many_bijectors_in_sequence_raises(self):
        """Should reject Sequence with more bijectors than parameters."""
        scale = lsl.Var.new_param(1.0, name="scale")
        loc = lsl.Var.new_param(0.0, name="loc")

        dist = lsl.Dist(tfd.Normal, loc=loc, scale=scale)

        # Normal has 2 parameters, provide 3 bijectors
        with pytest.raises(ValueError, match="Too many bijectors provided"):
            dist.biject_parameters(bijectors=["auto", "auto", "auto"])

    def test_too_few_bijectors_in_sequence_works(self):
        scale = lsl.Var.new_param(1.0, name="scale_var")
        concentration = lsl.Var.new_param(1.0, name="concentration_var")

        dist = lsl.Dist(tfd.InverseGamma, concentration=concentration, scale=scale)
        dist.biject_parameters(bijectors=["auto"])

        assert concentration.weak
        assert not scale.weak

    def test_var_is_weak(self):
        scale = lsl.Var.new_param(1.0, name="scale_var")
        concentration = lsl.Var.new_param(1.0, name="concentration_var")
        concentration.biject(tfb.Exp())
        assert concentration.weak

        dist = lsl.Dist(tfd.InverseGamma, concentration=concentration, scale=scale)
        with pytest.raises(ValueError, match="weak, but explicit bijector"):
            dist.biject_parameters(bijectors=["auto"])

    def test_var_has_auto_bijector(self):
        scale = lsl.Var.new_param(1.0, name="scale_var")
        concentration = lsl.Var.new_param(1.0, name="concentration_var")
        concentration.auto_transform = True
        assert concentration.strong

        dist = lsl.Dist(tfd.InverseGamma, concentration=concentration, scale=scale)
        with pytest.raises(ValueError, match="auto_transform=True"):
            dist.biject_parameters(bijectors=["auto"])

    def test_node_supplied_raises(self):
        scale = lsl.Value(1.0)
        concentration = lsl.Value(1.0)
        dist = lsl.Dist(tfd.InverseGamma, concentration=concentration, scale=scale)

        with pytest.raises(ValueError, match="only lsl.Var objects can be bijected"):
            dist.biject_parameters(bijectors=["auto"])

        with pytest.raises(ValueError, match="only lsl.Var objects can be bijected"):
            dist.biject_parameters(bijectors="auto")

        with pytest.raises(ValueError, match="only lsl.Var objects can be bijected"):
            dist.biject_parameters(bijectors={"scale": "auto"})

        with pytest.raises(ValueError, match="only lsl.Var objects can be bijected"):
            dist.biject_parameters(bijectors={"concentration": None, "scale": "auto"})

        dist.biject_parameters(bijectors={"concentration": None, "scale": None})

    def test_invalid_parameter_name_in_dict_raises(self):
        """Should reject dict with invalid parameter names."""
        scale = lsl.Var.new_param(1.0, name="scale")
        loc = lsl.Var.new_param(0.0, name="loc")

        dist = lsl.Dist(tfd.Normal, loc=loc, scale=scale)

        # "invalid_param" is not a valid parameter for Normal
        with pytest.raises(ValueError, match="Invalid parameter name.*invalid_param"):
            dist.biject_parameters(bijectors={"loc": "auto", "invalid_param": "auto"})

    def test_non_parameter_variable_warns_but_proceeds(self, local_caplog):
        """Should warn when bijecting a non-parameter variable."""

        # Create a variable that's NOT a parameter
        scale = lsl.Var.new_value(1.0, name="scale")  # not new_param!
        loc = lsl.Var.new_param(0.0, name="loc")

        dist = lsl.Dist(tfd.Normal, loc=loc, scale=scale)

        with local_caplog(level=logging.WARNING, name="liesel.model.nodes") as caplog:
            dist.biject_parameters(bijectors="auto")

        # Should have logged a warning
        assert len(caplog.records) > 0
        assert "parameter=False" in caplog.records[0].message
        assert "scale" in caplog.records[0].message

        # But should still proceed - scale should be weak
        assert scale.weak

    def test_bijector_sequence_and_unordered_kwargs(self):
        """Should reject Sequence with more bijectors than parameters."""
        scale = lsl.Var.new_param(1.0, name="scale_var")
        concentration = lsl.Var.new_param(1.0, name="concentration_var")

        dist = lsl.Dist(tfd.InverseGamma, scale=scale, concentration=concentration)
        dist.biject_parameters(bijectors=["auto"])
        assert scale.weak
        assert concentration.strong

        scale = lsl.Var.new_param(1.0, name="scale_var")
        concentration = lsl.Var.new_param(1.0, name="concentration_var")

        dist = lsl.Dist(tfd.InverseGamma, concentration=concentration, scale=scale)
        dist.biject_parameters(bijectors=["auto"])
        assert scale.strong
        assert concentration.weak

    def test_bijector_class_raises(self):
        scale = lsl.Var.new_param(1.0, name="scale_var")
        concentration = lsl.Var.new_param(1.0, name="concentration_var")

        dist = lsl.Dist(tfd.InverseGamma, scale=scale, concentration=concentration)
        with pytest.raises(TypeError, match="bijector class"):
            dist.biject_parameters(bijectors=[tfb.Identity])

    def test_inference(self):
        scale = lsl.Var.new_param(
            1.0, name="scale_var", inference=gs.MCMCSpec(gs.HMCKernel)
        )
        concentration = lsl.Var.new_param(1.0, name="concentration_var")

        dist = lsl.Dist(tfd.InverseGamma, scale=scale, concentration=concentration)
        with pytest.raises(ValueError, match="inference information"):
            dist.biject_parameters()

        with pytest.raises(ValueError, match="not supported"):
            dist.biject_parameters(inference=gs.MCMCSpec(gs.HMCKernel))

        dist.biject_parameters(inference="drop")
        assert scale.weak
        assert concentration.weak


class TestBijectParametersSuccess:
    """Test successful bijection cases."""

    def test_auto_all_keyword_parameters(self):
        """Auto bijectors work with all keyword parameters."""
        scale = lsl.Var.new_param(1.0, name="scale")
        concentration = lsl.Var.new_param(2.0, name="concentration")

        dist = lsl.Dist(tfd.Gamma, concentration=concentration, rate=scale)
        dist.biject_parameters(bijectors="auto")

        # Both should now be weak (transformed)
        assert scale.weak
        assert concentration.weak

    def test_auto_all_positional_parameters(self):
        """Auto bijectors work with all positional parameters."""
        concentration = lsl.Var.new_param(2.0, name="concentration")
        scale = lsl.Var.new_param(1.0, name="scale")

        dist = lsl.Dist(tfd.Gamma, concentration, scale)
        dist.biject_parameters(bijectors="auto")

        # Both should now be weak (transformed)
        assert scale.weak
        assert concentration.weak

    def test_dict_with_none_skips_parameter(self):
        """Dict bijectors with None should skip that parameter."""

        # pre-test: should biject both parameters
        scale = lsl.Var.new_param(1.0, name="scale")
        concentration = lsl.Var.new_param(0.0, name="concentration")

        dist = lsl.Dist(tfd.Weibull, concentration=concentration, scale=scale)
        dist.biject_parameters(bijectors="auto")

        assert scale.weak
        assert concentration.weak

        # post-test: should biject only scale
        scale = lsl.Var.new_param(1.0, name="scale")
        concentration = lsl.Var.new_param(0.0, name="concentration")

        dist = lsl.Dist(tfd.Weibull, concentration=concentration, scale=scale)
        dist.biject_parameters(bijectors={"scale": "auto", "concentration": None})

        # Only scale should be weak
        assert scale.weak
        assert concentration.strong

    def test_not_in_dict_means_skipping(self):
        # pre-test: should biject both parameters
        scale = lsl.Var.new_param(1.0, name="scale")
        concentration = lsl.Var.new_param(0.0, name="concentration")

        dist = lsl.Dist(tfd.Weibull, concentration=concentration, scale=scale)
        dist.biject_parameters(bijectors="auto")

        assert scale.weak
        assert concentration.weak

        # post-test: should biject only scale
        scale = lsl.Var.new_param(1.0, name="scale")
        concentration = lsl.Var.new_param(0.0, name="concentration")

        dist = lsl.Dist(tfd.Weibull, concentration=concentration, scale=scale)
        dist.biject_parameters(bijectors={"scale": "auto"})

        # Only scale should be weak
        assert scale.weak
        assert concentration.strong

    def test_dict_with_none_when_var_is_weak(self):
        """Dict bijectors with None should skip that parameter."""

        # post-test: should biject only scale
        scale = lsl.Var.new_param(1.0, name="scale")
        concentration = lsl.Var.new_param(0.0, name="concentration")
        concentration.biject(tfb.Exp())
        assert concentration.weak

        dist = lsl.Dist(tfd.Weibull, concentration=concentration, scale=scale)
        dist.biject_parameters(bijectors={"scale": "auto", "concentration": None})

        # Only scale should be weak
        assert scale.weak
        assert concentration.weak

    def test_dict_with_none_when_var_has_auto_transform(self):
        """Dict bijectors with None should skip that parameter."""

        # post-test: should biject only scale
        scale = lsl.Var.new_param(1.0, name="scale")
        concentration = lsl.Var.new_param(0.0, name="concentration")
        concentration.auto_transform = True
        assert concentration.strong

        dist = lsl.Dist(tfd.Weibull, concentration=concentration, scale=scale)
        dist.biject_parameters(bijectors={"scale": "auto", "concentration": None})

        # Only scale should be weak
        assert scale.weak
        assert concentration.strong

    def test_sequence_bijectors(self):
        """Sequence bijectors work correctly."""
        concentration = lsl.Var.new_param(2.0, name="concentration")
        scale = lsl.Var.new_param(1.0, name="scale")

        dist = lsl.Dist(tfd.Gamma, concentration, scale)
        # Transform only the first parameter
        dist.biject_parameters(bijectors=["auto", None])

        assert concentration.weak
        assert scale.strong

    def test_sequence_bijectors_with_optional_param(self):
        """Sequence bijectors work correctly."""
        concentration = lsl.Var.new_param(2.0, name="concentration")
        log_rate = lsl.Var.new_param(1.0, name="log_rate")

        dist = lsl.Dist(tfd.Gamma, concentration=concentration, log_rate=log_rate)
        # Transform only the first parameter
        dist.biject_parameters(bijectors=[None, "auto"])

        assert log_rate.weak
        assert concentration.strong

    def test_none_bijector_in_sequence_skips(self):
        """None in sequence should skip that parameter."""
        concentration = lsl.Var.new_param(2.0, name="concentration")
        scale = lsl.Var.new_param(1.0, name="scale")

        dist = lsl.Dist(tfd.Gamma, concentration, scale)
        dist.biject_parameters(bijectors=[None, "auto"])

        assert concentration.strong
        assert scale.weak

    def test_custom_bijector(self):
        """None in sequence should skip that parameter."""
        concentration = lsl.Var.new_param(2.0, name="concentration")
        scale = lsl.Var.new_param(1.0, name="scale")

        dist = lsl.Dist(tfd.Gamma, concentration, scale)
        exp = tfb.Exp()
        dist.biject_parameters(bijectors=[exp, "auto"])

        assert concentration.weak
        assert scale.weak

        assert concentration.value_node[0].value == pytest.approx(
            jnp.log(concentration.value)
        )

        concentration = lsl.Var.new_param(2.0, name="concentration")
        scale = lsl.Var.new_param(1.0, name="scale")

        dist = lsl.Dist(tfd.Gamma, concentration, scale)
        dist.biject_parameters(bijectors=[tfb.Identity(), "auto"])

        assert concentration.weak
        assert scale.weak

        assert concentration.value_node[0].value == pytest.approx(concentration.value)

    def test_return_self(self):
        concentration = lsl.Var.new_param(2.0, name="concentration")
        scale = lsl.Var.new_param(1.0, name="scale")

        dist = lsl.Dist(tfd.Gamma, concentration, scale)
        dist2 = dist.biject_parameters()
        assert dist is dist2

    def test_default_of_method_is_auto(self):
        concentration = lsl.Var.new_param(2.0, name="concentration")
        scale = lsl.Var.new_param(1.0, name="scale")

        dist = lsl.Dist(tfd.Gamma, concentration, scale)
        dist.biject_parameters()

        assert concentration.weak
        assert scale.weak

    def test_default_of_class_is_no_action(self):
        concentration = lsl.Var.new_param(2.0, name="concentration")
        scale = lsl.Var.new_param(1.0, name="scale")

        _ = lsl.Dist(tfd.Gamma, concentration, scale)

        assert concentration.strong
        assert scale.strong


class TestVarBiject:
    def test_bijected_var_manually(self):
        log_scale = lsl.Var.new_param(1.0, name="log_scale")
        scale = lsl.Var.new_calc(jnp.exp, log_scale)
        scale.bijected_var = log_scale

        assert scale.bijected_var is log_scale

    def test_bijected_var_is_no_var(self):
        log_scale = lsl.Value(1.0)
        scale = lsl.Var.new_calc(jnp.exp, log_scale)
        with pytest.raises(TypeError):
            scale.bijected_var = log_scale

    def test_bijected_var_is_no_input(self):
        log_scale = lsl.Var(1.0)
        scale = lsl.Var.new_value(1.0)
        with pytest.raises(ValueError):
            scale.bijected_var = log_scale

    def test_bijected_var_from_biject(self):
        scale = lsl.Var.new_param(1.0, name="scale")
        scale.biject(tfb.Exp())

        assert scale.bijected_var.name == "h(scale)"

    def test_bijected_var_from_transform(self):
        scale = lsl.Var.new_param(1.0, name="scale")
        scale.transform(tfb.Exp())

        assert scale.bijected_var.name == "scale_transformed"
