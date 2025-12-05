"""Tests for Dist.biject_parameters validation and behavior."""

import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

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
        import logging

        # Create a variable that's NOT a parameter
        scale = lsl.Var.new_value(1.0, name="scale")  # not new_param!
        loc = lsl.Var.new_param(0.0, name="loc")

        dist = lsl.Dist(tfd.Normal, loc=loc, scale=scale)

        with local_caplog(level=logging.WARNING, name="liesel.model.nodes") as caplog:
            dist.biject_parameters(bijectors="auto")

        # Should have logged a warning
        assert len(caplog.records) > 0
        assert "not marked as a parameter" in caplog.records[0].message
        assert "scale" in caplog.records[0].message

        # But should still proceed - scale should be weak
        assert scale.weak


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
        scale = lsl.Var.new_param(1.0, name="scale")
        loc = lsl.Var.new_param(0.0, name="loc")

        dist = lsl.Dist(tfd.Normal, loc=loc, scale=scale)
        dist.biject_parameters(bijectors={"scale": "auto", "loc": None})

        # Only scale should be weak
        assert scale.weak
        assert loc.strong

    def test_sequence_bijectors(self):
        """Sequence bijectors work correctly."""
        concentration = lsl.Var.new_param(2.0, name="concentration")
        scale = lsl.Var.new_param(1.0, name="scale")

        dist = lsl.Dist(tfd.Gamma, concentration, scale)
        # Transform only the first parameter
        dist.biject_parameters(bijectors=["auto", None])

        assert concentration.weak
        assert scale.strong

    def test_none_bijector_in_sequence_skips(self):
        """None in sequence should skip that parameter."""
        concentration = lsl.Var.new_param(2.0, name="concentration")
        scale = lsl.Var.new_param(1.0, name="scale")

        dist = lsl.Dist(tfd.Gamma, concentration, scale)
        dist.biject_parameters(bijectors=[None, "auto"])

        assert concentration.strong
        assert scale.weak
