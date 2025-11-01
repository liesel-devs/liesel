import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
from liesel.experimental.optim import Split


class TestSplit:
    def test_no_split(self):
        m = lsl.Var.new_param(0.0, name="m")
        x = lsl.Var.new_obs(
            jnp.arange(30),
            distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
            name="x",
        )
        model = lsl.Model([x])

        split = Split.from_share(
            position_keys=["x"],
            n=x.value.size,
            share_validate=0.0,
            share_test=0.0,
        )

        assert split.indices_train.size == x.value.size
        assert split.indices_test.size == 0
        assert split.indices_validate.size == 0

        pos = model.extract_position(["x"])
        split_pos = split.split_position(pos)
        assert split_pos.train["x"].size == x.value.size
        assert split_pos.validate["x"].size == 0
        assert split_pos.test["x"].size == 0

    def test_split(self):
        m = lsl.Var.new_param(0.0, name="m")
        x = lsl.Var.new_obs(
            jnp.arange(100),
            distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
            name="x",
        )
        model = lsl.Model([x])

        split = Split.from_share(
            position_keys=["x"],
            n=x.value.size,
            share_validate=0.2,
            share_test=0.2,
        )

        assert split.indices_train.size == 60
        assert split.indices_test.size == 20
        assert split.indices_validate.size == 20

        pos = model.extract_position(["x"])
        split_pos = split.split_position(pos)
        assert split_pos.train["x"].size == 60
        assert split_pos.validate["x"].size == 20
        assert split_pos.test["x"].size == 20

    def test_split_incoherent(self):
        with pytest.raises(ValueError):
            Split.from_share(
                position_keys=["x"],
                n=100,
                share_validate=0.9,
                share_test=0.2,
            )

        with pytest.raises(ValueError):
            Split.from_share(
                position_keys=["x"],
                n=100,
                share_validate=1.1,
                share_test=0.0,
            )

        with pytest.raises(ValueError):
            Split.from_share(
                position_keys=["x"],
                n=100,
                share_validate=0.0,
                share_test=1.1,
            )

        with pytest.raises(ValueError):
            Split.from_share(
                position_keys=["x"],
                n=100,
                share_validate=-0.3,
                share_test=0.5,
            )
