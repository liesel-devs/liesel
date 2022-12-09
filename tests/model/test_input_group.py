import pytest
import tensorflow_probability.substrates.numpy.bijectors as tfb
import tensorflow_probability.substrates.numpy.distributions as tfd

from liesel.model import Calc, InputGroup, Model


def test_input_group():
    def fn(at, dargs, bargs):
        d = tfd.Normal(*dargs.args, **dargs.kwargs)
        b = tfb.Softplus(*bargs.args, **bargs.kwargs)
        td = tfd.TransformedDistribution(d, b)
        return td.log_prob(at)

    c = Calc(
        fn,
        1.0,
        dargs=InputGroup(0.0, 1.0),
        bargs=InputGroup(hinge_softness=1.0),
        _name="c",
    )

    assert c.update().value == pytest.approx(-0.6067796945571899)

    m = Model([c])
    m.update()

    assert m.nodes["c"].value == pytest.approx(-0.6067796945571899)
