Some tests
===========

Internal Reference:
:class:`liesel.liesel.model.Model`, :class:`~.model.Model`, :class:`.Model`

Python standard library link:
:class:`pathlib.Path`

Numpy link:
:class:`numpy.ndarray`, :class:`np.ndarray`

Scipy link:
:func:`scipy.linalg.solve`

Jax link:
:func:`jax.grad`

TFP link:
:class:`tfp.bijectors.Bijector`, :class:`tensorflow_probability.bijectors.Bijector`


We can add codeblocks::

    import numpy as np

    import liesel.liesel as lsl

    n_loc = lsl.Parameter(0.0, name="loc")
    n_scale = lsl.Parameter(1.0, name="scale")

    n_y = lsl.Node(
        value=np.array([1.314, 0.861, -1.813, 0.587, -1.408]),
        distribution=lsl.NodeDistribution("Normal", loc=n_loc, scale=n_scale),
        name="y",
    )

    model = lsl.Model([n_loc, n_scale, n_y])


We can also have nice-looking boxes:

.. note::
    This is a note.

.. warning::
    This is a warning.

