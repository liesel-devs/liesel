Liesel: A Probabilistic Programming Framework
=============================================

.. include:: welcome.md
   :parser: myst_parser.sphinx_

.. toctree::
   :caption: Guides
   :hidden:
   :maxdepth: 1

   model
   goose
   tutorials_overview


API Reference
-------------

.. autosummary::
    :toctree: generated
    :caption: API Reference
    :recursive:

    ~liesel.model
    ~liesel.goose
    ~liesel.contrib
    ~liesel.distributions
    ~liesel.bijectors

.. autosummary::
    :toctree: generated
    :caption: Experimental API
    :recursive:

    ~liesel.experimental


Effort-Based Versioning
-----------------------


Starting with v0.4.0, we will be using effort-based versioning.
See the EffVer documentation at https://jacobtomlinson.dev/effver/

The JAX developers provide a wonderful summary:
https://docs.jax.dev/en/latest/jep/25516-effver.html

The following description is almost entirely quoted from the linked JAX page,
but it describes what we intend with effort-based versioning perfectly.

Effort-based versioning is a three-number versioning system,
similar to the better-known semantic versioning (SemVer: https://semver.org/).
It uses a three-number format: ``MACRO.MESO.MICRO``, where version numbers
are incremented based on the expected effort required to adapt to the change.

As an example, consider software with current version ``2.3.4``:

1. Increasing the *micro* version (i.e. releasing ``2.3.5``)
   signals to users that little to no effort is necessary on their part
   to adapt to the changes.
2. Increasing the *meso* version (i.e. releasing ``2.4.0``)
   signals to users that some small effort will be required
   for existing code to work with the changes.
3. Increasing the *macro* version (i.e. releasing ``3.0.0``)
   signals to users that significant effort may be required
   to update to the changes.

In some ways, this captures the essence of more commonly-used semantic versioning,
but avoids phrasing in terms of compatibility guarantees that are hard to meet in practice.

Zero Version
************

In addition, EffVer gives special meaning to the *zero version*.
Early releases of software are often versioned ``0.X.Y``, and in this case:

- ``X`` has the characteristics of the macro version.
- ``Y`` has the characteristics of the meso version.

Liesel has been in a zero-version state since its initial release,
and EffVer's zero-version case is a good post-facto description
of the implicit intent behind Liesel's releases to date.

In EffVer, bumping from ``0.X.Y`` to version ``1.0.0`` is recommended
when a certain level of stability has been reached in practice:
If you end up on a version like ``0.9.x`` for many months,
it is a good signal that things are pretty stable
and that it's time to switch to a ``1.0.0`` release.


Indices and Search
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
