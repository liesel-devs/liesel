import jax.numpy as jnp

from liesel.goose.pytree import concatenate_leaves, slice_leaves, stack_leaves


def test_slice():
    tree = {
        "foo": jnp.zeros((3, 3, 1)),
        "bar": (jnp.ones((3, 1)), jnp.arange(18).reshape((3, 3, 2))),
    }
    slobj = jnp.s_[:, ..., 0]
    nt = slice_leaves(tree, slobj)
    assert nt["foo"].shape == (3, 3)
    assert nt["bar"][0].shape == (3,)
    assert nt["bar"][1].shape == (3, 3)

    assert jnp.all(tree["foo"][slobj] == nt["foo"])


def test_stack():
    tree = {
        "foo": jnp.zeros((3, 3, 1)),
        "bar": (jnp.ones((3, 1)), jnp.arange(18).reshape((3, 3, 2))),
    }

    nt = stack_leaves([tree, tree])

    assert nt["foo"].shape == (2, 3, 3, 1)
    assert nt["bar"][0].shape == (2, 3, 1)
    assert nt["bar"][1].shape == (2, 3, 3, 2)
    assert jnp.all(jnp.stack([tree["foo"], tree["foo"]]) == nt["foo"])

    nt = stack_leaves([tree, tree], axis=1)

    assert nt["foo"].shape == (3, 2, 3, 1)
    assert nt["bar"][0].shape == (3, 2, 1)
    assert nt["bar"][1].shape == (3, 2, 3, 2)
    assert jnp.all(jnp.stack([tree["foo"], tree["foo"]], axis=1) == nt["foo"])


def test_concatenate():
    tree = {
        "foo": jnp.zeros((3, 3, 1)),
        "bar": (jnp.ones((3, 1)), jnp.arange(18).reshape((3, 3, 2))),
    }

    nt = concatenate_leaves([tree, tree])

    assert nt["foo"].shape == (6, 3, 1)
    assert nt["bar"][0].shape == (6, 1)
    assert nt["bar"][1].shape == (6, 3, 2)
    assert jnp.all(jnp.concatenate([tree["foo"], tree["foo"]]) == nt["foo"])

    nt = concatenate_leaves([tree, tree], axis=1)

    assert nt["foo"].shape == (3, 6, 1)
    assert nt["bar"][0].shape == (3, 2)
    assert nt["bar"][1].shape == (3, 6, 2)
    assert jnp.all(jnp.concatenate([tree["foo"], tree["foo"]], axis=1) == nt["foo"])
