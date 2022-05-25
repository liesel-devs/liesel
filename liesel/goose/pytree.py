"""
# Pytree utilities
"""

import dataclasses
from typing import TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util

T = TypeVar("T")


def register_dataclass_as_pytree(cls):
    """Decorator for registering dataclasses as pytrees."""

    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} must be a dataclass")

    def flatten(cls_instance):
        # don't use dataclasses.asdict() here, because it converts nested dataclasses
        # to dicts recursively

        return jax.tree_util.tree_flatten(cls_instance.__dict__)

    def unflatten(aux_data, children):
        d = jax.tree_util.tree_unflatten(aux_data, children)
        rv = cls.__new__(cls)
        rv.__dict__.update(d)
        return rv

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)

    return cls


def slice_leaves(pytree, idx):
    """
    Performs the same slice operation on every leaf.

    `idx` can be constructed with `jnp.s_` or `np.s_`, for example:

    ```python
    jnp.s_[0]
    jnp.s_[0:3, :, 2]
    ```
    """
    return jax.tree_util.tree_map(lambda x: x[idx], pytree)


def stack_leaves(pytrees, axis=0):
    """
    Stacks all leaves in the list of pytrees along the given axis.

    The stack operation creates a new axis.
    """
    return jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=axis),
        *pytrees,
    )


def concatenate_leaves(pytrees, axis=0):
    """
    Concatenates all leaves in the list of pytrees along the given axis.

    The concatenate operation does not create a new axis.
    """
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=axis), *pytrees)


def split_leaves(pytree, indices_or_sections, axis=0):
    """
    Splits all leaves in a pytree into multiple sub-arrays.

    The function applies `jnp.split` to all leaves.
    """
    return jax.tree_util.tree_map(
        lambda x: jnp.split(x, indices_or_sections, axis), pytree
    )


def squeeze_leaves(pytree, axis=0):
    """
    Squeezes all leaves in a pytree.
    """
    return jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis), pytree)


def split_and_transpose(pytree, axis=0):
    """
    Splits the leaves in a pytree along one axis and transposes the tree such
    that it's a list of pytrees. It assumes that all leaves have the same
    dimensionality along the chosen axis.
    """
    dim = jax.tree_util.tree_leaves(pytree)[0].shape[axis]
    spytree = split_leaves(pytree, dim, axis=axis)
    td_inner = jax.tree_util.tree_structure([0 for _ in range(dim)])
    td_outer = jax.tree_util.tree_structure(pytree)
    return jax.tree_util.tree_transpose(td_outer, td_inner, spytree)


def as_strong_pytree(pytree: T) -> T:
    """
    Converts every leaf in a pytree to a non-weak `DeviceArray`.

    See <https://jax.readthedocs.io/en/latest/type_promotion.html>.
    """
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=jnp.asarray(x).dtype), pytree
    )
