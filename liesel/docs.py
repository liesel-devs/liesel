from functools import wraps
from inspect import getmembers, isfunction


def usedocs(of: object, greedy: bool = True):
    """
    Replaces the docstring of the decorated object with the docstring of the specified
    object.

    Works on both functions and classes. Some notes on the details:

    1. By default, this decorator will paste the class docstring from the object
       ``of`` to the decorated object.
    2. If the decorated object is a class, the decorator will paste the docstrings of
       methods and properties on the object ``of`` to their counterparts on the
       decorated object. This only works for methods and properties that have the same
       name in both objects. This behavior can be turned off by setting the
       argument ``greedy`` to ``False``.
    3. It is also possible to decorate individual methods and properties to reuse
       docstrings from specific objects.
    4. Existing docstrings will never be overwritten.

    Class and instance attribute docstrings are not covered by this decorator.

    Usage example::

        class ParentClass:
            '''Parent class documentation.'''

            def parent_method(self):
                '''Method documentation.'''

        @usedocs(ParentClass)
        class InheritingClass:

            def parent_method(self):
                pass
    """

    def _usedocs(obj):
        @wraps(obj)
        def wrapper():
            if not obj.__doc__:
                obj.__doc__ = of.__doc__

            if not greedy:
                return obj

            methods_list = getmembers(obj, predicate=_is_prop_or_method)
            for name, method in methods_list:
                if method.__doc__:
                    continue

                try:
                    parent_method = getattr(of, name)
                except AttributeError:
                    continue

                method.__doc__ = parent_method.__doc__

            return obj

        return wrapper()

    return _usedocs


def _is_prop_or_method(x: object) -> bool:
    """
    Helper for selecting methods and properties of classes with
    :func:`inspect.getmembers`.

    Does not use :func:`inspect.ismethod`, because that is *True* only for methods on
    class instances, not on classes themselves.
    """
    return isfunction(x) or isinstance(x, property)
