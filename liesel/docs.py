from functools import wraps


def usedocs(of: object):
    """
    Replaces the docstring of the decorated object with the docstring of the specified
    object.

    In classes, methods need to be decorated individually. Attribute docstrings are not
    covered by this decorator.

    Usage example::

        class ParentClass:
            '''Parent class documentation.'''

            def parent_method(self):
                '''Method documentation.'''

        @usedocs(ParentClass)
        class InheritingClass:

            @usedocs(ParentClass.parent_method)
            def parent_method(self):
                pass
    """

    def _usedocs(obj):
        @wraps(obj)
        def wrapper():
            obj.__doc__ = of.__doc__
            return obj

        return wrapper()

    return _use_docs
