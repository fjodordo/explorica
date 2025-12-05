"""
Protocol-based type validators for runtime type checking.

This module provides singleton classes that emulate type behavior through
the `__instancecheck__` protocol. These are not real types but descriptors
that can be used with `isinstance()` for custom validation logic.

Classes
-------
NaturalNumber
    Validator for natural numbers (positive integers).

Notes
-----
- These are singleton instances, not actual types
- Use with `isinstance(obj, natural_number)` for validation
- Return meaningful names via `__repr__` and `__name__`

Examples
--------
Basic type validation
~~~~~~~~~~~~~~~~~~~~~
>>> from explorica._utils import natural_number

>>> # Check if values are natural numbers
>>> isinstance(5, natural_number)     # True
>>> isinstance(0, natural_number)     # False (not positive)
>>> isinstance(3.14, natural_number)  # False (not integer)
>>> isinstance(-1, natural_number)    # False (negative)
"""

from numbers import Number


class NaturalNumberMeta(type):
    """Metaclass to enable isinstance checks for natural numbers."""

    def __instancecheck__(cls, instance):
        """Return True if instance is a positive integer."""
        return (
            isinstance(instance, Number) and instance > 0 and instance == int(instance)
        )

    def __repr__(cls):
        return "NaturalNumber"

    def __init__(cls, *_):
        cls.__name__ = "NaturalNumber"

# pylint: disable=R0903
class NaturalNumber(metaclass=NaturalNumberMeta):
    """
    Type descriptor for natural numbers (positive integers).

    This descriptor implements the `__instancecheck__` protocol to validate
    that an object is a natural number - a positive integer greater than zero.
    It accepts both `int` and `float` instances if they represent whole numbers.

    Examples
    --------
    >>> from explorica._utils import natural_number

    >>> isinstance(1, natural_number)     # True
    >>> isinstance(1.0, natural_number)   # True (whole number)
    >>> isinstance(0, natural_number)     # False
    >>> isinstance(-5, natural_number)    # False
    >>> isinstance(3.14, natural_number)  # False (not whole)
    >>> isinstance("5", natural_number)   # False (not numeric)
    """
# disabled, because `natural_number` will be removed in future versions
# pylint: disable=C0103
natural_number = NaturalNumber
