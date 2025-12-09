"""
Backward-compatibility aliases for legacy type imports.

This module exists solely to provide temporary compatibility for codebases
that previously imported ``NaturalNumber`` from ``explorica._utils``.
The canonical and stable location for this type is now :mod:`explorica.types`.

Notes
-----
- This module is deprecated and will be removed in a future major release.
- Users should update all imports to use::

      from explorica.types import NaturalNumber

- The alias ``natural_number`` is preserved for legacy support but is also
  deprecated and will be removed.

Examples
--------
>>> # Legacy (deprecated)
>>> from explorica._utils.types import natural_number

>>> # Recommended
>>> from explorica.types import NaturalNumber
"""

from explorica.types import NaturalNumber


# disabled, because `natural_number` will be removed in future versions
# pylint: disable=C0103
natural_number = NaturalNumber
