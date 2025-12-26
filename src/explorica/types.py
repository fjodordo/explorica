"""
Type utilities and descriptors used throughout the Explorica framework.

This module defines shared type structures that provide a consistent interface
for handling visualization outputs, validating numeric inputs, and enabling
stronger semantic typing across Explorica components. These types are not
intended to replace Python's built-in types; rather, they serve as lightweight
descriptors and containers that improve clarity, correctness, and
interoperability inside the framework.

Classes
-------
VisualizationResult
    Container object returned by all visualization functions. Stores the
    generated figure, axes (if applicable), engine metadata, sizing
    information, and arbitrary additional details.
NaturalNumber
    Type descriptor enabling `isinstance(x, NaturalNumber)` checks for
    positive integers (natural numbers). Used in parameter validation across
    plotting and preprocessing utilities.
TableResult
    Standardized container for tabular results in Explorica.


Notes
-----
- The types defined in this module are part of the public API. They are designed
  to be stable, lightweight, and safe for direct user interaction.
- `NaturalNumber` is implemented via a metaclass and acts like a pseudo-type.
  It cannot be instantiated and carries no behavior beyond validation.
- Additional pseudo-types and data descriptors may be added in future releases
  to support broader patterns of type-driven validation within Explorica.

Examples
--------

# Using `VisualizationResult`
>>> import explorica.visualizations as viz
>>> res = viz.scatterplot(
...     x=[1, 2, 3], y=[3, 2, 1], title="Example"
... )
>>> res.figure          # access underlying Matplotlib or Plotly figure
>>> res.axes            # Matplotlib only
>>> res.title
'Example'

# Using `NaturalNumber`
>>> from explorica.types import NaturalNumber
>>> isinstance(5, NaturalNumber)
True
>>> isinstance(5.0, NaturalNumber)
True
>>> isinstance(-1, NaturalNumber)
False
>>> isinstance("3", NaturalNumber)
False
"""

from numbers import Number
from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class _NaturalNumberMeta(type):
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
class NaturalNumber(metaclass=_NaturalNumberMeta):
    """
    Type descriptor for natural numbers (positive integers).

    This class defines a numeric type representing natural numbers, i.e.,
    positive integers greater than zero. It is primarily intended for type
    checking in Explorica and related libraries, and can be used wherever
    one wants to enforce that a numeric input is a natural number.

    The descriptor implements the `__instancecheck__` protocol, so that
    `isinstance(value, NaturalNumber)` returns True if and only if `value`
    satisfies all of the following:

    1. It is a numeric type (`int` or `float`).
    2. It is strictly greater than zero.
    3. It represents a whole number (integer value).

    Notes
    -----
    - Floats are accepted **only if they are exact integers**, e.g. `1.0`.
    - Non-numeric types (str, list, etc.) always return False.
    - Zero and negative numbers are not considered natural numbers.
    - This class is a **type descriptor**, not a numeric class. It cannot be
      instantiated or used for arithmetic. Its purpose is **type validation**.
    - The class is singleton-like: all checks are done via the metaclass,
      so `isinstance` works directly on the class.

    Examples
    --------
    >>> from explorica.types import NaturalNumber

    >>> isinstance(1, NaturalNumber)
    True
    >>> isinstance(1.0, NaturalNumber)
    True
    >>> isinstance(0, NaturalNumber)
    False
    >>> isinstance(-5, NaturalNumber)
    False
    >>> isinstance(3.14, NaturalNumber)
    False
    >>> isinstance("5", NaturalNumber)
    False
    """


@dataclass
class VisualizationResult:
    """
    Standardized container for the output of all Explorica visualization
    functions.

    This dataclass provides a unified structure for accessing the generated
    figure, axes, metadata, and rendering backend. All visualization functions
    across Explorica return a :class:`VisualizationResult`, ensuring that users
    always interact with plots through a consistent and predictable interface.

    The container is engine-agnostic and supports both Matplotlib and Plotly.
    This allows downstream processing, inspection, chaining, or exporting of
    visualizations without needing to know which plotting backend produced them.

    Parameters
    ----------
    figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The figure object produced by the visualization function.
        - For Matplotlib, this is an instance of ``matplotlib.figure.Figure``.
        - For Plotly, this is an instance of ``plotly.graph_objects.Figure``.
    axes : matplotlib.axes.Axes or None, default=None
        The primary axes object when using Matplotlib.
        Plotly visualizations do not use axes and set this attribute to ``None``.
    engine : {'matplotlib', 'plotly'}
        Name of the plotting engine used to generate the visualization.
        Useful for backend-specific post-processing.
    width : int or None
        Width of the figure:
        - Measured in inches for Matplotlib.
        - Measured in pixels for Plotly.
    height : int or None
        Height of the figure:
        - Measured in inches for Matplotlib.
        - Measured in pixels for Plotly.
    title : str or None
        Title of the generated visualization.
        This duplicates ``figure.title`` for convenience and consistency.
    extra_info : dict or None
        Optional metadata dictionary containing additional details about
        the visualization.
        Typical fields may include:
        - `'palette'`: the color palette used,
        - `'trendline'`: trendline model or parameters (for scatterplots),
        - `'layout'`: Plotly layout overrides,
        - `'transform'`: preprocessing steps applied to input data,
        - or any backend-specific information used for reproducibility.

    Notes
    -----
    The purpose of this class is to standardize the interface between
    visualization generators and downstream user code. For example:

    - Saving a plot to disk using ``result.figure`` works uniformly across engines.
    - Accessing axes-based methods (e.g., ``set_xlim``) is only valid for Matplotlib.
    - Tools that inspect plot metadata (e.g., model diagnostics, layout exporters)
      can rely on ``extra_info`` instead of backend-specific properties.

    Notes
    -----
    `VisualizationResult` provides a unified interface for interacting with
    figures generated by different plotting engines. While the returned object
    always contains a figure-like object in ``figure``, its behavior depends on
    the backend:

    - ``matplotlib``:
      ``figure`` is a :class:`matplotlib.figure.Figure`, and ``axes`` contains
      the primary :class:`matplotlib.axes.Axes` instance.

    - ``plotly``:
      ``figure`` is a :class:`plotly.graph_objects.Figure`, and ``axes`` is
      always ``None``.

    Common operations such as ``figure.show()`` work for both backends, though
    the resulting UI differs (Matplotlib uses the local renderer or notebook
    backend; Plotly opens an interactive HTML-based viewer).

    Backend-specific methods remain available. For example:

    - Matplotlib: ``figure.savefig(...)``
    - Plotly: ``figure.write_html(...)`` or ``figure.to_json()``

    This design allows both standardized downstream usage (e.g. consistent access
    to ``title`` or ``extra_info``) and full access to the native API of the
    underlying visualization engine.

    Examples
    --------
    Basic usage with a Matplotlib-based visualization:
    >>> import explorica.visualizations as vis
    >>> result = vis.scatterplot([1, 2, 3], [2, 4, 6], title="Demo Plot")
    >>> result.figure        # Matplotlib Figure
    >>> result.axes          # Matplotlib Axes
    >>> result.engine
    'matplotlib'
    >>> result.title
    'Demo Plot'

    Basic usage with a Plotly-based visualization:
    >>> from explorica.visualizations import mapbox
    >>> result = mapbox(
    ...     lat=[34.05, 40.71],
    ...     lon=[-118.24, -74.00],
    ...     title="Cities Map"
    ... )
    >>> result.figure        # Plotly Figure
    >>> result.axes is None
    True
    >>> result.engine
    'plotly'
    >>> result.title
    'Cities Map'

    Accessing extended metadata:
    >>> result.extra_info
    {'palette': 'viridis'}
    """

    figure: Union[Figure]
    axes: Optional[Axes] = None
    engine: str = "matplotlib"
    width: Optional[int] = None
    height: Optional[int] = None
    title: Optional[str] = None
    extra_info: dict = None


@dataclass
class TableResult:
    """
    Standardized container for tabular results in Explorica.

    This class represents a structured tabular artifact produced during
    exploratory data analysis (EDA), such as summary statistics, quality
    diagnostics, or interaction analysis results.

    At the current stage, `TableResult` serves as a lightweight wrapper
    around a pandas DataFrame.

    Parameters
    ----------
    table : pandas.DataFrame
        Tabular data. The DataFrame is expected to use a flat structure:
        no MultiIndex on rows and no MultiIndex on columns.
    title : str, optional
        Short human-readable title describing the table contents.
    description : str, optional
        Longer description providing context or interpretation
        guidelines for the table.
    render_extra : dict, optional
        Optional dictionary controlling rendering behavior for this table.
        Keys may include:
        - ``show_index`` : bool, default True - whether to display the row index
          in rendered output (HTML or PDF).
        - ``show_header`` : bool, default True - whether to display column names.
        - Any additional rendering hints may be added in the future.

    Notes
    -----
    - `TableResult` is a passive data container and does not implement
      analytical logic or rendering behavior.
    - This class may be extended in the future to include additional
      metadata (e.g., per-column annotations or semantic roles).

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.reports import TableResult

    >>> df = pd.DataFrame({
    ...     "feature": ["age", "income"],
    ...     "mean": [35.2, 52000],
    ...     "std": [8.1, 12000],
    ... })

    >>> table = TableResult(
    ...     table=df,
    ...     title="Feature Summary Statistics",
    ...     description="Basic central tendency and dispersion measures."
    ... )
    """

    table: pd.DataFrame
    title: Optional[str] = None
    description: Optional[str] = None
    render_extra: Optional[dict] = field(default_factory=dict)
