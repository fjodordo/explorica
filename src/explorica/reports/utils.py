"""
Low-level utilities for Explorica's reports module

Low-level utility functions for standardizing visualization objects
into the `VisualizationResult` format. This module provides a
single public function that can be used in user code to normalize
Matplotlib and Plotly figures for downstream report generation
or further processing.

Functions
---------
normalize_visualization(figure)
    Convert a Matplotlib or Plotly figure into a standardized
    `VisualizationResult` dataclass with extracted metadata.
normalize_table(data)
    Normalize tabular data into a standardized TableResult object.

Examples
--------
>>> from explorica.reports.utils import normalize_visualization

# Usage with matplotlib figure
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3], [4, 5, 6])
>>> result = normalize_visualization(fig)
>>> result.engine
'matplotlib'
>>> result.width, result.height
(6.0, 4.0)

# Usage with plotly figure
>>> import plotly.graph_objects as go
>>> fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
>>> result = normalize_visualization(fig)
>>> result.engine
'plotly'
>>> result.width, result.height
(800, 600)  # default if not specified in layout
"""

from typing import Union, Sequence, Any, Mapping

import pandas as pd
import matplotlib.figure
import plotly.graph_objects

from .._utils import convert_dataframe
from ..types import VisualizationResult, TableResult


def normalize_visualization(
    figure: Union[
        matplotlib.figure.Figure, plotly.graph_objects.Figure, VisualizationResult
    ],
) -> VisualizationResult:
    """
    Normalize a visualization object into a standardized `VisualizationResult`.

    This function converts a Matplotlib or Plotly figure into a
    `VisualizationResult` dataclass, extracting common metadata
    such as engine, axes, width, height, and title. This allows
    downstream rendering or report composition functions to work
    with a uniform interface.

    Parameters
    ----------
    figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The input figure to normalize. Can be:
        - A Matplotlib figure
        - A Plotly figure
        - A pre-normalized `VisualizationResult` (in which
          case it is returned as-is)

    Returns
    -------
    VisualizationResult
        A dataclass containing:
        - figure : The original figure object.
        - engine : str, either "matplotlib" or "plotly".
        - axes : List of Matplotlib axes (for Matplotlib) or None (for Plotly).
        - width : Figure width in inches (Matplotlib) or pixels (Plotly).
        - height : Figure height in inches (Matplotlib) or pixels (Plotly).
        - title : Optional figure title.

    Raises
    ------
    TypeError
        If `figure` is not an instance of Matplotlib,
        Plotly Figure, or `VisualizationResult`.

    Notes
    -----
    - For Matplotlib figures, `width` and `height` are measured in inches.
    - For Plotly figures, `width` and `height` are measured in pixels.
    - The original figure is preserved in the `figure` attribute of the
      returned `VisualizationResult`.

    Examples
    --------
    >>> from explorica.reports.utils import normalize_visualization

    # Usage with matplotlib figure
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> result = normalize_visualization(fig)
    >>> result.engine
    'matplotlib'
    >>> result.width, result.height
    (6.0, 4.0)

    # Usage with plotly figure
    >>> import plotly.graph_objects as go
    >>> fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    >>> result = normalize_visualization(fig)
    >>> result.engine
    'plotly'
    >>> result.width, result.height
    (800, 600)  # default if not specified in layout
    """
    if isinstance(figure, matplotlib.figure.Figure):
        vis_result = VisualizationResult(figure=figure)
        vis_result.engine = "matplotlib"
        vis_result.axes = vis_result.figure.get_axes()
        w, h = figure.get_size_inches()
        vis_result.width = w
        vis_result.height = h
    elif isinstance(figure, plotly.graph_objects.Figure):
        vis_result = VisualizationResult(figure=figure)
        vis_result.engine = "plotly"
        vis_result.axes = None  # should be None
        vis_result.width = figure.layout.width
        vis_result.height = figure.layout.height
    elif isinstance(figure, VisualizationResult):
        vis_result = figure
    else:
        raise TypeError(
            f"Unsupported figure type: {type(figure)}."
            f"Please provide 'matplotlib.figure.Figure',"
            f"'plotly.graph_objects.Figure', or 'VisualizationResult'."
        )
    return vis_result


def normalize_table(
    data: (
        Sequence[float]
        | Sequence[Sequence[float]]
        | Mapping[str, Sequence[Any]]
        | TableResult
    ),
) -> TableResult:
    """
    Normalize tabular data into a standardized TableResult object.

    This function converts input data of various formats (1D/2D sequences, mappings
    or `TableResult`) into a `TableResult` instance containing a Pandas DataFrame.
    This ensures consistent handling of tabular results across Explorica reports.

    Parameters
    ----------
    data : Sequence, Mapping or TableResult
        Tabular data to normalize. Supported types include:
        - 1D or 2D sequences (e.g., list, tuple of lists)
        - Mapping[str, Sequence] (e.g., dict of column_name -> values)
        MultiIndex rows or columns are not supported.

    Returns
    -------
    TableResult
        A standardized container wrapping a Pandas DataFrame.

    Raises
    ------
    ValueError
        If the input DataFrame has a MultiIndex in rows or columns.

    Examples
    --------
    >>> data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
    >>> table_result = normalize_table(data)
    >>> isinstance(table_result, TableResult)
    True
    >>> table_result.table.shape
    (3, 2)
    """
    if isinstance(data, TableResult):
        return data
    df = convert_dataframe(data)
    if isinstance(df.index, pd.MultiIndex):
        raise ValueError("MultiIndex in rows is not supported in normalize_table.")
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("MultiIndex in columns is not supported in normalize_table.")

    return TableResult(table=df)
