"""
High-level plotting utilities for Explorica visualizations.

This module defines high-level functions for exploring distributions and relationships
between numeric variables. Each function is independent and returns a
`VisualizationResult` dataclass encapsulating the figure, axes, engine, and metadata.

Functions
---------
distplot(data, bins = 30, kde = True, **kwargs)
    Plots a histogram of numeric data with optional Kernel Density Estimation (KDE).
boxplot(data, **kwargs)
    Draws a boxplot for a numeric variable to visualize distribution, median,
    and potential outliers.
hexbin(data, target, **kwargs)
    Creates a hexbin plot for two numeric variables. Useful for visualizing dense
    scatter data.
heatmap(data, **kwargs)
    Generates a heatmap to visualize a 2D array of numeric values.

Notes
-----
- All plotting functions return a :class:`explorica.types.VisualizationResult`,
  which provides a consistent interface for accessing the figure, axes (if applicable),
  plotting engine, and additional metadata.
- `plot_kws` allows passing keyword arguments directly to the underlying plotting
  function used by the engine (Matplotlib, Seaborn, or Plotly).
      This provides fine-grained control over styling and behavior
      specific to that function.

Examples
--------
>>> import explorica.visualizations as vis
>>> import numpy as np

# Distribution plot with custom bins, KDE, and figure size
>>> data = np.random.normal(loc=0, scale=1, size=100)
>>> result = vis.distplot(
... data,
... bins=30,
... kde=True,
... title="Normal Distribution Example",
... plot_kws={"figsize": (8, 5)}
... )
>>> result.figure.show()

# Boxplot with figure saving
>>> result = vis.boxplot(
...     data,
...     title="Boxplot Example",
...     directory="./plots",
...     figsize = (6, 4)
... )
>>> Path("./plots/Boxplot Example.html").exists()
True

# Hexbin plot with color map and point sizing via plot_kws
>>> x = np.random.randn(500)
>>> y = x*0.5 + np.random.randn(500)*0.5
>>> result = vis.hexbin(
...     x, y,
...     gridsize=25,
...     colormap="plasma",
...     title="Hexbin Example",
...     figsize = (7, 6),
...     plot_kws={"mincnt": 1}
... )
>>> result.figure.show()

# Heatmap with annotations, custom colormap, and figure size
>>> matrix = np.random.rand(5, 5)
>>> result = vis.heatmap(
...     matrix,
...     annot=True,
...     cmap="coolwarm",
...     title="Annotated Heatmap",
...     figsize=(6, 5)
... )
>>> result.figure.show()
"""

import warnings
from numbers import Number
from typing import Sequence, Mapping, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from explorica.types import VisualizationResult, NaturalNumber
from explorica._utils import (
    convert_dataframe,
    convert_series,
    handle_nan,
    validate_lengths_match,
)
from ._utils import (
    temp_plot_theme,
    save_plot,
    get_empty_plot,
    DEFAULT_MPL_PLOT_PARAMS,
    WRN_MSG_EMPTY_DATA,
    ERR_MSG_ARRAYS_LENS_MISMATCH,
)


def distplot(
    data: Sequence[float] | Mapping[str, Sequence[float]],
    bins: int = 30,
    kde: bool = True,
    **kwargs,
) -> VisualizationResult:
    """
    Plot a histogram with optional kernel density estimate.

    This function generates a histogram for univariate data with an optional
    kernel density estimate (KDE) curve overlay. It automatically handles
    data conversion, NaN values, and provides flexible styling options
    through integration with Seaborn's plotting system.

    Under the hood, the function uses Seaborn's `seaborn.histplot` function
    and applies Seaborn styles for aesthetic defaults. This allows passing
    additional kwargs directly to the underlying Seaborn calls via `plot_kws`.
    For complete parameter documentation and
    advanced customization options, see urls below.

    Parameters
    ----------
    data : Sequence[float] | Mapping[str, Sequence[float]]
        Numeric input data. Can be:
        - 1D sequence of numbers
        - Dictionary with single key-value pair (value is numeric sequence)
        - pandas Series or single-column DataFrame
        Must be one-dimensional.
    bins : int, default=30
        Number of bins in the histogram. Must be a positive integer.
    kde : bool, default=True
        If True, adds kernel density estimate curve.
    opacity : float, default=0.5
        Transparency of the histogram bars (alpha value). Must be between 0 and 1.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the X-axis.
    ylabel : str, optional
        Label for the Y-axis.
    figsize : tuple[float, float], default=(10, 6)
        Figure size (width, height) in inches.
    style : str, optional
        Seaborn style context (e.g., "whitegrid", "darkgrid").
    palette : str, optional
       Seaborn color palette (e.g., "viridis", "husl").
    plot_kws : dict, optional
        Dictionary of keyword arguments passed directly to the underlying seaborn
        function (`sns.histplot`). This allows overriding any default plotting
        behavior. If not provided, the function internally constructs a dictionary
        from its own relevant parameters. Keys provided in `plot_kws` take precedence
        over internally generated defaults. For complete parameter documentation and
        advanced customization options, see urls below.
    directory : str, optional
        File path to save figure (e.g., "./plot.png").
    nan_policy : str | Literal['drop', 'raise'],
                 default='drop'
        Policy for handling NaN values in input data:
        - 'raise' : raise ValueError if any NaNs are present in `data`.
        - 'drop'  : drop rows (axis=0) containing NaNs before computation. This
                    does **not** drop entire columns.
    verbose : bool, default=False
        If True, enables informational logging during plot generation.

    Returns
    -------
    VisualizationResult
        A dataclass encapsulating the result of a visualization.
        See also :class:`explorica.types.VisualizationResult` for full attribute
        details.

    Raises
    ------
    ValueError
        If input data contains multiple columns (not 1-dimensional).
        If `bins` is not a positive integer.

    Warns
    -----
    UserWarning
        Raised if the input data is empty. An empty plot with a warning message
        will be returned in this case.

    Notes
    -----
    This function uses seaborn.histplot under the hood. For complete parameter
    documentation and advanced customization options, see:
        Seaborn histplot:
        https://seaborn.pydata.org/generated/seaborn.histplot.html#seaborn-histplot
    - Vectorization support is planned to be added.
    - Empty data returns a placeholder plot with informative message.

    Examples
    --------
    >>> import explorica.visualizations as vis

    # Simple distribution plot with KDE
    >>> data = [1, 2, 2, 3, 3, 3, 4]
    >>> result = vis.distplot(data, kde=True, title="Small Dataset Example")
    >>> result.figure.show()

    # Distribution plot with figure saving
    >>> data = [10, 20, 20, 30, 40, 40, 50]
    >>> result = vz.distplot(
    ...     data,
    ...     bins=5,
    ...     title="Saved Plot Example",
    ...     directory="./plots"
    ... )
    >>> Path("./plots/Saved Plot Example.html").exists()
    True

    # Normal distribution with custom bins and figure size
    >>> import numpy as np
    >>>
    >>> data = np.random.normal(loc=50, scale=15, size=200)
    >>> result = vis.distplot(
    ...     data,
    ...     bins=25,
    ...     kde=True,
    ...     title="Normal Distribution Example",
    ...     figsize=(8, 5)
    ...     plot_kws={"color": "skyblue"}
    ... )
    >>> result.figure.show()
    """
    params = {**DEFAULT_MPL_PLOT_PARAMS, "opacity": 0.5, "palette": None, **kwargs}
    plot_kws_merged = {
        "alpha": params["opacity"],
        "bins": bins,
        "kde": kde,
        **params.get("plot_kws", {}),
    }
    series = convert_series(data)
    series = handle_nan(
        series,
        params["nan_policy"],
        supported_policy=("drop", "raise"),
        is_dataframe=False,
    )
    if not isinstance(bins, NaturalNumber):
        raise ValueError("'bins' must be a positive integer.")
    series = series.squeeze(axis=1)
    with temp_plot_theme(palette=params["palette"], style=params["style"]):
        if not series.empty:
            fig, ax = plt.subplots(figsize=params["figsize"])
            sns.histplot(series, ax=ax, **plot_kws_merged)
        else:
            fig, ax = get_empty_plot(figsize=params["figsize"])
            warnings.warn(WRN_MSG_EMPTY_DATA.format("distplot"))
        ax.set_title(params["title"])
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        if params["directory"]:
            save_plot(
                fig,
                directory=params["directory"],
                verbose=params["verbose"],
                plot_name="distplot",
            )
    return VisualizationResult(
        figure=fig,
        axes=ax,
        engine="matplotlib",
        width=params["figsize"][0],
        height=params["figsize"][1],
        title=params["title"],
    )


def boxplot(
    data: Sequence[float] | Mapping[str, Sequence[float]], **kwargs
) -> VisualizationResult:
    """
    Draw a boxplot for a numeric variable.

    This function generates a standard boxplot to visualize the distribution,
    median, quartiles, and potential outliers of numeric data.

    Under the hood, the function uses Matplotlib's `matplotlib.axes.Axes.boxplot`
    function and applies Seaborn styles for aesthetic defaults. This allows passing
    additional kwargs directly to the underlying Matplotlib calls via `plot_kws`.
    For complete parameter documentation and advanced customization options, see
    urls below.

    Parameters
    ----------
    data : Sequence[float] | Mapping[str, Sequence[float]]
        Numeric input data. Can be:
        - 1D sequence of numbers
        - Dictionary with single key-value pair (value is numeric sequence)
        - pandas Series or single-column DataFrame

    Other Parameters
    ----------------
    title : str, optional
        Title of the chart.
    xlabel : str, optional
        Label for the X-axis.
    ylabel : str, optional
        Label for the Y-axis.
    palette : str or list, optional
        Color palette to use for the plot.
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    plot_kws : dict, optional
        Dictionary of keyword arguments passed directly to the underlying matplotlib
        function (`matplotlib.axes.Axes.boxplot`). This allows overriding any default
        plotting behavior. If not provided, the function internally constructs a
        dictionary from its own relevant parameters. Keys provided in `plot_kws` take
        precedence over internally generated defaults. For complete parameter
        documentation and advanced customization options, see urls below.
    directory : str, optional
        If provided, the plot will be saved to this directory.
    nan_policy : {"drop", "raise"}, default="drop"
        How to handle NaN values.
    verbose : bool, default=False
        If True, enables informational logging during plot generation.

    Returns
    -------
    VisualizationResult
        A dataclass encapsulating the result of a visualization.
        See also :class:`explorica.types.VisualizationResult` for full attribute
        details.

    Warns
    -----
    UserWarning
        Raised if the input data is empty. An empty plot with a warning message
        will be returned in this case.

    Notes
    -----
    This function uses matplotlib.axes.Axes.boxplot under the hood. For complete
    parameter documentation and advanced customization options, see:
        Matplotlib boxplot:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.boxplot.html

    Examples
    --------
    >>> import explorica.visualizations as vis
    >>> from pathlib import Path

    # Basic boxplot
    >>> plot = vis.boxplot([5, 7, 8, 5, 6, 9, 12], title="Simple Boxplot")
    >>> plot.figure.show()

    # Saving the plot to disk
    >>> plot = vis.boxplot([4, 6, 5, 7, 8], directory="./plots", title="Saved Boxplot")
    >>> Path("./plots/Saved Boxplot.html").exists()
    True

    # Detecting potential outliers
    >>> data_with_outliers = [10, 12, 11, 14, 100, 13, 12, 9, 105]
    >>> plot = vis.boxplot(data_with_outliers, title="Boxplot with Outliers")
    >>> plot.figure.show()
    """
    params = {
        **DEFAULT_MPL_PLOT_PARAMS,
        "palette": None,
        **kwargs,
    }
    series = convert_series(
        handle_nan(
            data,
            params["nan_policy"],
            supported_policy=("drop", "raise"),
            is_dataframe=False,
        )
    )
    with temp_plot_theme(palette=params["palette"], style=params["style"]):
        if not series.empty:
            fig, ax = plt.subplots(figsize=params["figsize"])
            ax.boxplot(series, **params.get("plot_kws", {}))
        else:
            fig, ax = get_empty_plot(figsize=params["figsize"])
            warnings.warn(WRN_MSG_EMPTY_DATA.format("boxplot"))
        ax.set_title(params["title"])
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        if params["directory"] is not None:
            save_plot(
                fig, params["directory"], verbose=params["verbose"], plot_name="boxplot"
            )
    return VisualizationResult(
        figure=fig,
        axes=ax,
        engine="matplotlib",
        width=params["figsize"][0],
        title=params["title"],
        height=params["figsize"][1],
    )


def hexbin(
    data: Sequence[Number], target: Sequence[Number], **kwargs
) -> VisualizationResult:
    """
    Create a hexbin plot for two numeric variables to visualize point density.

    A hexbin plot is a bivariate histogram that uses hexagonal bins to display
    the density of points in a 2D space. It is particularly useful for large
    datasets where scatter plots become overcrowded.

    Under the hood, the function uses Matplotlib's `matplotlib.axes.Axes.hexbin`
    function and applies Seaborn styles for aesthetic defaults. This allows passing
    additional kwargs directly to the underlying Matplotlib calls via `plot_kws`.
    For complete parameter documentation and advanced customization options, see
    urls below.

    Parameters
    ----------
    data : Sequence[Number]
        First numeric variable (plotted on x-axis).
    target : Sequence[Number]
        Second numeric variable (plotted on y-axis).

    Other Parameters
    ----------------
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap used to color hexagons by count density. Defaults to None.
    opacity : float, default=1
        Opacity of the hexagons (0 = fully transparent, 1 = fully opaque).
    gridsize : int, default=30
        The number of hexagons in the x-direction. Larger values produce smaller
        hexagons and higher resolution.
    title : str, optional
        Title of the chart.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    style : str, default=None
        Seaborn style context (e.g., "whitegrid", "darkgrid").
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    plot_kws : dict, optional
        Dictionary of keyword arguments passed directly to the underlying matplotlib
        function (`matplotlib.axes.Axes.hexbin`). This allows overriding any default
        plotting behavior. If not provided, the function internally constructs a
        dictionary from its own relevant parameters. Keys provided in `plot_kws` take
        precedence over internally generated defaults. For complete parameter
        documentation and advanced customization options, see urls below.
    directory : str, optional
        If provided, the plot will be saved to this directory.
    nan_policy : {"drop", "raise"}, default="drop"
        How to handle NaN values in the data.
    verbose : bool, default=False
        If True, enables informational logging during plot generation.

    Returns
    -------
    VisualizationResult
        A dataclass encapsulating the result of a visualization.
        See also :class:`explorica.types.VisualizationResult` for full attribute
        details.

    Raises
    ------
    ValueError
        If `data` and `target` have different lengths.
        If NaN values are present and `nan_policy="raise"`.
        If `gridsize` is not a positive integer.

    Warns
    -----
    UserWarning
        Raised if the input data is empty. An empty plot with a warning message
        will be returned in this case.

    Notes
    -----
    This function uses matplotlib.axes.Axes.hexbin under the hood.
    For complete parameter documentation and advanced customization options, see:
        Matplotlib hexbin:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hexbin.html
    - Hexbin plots are most effective with large datasets (>1000 points)
    - The color intensity represents the count of points in each hexagon
    - For very sparse data, consider using a scatter plot instead

    Examples
    --------
    import explorica.visualizations as vis

    # Simple usage
    >>> plot = vis.hexbin(
    ...     [1, 2, 3, 4, 5],
    ...     [2, 4, 6, 8, 10],
    ...     xlabel="X Variable",
    ...     ylabel="Y Variable",
    ...     gridsize=20
    ... )
    >>> plot.figure.show()

    # Saving the plot to a directory
    >>> plot = vis.hexbin(
    ...     [1, 2, 3, 4, 5],
    ...     [2, 4, 6, 8, 10],
    ...     title="Hexbin Example",
    ...     directory="plots",
    ...     figsize=(8, 5)
    ... )
    >>> # The figure is saved to the 'plots' directory with filename 'hexbin.png'

    # Passing additional Matplotlib options via plot_kws
    >>> plot = vis.hexbin(
    ...     [1, 2, 3, 4, 5],
    ...     [2, 4, 6, 8, 10],
    ...     plot_kws={"cmap": "viridis", "mincnt": 1},
    ...     gridsize=25
    ... )
    >>> plot.figure.show()
    """
    params = {
        **DEFAULT_MPL_PLOT_PARAMS,
        "cmap": None,
        "gridsize": 30,
        "opacity": 1,
        **kwargs,
    }
    plot_kws_merged = {
        "gridsize": params["gridsize"],
        "alpha": params["opacity"],
        **params.get("plot_kws", {}),
    }
    # NaturalNumber is descriptor, isinstace ensures positive integer check
    if not isinstance(params["gridsize"], NaturalNumber):
        raise ValueError("'gridsize' must be a positive integer.")
    x_series, y_series = convert_series(data), convert_series(target)

    validate_lengths_match(
        x_series,
        y_series,
        err_msg=ERR_MSG_ARRAYS_LENS_MISMATCH.format("data", "target"),
    )
    df = pd.DataFrame({"x": x_series, "y": y_series})
    df = handle_nan(
        df,
        params["nan_policy"],
        supported_policy=("drop", "raise"),
        data_name="data or target",
    )
    with temp_plot_theme(cmap=params["cmap"], style=params["style"]):
        if not df.empty:
            fig, ax = plt.subplots(figsize=params["figsize"])
            ax.hexbin(df["x"], df["y"], **plot_kws_merged)
        else:
            fig, ax = get_empty_plot(figsize=params["figsize"])
            warnings.warn(WRN_MSG_EMPTY_DATA.format("hexbin"))
        ax.set_title(params["title"])
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        if params["directory"] is not None:
            save_plot(
                fig,
                params["directory"],
                verbose=params["verbose"],
                plot_name="hexbin",
            )
    return VisualizationResult(
        figure=fig,
        axes=ax,
        engine="matplotlib",
        width=params["figsize"][0],
        height=params["figsize"][1],
        title=params["title"],
    )


def heatmap(
    data: Sequence[float] | Sequence[Sequence[float]] | Mapping[Any, Sequence[float]],
    **kwargs,
) -> VisualizationResult:
    """
    Draw a heatmap from the provided data using Matplotlib and Seaborn.

    Create a heatmap visualization from numerical data, supporting 1D, 2D sequences,
    or mappings of keys to sequences. The function automatically handles NaN values
    according to nan_policy, and provides options for figure size,
    annotations, and saving the plot to a directory.

    Under the hood, the function uses Seaborn's `seaborn.heatmap` function
    and applies Seaborn styles for aesthetic defaults. This allows passing
    additional kwargs directly to the underlying Seaborn calls via `plot_kws`.
    For complete parameter documentation and advanced customization options, see
    urls below.

    Parameters
    ----------
    data : sequence of floats, sequence of sequences of floats, or mapping
        Input data for the heatmap. Can be:
        - 1D sequence of numerical values (converted to a 1-row heatmap),
        - 2D sequence (list of lists, NumPy array, etc.),
        - Mapping of keys to sequences (converted to a DataFrame).

    Other Parameters
    ----------------
    annot : bool, default=True
        Whether to annotate the heatmap cells with their numeric values.
    cmap : str, optional
        Color map for the heatmap.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    figsize : tuple[float, float], default=(10, 6)
        Figure size.
    plot_kws : dict, optional
        Dictionary of keyword arguments passed directly to the underlying seaborn
        function (`seaborn.heatmap`). This allows overriding any default plotting
        behavior. If not provided, the function internally constructs a dictionary
        from its own relevant parameters. Keys provided in `plot_kws` take precedence
        over internally generated defaults. For complete parameter documentation and
        advanced customization options, see urls below.
    directory : str or None, optional
        Directory path to save the figure. If None, the figure is not saved.
    nan_policy : str, default="drop"
        Policy for handling NaN values in input data. Supports 'drop'
        (removes rows with NaNs) or 'raise' (raises an error).
    verbose : bool, default=False
        Whether to print additional messages during plotting.

    Returns
    -------
    VisualizationResult
        A dataclass encapsulating the result of a visualization.
        See also :class:`explorica.types.VisualizationResult` for full attribute
        details.

    Raises
    ------
    ValueError
        If NaN values are present and `nan_policy="raise"`.

    Warns
    -----
    UserWarning
        Raised if the input data is empty. An empty plot with a warning message
        will be returned in this case.

    Notes
    -----
    This function uses seaborn.heatmap under the hood. For complete parameter
    documentation and advanced customization options, see:
        Seaborn heatmap:
        https://seaborn.pydata.org/generated/seaborn.heatmap.html

    Examples
    --------
    import explorica.visualizations as vis

    # Simple usage
    >>> plot = vis.heatmap(
    ...     [[1, 2, 3, 4, 5],
    ...     [5, 4, 3, 2, 1],
    ...     [2, 4, 6, 8, 10]],
    ...     xlabel="X Variable",
    ...     ylabel="Y Variable",
    ...     cmap="viridis"
    ... )
    >>> plot.figure.show()

    # Saving the plot to a directory
    >>> plot = vis.heatmap(
    ...     [[1, 2, 3, 4, 5],
    ...     [5, 4, 3, 2, 1],
    ...     [2, 4, 6, 8, 10]],
    ...     title="Heatmap Example",
    ...     directory="plots",
    ...     figsize=(8, 5)
    ... )
    >>> # The figure is saved to the 'plots' directory with filename 'heatmap.png'

    # Passing additional Matplotlib options via plot_kws
    >>> plot = vis.heatmap(
    ...     [[1, 2, 3, 4, 5],
    ...     [5, 4, 3, 2, 1],
    ...     [2, 4, 6, 8, 10]],
    ...     plot_kws={"cmap": "viridis", "cbar": True},
    ...     annot=False
    ... )
    >>> plot.figure.show()
    """
    params = {
        **DEFAULT_MPL_PLOT_PARAMS,
        "annot": True,
        "cmap": None,
        "cbar": True,
        "fmt": ".2f",
        "square": True,
        **kwargs,
    }
    df = convert_dataframe(data)
    df = handle_nan(df, params["nan_policy"], supported_policy=("drop", "raise"))
    plot_kws_merged = {
        "annot": params["annot"],
        "cmap": params["cmap"],
        "cbar": params["cbar"],
        "fmt": params["fmt"],
        "square": params["square"],
        **params.get("plot_kws", {}),
    }
    if not df.empty:
        fig, ax = plt.subplots(figsize=params["figsize"])
        sns.heatmap(df, ax=ax, **plot_kws_merged)
    else:
        fig, ax = get_empty_plot(figsize=params["figsize"])
        warnings.warn(WRN_MSG_EMPTY_DATA.format("heatmap"), stacklevel=2)
    ax.set_title(params["title"])
    ax.set_xlabel(params["xlabel"])
    ax.set_ylabel(params["ylabel"])
    if params["directory"] is not None:
        save_plot(fig, params["directory"], plot_name="heatmap")
    return VisualizationResult(
        figure=fig,
        axes=ax,
        engine="matplotlib",
        width=params["figsize"][0],
        height=params["figsize"][1],
        title=params["title"],
    )
