"""
Module provides statistical visualization tools for numerical data.

This module defines the :class:`StatisticalVisualizer`, which offers
methods for exploring distributions and relationships between numeric variables.
It can be used as part of the high-level :class:`DataVisualizer` facade
or independently for focused statistical analysis.

Methods
-------
distplot(data, bins, kde, title, **kwargs)
    Plots a distribution histogram for a numeric variable with optional
    Kernel Density Estimation (KDE).
boxplot(data, title, **kwargs)
    Draws a boxplot for a numeric variable to visualize distribution,
    median, and potential outliers.
hexbin(data, target, colormap, **kwargs)
    Creates a hexbin plot for two numeric variables using the default colormap.
    Useful for visualizing dense scatter data.
scatterplot(data, target, category, **kwargs)
    Draws a scatterplot with optional categorical coloring and trendline.
    Supported trendline types include 'linear', 'exp', etc.
heatmap(data, **kwargs)
    Generates a heatmap to visualize a 2D array of numeric values.
"""

import warnings
from numbers import Number
from typing import Sequence, Mapping, Callable, Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import seaborn as sns

from explorica._utils import (convert_dataframe, convert_series,
                              NaturalNumber, handle_nan,
                              read_config, validate_lengths_match,
                              validate_string_flag, convert_from_alias)
from ._utils import (temp_plot_theme, save_plot, get_empty_plot,
                     DEFAULT_MPL_PLOT_PARAMS, VisualizationResult)

WRN_MSG_CATEGORIES_EXCEEDS_PALETTE_F = read_config("messages")[
    "warns"]["DataVisualizer"]["categories_exceeds_palette_f"]
WRN_MSG_EMPTY_DATA = read_config("messages")["warns"]["DataVisualizer"]["empty_data_f"]
ERR_MSG_ARRAYS_LENS_MISMATCH_F = read_config("messages")["errors"][
    "arrays_lens_mismatch_f"]
ERR_MSG_UNSUPPORTED_METHOD_F = read_config("messages")["errors"]["unsupported_method_f"]
ERR_MSG_ARRAYS_LENS_MISMATCH = read_config(
    "messages")["errors"]["arrays_lens_mismatch_f"]

def distplot(data: Sequence[float] | Mapping[str, Sequence[float]],
             bins: int = 30,
             kde: bool = True,
             **kwargs) -> tuple[Figure, Axes]:
    """
    Plot a histogram with optional kernel density estimate.

    This function generates a histogram for univariate data with an optional
    kernel density estimate (KDE) curve overlay. It automatically handles
    data conversion, NaN values, and provides flexible styling options
    through integration with Seaborn's plotting system.

    Parameters
    ----------
    data : Sequence[float] | Mapping[str, Sequence[float]]
        Numeric input data. Can be:
        - 1D sequence of numbers
        - Dictionary with single key-value pair (value is numeric sequence)
        - pandas Series or single-column DataFrame
    bins : int, default=30
        Number of bins in the histogram.
    kde : bool, default=True
        If True, adds kernel density estimate curve.
    opacity : float, default=0.5
        Transparency of the histogram bars (alpha value). Must be between 0 and 1.
    title : str, default=""
        Plot title.
    xlabel : str, default=""
        Label for the X-axis.
    ylabel : str, default=""
        Label for the Y-axis.
    figsize : tuple[float, float], default=(10, 6)
        Figure size (width, height) in inches.
    style : str, default=None
        Seaborn style context (e.g., "whitegrid", "darkgrid").
    palette : str, default=None
       Seaborn color palette (e.g., "viridis", "husl").
    plot_kws : dict, optional
        Dictionary of keyword arguments passed directly to the underlying seaborn
        function (`sns.histplot`). This allows overriding any default plotting
        behavior. If not provided, the function internally constructs a dictionary
        from its own relevant parameters. Keys provided in `plot_kws` take precedence
        over internally generated defaults.
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
        Contains the following attributes:
            - figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
            The generated figure object ready for display or saving.
            - axes : matplotlib.axes.Axes or None
            For Matplotlib figures, the primary Axes object; None for Plotly figures.
            - engine : str
            The plotting engine used, either 'matplotlib' or 'plotly'.
            - width : int or None
            Figure width in pixels (Plotly) or inches (Matplotlib);
            None if not specified.
            - height : int or None
            Figure height in pixels (Plotly) or inches (Matplotlib);
            None if not specified.
            - title : str or None
            The title of the visualization, if specified.
            - extra_info : dict or None
            Optional dictionary storing additional metadata about the visualization,
            such as color palettes, zoom levels, legend settings, or any other info
            relevant to downstream processing or reproducibility.

    Raises
    ------
    ValueError
        If input data contains multiple columns (not 1-dimensional).
        If `bins` is not a positive integer.

    Warns
    -----
    UserWarning
        If input data is empty after NaN handling.
        
    Notes
    -----
    - Vectorization support is planned to be added.
    - Empty data returns a placeholder plot with informative message.
    
    Examples
    --------
    Basic usage with a numeric sequence:

    >>> from explorica.visualizations import distplot
    >>> result = distplot([1, 2, 2, 3, 3, 3])
    >>> isinstance(result.figure, object)
    True
    >>> result.axes is not None
    True

    Displaying the plot interactively (not executed in tests):

    >>> result = distplot([1, 2, 3, 4, 5])
    >>> result.figure.show()  # doctest: +SKIP
    """
    params = {
        **DEFAULT_MPL_PLOT_PARAMS,
        "opacity": 0.5,
        "palette": None,
        **kwargs
    }
    plot_kws_merged = {
        "alpha": params["opacity"],
        "bins": bins,
        "kde": kde,
        **(params["plot_kws"] or {})
    }
    series = convert_series(data)
    series = handle_nan(
        series,
        params["nan_policy"],
        supported_policy = ("drop", "raise"),
        is_dataframe=False)
    if not isinstance(bins, NaturalNumber):
        raise ValueError(
            "'bins' must be a positive integer."
        )
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
                plot_name="distplot")
    return VisualizationResult(figure=fig, axes=ax, engine="matplotlib",
                               width=params["figsize"][0],
                               height=params["figsize"][1],
                               title=params["title"])

def boxplot(data: Sequence[float] | Mapping[str, Sequence[float]],
            **kwargs
            ) -> tuple[Figure, Axes]:
    """
    Draw a boxplot for a numeric variable.

    Parameters
    ----------
    data : Sequence[float] | Mapping[str, Sequence[float]]
        Numeric input data. Can be:
        - 1D sequence of numbers
        - Dictionary with single key-value pair (value is numeric sequence)
        - pandas Series or single-column DataFrame
    title : str, optional
        Title of the chart.
    xlabel : str, default=""
        Label for the X-axis.
    ylabel : str, default=""
        Label for the Y-axis.
    palette : str or list, optional
        Color palette to use for the plot.
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    plot_kws : dict, optional
        Dictionary of keyword arguments passed directly to the underlying matplotlib
        function (`plt.boxplot`). This allows overriding any default plotting
        behavior. If not provided, the function internally constructs a dictionary
        from its own relevant parameters. Keys provided in `plot_kws` take precedence
        over internally generated defaults.
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
        Contains the following attributes:
            - figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
            The generated figure object ready for display or saving.
            - axes : matplotlib.axes.Axes or None
            For Matplotlib figures, the primary Axes object; None for Plotly figures.
            - engine : str
            The plotting engine used, either 'matplotlib' or 'plotly'.
            - width : int or None
            Figure width in pixels (Plotly) or inches (Matplotlib);
            None if not specified.
            - height : int or None
            Figure height in pixels (Plotly) or inches (Matplotlib);
            None if not specified.
            - title : str or None
            The title of the visualization, if specified.
            - extra_info : dict or None
            Optional dictionary storing additional metadata about the visualization,
            such as color palettes, zoom levels, legend settings, or any other info
            relevant to downstream processing or reproducibility.

    Examples
    --------
    >>> # Single dataset
    >>> plot = boxplot([1, 2, 3, 4, 5], title="Boxplot Example")
    >>> plot.figure.show()
    >>> 
    >>> # With NaN values and drop policy
    >>> import numpy as np
    >>> data_with_nan = [1, 2, np.nan, 4, 5]
    >>> plot = boxplot(data_with_nan, nan_policy="drop")
    >>> plot.figure.show()
    """
    params = {
        **DEFAULT_MPL_PLOT_PARAMS,
        "palette": None,
        **kwargs,
    }
    series = convert_series(
        handle_nan(data, params["nan_policy"],
        supported_policy=("drop", "raise"), is_dataframe=False))

    with temp_plot_theme(palette=params["palette"], style=params["style"]):
        if not series.empty:
            fig, ax = plt.subplots(figsize = params["figsize"])
            sns.boxplot(series, ax=ax, **(params["plot_kws"] or {}))
        else:
            fig, ax = get_empty_plot(figsize=params["figsize"])
            warnings.warn(WRN_MSG_EMPTY_DATA.format("boxplot"))
        ax.set_title(params["title"])
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        if params["directory"] is not None:
            save_plot(fig, params["directory"],
                      plot_name="boxplot",
                      verbose=params["verbose"])
    return VisualizationResult(figure=fig, axes=ax, engine="matplotlib",
                               width=params["figsize"][0],
                               height=params["figsize"][1],
                               title=params["title"])



def hexbin(data: Sequence[Number],
           target: Sequence[Number],
           **kwargs
           ) -> tuple[Figure, Axes]:
    """
    Create a hexbin plot for two numeric variables to visualize point density.
    
    A hexbin plot is a bivariate histogram that uses hexagonal bins to display
    the density of points in a 2D space. It is particularly useful for large
    datasets where scatter plots become overcrowded.
    
    Parameters
    ----------
    data : Sequence[Number]
        First numeric variable (plotted on x-axis).
    target : Sequence[Number]
        Second numeric variable (plotted on y-axis).
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
        function (`plt.hexbin`). This allows overriding any default plotting
        behavior. If not provided, the function internally constructs a dictionary
        from its own relevant parameters. Keys provided in `plot_kws` take precedence
        over internally generated defaults.
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
        Contains the following attributes:
            - figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
            The generated figure object ready for display or saving.
            - axes : matplotlib.axes.Axes or None
            For Matplotlib figures, the primary Axes object; None for Plotly figures.
            - engine : str
            The plotting engine used, either 'matplotlib' or 'plotly'.
            - width : int or None
            Figure width in pixels (Plotly) or inches (Matplotlib);
            None if not specified.
            - height : int or None
            Figure height in pixels (Plotly) or inches (Matplotlib);
            None if not specified.
            - title : str or None
            The title of the visualization, if specified.
            - extra_info : dict or None
            Optional dictionary storing additional metadata about the visualization,
            such as color palettes, zoom levels, legend settings, or any other info
            relevant to downstream processing or reproducibility.

    Raises
    ------
    ValueError
        If `data` and `target` have different lengths.
        If NaN values are present and `nan_policy="raise"`.
        If `gridsize` is not a positive integer.
    
    Notes
    -----
    - Hexbin plots are most effective with large datasets (>1000 points)
    - The color intensity represents the count of points in each hexagon
    - For very sparse data, consider using a scatter plot instead
    
    Examples
    --------
    >>> plot = hexbin(
    ...     [1, 2, 3, 4, 5],
    ...     [2, 4, 6, 8, 10],
    ...     xlabel="X Variable",
    ...     ylabel="Y Variable",
    ...     gridsize=20
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
        **params["plot_kws"]
        }
    # NaturalNumber is descriptor, isinstace ensures positive integer check
    if not isinstance(params["gridsize"], NaturalNumber):
        raise ValueError(
            "'gridsize' must be a positive integer."
        )
    x_series, y_series = convert_series(data), convert_series(target)

    validate_lengths_match(
        x_series, y_series,
        err_msg=ERR_MSG_ARRAYS_LENS_MISMATCH.format("data", "target"))
    df = pd.DataFrame({"x": x_series, "y": y_series})
    df = handle_nan(df, params["nan_policy"], supported_policy=("drop", "raise"),
                    data_name="data or target")
    with temp_plot_theme(cmap=params["cmap"], style=params["style"]):
        if not df.empty:
            fig, ax = plt.subplots(figsize = params["figsize"])
            ax.hexbin(df["x"], df["y"], **plot_kws_merged)
        else:
            fig, ax = get_empty_plot(figsize=params["figsize"])
            warnings.warn(WRN_MSG_EMPTY_DATA.format("hexbin"))
        ax.set_title(params["title"])
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        if params["directory"] is not None:
            save_plot(fig, params["directory"],
                      plot_name="hexbin",
                      verbose=params["verbose"])
    return VisualizationResult(figure=fig, axes=ax, engine="matplotlib",
                               width=params["figsize"][0],
                               height=params["figsize"][1],
                               title=params["title"])

def scatterplot(data: Sequence[Number],
                target: Sequence[Number],
                category: Sequence[Any] = None,
                **kwargs
                ) -> tuple[Figure, Axes]:
    """
    Generates a scatter plot with optional categorization and a trendline.

    The function supports coloring points by category and displaying a fitted trendline.
    The trendline can be automatically calculated using Ordinary Least Squares (OLS) 
    for linear or polynomial regression, or a custom pre-calculated user function 
    can be provided.

    Parameters
    ----------
    data : Sequence[Number]
        Data for the X-axis.
    target : Sequence[Number]
        Data for the Y-axis.
    category : Sequence[Any], optional
        Categorical data used for coloring points and generating a legend.
        Defaults to None (no categorization).
    
    Other Parameters
    ----------------
    title : str, default=""
        Plot title.
    xlabel : str, default=""
        X-axis label.
    ylabel : str, default=""
        Y-axis label.
    title_legend : str, default="Category"
        Title for the category legend.
    show_legend : bool, default=False
        If True, displays the category legend (if categories exist).
    opacity : float, default=1.0
        Transparency level for scatter points (0.0 to 1.0).
    palette : str or list or None, default=None
        Color palette name (e.g., 'viridis') or a list of specific colors.
    cmap : str or None, default=None
        Colormap name for continuous data (rarely used in scatter plots
        but included for theme compatibility).
    style : str or None, default=None
        Matplotlib style context (e.g., 'seaborn-v0_8').
    figsize : tuple[int, int], default=(10, 6)
        Figure size (width, height) in inches.
    trendline : str or Callable or None, default=None
        Method to draw a trendline. Supports 'linear', 'polynomial', or a custom
        callable function. If a string ('linear' or 'polynomial') is provided,
        the trendline is automatically fitted using Ordinary Least Squares (OLS)
        under the hood. If a callable function is provided, it must implement a mapping
        from a single numeric input to a single numeric output (y = f(x));
        the function itself is not modified and will be automatically vectorized
        over the X-domain for plotting.
    trendline_kws : dict, default=None
        Additional arguments for the trendline function. Keys include:
        * **'color'** : str, default=None
            Color of the trendline.
        * **'linestyle'** : str, default='--'
            Linestyle of the trendline (e.g., '-', '--', ':', '-.').
        * **'linewidth'** : int, default=2
            Thickness of the trendline.
        * **'x_range'** : tuple[float, float], default=None
            The domain (min, max) for which the trendline should be calculated 
            and plotted. If None, it uses the min and max of the input `data` (x).
        * **'degree'** : int, default=2
            The degree of the polynomial to fit. Only used when
            `trendline='polynomial'`.
        * **'dots'** : int, default=1000
            The number of data points used to draw the smooth trendline curve.
    plot_kws : dict, optional
        Dictionary of keyword arguments passed directly to the underlying matplotlib
        function (`plt.scatter`). This allows overriding any default plotting
        behavior. If not provided, the function internally constructs a dictionary
        from its own relevant parameters. Keys provided in `plot_kws` take precedence
        over internally generated defaults. Doesn't work if category is provided.
    nan_policy : str, default="drop"
        Policy for handling NaN values in input data. Supports 'drop' 
        (removes rows with NaNs) or 'raise' (raises an error).
    directory : str or None, default=None
        Directory path to save the plot. If None, the plot is not saved.
    verbose : bool, default=False
        If True, prints save messages.

    Returns
    -------
    VisualizationResult
        A dataclass encapsulating the result of a visualization.
        Contains the following attributes:
            - figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
            The generated figure object ready for display or saving.
            - axes : matplotlib.axes.Axes or None
            For Matplotlib figures, the primary Axes object; None for Plotly figures.
            - engine : str
            The plotting engine used, either 'matplotlib' or 'plotly'.
            - width : int or None
            Figure width in pixels (Plotly) or inches (Matplotlib);
            None if not specified.
            - height : int or None
            Figure height in pixels (Plotly) or inches (Matplotlib);
            None if not specified.
            - title : str or None
            The title of the visualization, if specified.
            - extra_info : dict or None
            Optional dictionary storing additional metadata about the visualization,
            such as color palettes, zoom levels, legend settings, or any other info
            relevant to downstream processing or reproducibility.
    
    Raises
    ------
    ValueError
        If `trendline` is a string and is not one of the supported methods
        ('linear', 'polynomial').
        If `trendline` is a callable and its output is not a 1D sequence of
        numbers with the same length as `x_domain`.
        If `data`, `target`, or `category` (if provided) are not 1D sequences
        or their lengths do not match.
        If any of `data`, `target`, or `category` contains NaNs and
        `nan_policy='raise'`.
        If `trendline_kws['degree']` or `trendline_kws['dots']` are not
        natural numbers.

    Warns
    -----
    UserWarning
        If the resulting DataFrame after preprocessing is empty (no data to plot).
        If the number of unique objects to visualize (categories + trendline)
        exceeds the number of available colors in the chosen palette.

    Examples
    --------
    >>> data = [1, 2, 3, 4, 5, 6, 7]
    >>> target = [2.5, 3.2, 4.8, 5.1, 7.5, 9.0, 10.5]
    >>> category = ['A', 'A', 'B', 'B', 'A', 'B', 'A']
    
    # 1. Basic scatterplot with trendline
    >>> plot = scatterplot(data, target, trendline='linear', show_legend=False, 
    ...                       title='basic scatterplot')
    >>> plot.figure.show()
    """
    params = {
        **DEFAULT_MPL_PLOT_PARAMS,
        "title_legend": "",
        "show_legend": False,
        "opacity": 1.0,
        "palette": None,
        "trendline": None,
        "trendline_kws": None,
        "cmap": None,
        **kwargs,
    }
    plot_kws_merged = {
        "alpha": params["opacity"],
        **(params["plot_kws"] or {})
    }
    params["trendline_kws"] = params["trendline_kws"] or {
        "color": None,
        "linestyle": '-',     
        "linewidth": 2,
        "dots": 1000,
        "degree": 2,
        "x_range": None}
    df, params["trendline"] = _scatterplot_data_preprocess(
        data, target, category, params["nan_policy"], trendline=params["trendline"])

    with temp_plot_theme(palette=params["palette"],
                         style=params["style"],
                         cmap=params["cmap"]):
        if not df.empty:
            fig, ax = _scatterplot_get_scatter_with_trendline(
                df,
                have_category=category is not None,
                show_legend=params["show_legend"],
                opacity=params["opacity"],
                figsize=params["figsize"],
                trendline=params["trendline"],
                title_legend=params["title_legend"],
                trendline_kws = params["trendline_kws"],
                scatter_kws=plot_kws_merged
            )
        else:
            fig, ax = get_empty_plot(figsize=params["figsize"])
            warnings.warn(WRN_MSG_EMPTY_DATA.format("scatterplot"))
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        ax.set_title(params["title"])
        if params["directory"] is not None:
            save_plot(fig, params["directory"],
                      plot_name="scatterplot",
                      verbose=params["verbose"])
    return VisualizationResult(figure=fig, axes=ax, engine="matplotlib",
                               width=params["figsize"][0],
                               height=params["figsize"][1],
                               title=params["title"])

def _scatterplot_data_preprocess(data: Sequence[Number],
                                     target: Sequence[Number],
                                     category: Sequence[Any],
                                     nan_policy: str,
                                     trendline: str|Callable|None):
    """
    Processes, validates, and prepares input dataspecifically for the
    `scatterplot` function.

    This utility function performs all preliminary checks before plotting the
    scatter chart:
    1. Converts input sequences (data, target, category) to pandas Series.
    2. Validates that the lengths of all provided arrays match.
    3. Applies the specified policy for handling missing values (`nan_policy`).
    4. Valicates the trendline method: resolves aliases (e.g., 'lin' for 'linear')
       and checks if the method is in the list of supported strings.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing cleaned and ready-to-use 'x', 'y', and 'category' 
        columns (if provided).
    trendline : str or Callable or None
        The validated trendline method (string, function, or None).
    """
    x = convert_series(data)
    y = convert_series(target)
    validate_lengths_match(
        x,
        y,
        ERR_MSG_ARRAYS_LENS_MISMATCH_F.format("x", "y"),
    )
    df = pd.DataFrame({"x": x, "y": y})
    if category is not None:
        category_series = convert_series(category)
        validate_lengths_match(
            x,
            category_series,
            ERR_MSG_ARRAYS_LENS_MISMATCH_F.format("data and target", "category"),
            )
        df["category"] = category_series
    df = handle_nan(
        df, nan_policy, ("drop", "raise"),
        data_name="data, target or category")
    if trendline is not None:
        if isinstance(trendline, str):
            trendline = convert_from_alias(trendline, ("linear", "polynomial"))
            validate_string_flag(
                trendline, ("linear", "polynomial"),
                ERR_MSG_UNSUPPORTED_METHOD_F.format(
                    trendline, ("linear", "polynomial")))
        elif not callable(trendline):
            raise ValueError(
                ERR_MSG_UNSUPPORTED_METHOD_F.format(
                    trendline, ("linear", "polynomial", "callable function")))
    return df, trendline

def _scatterplot_get_scatter_with_trendline(df,
                                   have_category,
                                   show_legend,
                                   opacity,
                                   **kwargs):
    """
    Creates the Matplotlib Figure and Axes, plots scatter points,
    and adds a trendline for the `scatterplot` function.

    This function handles the core plotting logic, including:
    1. Plotting categorical data by iterating over unique categories
       and explicitly assigning colors from the current Matplotlib color
       cycle (defined by 'palette' in the theme).
    2. Plotting uncategorized data.
    3. Fitting and plotting the trendline (if requested).
    4. Adding the legend for categorical data.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared DataFrame containing 'x', 'y', and optionally 'category'.
    have_category : bool
        Flag indicating if categorical data is present in the DataFrame.
    show_legend : bool
        If True, displays the category legend.
    opacity : float
        Transparency level for scatter points.
    **kwargs
        Parameters related to styling, trendline, and legend titles.

    Returns
    -------
    tuple[Figure, Axes]
        The Matplotlib Figure and Axes objects.
    """
    fig, ax = plt.subplots(figsize=kwargs.get("figsize"))
    colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    trendline_kws = kwargs.get("trendline_kws", {}).copy()

    # CRITICAL: Create a copy of the trendline kwargs to prevent side effects
    # when setting a default color below.
    if have_category:
        unique_categories = df["category"].unique()

        # Calculate the total number of unique
        # elements (categories + trendline, if present)
        unique_objects = (len(unique_categories) + 1
                          if kwargs.get("trendline") is not None else 0)
        for i, cat in enumerate(unique_categories):
            subset_df = df.loc[df["category"] == cat]
            ax.scatter(
                subset_df["x"],
                subset_df["y"],
                label=cat if show_legend else None,
                color=colors[i % len(colors)],
                alpha=opacity,
            )
        # Assign the next available color
        # from the cycle to the trendline (if not manually set)
        trendline_kws["color"] = trendline_kws.get(
            "color", colors[(len(unique_categories) + 1) % len(colors)])
    else:
        unique_objects = 1 + (1 if kwargs.get("trendline") is not None else 0)
        # Plot uncategorized data using the first color in the cycle
        ax.scatter(
            df["x"],
            df["y"],
            **kwargs.get("scatter_kws", {})
        )

        # Assign the second color to the trendline (if not manually set)
        trendline_kws["color"] = trendline_kws.get("color", colors[1 % len(colors)])
    if unique_objects > len(colors):
        warnings.warn(
            WRN_MSG_CATEGORIES_EXCEEDS_PALETTE_F.format(unique_objects, len(colors)),
            UserWarning,
        )
    if kwargs.get("trendline") is not None:

        # Calculate the trendline coordinates
        trendline = scatterplot_fit_trendline(
            df["x"], df["y"], kwargs.get("trendline"),
            trendline_kws={
                "degree": trendline_kws.get("degree"),
                "x_range": trendline_kws.get("x_range"),
                "dots": trendline_kws.get("dots")})

        # Plot the calculated trendline
        ax.plot(trendline[0], trendline[1],
                color = trendline_kws.get("color"),
                linestyle=trendline_kws.get("linestyle", "-"),
                linewidth=trendline_kws.get("linewidth", 2))
    if have_category and show_legend:

        # Place the legend outside the plot area
        ax.legend(
            title=kwargs.get("title_legend"),
            bbox_to_anchor=(1.05, 1), loc="upper left")
    return fig, ax

def scatterplot_fit_trendline(x: pd.Series,
                   y: pd.Series,
                   method: str|Callable,
                   **trendline_kws
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatcher function to calculate trendline coordinates based on the method provided.

    Parameters
    ----------
    x : pandas.Series
        The independent variable data (X).
    y : pandas.Series
        The dependent variable data (Y).
    method : str or Callable
        The method to use: 'linear', 'polynomial', or a custom function.
    trendline_kws : dict
        A dictionary of keyword arguments passed to the specific fitting function.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The (x_domain, y_predicted) arrays for plotting the trendline.

    Raises
    ------
    ValueError
        If 'dots' is not a positive integer.
    """
    if not isinstance(trendline_kws.get("dots", 1000), NaturalNumber):
        raise ValueError("'dots' must be a positive integer.")

    if callable(method):
        # Handle custom callable function for trendline

        dots = trendline_kws.get("dots", 1000)
        x_range = trendline_kws.get("x_range")

        x_min, x_max = x.min(), x.max()
        if x_range is not None:
            x_min, x_max = x_range

        x_domain = np.linspace(x_min, x_max, dots)
        y_pred = method(x_domain)
        validate_lengths_match(y_pred, x_domain,
                               err_msg=ERR_MSG_ARRAYS_LENS_MISMATCH.format(
            "x_domain", "y_predicted from custom trendline function"))
        trendline = (x_domain, y_pred)
    elif method == "linear":
        trendline = _scatterplot_fit_linear_ols(x, y, trendline_kws=trendline_kws)
    elif method == "polynomial":
        trendline = _scatterplot_fit_polinomial_ols(
            x, y, trendline_kws=trendline_kws)
    else:
        trendline = None

    return trendline

def _scatterplot_fit_polinomial_ols(
    x: pd.Series,
    y: pd.Series,
    **trendline_kws,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the polynomial trendline using Ordinary Least Squares (OLS).

    This function is a helper for `_fit_trendline` and returns coordinates
    ready for plotting.

    Parameters
    ----------
    x : pandas.Series
        The independent variable data (X).
    y : pandas.Series
        The dependent variable data (Y).
    **trendline_kws
        Keyword arguments, primarily expects 'degree' (int), 'dots' (int), 
        and optional 'x_range' (tuple).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The (x_domain, y_predicted) arrays for plotting the curve.

    Raises
    ------
    ValueError
        If 'degree' is not a positive integer.
    """
    if not isinstance(trendline_kws.get("degree", 2), NaturalNumber):
        raise ValueError("'degree' must be a positive integer.")

    if trendline_kws.get("x_range") is None:
        x_range = [x.min(), x.max()]
    else:
        x_range = trendline_kws.get("x_range")

    coefficients = np.polyfit(x, y, deg=trendline_kws.get("degree", 2))
    polynomial = np.poly1d(coefficients)

    x_domain = np.linspace(x_range[0], x_range[1], trendline_kws.get("dots", 1000))
    y_pred = polynomial(x_domain)
    return (x_domain, y_pred)

def _scatterplot_fit_linear_ols(
    x: np.ndarray,
    y: np.ndarray,
    **trendline_kws
):
    """
    Calculates the linear trendline (degree 1) using Ordinary Least Squares (OLS).

    Parameters
    ----------
    x : pandas.Series or np.ndarray
        The independent variable data (X).
    y : pandas.Series or np.ndarray
        The dependent variable data (Y).
    **trendline_kws
        Keyword arguments, primarily expects 'dots'
        (int) and optional 'x_range' (tuple).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The (x_domain, y_predicted) arrays for plotting the line.
    """
    if trendline_kws.get("x_range") is None:
        x_range = [x.min(), x.max()]
    else:
        x_range = trendline_kws.get("x_range")
    x_with_intercept = np.column_stack([x, np.ones_like(x)])

    coefficients = np.linalg.lstsq(x_with_intercept, y, rcond=None)[0]
    x_domain = np.linspace(x_range[0], x_range[1], trendline_kws.get("dots", 1000))
    y_pred = x_domain * coefficients[0] + coefficients[1]
    return (x_domain, y_pred)



def heatmap(data: Sequence[float]|Sequence[Sequence[float]]|
            Mapping[Any, Sequence[float]],
            **kwargs):
    """
    Draw a heatmap from the provided data using Matplotlib and Seaborn.

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
    cmap : str or Colormap, default=None
        Color map for the heatmap.
    title : str, default=""
        Plot title.
    xlabel : str, default=""
        X-axis label.
    ylabel : str, default=""
        Y-axis label.
    figsize : tuple of (float, float), default=(10, 6)
        Figure size.
    plot_kws : dict, optional
        Dictionary of keyword arguments passed directly to the underlying seaborn
        function (`sns.heatmap`). This allows overriding any default plotting
        behavior. If not provided, the function internally constructs a dictionary
        from its own relevant parameters. Keys provided in `plot_kws` take precedence
        over internally generated defaults.
    directory : str or None, default=None
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
        Contains the following attributes:
            - figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
            The generated figure object ready for display or saving.
            - axes : matplotlib.axes.Axes or None
            For Matplotlib figures, the primary Axes object; None for Plotly figures.
            - engine : str
            The plotting engine used, either 'matplotlib' or 'plotly'.
            - width : int or None
            Figure width in pixels (Plotly) or inches (Matplotlib);
            None if not specified.
            - height : int or None
            Figure height in pixels (Plotly) or inches (Matplotlib);
            None if not specified.
            - title : str or None
            The title of the visualization, if specified.
            - extra_info : dict or None
            Optional dictionary storing additional metadata about the visualization,
            such as color palettes, zoom levels, legend settings, or any other info
            relevant to downstream processing or reproducibility.
    
    Raises
    ------
    ValueError
        If NaN values are present and `nan_policy="raise"`.

    Warns
    -----
    UserWarning
        Raised if the input data is empty. An empty plot with a warning message
        will be returned in this case.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, 5, 6]])
    >>> plot = heatmap(data, title="Example Heatmap", figsize=(8, 5))
    >>> plot.axes.get_title()
    'Example Heatmap'
    """
    params = {
        **DEFAULT_MPL_PLOT_PARAMS,
        "annot": True,
        "cmap": None,
        "cbar": True,
        "fmt": ".2f",
        "square": True,
        **kwargs}
    df = convert_dataframe(data)
    df = handle_nan(df, params["nan_policy"], supported_policy = ("drop", "raise"))
    plot_kws_merged = {
            "annot": params["annot"],
            "cmap": params["cmap"],
            "cbar": params["cbar"],
            "fmt": params["fmt"],
            "square": params["square"],
            **(params["plot_kws"] or {})
        }
    if not df.empty:
        fig, ax = plt.subplots(figsize = params["figsize"])
        sns.heatmap(df, ax=ax, **plot_kws_merged)
    else:
        fig, ax = get_empty_plot(figsize=params["figsize"])
        warnings.warn(WRN_MSG_EMPTY_DATA.format("heatmap"), stacklevel=2)
    ax.set_title(params["title"])
    ax.set_xlabel(params["xlabel"])
    ax.set_ylabel(params["ylabel"])
    if params["directory"] is not None:
        save_plot(fig, params["directory"], plot_name="heatmap")
    return VisualizationResult(figure=fig, axes=ax, engine="matplotlib",
                               width=params["figsize"][0],
                               height=params["figsize"][1],
                               title=params["title"])
