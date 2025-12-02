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
"""

import warnings
from numbers import Number
from typing import Sequence, Union, Sequence, Mapping, Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

from ._utils import temp_plot_theme, save_plot, get_empty_plot
from explorica._utils import (convert_dataframe, convert_series,
                              natural_number, handle_nan,
                              read_config, validate_lengths_match,
                              validate_string_flag)

WRN_MSG_CATEGORIES_EXCEEDS_PALETTE_F = read_config("messages")[
    "warns"]["DataVisualizer"]["categories_exceeds_palette_f"]
WRN_MSG_EMPTY_DATA = read_config("messages")["warns"]["DataVisualizer"]["empty_data_f"]
ERR_MSG_ARRAYS_LENS_MISMATCH_F = read_config("messages")["errors"][
    "arrays_lens_mismatch_f"]
ERR_MSG_UNSUPPORTED_METHOD_F = read_config("messages")["errors"]["unsupported_method_f"]

def distplot(data: Sequence[float] | Mapping[str, Sequence[Any]],
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
    data : Sequence[float] | Mapping[str, Sequence[Any]]
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
    tuple (matplotlib.figure.Figure, matplotlib.axes.Axes)
        Figure and axes objects for further customization.
    
    Raises
    ------
    ValueError
        If input data contains multiple columns (not 1-dimensional).
    ValueError  
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
    Basic one-liner usage with pandas Series:

    >>> import pandas as pd
    >>> import explorica.visualizations as visualizations

    >>> data = pd.Series([0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5])
    >>> visualizations.distplot(data) # Will show a histogram

    Customized plot with styling and saving:

    >>> visualizations.distplot(
    ...     data, 
    ...     bins=20,
    ...     kde=False,
    ...     style="darkgrid", 
    ...     palette="viridis",
    ...     title="Distribution",
    ...     directory="./histogram.png")
        ./
        ├── main.py
        └── some_plot.png
    """
    params = {
        "palette": None,
        "opacity": 0.5,

        "title": "",
        "xlabel": "",
        "ylabel": "",
        "style": None,
        "figsize": (10, 6),
        "directory": None,
        "nan_policy": "drop",
        "verbose": False,
        **kwargs
    }
    series = convert_series(data)
    series = handle_nan(
        series,
        params["nan_policy"],
        supported_policy = ("drop", "raise"),
        is_dataframe=False)
    if not isinstance(bins, natural_number):
        raise ValueError(
            "'bins' must be a positive integer."
        )
    series = series.squeeze(axis=1)
    with temp_plot_theme(palette=params["palette"], style=params["style"]):
        if not series.empty:
            fig, ax = plt.subplots(figsize=params["figsize"])
            sns.histplot(series, bins=bins, kde=kde,
                         alpha=params["opacity"], ax=ax)
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
        return fig, ax



def boxplot(data: Sequence[Number],
            title: str = "",
            **kwargs
            ) -> tuple[Figure, Axes]:
    """
    Draw a boxplot for a numeric variable.

    Parameters
    ----------
    x : array-like of shape (n,)
        Numeric values to visualize in the boxplot.
    title : str, default=""
        Title of the chart.
    xlabel : str or None, optional
        Label for the x-axis. If None, no label is shown.
    ylabel : str, default="Quantiles"
        Label for the y-axis.
    return_plot : bool, default=False
        If True, return the matplotlib Figure object.
    show_plot : bool, default=True
        If True, display the chart immediately.
    dir : str or None, optional
        Directory path where the chart is saved. If None, the chart is not saved.

    Returns
    -------
    matplotlib.figure.Figure or None
        The resulting matplotlib Figure object if `return_plot=True`,
        otherwise None.

    Raises
    ------
    ValueError
        If `x` contains null values.
    """
    params = {
        "directory": None,
        "figsize": (10, 6),
        "xlabel": None,
        "ylabel": "Quantiles",
        "style": None,
        "palette": None,
        "nan_policy": "drop",
        "verbose": False,
        **kwargs,
    }
    series = convert_dataframe(data).iloc[:, 0]
    series = handle_nan(series, params["nan_policy"],
                        supported_policy=("drop", "raise"),)

    with temp_plot_theme(palette=params["palette"], style=params["style"]):
        fig, ax = plt.subplots(figsize = params["figsize"])
        sns.boxplot(series, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        if params["directory"] is not None:
            save_plot(fig, params["directory"],
                      plot_name="boxplot",
                      verbose=params["verbose"])
    return fig, ax



def hexbin(data: Sequence[Number],
           target: Sequence[Number],
           colormap: str = "viridis",
           **kwargs,
           ) -> Union[None | plt.Figure]:
    """
    Create a hexbin plot for two numeric variables to visualize point density.

    Parameters
    ----------
    x : array-like of shape (n,)
        First numeric variable (x-axis).
    y : array-like of shape (n,)
        Second numeric variable (y-axis).
    colormap : str, default="viridis"
        Colormap used to color hexagons by count density.
    gridsize : int, default=30
        The number of hexagons in the x-direction. Larger values produce smaller
        hexagons.
    title : str, default=""
        Title of the chart.
    xlabel : str or None, optional
        Label for the x-axis. If None, no label is shown.
    ylabel : str or None, optional
        Label for the y-axis. If None, no label is shown.
    return_plot : bool, default=False
        If True, return the matplotlib Figure object.
    show_plot : bool, default=True
        If True, display the chart immediately.
    dir : str or None, optional
        Directory path where the chart is saved. If None, the chart is not saved.

    Returns
    -------
    matplotlib.figure.Figure or None
        The resulting matplotlib Figure object if `return_plot=True`,
        otherwise None.

    Raises
    ------
    ValueError
        If `x` or `y` contains null values.
    """
    params = {
        "title": "",
        "return_plot": False,
        "show_plot": True,
        "directory": None,
        "xlabel": None,
        "ylabel": "Quantiles",
        "params": None,
        "gridsize": 30,
        "figsize": (10, 6),
        "nan_policy": "drop",
        "verbose": False,
        **kwargs,
    }
    x_series = convert_dataframe(data).iloc[:, 0]
    y_series = convert_dataframe(target).iloc[:, 0]
    x_series = handle_nan(x_series, params["nan_policy"], ("drop", "raise"), data_name="data")
    y_series = handle_nan(y_series, params["nan_policy"], ("drop", "raise"), data_name="target")

    with temp_plot_theme(cmap=colormap, style=params["style"]):
        fig, ax = plt.subplots(figsize = params["figsize"])
        plt.hexbin(x_series, y_series, gridsize=params["gridsize"], ax=ax)
        ax.set_title(params["title"])
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["xlabel"])
        # ax.set_colorbar(label="Counts")
        if params["directory"] is not None:
            save_plot(fig, params["directory"],
                      plot_name="hexbin",
                      verbose=params["verbose"])
    return fig, ax

def scatterplot(data: Sequence[Number],
                target: Sequence[Number],
                category: Sequence = None,
                **kwargs
                ) -> Union[None | plt.Figure]:
    """
    Draw a scatterplot with optional categorical coloring and trend line fitting.

    Supported trend line methods
    ----------------------------
    - 'linear'      : y = a * x + b
    - 'binomial'    : y = a * x² + b * x + c
    - 'exp'         : y = b * exp(a * x)
    - 'ln'          : y = a * ln(x) + b
    - 'hyperbolic'  : y = a / x + b
    - 'power'       : y = a * xᵇ
    - 'custom'      : y = custom_function(x)

    Parameters
    ----------
    x : array-like of shape (n,)
        Numerical values for the x-axis. Must not contain nulls.
    y : array-like of shape (n,)
        Numerical values for the y-axis. Must not contain nulls.
    category : array-like of shape (n,), optional
        Categorical labels for each point. If provided, must match the length of
        `x` and `y`, and contain no nulls. Points will be colored by category.
    trend_line : {'linear', 'binomial', 'exp', 'ln', 'hyperbolic', 'power',
                  'custom'}, optional
        Specifies the type of trend line to fit. If None, no trend line is shown.
    custom_function : callable, optional
        Custom function to be used for fitting when `trend_line="custom"`.
        Must accept an array of floats and return an array of floats.
    palette : iterable of str, optional
        Iterable of color values (e.g., hex codes or named colors). If not provided,
        the default class palette will be used. If the number of unique categories
        exceeds the number of colors, the palette repeats and a warning is issued.
    opacity : float, default=1.0
        Opacity level of points, in the interval [0.0, 1.0].
    show_legend : bool, default=False
        Whether to display a legend (only works if `category` is provided).
    title : str, default=""
        Title of the plot.
    xlabel : str or None, optional
        Label for the x-axis. If None, no label is shown.
    ylabel : str or None, optional
        Label for the y-axis. If None, no label is shown.
    return_plot : bool, default=False
        If True, return the matplotlib Figure object.
    show_plot : bool, default=True
        If True, display the chart immediately.
    dir : str or None, optional
        Directory path where the chart is saved. If None, the chart is not saved.
    
    Returns
    -------
    matplotlib.figure.Figure or None
        The resulting matplotlib Figure object if `return_plot=True`,
        otherwise None.

    Raises
    ------
    ValueError
        If input sizes mismatch.
        If any of `x`, `y`, or `category` contains null values.
        If an unsupported `trend_line` method is provided.
        If `trend_line="custom"` but `custom_function` is not supplied.
        If `category` is provided but its length does not match `x` and `y`.

    Warns
    -----
    UserWarning
        If the number of unique categories exceeds the palette length, causing
        colors to repeat.
        If trend line fitting fails (e.g., model does not converge), the trend
        line is not displayed.

    Notes
    -----
    For some trend types, `x` values are filtered automatically to avoid
    mathematical errors:

    - 'exp'         : exclude x > ~350 to avoid numerical overflow.
    - 'ln'          : exclude x <= 0.
    - 'hyperbolic'  : exclude x == 0.
    - 'power'       : exclude x == 0.

    Trend lines are fitted using ``scipy.optimize.curve_fit`` and drawn smoothly
    over the filtered domain. If the fit fails due to invalid data or convergence
    issues, no trend line is shown. The trend line is drawn using the next color
    in the palette after the last category color.
    """
    params = {
        "title": "",
        "c_title": "",
        "return_plot": False,
        "show_plot": True,
        "dir": None,
        "xlabel": None,
        "ylabel": None,
        "show_legend": False,
        "figsize": (10, 6),
        "opacity": 1.0,
        "palette": None,
        "style": None,
        "custom_function": None,
        "trend_line": None,
        "nan_policy": "drop",
        "verbose": False,
        **kwargs,
    }
    models = {
        "linear": lambda x, a, b: a * x + b,
        "binomial": lambda x, a, b, c: a * x**2 + b * x + c,
        "exp": lambda x, a, b: b * np.exp(a * x),
        "ln": lambda x, a, b: a * np.log(x) + b,
        "hyperbolic": lambda x, a, b: a / x + b,
        "power": lambda x, a, b: a * x**b,
        "custom": params["custom_function"],
    }
    def filter_x_by_domain(x: pd.Series, y: pd.Series, func_type: str) -> pd.Series:
        if func_type == "ln":
            y = y[x > 0]
            x = x[x > 0]
        elif func_type in {"hyperbolic", "power"}:
            y = y[x != 0]
            x = x[x != 0]
        elif func_type == "exp":
            y = y[x <= 350]
            x = x[x <= 350]
        return x, y
    x_series = convert_dataframe(data).iloc[:, 0]
    y_series = convert_dataframe(target).iloc[:, 0]
    category_series = (
        convert_dataframe(category).iloc[:, 0]
        if category is not None
        else None
    )
    # Parameter validation
    x_series = handle_nan(x_series, params["nan_policy"], ("drop", "raise"), data_name="data")
    y_series = handle_nan(y_series, params["nan_policy"], ("drop", "raise"), data_name="target")
    validate_lengths_match(
        x_series,
        y_series,
        ERR_MSG_ARRAYS_LENS_MISMATCH_F.format("x", "y"),
    )
    if params["trend_line"] is not None:
        validate_string_flag(
            params["trend_line"],
            models,
            ERR_MSG_UNSUPPORTED_METHOD_F.format(
                params["trend_line"], set(models.keys())
            ),
        )
    if params["trend_line"] == "custom" and params["custom_function"] is None:
        raise ValueError("Custom function must be provided when method='custom'")
    if isinstance(params["palette"], str):
        params["palette"] = sns.color_palette(params["palette"])

    with temp_plot_theme(palette=params["palette"],
                         style=params["style"]):
        fig, ax = plt.subplots(figsize=params["figsize"])
        if category_series is not None:
            validate_string_flag(
                x_series,
                category_series,
                ERR_MSG_ARRAYS_LENS_MISMATCH_F.format(
                    "category", "x' and 'y"
                ),
                n_dim=1,
            )
            category_series = handle_nan(category_series, params["nan_policy"],
                                         ("drop", "raise"), data_name="categories")
            unique_categories = category_series.unique()
            n_unique_categories = category_series.nunique()
            if n_unique_categories > len(params["palette"]):
                warnings.warn(
                    WRN_MSG_CATEGORIES_EXCEEDS_PALETTE_F.format(n_unique_categories, len(params["palette"])),
                    UserWarning,
                )
            colors = {
                cat: params["palette"][i % len(params["palette"])]
                for i, cat in enumerate(unique_categories)
            }
            for cat in unique_categories:
                plt.scatter(
                    x_series[category_series == cat],
                    y_series[category_series == cat],
                    label=cat if params["show_legend"] else None,
                    alpha=params["opacity"],
                    color=colors[cat],
                    ax=ax
                )
            trend_line_color = params["palette"][
                n_unique_categories % len(params["palette"])
            ]
        else:
            plt.scatter(
                x_series,
                y_series,
                alpha=params["opacity"],
                color=params["palette"][0],
                ax=ax
            )
            trend_line_color = params["palette"][1 % len(params["palette"])]
        if params["trend_line"]:
            if params["trend_line"] == "custom":
                model_x = np.linspace(x_series.min(), x_series.max(), 1000)
                model_values = models[params["trend_line"]](model_x)
                plt.plot(model_x, model_values, color=trend_line_color, ax=ax)
            else:
                model_x, model_y = filter_x_by_domain(
                    x_series, y_series, params["trend_line"]
                )
                try:
                    coeffs = curve_fit(
                        models[params["trend_line"]], model_x, model_y, maxfev=10000
                    )[0]
                    model_x = np.linspace(model_x.min(), model_x.max(), 1000)
                    model_values = models[params["trend_line"]](model_x, *coeffs)
                    plt.plot(model_x, model_values, color=trend_line_color, ax=ax)
                except RuntimeError as e:
                    warnings.warn(
                        f"Could not fit '{params['trend_line']}' trend: {e}",
                        UserWarning,
                    )
        if category_series is not None and params["show_legend"]:
            ax.legend(
                title=params["c_title"], bbox_to_anchor=(1.05, 1), loc="upper left"
            )
        # None < pd.Series.name < "x/y_label" argument priority
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        ax.set_title(params["title"])
        if params["directory"] is not None:
            save_plot(fig, params["directory"],
                      plot_name="scatterplot",
                      verbose=params["verbose"])
    return fig, ax


def heatmap(data: Sequence[float],
            annot: bool = True,
            title: str = "",
            cmap: str = None,
            **kwargs):
        """
        Plots a heatmap to visualize correlation matrices & time series.

        Parameters
        ----------
        dataframe : pd.DataFrame
            2-dimensional numeric dataset.
        annot : bool
            Whether to annotate cells with correlation or other values.
        title : str
            Plot title.
        colormap : str
            Color map.
        """
        params = {
            "style": None,
            "figsize": (10, 6),
            "directory": None,
            **kwargs}
        df = convert_dataframe(data)
        with temp_plot_theme(cmap=cmap, style=params["style"]):
            fig, ax = plt.subplots(figsize = params["figsize"])
            sns.heatmap(df, annot=annot, fmt=".2f", square=True, ax=ax)
            ax.set_title(title)
            if params["directory"] is not None:
                save_plot(fig, params["directory"], plot_name="heatmap")
        return fig, ax