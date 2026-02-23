r"""
High-level plotting utilities for Explorica visualizations.

This module provides a high-level interface for generating scatter plots with
optional categorization and trendline fitting. It is designed to offer a
consistent and expressive API built on top of Matplotlib, with seamless support
for themes, palettes, NaN handling, and plot saving.

Functions
---------
scatterplot(data, target, category=None, \**kwargs)
    Generates a scatter plot with optional categorization and a trendline.

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
>>> data = [1, 2, 3, 4, 5, 6, 7]
>>> target = [2.5, 3.2, 4.8, 5.1, 7.5, 9.0, 10.5]
>>> category = ['A', 'A', 'B', 'B', 'A', 'B', 'A']
>>> # Basic scatterplot with linear trendline
>>> plot = vis.scatterplot(data, target, trendline='linear', show_legend=False,
...                       title='Basic Scatterplot')
>>> plot.figure.show() # doctest: +SKIP

>>> # Scatterplot with categories and saving
>>> plot = vis.scatterplot( # doctest: +SKIP
...     data, target, category=category, show_legend=True,
...     title='Scatterplot with Categories', directory='plots',
...     figsize=(8, 5))
>>> # The figure is saved to the 'plots' directory with filename 'scatterplot.png'

>>> # Scatterplot with polynomial trendline and custom plot_kws
>>> plot = vis.scatterplot(data, target, trendline='polynomial',
...                       plot_kws={'s': 100, 'marker': 'o', 'c': 'orange',
...                                 'edgecolor': 'black'},
...                       title='Scatterplot with Polynomial Trendline')
>>> plot.figure.show() # doctest: +SKIP
>>> plt.close(plot.figure)
"""

import warnings
from typing import Any, Callable, Sequence
from numbers import Number

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from explorica._utils import (
    validate_lengths_match,
    validate_string_flag,
    convert_from_alias,
    convert_series,
    handle_nan,
)
from explorica.visualizations._utils import (
    temp_plot_theme,
    save_plot,
    get_empty_plot,
    DEFAULT_MPL_PLOT_PARAMS,
)
from explorica.types import VisualizationResult, NaturalNumber
from ._utils import (
    WRN_MSG_CATEGORIES_EXCEEDS_PALETTE_F,
    WRN_MSG_EMPTY_DATA,
    ERR_MSG_ARRAYS_LENS_MISMATCH_F,
    ERR_MSG_UNSUPPORTED_METHOD_F,
    ERR_MSG_ARRAYS_LENS_MISMATCH,
)

__all__ = [
    "scatterplot",
]


def scatterplot(
    data: Sequence[Number],
    target: Sequence[Number],
    category: Sequence[Any] = None,
    **kwargs,
) -> VisualizationResult:
    """
    Generate a scatter plot with optional categorization and a trendline.

    The function supports coloring points by category and displaying a fitted trendline.
    The trendline can be automatically calculated using Ordinary Least Squares (OLS)
    for linear or polynomial regression, or a custom pre-calculated user function
    can be provided.

    Under the hood, the function uses Matplotlib's `matplotlib.axes.Axes.scatter`
    function and applies Seaborn styles for aesthetic defaults. This allows passing
    additional kwargs directly to the underlying Matplotlib calls via `plot_kws`.
    For complete parameter documentation and
    advanced customization options, see urls below.

    Parameters
    ----------
    data : Sequence[Number]
        Data for the X-axis.
    target : Sequence[Number]
        Data for the Y-axis.
    category : Sequence[Any], optional
        Categorical data used for coloring points and generating a legend.
        Defaults to None (no categorization).

    Returns
    -------
    VisualizationResult
        A dataclass encapsulating the result of a visualization.
        See also :class:`explorica.types.VisualizationResult` for full attribute
        details.

    Other Parameters
    ----------------
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    title_legend : str, default="Category"
        Title for the category legend.
    show_legend : bool, default=False
        If True, displays the category legend (if categories exist).
    opacity : float, optional
        Transparency level for scatter points (0.0 to 1.0).
    palette : str or list or None, optional
        Color palette name (e.g., 'viridis') or a list of specific colors.
    cmap : str or None, optional
        Colormap name for continuous data (rarely used in scatter plots
        but included for theme compatibility).
    style : str or None, optional
        Matplotlib style context (e.g., 'seaborn-v0_8').
    figsize : tuple[int, int], default=(10, 6)
        Figure size (width, height) in inches.
    trendline : str or Callable, optional
        Method to draw a trendline. Supports 'linear', 'polynomial', or a custom
        callable function. If a string ('linear' or 'polynomial') is provided,
        the trendline is automatically fitted using Ordinary Least Squares (OLS)
        under the hood. If a callable function is provided, it must implement a mapping
        from a single numeric input to a single numeric output (y = f(x));
        the function itself is not modified and will be automatically vectorized
        over the X-domain for plotting.
    trendline_kws : dict, optional
        Additional arguments for the trendline function. Keys include:

        * **'color'** : str, optional
            Color of the trendline.
        * **'linestyle'** : str, default='--'
            Linestyle of the trendline (e.g., '-', '--', ':', '-.').
        * **'linewidth'** : int, default=2
            Thickness of the trendline.
        * **'x_range'** : tuple[float, float], optional
            The domain (min, max) for which the trendline should be calculated
            and plotted. If None, it uses the min and max of the input `data` (x).
        * **'degree'** : int, default=2
            The degree of the polynomial to fit. Only used when
            `trendline='polynomial'`.
        * **'dots'** : int, default=1000
            The number of data points used to draw the smooth trendline curve.
    plot_kws : dict, optional
        Dictionary of keyword arguments passed directly to the underlying seaborn
        function (`matplotlib.axes.Axes.scatter`). This allows overriding any default
        plotting behavior. If not provided, the function internally constructs a
        dictionary from its own relevant parameters. Keys provided in `plot_kws` take
        precedence over internally generated defaults. For complete parameter
        documentation and advanced customization options, see urls below.
    nan_policy : str, default="drop"
        Policy for handling NaN values in input data. Supports 'drop'
        (removes rows with NaNs) or 'raise' (raises an error).
    directory : str or None, optional
        Directory path to save the plot. If None, the plot is not saved.
    verbose : bool, default=False
        If True, prints save messages.

    Raises
    ------
    ValueError
        - If `trendline` is a string and is not one of the supported methods
          ('linear', 'polynomial').
        - If `trendline` is a callable and its output is not a 1D sequence of
          numbers with the same length as `x_domain`.
        - If `data`, `target`, or `category` (if provided) are not 1D sequences
          or their lengths do not match.
        - If any of `data`, `target`, or `category` contains NaNs and
          `nan_policy='raise'`.
        - If `trendline_kws['degree']` or `trendline_kws['dots']` are not
          natural numbers.

    Warns
    -----
    UserWarning
        if the input data is empty. An empty plot with a warning message
        will be returned in this case.
        If the number of unique objects to visualize (categories + trendline)
        exceeds the number of available colors in the chosen palette.

    Notes
    -----
    This function uses matplotlib.axes.Axes.scatter under the hood. For complete
    parameter documentation and advanced customization options, see:
    `matplotlib scatterplot`_.

    .. _matplotlib scatterplot: https://matplotlib.org/stable/api/
       _as_gen/matplotlib.axes.Axes.scatter.html>`_

    Examples
    --------
    >>> import explorica.visualizations as vis
    >>> data = [1, 2, 3, 4, 5, 6, 7]
    >>> target = [2.5, 3.2, 4.8, 5.1, 7.5, 9.0, 10.5]
    >>> category = ['A', 'A', 'B', 'B', 'A', 'B', 'A']
    >>>
    >>> # Basic scatterplot with linear trendline
    >>> plot = vis.scatterplot(
    ...     data,
    ...     target,
    ...     trendline='linear',
    ...     show_legend=False,
    ...     title='Basic Scatterplot'
    ... )
    >>> plot.figure.show() # doctest: +SKIP

    >>> # Scatterplot with categories and saving the plot
    >>> plot = vis.scatterplot( # doctest: +SKIP
    ...     data,
    ...     target,
    ...     category=category,
    ...     show_legend=True,
    ...     title='Scatterplot with Categories',
    ...     directory='plots',
    ...     figsize=(8, 5)
    ... )
    >>> # The figure is saved to the 'plots' directory with filename 'scatterplot.png'

    >>> # Passing additional Matplotlib options via plot_kws and a polynomial trendline
    >>> plot = vis.scatterplot(
    ...     data,
    ...     target,
    ...     trendline="polynomial",
    ...     plot_kws={'s': 100, 'marker': 'o', 'c': 'orange', 'edgecolor': 'black'},
    ...     title='Scatterplot with Polynomial Trendline'
    ... )
    >>> plot.figure.show() # doctest: +SKIP
    >>> plt.close(plot.figure)
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
    plot_kws_merged = {"alpha": params["opacity"], **(params["plot_kws"] or {})}
    params["trendline_kws"] = params["trendline_kws"] or {
        "color": None,
        "linestyle": "-",
        "linewidth": 2,
        "dots": 1000,
        "degree": 2,
        "x_range": None,
    }
    df, params["trendline"] = _scatterplot_data_preprocess(
        data, target, category, params["nan_policy"], trendline=params["trendline"]
    )

    with temp_plot_theme(
        palette=params["palette"], style=params["style"], cmap=params["cmap"]
    ):
        if not df.empty:
            fig, ax = _scatterplot_get_scatter_with_trendline(
                df,
                have_category=category is not None,
                show_legend=params["show_legend"],
                opacity=params["opacity"],
                figsize=params["figsize"],
                trendline=params["trendline"],
                title_legend=params["title_legend"],
                trendline_kws=params["trendline_kws"],
                scatter_kws=plot_kws_merged,
            )
        else:
            fig, ax = get_empty_plot(figsize=params["figsize"])
            warnings.warn(WRN_MSG_EMPTY_DATA.format("scatterplot"))
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        ax.set_title(params["title"])
        if params["directory"] is not None:
            save_plot(
                fig,
                params["directory"],
                plot_name="scatterplot",
                verbose=params["verbose"],
            )
    return VisualizationResult(
        engine="matplotlib",
        figure=fig,
        axes=ax,
        width=params["figsize"][0],
        height=params["figsize"][1],
        title=params["title"],
    )


def _scatterplot_data_preprocess(
    data: Sequence[Number],
    target: Sequence[Number],
    category: Sequence[Any],
    nan_policy: str,
    trendline: str | Callable | None,
):
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
        df, nan_policy, ("drop", "raise"), data_name="data, target or category"
    )
    if trendline is not None:
        if isinstance(trendline, str):
            trendline = convert_from_alias(trendline, ("linear", "polynomial"))
            validate_string_flag(
                trendline,
                ("linear", "polynomial"),
                ERR_MSG_UNSUPPORTED_METHOD_F.format(
                    trendline, ("linear", "polynomial")
                ),
            )
        elif not callable(trendline):
            raise ValueError(
                ERR_MSG_UNSUPPORTED_METHOD_F.format(
                    trendline, ("linear", "polynomial", "callable function")
                )
            )
    return df, trendline


def _scatterplot_get_scatter_with_trendline(
    df, have_category, show_legend, opacity, **kwargs
):
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
    colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
    trendline_kws = kwargs.get("trendline_kws", {}).copy()

    # CRITICAL: Create a copy of the trendline kwargs to prevent side effects
    # when setting a default color below.
    if have_category:
        unique_categories = df["category"].unique()

        # Calculate the total number of unique
        # elements (categories + trendline, if present)
        unique_objects = (
            len(unique_categories) + 1 if kwargs.get("trendline") is not None else 0
        )
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
            "color", colors[(len(unique_categories) + 1) % len(colors)]
        )
    else:
        unique_objects = 1 + (1 if kwargs.get("trendline") is not None else 0)
        # Plot uncategorized data using the first color in the cycle
        ax.scatter(df["x"], df["y"], **kwargs.get("scatter_kws", {}))

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
            df["x"],
            df["y"],
            kwargs.get("trendline"),
            trendline_kws={
                "degree": trendline_kws.get("degree"),
                "x_range": trendline_kws.get("x_range"),
                "dots": trendline_kws.get("dots"),
            },
        )

        # Plot the calculated trendline
        ax.plot(
            trendline[0],
            trendline[1],
            color=trendline_kws.get("color"),
            linestyle=trendline_kws.get("linestyle", "-"),
            linewidth=trendline_kws.get("linewidth", 2),
        )
    if have_category and show_legend:

        # Place the legend outside the plot area
        ax.legend(
            title=kwargs.get("title_legend"), bbox_to_anchor=(1.05, 1), loc="upper left"
        )
    return fig, ax


def scatterplot_fit_trendline(
    x: pd.Series, y: pd.Series, method: str | Callable, **trendline_kws
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
        validate_lengths_match(
            y_pred,
            x_domain,
            err_msg=ERR_MSG_ARRAYS_LENS_MISMATCH.format(
                "x_domain", "y_predicted from custom trendline function"
            ),
        )
        trendline = (x_domain, y_pred)
    elif method == "linear":
        trendline = _scatterplot_fit_linear_ols(x, y, trendline_kws=trendline_kws)
    elif method == "polynomial":
        trendline = _scatterplot_fit_polinomial_ols(x, y, trendline_kws=trendline_kws)
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


def _scatterplot_fit_linear_ols(x: np.ndarray, y: np.ndarray, **trendline_kws):
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
