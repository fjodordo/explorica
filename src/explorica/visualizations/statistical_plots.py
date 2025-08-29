"""
Module provides statistical visualization tools for numerical data.

This module defines the :class:`StatisticalVisualizer`, which offers
methods for exploring distributions and relationships between numeric variables.
It can be used as part of the high-level :class:`DataVisualizer` facade
or independently for focused statistical analysis.

Classes
-------
StatisticalVisualizer
    Encapsulates plotting methods for distributions (histograms, KDE, boxplots)
    and numeric relationships (scatterplots, hexbin).
"""

import warnings
from numbers import Number
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

from explorica._utils import ConvertUtils as cutils
from explorica._utils import ValidationUtils as vutils
from explorica._utils import read_messages
from explorica.visualizations._utils import handle_plot_output_matplotlib


class StatisticalVisualizer:
    """
    Provides statistical visualization methods for numerical data.

    This class focuses on plots that describe distributions and relationships
    between numeric variables. It is designed to be used both as part of the
    high-level :class:`DataVisualizer` facade or independently for targeted
    statistical analysis.

    Class Attributes
    ----------------
    style : str
        Default seaborn style to apply to matplotlib-based plots
        (e.g., 'whitegrid', 'dark', 'ticks').

    palette : seaborn color palette
        Default color palette used for plots. This must be a palette object returned by
        `sns.color_palette(...)`, supporting `.as_hex()` and `.as_rgb()` methods.

    colormap : str
        Default seaborn colormap to apply to matplotlib-based plots
        (e.g., 'coolwarm', 'viridis', 'plasma').

    Methods
    -------
    distplot(x: Sequence[Number], bins: int = 30, kde: bool = True, **kwargs)
        Plots a distribution histogram for a numeric variable with optional
        Kernel Density Estimation (KDE).

    boxplot(x: Sequence[Number], title: str = "", **kwargs)
        Draws a boxplot for a numeric variable to visualize distribution,
        median, and potential outliers.

    hexbin(x: Sequence[Number], y: Sequence[Number], colormap: str = "viridis",
           **kwargs)
        Creates a hexbin plot for two numeric variables using the default colormap.
        Useful for visualizing dense scatter data.

    scatterplot(x: Sequence[Number], y: Sequence[Number], category: Sequence = None,
                **kwargs)
        Draws a scatterplot with optional categorical coloring and trendline.
        Supported trendline types include 'linear', 'exp', etc.
    """

    style = "whitegrid"
    palette = sns.color_palette("Set2")
    colormap = "coolwarm"
    errors = read_messages()["errors"]
    warns = read_messages()["warns"]

    @classmethod
    def distplot(
        cls, x: Sequence[Number], bins: int = 30, kde: bool = True, **kwargs
    ) -> Union[None | plt.Figure]:
        """
        Plots a histogram with optional KDE (kernel density estimate).

        This method provides a high-level interface for quickly visualizing
        numeric data. If a pandas Series or single-column DataFrame is
        provided, the x-axis label is automatically inferred from the
        Series name. The input is validated to ensure it does not contain NaNs.

        Parameters
        ----------
        x : Sequence[Number] or pd.Series or pd.DataFrame
            Numeric data to plot. Single-column DataFrame or Series are
            automatically converted and validated.
        bins : int, default=30
            Number of bins for the histogram.
        kde : bool, default=True
            Whether to include a KDE curve.
        **kwargs : optional
            Additional parameters controlling plot behavior:
            - title : str
                Plot title (default: "").
            - xlabel : str
                Label for the x-axis (default: inferred from Series/DataFrame).
            - ylabel : str
                Label for the y-axis (default: "Frequency").
            - return_plot : bool
                Whether to return the Matplotlib Figure object (default: False).
            - show_plot : bool
                Whether to call plt.show() (default: True).
            - dir : str
                File path to save the figure (default: None).

        Returns
        -------
        matplotlib.figure.Figure or None
            The Figure object if `return_plot=True`, otherwise None.

        Raises
        ------
        ValueError
            If the input data contains NaNs.
        """
        params = {
            "title": "",
            "return_plot": False,
            "show_plot": True,
            "dir": None,
            "xlabel": None,
            "ylabel": "Frequency",
            **kwargs,
        }
        if params["xlabel"] is None and isinstance(x, pd.Series):
            params["xlabel"] = x.name
        series = cutils.convert_dataframe(x).iloc[:, 0]
        vutils.validate_array_not_contains_nan(
            series, err_msg=cls.errors["array_contains_nans"]
        )
        with sns.axes_style(cls.style), sns.color_palette(cls.palette):
            sns.histplot(series, bins=bins, kde=kde)
            plt.title(params["title"])
            plt.xlabel(params["xlabel"])
            plt.ylabel(params["ylabel"])
            return handle_plot_output_matplotlib(
                plt.gcf(),
                show_plot=params["show_plot"],
                return_plot=params["return_plot"],
                directory=params["dir"],
            )

    @classmethod
    def boxplot(
        cls, x: Sequence[Number], title: str = "", **kwargs
    ) -> Union[None | plt.Figure]:
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
            "return_plot": False,
            "show_plot": True,
            "dir": None,
            "xlabel": None,
            "ylabel": "Quantiles",
            **kwargs,
        }
        series = cutils.convert_dataframe(x).iloc[:, 0]
        vutils.validate_array_not_contains_nan(
            series, err_msg=cls.errors["array_contains_nans"]
        )
        with sns.axes_style(cls.style), sns.color_palette(cls.palette):
            sns.boxplot(series)
            plt.title(title)
            plt.xlabel(params["xlabel"])
            plt.ylabel(params["ylabel"])
            return handle_plot_output_matplotlib(
                plt.gcf(),
                show_plot=params["show_plot"],
                return_plot=params["return_plot"],
                directory=params["dir"],
            )

    @classmethod
    def hexbin(
        cls,
        x: Sequence[Number],
        y: Sequence[Number],
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
            "dir": None,
            "xlabel": None,
            "ylabel": "Quantiles",
            "gridsize": 30,
            **kwargs,
        }
        x_series = cutils.convert_dataframe(x).iloc[:, 0]
        y_series = cutils.convert_dataframe(y).iloc[:, 0]
        vutils.validate_array_not_contains_nan(
            x_series, err_msg=cls.errors["array_contains_nans_f"].format("x")
        )
        vutils.validate_array_not_contains_nan(
            y_series, err_msg=cls.errors["array_contains_nans_f"].format("y")
        )
        if colormap is None:
            colormap = cls.colormap
        with sns.axes_style(cls.style):
            plt.hexbin(x_series, y_series, gridsize=params["gridsize"], cmap=colormap)
            plt.title(params["title"])
            plt.xlabel(params["xlabel"])
            plt.ylabel(params["xlabel"])
            plt.colorbar(label="Counts")
            return handle_plot_output_matplotlib(
                plt.gcf(),
                show_plot=params["show_plot"],
                return_plot=params["return_plot"],
                directory=params["dir"],
            )

    @classmethod
    def scatterplot(
        cls,
        x: Sequence[Number],
        y: Sequence[Number],
        category: Sequence = None,
        **kwargs,
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
            "opacity": 1.0,
            "palette": None,
            "custom_function": None,
            "trend_line": None,
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

        x_series = cutils.convert_dataframe(x).iloc[:, 0]
        y_series = cutils.convert_dataframe(y).iloc[:, 0]
        category_series = (
            cutils.convert_dataframe(category).iloc[:, 0]
            if category is not None
            else None
        )
        # Parameter validation
        vutils.validate_array_not_contains_nan(
            x_series, err_msg=cls.errors["array_contains_nans_f"].format("x")
        )
        vutils.validate_array_not_contains_nan(
            y_series, err_msg=cls.errors["array_contains_nans_f"].format("y")
        )
        vutils.validate_lenghts_match(
            x_series,
            y_series,
            cls.errors["arrays_lens_mismatch_f"].format("x", "y"),
            n_dim=1,
        )
        if params["trend_line"] is not None:
            vutils.validate_string_flag(
                params["trend_line"],
                models,
                cls.errors["usupported_method_f"].format(
                    params["trend_line"], set(models.keys())
                ),
            )

        if params["trend_line"] == "custom" and params["custom_function"] is None:
            raise ValueError("Custom function must be provided when method='custom'")

        if params["palette"] is None:
            params["palette"] = cls.palette[:]

        with sns.axes_style(cls.style):
            if category_series is not None:
                vutils.validate_lenghts_match(
                    x_series,
                    category_series,
                    cls.errors["arrays_lens_mismatch_f"].format(
                        "category", "x' and 'y"
                    ),
                    n_dim=1,
                )
                vutils.validate_array_not_contains_nan(
                    category_series,
                    err_msg=cls.errors["array_contains_nans_f"].format("category"),
                )

                unique_categories = category_series.unique()
                n_unique_categories = category_series.nunique()
                if n_unique_categories > len(params["palette"]):
                    warnings.warn(
                        cls.warns["DataVisualizer"][
                            "categories_exceeds_palette_f"
                        ].format(n_unique_categories, len(params["palette"])),
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
                )
                trend_line_color = params["palette"][1 % len(params["palette"])]
            if params["trend_line"]:
                if params["trend_line"] == "custom":
                    model_x = np.linspace(x_series.min(), x_series.max(), 1000)
                    model_values = models[params["trend_line"]](model_x)
                    plt.plot(model_x, model_values, color=trend_line_color)
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
                        plt.plot(model_x, model_values, color=trend_line_color)
                    except RuntimeError as e:
                        warnings.warn(
                            f"Could not fit '{params['trend_line']}' trend: {e}",
                            UserWarning,
                        )

            if category_series is not None and params["show_legend"]:
                plt.legend(
                    title=params["c_title"], bbox_to_anchor=(1.05, 1), loc="upper left"
                )

            # None < pd.Series.name < "x/y_label" argument priority
            plt.xlabel(params["xlabel"])
            plt.ylabel(params["ylabel"])
            plt.title(params["title"])

            return handle_plot_output_matplotlib(
                plt.gcf(),
                show_plot=params["show_plot"],
                return_plot=params["return_plot"],
                directory=params["dir"],
            )
