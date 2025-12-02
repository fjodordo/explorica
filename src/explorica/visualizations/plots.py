"""
This module contains the DataVisualizer class for visualizing different types of data.
It includes methods for creating various types of plots such as distplots, boxplots, and more.

Modules:
    - DataVisualizer: Class for visualizing data with methods like distplot, heatmap, etc.
"""
from typing import Optional, Sequence, Mapping, Any
import warnings
import logging

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import plotly.express as px
import pandas as pd

from explorica._utils import (convert_dataframe, convert_series, handle_nan,
                              read_config, validate_lengths_match,
                              validate_string_flag)
from ._utils import temp_plot_theme, save_plot, get_empty_plot

logger = logging.getLogger(__name__)

WRN_MSG_EMPTY_DATA = read_config("messages")["warns"]["DataVisualizer"]["empty_data_f"]
ERR_MSG_ARRAYS_LENS_MISMATCH = read_config("messages")["errors"]["arrays_lens_mismatch_f"]
ERR_MSG_MULTIDIMENSIONAL_DATA = read_config("messages")["errors"]["multidimensional_data_f"]
ERR_MSG_UNSUPPORTED_METHOD = read_config("messages")["errors"]["unsupported_method_f"]


def barchart(data: Sequence[float] | Mapping[Any, Sequence[float]],
             category: Sequence[Any] | Mapping[Any, Sequence[Any]],
             ascending: bool = None,
             horizontal: bool = False,
             **kwargs
             ) -> tuple[Figure, Axes]:
    """
    Plots a Bar Chart using categorical and numerical data series.

    This function creates a bar chart to visualize the relationship between 
    categorical labels and numerical values. It supports both vertical and 
    horizontal orientations, automatic sorting, and comprehensive styling 
    options through integration with Seaborn's visualization system.    

    Parameters:
    -----------
    data : Sequence[float] | Mapping[Any, Sequence[float]]
        A sequence containing numerical values (bar heights).
    category : Sequence[Any] | Mapping[Any, Sequence[Any]]
        A sequence containing categorical labels (bar names).
    ascending : bool, optional
        If True or False, sorts the bars by value in ascending or descending order, respectively. 
        If None (default), the original order is preserved.
    horizontal : bool, optional
        If True, plots a horizontal bar chart (barh) instead of a vertical one. 
        Defaults to False.
    opacity : float, default=0.5
        Transparency of the bars (alpha value). Must be between 0 and 1.
    title : str, optional
        The title of the chart. Defaults to an empty string.
    xlabel : str, optional
        The label for the X-axis. Overrides the automatic label.
    ylabel : str, optional
        The label for the Y-axis. Overrides the automatic label.
    figsize : tuple[float, float], optional
        The Matplotlib figure size (width, height) in inches. Defaults to (10, 6).
    palette : str or dict, optional
        The Seaborn/Matplotlib color palette to use for the plot.
    style : str, optional
        The Matplotlib/Seaborn style to apply to the figure
        (e.g., 'whitegrid', 'darkgrid').
    nan_policy : str | Literal['drop', 'raise'], default='drop'
            Policy for handling NaN values in input data:
            - 'raise' : raise ValueError if any NaNs are present in `data`.
            - 'drop'  : drop rows (axis=0) containing NaNs before computation. This
                        does **not** drop entire columns.
    directory : str, optional
        The path to the directory for saving the plot. If None (default), the plot is not saved.
    verbose : bool, optional
        If True, prints messages about the saving process. Defaults to False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib Figure object generated.
    ax : matplotlib.axes.Axes
        The Matplotlib Axes object generated.

    Raises:
    -------
    ValueError
        If the lengths of the 'data' and 'category' input series do not match.
        If the 'data' or 'category' input contains more than one column/dimension.
        If nan_policy='raise' and missing values (NaN/null) are found in the data.
    
    Examples
    --------
    Simple vertical Bar Chart:

    >>> import numpy as np
    >>> values = [25, 40, 15, 60, 35]
    >>> labels = ['Apple', 'Banana', 'Cherry', 'Date', 'Elderberry']
    >>> fig, ax = barchart(values, labels, title="Продажи фруктов")
    >>> # plt.show()

    Horizontal Bar Chart with descending sort:

    >>> values_h = [150, 80, 220]
    >>> labels_h = ['Group A', 'Group B', 'Group C']
    >>> fig, ax = barchart(values_h, labels_h, 
    ...                    horizontal=True, 
    ...                    ascending=False,
    ...                    palette='viridis')
    >>> # plt.show()
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

    series_category = convert_series(category)
    series_value = convert_series(data)

    validate_lengths_match(
        series_category, series_value,
        err_msg=ERR_MSG_ARRAYS_LENS_MISMATCH.format("data", "category"))

    df = pd.DataFrame({"category": series_category,
                       "value": series_value})

    df = handle_nan(df, nan_policy=params["nan_policy"],
                          supported_policy=("drop", "raise"), data_name="data or category")

    if ascending is not None:
        df = df.sort_values(
         "value", ascending=ascending)

    with temp_plot_theme(palette=params["palette"],
                         style=params["style"]):
        if not df.empty:
            fig, ax = plt.subplots(figsize=params["figsize"])
            if horizontal:
                ax.barh(df["category"], df["value"], alpha=params["opacity"])
            else:
                ax.bar(df["category"], df["value"], alpha=params["opacity"])
        else:
            fig, ax = get_empty_plot(figsize=params["figsize"])
            warnings.warn(WRN_MSG_EMPTY_DATA.format("barchart"))
        ax.set_title(params["title"])
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        if params["directory"] is not None:
            save_plot(fig, directory=params["directory"],
                      plot_name="barchart", verbose=params["verbose"])
    return fig, ax


def piechart(data: Sequence[float],
             category: Sequence[Any],
             autopct_method: str = "value",
             **kwargs
             ) -> tuple[Figure, Axes]:
    """
    Draws a pie chart based on categorical and corresponding numerical data.

    This function generates a pie chart where each segment represents a category
    from the input data. The size of each segment is proportional to the corresponding
    numerical value in `data`. The chart supports automatic display of percentages,
    raw values, or both on each segment.

    Parameters:
    -----------
    data : Sequence[float]
        A numerical sequence representing the sizes of the segments.
    category : Sequence[Any]
        A categorical sequence representing the pie chart segments.
    autopct_method : str, default="value"
        Determines how the values are displayed on the pie chart.
        Supported options: "percent", "value", "both".
    title : str, optional
        Title of the pie chart.
    xlabel : str, optional
        The label for the X-axis. Overrides the automatic label.
    ylabel : str, optional
        The label for the Y-axis. Overrides the automatic label.
    show_legend : bool, default=True
        Whether to display a legend.
    show_labels : bool, default=True
        Whether to display category labels directly on the chart.
    palette : str or list, optional
        Color palette to use for the plot.
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    directory : str, optional
        If provided, the plot will be saved to this directory.
    nan_policy : {"drop", "raise"}, default="drop"
        How to handle NaN values.
    verbose : bool, default=False
        If True, print additional information.

    Returns:
    --------
    tuple[Figure, Axes]
        matplotlib figure and axes objects

    Raises:
    -------
    ValueError
        If input sizes mismatch.
        If invalid autopct method is provided.
    """
    params = {
        "show_legend": True,
        "show_labels": True,
        "palette": None,

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

    # Parameter validation
    supported_autopct = {"percent", "value", "both"}
    validate_string_flag(autopct_method, supported_autopct,
                         err_msg=ERR_MSG_UNSUPPORTED_METHOD.format(
                             autopct_method, supported_autopct))

    series_category = convert_series(category)
    series_value = convert_series(data)

    validate_lengths_match(
        series_category, series_value,
        err_msg=ERR_MSG_ARRAYS_LENS_MISMATCH.format("data", "category"))

    df = pd.DataFrame({"category": series_category,
                       "value": series_value})

    df = handle_nan(df, nan_policy=params["nan_policy"],
                          supported_policy=("drop", "raise"), data_name="data or category")

    labels = df["category"] if params["show_labels"] else None
    with temp_plot_theme(palette=params["palette"], style=params["style"]):
        if not df.empty:
            fig, ax = plt.subplots(figsize = params["figsize"])
            wedges, _, _ = ax.pie(
            df["value"],
            labels=labels,
            autopct=_make_autopct(df["value"], autopct_method),
            startangle=90,)
            if params["show_legend"]:
                ax.legend(wedges, df["category"], loc="center left", bbox_to_anchor=(1, 0.5))
        else:
            fig, ax = get_empty_plot(figsize=params["figsize"])
            warnings.warn(WRN_MSG_EMPTY_DATA.format("piechart"))
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        ax.set_title(params["title"])
        if params["directory"] is not None:
            save_plot(fig, directory=params["directory"],
                      plot_name="piechart", verbose=params["verbose"])
    return fig, ax


def _make_autopct(values: pd.Series, method: str):
    """
    Internal helper to format piechart percentage labels.
    """
    def formatter(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        if method == "percent":
            return f"{pct:.1f}%"
        if method == "value":
            return f"{val}"
        return f"{pct:.1f}%\n({val})"
    return formatter


def mapbox(lat: Sequence[float],
           lon: Sequence[float],
           category: Optional[Sequence] = None,
           dot_names: Optional[Sequence] = None,
           size: Optional[Sequence[float]] = None,
           title: Optional[str] = None,
           show_legend: Optional[bool] = True,
           palette: Optional[Sequence[str]] = None,
           map_style: Optional[str] = "open-street-map"
           ) -> None:
    """
    Displays a geographic scatter plot (Mapbox) with optional
    category-based coloring, size scaling,
    and hover labels.

        This method is intended for quick visualization of spatial data using latitude and longitude
        coordinates. It supports additional inputs like category, size, and dot names to enhance the
        plot’s readability without requiring complex setup.

        Parameters
        ----------
        lat : Iterable
            Latitude values for each point. Cannot contain null values.
        lon : Iterable
            Longitude values for each point. Must match length of `lat` and cannot contain nulls.
        category : Iterable, optional
            Categorical labels used to color the points. 
            If provided, must be the same lengths as `lat` and `lon`
            and contain no null values.
        dot_names : Iterable, optional
            Names to be displayed on hover. Should match the lengths of `lat` and `lon` if provided.
        size : Iterable, optional
            Numerical values used to scale the size of the points. 
            Must match the lengths of `lat` and `lon`
            and contain no nulls.
        title : str, optional
            Plot title to be displayed at the top of the map.
        show_legend : bool, optional, default=True
            Whether to display the legend (relevant only if `category` is provided).
        palette : Iterable, optional
            List of color values (hex or named) to use for coloring categories. If not provided,
            the default class palette is used. If the number of unique categories exceeds the number
            of colors, a warning is raised and colors may repeat.
        map_style : str, optional
            Mapbox style to be used for rendering the map. Default is "open-street-map".
            Other options include "carto-positron", "carto-darkmatter", etc.

        Raises
        ------
        ValueError
            If lat/lon are missing, contain nulls, or their lengths don't match.
            If any of the optional inputs (`category`, `dot_names`, `size`) are provided but
            contain nulls or do not match the lengths of `lat` and `lon`.

        Warnings
        --------
        UserWarning
            If the number of unique categories exceeds the number of colors in the palette.

        Examples
        --------
        >>> DataVisualizer.mapbox(
        ...     lat=df["latitude"],
        ...     lon=df["longitude"],
        ...     category=df["region"],
        ...     dot_names=df["city"],
        ...     title="Distribution of Cities",
        ...     palette=sns.color_palette("tab10").as_hex()
        ... )
    """

    if lat is None or lon is None:
        raise ValueError("latitude and longitude must be provided")
    if pd.isna(lat).any() or pd.isna(lon).any():
        raise ValueError("The input 'lat' or 'lon' contains NaN values."
                         "Please clean or impute missing data before visualization.")
    if len(lat) != len(lon):
        raise ValueError("The length of latitude must match the length of longitude.")
    names_optional_iterables = ("category", "dot_names", "size")
    for i, array in enumerate((category, dot_names, size)):
        if array is None:
            continue
        if pd.isna(array).any():
            raise ValueError(f"The input '{names_optional_iterables[i]}' contains NaN values."
                             f"Please clean or impute missing data before visualization.")
        if len(array) != len(lat):
            raise ValueError(f"The length of '{names_optional_iterables[i]}' "
                             f"must match the length of latitude and longitude.")
    if palette is not None:
        colors = palette
    else:
        colors = palette.as_hex()
    if category is not None:
        n_unique_categories = len(set(category))
        if n_unique_categories > len(colors):
            warnings.warn(f"Number of categories ({n_unique_categories}) "
                          f"exceeds the palette size ({len(colors)}). Colors may repeat.",
                        UserWarning)
    fig = px.scatter_map(
        lat=lat,
        lon=lon,
        color=category,
        hover_name=dot_names,
        size=size,
        zoom=1,
        height=600,
        color_discrete_sequence=colors)
    fig.update_layout(mapbox_style=map_style, showlegend=show_legend)
    if title is not None:
        fig.update_layout(title=title, title_x=0.5, title_y=0.98,
                          title_font_size=22, margin={"r":0,"t":40,"l":0,"b":0})
    else:
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

class DataVisualizer:
    """
    A utility class for quick and consistent data visualizations 
    using seaborn, matplotlib, and Plotly.

    This class simplifies the creation of common plots with pre-configured styles and layout 
    options, helping to keep Jupyter notebooks clean, readable, and standardized.

    Class Attributes
    ----------------
    style : str
        Default seaborn style to apply to matplotlib-based plots 
        (e.g., 'whitegrid', 'dark', 'ticks').

    palette : seaborn color palette
        Default color palette used for plots. This must be a palette object returned by 
        `sns.color_palette(...)`, supporting `.as_hex()` and `.as_rgb()` methods.

    colormap : str
        Default seaborn colormap to apply to atplotlib-based plots 
        (e.g., 'coolwarm', 'viridis', 'plasma')

    Methods
    -------
    set_theme(palette=None, style=None, colormap=None)
        Updates class-wide style settings, including seaborn palette, matplotlib style, 
        and colormap for use in visualizations.

    distplot(series, kde=True)
        Plots a distribution histogram for a numeric variable with optional KDE.

    boxplot(series)
        Draws a boxplot for a numeric variable.

    hexbin(x, y)
        Plots a hexbin plot for two numeric variables using the default colormap.

    heatmap(dataframe)
        Plots a heatmap of correlations or numerical matrix using the default colormap.

    piechart(categories, values)
        Draws a pie chart with percentage annotations. Supports custom or auto-generated labels.

    barchart(categories, values)
        Plots both a vertical and horizontal bar chart from a categorical series.

    scatterplot(x, y, category=None, trend_line=None)
        Draws a scatterplot with optional trendline (user-defined or fitted by type: 'linear', 
        'exp', etc.).

    mapbox(lat, lon, category=None, dot_names=None)
        Plots geospatial points on an interactive Plotly mapbox. Supports optional point 
        categories and sizes.

    _make_autopct(values, method)
        Internal static helper to format piechart percentage labels.
        
    Notes
    -----
    This class is intended for internal exploratory data analysis (EDA), dashboard prototyping, or 
    report visualizations where quick, clean outputs are more important than deep customization.
    """
