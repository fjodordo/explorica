r"""
High-level plotting utilities for Explorica visualizations.

This module provides a set of functions to generate common plots using Matplotlib,
Seaborn, and Plotly. It standardizes plot outputs through the `VisualizationResult`
dataclass and supports flexible styling, color palettes, and interactivity.

Methods
-------
barchart(data, category, ascending=None, horizontal=False, \**kwargs)
    Plots a bar chart from categorical and numerical data. Supports vertical
    or horizontal orientation, automatic sorting, and styling through Seaborn.
piechart(data, category, autopct_method='value', \**kwargs)
    Draws a pie chart based on categorical and numerical data. Supports
    value, percent, or combined display on each segment.
mapbox(lat, lon, category=None, \**kwargs)
    Generates an interactive geographic scatter plot using Plotly Mapbox.
    Supports categorical coloring, point scaling, hover labels, and Mapbox
    styling.

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
>>> # Basic vertical bar chart (Matplotlib)
>>> data = [3, 7, 5]
>>> categories = ['A', 'B', 'C']
>>> result = vis.barchart(data, categories,
...                       plot_kws={'color':'skyblue', 'edgecolor':'black'})
>>> result.figure.show()  # doctest: +SKIP

>>> # Pie chart with percentages displayed
>>> result = vis.piechart(data, categories, autopct_method='percent')
>>> result.figure.show() # doctest: +SKIP
>>> result.extra_info
{'autopct_method': 'percent'}

>>> # Mapbox scatter plot with categorical coloring
>>> lat = [34.05, 40.71, 37.77]
>>> lon = [-118.24, -74.00, -122.42]
>>> categories = ['City1', 'City2', 'City3']
>>> result = vis.mapbox(lat, lon, category=categories)
>>> # Show interactive map with hover labels
>>> result.figure.show() # doctest: +SKIP
"""

from typing import Optional, Sequence, Mapping, Any
import warnings
import logging

import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

from explorica.types import VisualizationResult
from explorica._utils import (
    convert_series,
    handle_nan,
    validate_lengths_match,
    validate_string_flag,
)
from ._utils import (
    temp_plot_theme,
    save_plot,
    get_empty_plot,
    resolve_plotly_palette,
    DEFAULT_MPL_PLOT_PARAMS,
    WRN_MSG_EMPTY_DATA,
    ERR_MSG_ARRAYS_LENS_MISMATCH,
    ERR_MSG_UNSUPPORTED_METHOD,
    WRN_MSG_CATEGORIES_EXCEEDS_PALETTE_F,
)

__all__ = ["barchart", "piechart", "mapbox"]

logger = logging.getLogger(__name__)


def barchart(
    data: Sequence[float] | Mapping[Any, Sequence[float]],
    category: Sequence[Any] | Mapping[Any, Sequence[Any]],
    ascending: bool = None,
    horizontal: bool = False,
    **kwargs,
) -> VisualizationResult:
    """
    Plot a Bar Chart using categorical and numerical data series.

    This function creates a bar chart to visualize the relationship between
    categorical labels and numerical values. It supports both vertical and
    horizontal orientations, automatic sorting, and comprehensive styling
    options through integration with Seaborn's visualization system.

    Under the hood, the function uses Matplotlib's `matplotlib.axes.Axes.bar` and
    `matplotlib.axes.Axes.barh` functions and applies Seaborn styles for aesthetic
    defaults. This allows passing additional kwargs directly to the underlying
    Matplotlib calls via `plot_kws`. For complete parameter documentation and
    advanced customization options, see urls below

    Parameters
    ----------
    data : Sequence[float] | Mapping[Any, Sequence[float]]
        A sequence containing numerical values (bar heights).
    category : Sequence[Any] | Mapping[Any, Sequence[Any]]
        A sequence containing categorical labels (bar names).
    ascending : bool, optional
        If True or False, sorts the bars by value
        in ascending or descending order, respectively.
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
    plot_kws : dict, optional
        Dictionary of keyword arguments passed directly to the underlying matplotlib
        functions (`matplotlib.axes.Axes.bar` & `matplotlib.axes.Axes.barh`). This
        allows overriding any default plotting behavior. If not provided, the function
        internally constructs a dictionary from its own relevant parameters. Keys
        provided in `plot_kws` take precedence over internally generated defaults.
        For complete parameter documentation and advanced customization options, see
        urls below.
    nan_policy : {'drop', 'raise'}, default='drop'
            Policy for handling NaN values in input data:

            - 'raise' : raise ValueError if any NaNs are present in `data`.
            - 'drop'  : drop rows (axis=0) containing NaNs before computation. This
              does **not** drop entire columns.
    directory : str, optional
        The path to the directory for saving the plot. If None (default),
        the plot is not saved.
    verbose : bool, optional
        If True, prints messages about the saving process. Defaults to False.

    Returns
    -------
    VisualizationResult
        A dataclass encapsulating the result of a visualization.
        See also :class:`explorica.types.VisualizationResult` for full attribute
        details.

    Raises
    ------
    ValueError
        If the lengths of the 'data' and 'category' input series do not match.
        If the 'data' or 'category' input contains more than one column/dimension.
        If nan_policy='raise' and missing values (NaN/null) are found in the data.

    Warns
    -----
    UserWarning
        Raised if the input data is empty. An empty plot with a warning message
        will be returned in this case.

    Notes
    -----
    This function uses matplotlib.axes.Axes.bar and matplotlib.axes.Axes.barh under the
    hood. For complete parameter documentation and advanced customization options, see:
    `matplotlib bar`_.

    .. _matplotlib bar: https://matplotlib.org/stable/api/
       _as_gen/matplotlib.axes.Axes.html>`_

    `matplotlib barh`_.

    .. _matplotlib barh: https://matplotlib.org/stable/api/
       _as_gen/matplotlib.axes.Axes.barh.html>`_

    Examples
    --------
    >>> import explorica.visualizations as vis
    >>> # Simple vertical Bar Chart
    >>> values = [25, 40, 15, 60, 35]
    >>> labels = ['Apple', 'Banana', 'Cherry', 'Date', 'Elderberry']
    >>> plot = vis.barchart(values, labels, title="Fruit sales")
    >>> plot.figure.show() # doctest: +SKIP

    >>> # Horizontal Bar Chart with descending sort:
    >>> values_h = [150, 80, 220]
    >>> labels_h = ['Group A', 'Group B', 'Group C']
    >>> plot = barchart(values_h, labels_h,
    ...                 horizontal=True,
    ...                 ascending=False,
    ...                 palette='viridis')
    >>> plot.figure.show() # doctest: +SKIP
    """
    params = {**DEFAULT_MPL_PLOT_PARAMS, "palette": None, "opacity": 0.5, **kwargs}
    plot_kws_merged = {"alpha": params["opacity"], **params.get("plot_kws", {})}

    series_category = convert_series(category)
    series_value = convert_series(data)

    validate_lengths_match(
        series_category,
        series_value,
        err_msg=ERR_MSG_ARRAYS_LENS_MISMATCH.format("data", "category"),
    )

    df = pd.DataFrame({"category": series_category, "value": series_value})

    df = handle_nan(
        df,
        nan_policy=params["nan_policy"],
        supported_policy=("drop", "raise"),
        data_name="data or category",
    )

    if ascending is not None:
        df = df.sort_values("value", ascending=ascending)

    with temp_plot_theme(palette=params["palette"], style=params["style"]):
        if not df.empty:
            fig, ax = plt.subplots(figsize=params["figsize"])
            if horizontal:
                ax.barh(df["category"], df["value"], **plot_kws_merged)
            else:
                ax.bar(df["category"], df["value"], **plot_kws_merged)
        else:
            fig, ax = get_empty_plot(figsize=params["figsize"])
            warnings.warn(WRN_MSG_EMPTY_DATA.format("barchart"))
        ax.set_ylabel(params["ylabel"])
        ax.set_title(params["title"])
        ax.set_xlabel(params["xlabel"])
        if params["directory"] is not None:
            save_plot(
                fig,
                directory=params["directory"],
                plot_name="barchart",
                verbose=params["verbose"],
            )
    return VisualizationResult(
        figure=fig,
        axes=ax,
        title=params["title"],
        engine="matplotlib",
        width=params["figsize"][0],
        height=params["figsize"][1],
        extra_info={
            "horizontal": horizontal,
        },
    )


def piechart(
    data: Sequence[float],
    category: Sequence[Any],
    autopct_method: str = "value",
    **kwargs,
) -> VisualizationResult:
    """
    Draw a pie chart based on categorical and corresponding numerical data.

    This function generates a pie chart where each segment represents a category
    from the input data. The size of each segment is proportional to the corresponding
    numerical value in `data`. The chart supports automatic display of percentages,
    raw values, or both on each segment.

    Under the hood, the function uses Matplotlib's `matplotlib.axes.Axes.pie` function
    and applies Seaborn styles for aesthetic defaults. This allows passing
    additional kwargs directly to the underlying Matplotlib calls via `plot_kws`.
    For complete parameter documentation and
    advanced customization options, see urls below

    Parameters
    ----------
    data : Sequence[float]
        A numerical sequence representing the sizes of the segments.
    category : Sequence[Any]
        A categorical sequence representing the pie chart segments.
    autopct_method : str, default="value"
        Determines how the values are displayed on the pie chart.
        Supported options: "percent", "value", "both".

    Returns
    -------
    VisualizationResult
        A dataclass encapsulating the result of a visualization.
        See also :class:`explorica.types.VisualizationResult` for full attribute
        details.

    Other Parameters
    ----------------
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
    plot_kws : dict, optional
        Dictionary of keyword arguments passed directly to the underlying matplotlib
        function (`matplotlib.Axes.ax.pie`). This allows overriding any default plotting
        behavior. If not provided, the function internally constructs a dictionary
        from its own relevant parameters. Keys provided in `plot_kws` take precedence
        over internally generated defaults. For complete parameter documentation and
        advanced customization options, see urls below.
    directory : str, optional
        If provided, the plot will be saved to this directory.
    nan_policy : {"drop", "raise"}, default="drop"
        How to handle NaN values.
    verbose : bool, default=False
        If True, print additional information.

    Raises
    ------
    ValueError
        If input sizes mismatch.
        If invalid autopct method is provided.

    Warns
    -----
    UserWarning
        Raised if the input data is empty. An empty plot with a warning message
        will be returned in this case.

    Notes
    -----
    This function uses matplotlib.axes.Axes.pie under the hood. For complete parameter
    documentation and advanced customization options, see:
    `matplotlib pie`_.

    .. _matplotlib pie: https://matplotlib.org/stable/api/
       _as_gen/matplotlib.pyplot.pie.html>`_

    Examples
    --------
    >>> import explorica.visualizations as vis
    >>> # Simple pie chart displaying raw values
    >>> data = [15, 30, 45, 10]
    >>> categories = ["A", "B", "C", "D"]
    >>> result = vis.piechart(data, categories, autopct_method="value",
    ...                       title="Simple Pie")
    >>>  # Display the chart
    >>> result.figure.show() # doctest: +SKIP
    >>> result.title
    'Simple Pie'

    >>> # Pie chart showing percentages on each segment
    >>> data = [50, 25, 25]
    >>> categories = ["Apples", "Bananas", "Cherries"]
    >>> result = vis.piechart(data, categories,
    ...     autopct_method="percent", show_legend=True)
    >>> result.figure.show() # doctest: +SKIP
    >>> result.extra_info["autopct_method"]
    'percent'
    >>> plt.close(result.figure)
    """
    params = {
        **DEFAULT_MPL_PLOT_PARAMS,
        "show_legend": True,
        "show_labels": True,
        "palette": None,
        **kwargs,
    }
    plot_kws_merged = {
        "labels": None,
        "startangle": 90,
        "autopct": None,
        **params.get("plot_kws", {}),
    }
    # Parameter validation
    supported_autopct = {"percent", "value", "both"}
    validate_string_flag(
        autopct_method,
        supported_autopct,
        err_msg=ERR_MSG_UNSUPPORTED_METHOD.format(autopct_method, supported_autopct),
    )

    series_category = convert_series(category)
    series_value = convert_series(data)

    validate_lengths_match(
        series_category,
        series_value,
        err_msg=ERR_MSG_ARRAYS_LENS_MISMATCH.format("data", "category"),
    )

    df = pd.DataFrame({"category": series_category, "value": series_value})

    df = handle_nan(
        df,
        nan_policy=params["nan_policy"],
        supported_policy=("drop", "raise"),
        data_name="data or category",
    )

    labels = df["category"] if params["show_labels"] else None
    with temp_plot_theme(palette=params["palette"], style=params["style"]):
        if not df.empty:
            # construct labels and autopct if not provided
            plot_kws_merged["labels"] = (
                labels
                if plot_kws_merged["labels"] is None
                else plot_kws_merged["labels"]
            )
            plot_kws_merged["autopct"] = (
                _make_autopct(df["value"], autopct_method)
                if plot_kws_merged["autopct"] is None
                else plot_kws_merged["autopct"]
            )
            fig, ax = plt.subplots(figsize=params["figsize"])
            wedges, _, _ = ax.pie(df["value"], **plot_kws_merged)
            if params["show_legend"]:
                ax.legend(
                    wedges, df["category"], loc="center left", bbox_to_anchor=(1, 0.5)
                )
        else:
            fig, ax = get_empty_plot(figsize=params["figsize"])
            warnings.warn(WRN_MSG_EMPTY_DATA.format("piechart"))
        ax.set_xlabel(params["xlabel"])
        ax.set_title(params["title"])
        ax.set_ylabel(params["ylabel"])
        if params["directory"] is not None:
            save_plot(
                fig,
                directory=params["directory"],
                plot_name="piechart",
                verbose=params["verbose"],
            )
    return VisualizationResult(
        figure=fig,
        axes=ax,
        engine="matplotlib",
        height=params["figsize"][1],
        title=params["title"],
        extra_info={"autopct_method": autopct_method},
        width=params["figsize"][0],
    )


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


def mapbox(
    lat: Sequence[float],
    lon: Sequence[float],
    category: Optional[Sequence] = None,
    **kwargs,
) -> VisualizationResult:
    """
    Display an interactive geographic scatter plot (Mapbox).

    This method provides a high-level interface for visualizing spatial data
    using latitude and longitude coordinates. It supports categorical coloring,
    dynamic point sizing, custom hover labels, and Plotly Mapbox styling.

    Under the hood, the function uses Plotly's `plotly.express.scatter_map` function
    and applies plotly styles for aesthetic defaults. This allows passing
    additional kwargs directly to the underlying Plotly calls via `plot_kws`.
    For complete parameter documentation and
    advanced customization options, see urls below

    Parameters
    ----------
    lat : Sequence[float]
        Latitude values for each point. Cannot contain null values.
    lon : Sequence[float]
        Longitude values for each point. Must match the length of `lat` and
        cannot contain null values.
    category : Sequence[Any], optional
        Categorical labels used to color the points. Must match the length of
        `lat` and `lon`, cannot contain nulls, and determines the number of
        discrete colors in the plot.

    Returns
    -------
    VisualizationResult
        A dataclass encapsulating the result of a visualization.
        See also :class:`explorica.types.VisualizationResult` for full attribute
        details.

    Other Parameters
    ----------------
    hover_name : Sequence[Any], optional
        Labels to show on hover. Must match the length of `lat` and `lon`.
    size : Sequence[float], optional
        Numerical values used to scale point sizes. Must match the length
        of `lat` and `lon`.
    title : str, optional
        Plot title.
    show_legend : bool, default=True
        Whether to display the legend. Relevant only if `category` is provided.
    palette : Sequence[str], optional
        List of colors (hex or named) for categories. If not provided, the default
        Plotly color sequence (px.colors.qualitative.Plotly) is used.
    opacity : float, default=0.7
        Marker opacity.
    height : int, default=600
        Figure height in pixels.
    width : int, default=800
        Figure width in pixels.
    template : str, default="plotly_white"
        Plotly template used for styling. E.g., "plotly_dark", "ggplot2", "seaborn".
    map_style : str, default="open-street-map"
        Mapbox style used for rendering the map. E.g., "carto-positron",
        "carto-darkmatter", "stamen-terrain", "open-street-map".
    plot_kws : dict, optional
        Dictionary of keyword arguments passed directly to the underlying Plotly
        function (`px.scatter_mapbox`). This allows overriding any default plotting
        behavior. If not provided, the function internally constructs a dictionary
        from its own relevant parameters. Keys provided in `plot_kws` take precedence
        over internally generated defaults. For complete parameter documentation and
        advanced customization options, see urls below.
    nan_policy : str, default="drop"
        Policy for handling NaN values in input data. Supports 'drop'
        (removes rows with NaNs) or 'raise' (raises an error).
    directory : str or Path, optional
        Path to save the figure as HTML.
    verbose : bool, default=False
        Enable logging.

    Raises
    ------
    ValueError
        If `lat`, `lon`, or any optional input contain nulls or mismatched
        lengths.

    Warns
    -----
    UserWarning
        Raised if the input data is empty. An empty plot with a warning message
        will be returned in this case.

    Notes
    -----
    - This function uses plotly.express.scatter_map under the hood. For complete
      parameter documentation and advanced customization options, see:
      `plotly scatter map`_.

      .. _plotly scatter map: https://plotly.com/python-api-reference/
         generated/plotly.express.scatter_map.html

    - The plot is saved as an interactive HTML file when `directory` is set.
    - Color resolution is handled internally using `resolve_plotly_palette`.
    - This function is intended for rapid map-based EDA rather than full
      cartographic customization.
    - `Supported Plotly templates <https://plotly.com/python/templates/>`_.
    - `Supported Mapbox styles <https://plotly.com/python/mapbox-layers/>`_.

    Examples
    --------
    >>> from pathlib import Path
    >>> import explorica.visualizations as vis
    >>> # Basic Mapbox scatter plot usage
    >>> lat = [34.05, 40.71, 37.77]
    >>> lon = [-118.24, -74.00, -122.42]
    >>> result = vis.mapbox(lat, lon, title="Major US Cities")
    >>> result.figure.show() # doctest: +SKIP

    >>> # Mapbox scatter plot with categorical coloring usage
    >>> lat = [34.05, 40.71, 37.77, 51.50]
    >>> lon = [-118.24, -74.00, -122.42, -0.12]
    >>> category = ["US", "US", "US", "UK"]
    >>> result = vis.mapbox(lat, lon, category=category, title="USA vs UK Cities")
    >>> result.figure.show() # doctest: +SKIP

    >>> # HTML saving example
    >>> lat = [34.05, 40.71]
    >>> lon = [-118.24, -74.00]
    >>> result = vis.mapbox( # doctest: +SKIP
    ...     lat, lon,
    ...     plot_kws={"zoom": 4},
    ...     directory="./plots",
    ...     title="Saved Map"
    ... )
    >>> result.figure.show() # doctest: +SKIP
    >>> # Check that the file actually created
    >>> Path("./plots/Saved Map.html").exists() # doctest: +SKIP
    True
    """
    params = {
        "hover_name": None,
        "size": None,
        "title": None,
        "show_legend": True,
        "palette": None,
        "opacity": 0.7,
        "zoom": 1,
        "height": 600,
        "width": 800,
        "template": "plotly_white",
        "map_style": "open-street-map",
        "plot_kws": {},
        "nan_policy": "drop",
        "directory": None,
        "verbose": False,
        **kwargs,
    }

    plot_kws_merged = {
        "opacity": params["opacity"],
        "height": params["height"],
        "width": params["width"],
        "template": params["template"],
        "color_discrete_sequence": params["palette"],
        "zoom": params["zoom"],
        **params.get("plot_kws", {}),
    }

    lat_series, lon_series, category_series = (
        convert_series(lat),
        convert_series(lon),
        convert_series(category),
    )
    params["hover_name"] = convert_series(params["hover_name"])
    params["size"] = convert_series(params["size"])

    validate_lengths_match(
        lat_series,
        lon_series,
        err_msg=ERR_MSG_ARRAYS_LENS_MISMATCH.format("lat", "lon"),
    )
    df = pd.DataFrame({"lat": convert_series(lat), "lon": convert_series(lon)})

    for key, value in {
        "category": category_series,
        "hover_name": params["hover_name"],
        "size": params["size"],
    }.items():
        if value.empty:
            continue
        validate_lengths_match(
            lat_series,
            value,
            err_msg=ERR_MSG_ARRAYS_LENS_MISMATCH.format("lat and lon", key),
        )
        df[key] = value
    df = handle_nan(
        df,
        nan_policy=params["nan_policy"],
        supported_policy=("drop", "raise"),
        data_name="input sequences",
    )
    unique_categories = df["category"].nunique() if "category" in df else 1
    if "category" in df:
        plot_kws_merged["color_discrete_sequence"] = resolve_plotly_palette(
            params["palette"]
        )
        if len(plot_kws_merged["color_discrete_sequence"]) < unique_categories:
            warnings.warn(
                WRN_MSG_CATEGORIES_EXCEEDS_PALETTE_F.format(
                    unique_categories, len(plot_kws_merged["color_discrete_sequence"])
                ),
                UserWarning,
            )
    if not df.empty:
        fig = px.scatter_map(
            lat=df["lat"],
            lon=df["lon"],
            color=df["category"] if "category" in df else None,
            hover_name=df["hover_name"] if "hover_name" in df else None,
            size=df["size"] if "size" in df else None,
            **plot_kws_merged,
        )
        fig.update_layout(
            mapbox_style=params["map_style"], showlegend=params["show_legend"]
        )
    else:
        fig = get_empty_plot(
            figsize=(params["width"], params["height"]), engine="plotly"
        )
        warnings.warn(WRN_MSG_EMPTY_DATA.format("mapbox"))
    if params["title"] is not None:
        fig.update_layout(
            title=params["title"],
            title_x=0.5,
            title_y=0.98,
            title_font_size=22,
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
        )
    else:
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    if params["directory"] is not None:
        save_plot(
            fig,
            directory=params["directory"],
            plot_name="mapbox",
            verbose=params["verbose"],
            engine="plotly",
        )
    return VisualizationResult(
        figure=fig,
        engine="plotly",
        width=params["width"],
        height=params["height"],
        title=params["title"],
    )
