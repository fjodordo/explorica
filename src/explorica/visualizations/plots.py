"""
This module contains the DataVisualizer class for visualizing different types of data.
It includes methods for creating various types of plots such as distplots, boxplots,
and more.

Modules:
    - DataVisualizer: Class for visualizing data with methods like distplot,
      heatmap, etc.
"""
from typing import Optional, Sequence, Mapping, Any
import warnings
import logging

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import plotly.express as px
import pandas as pd

from explorica._utils import (convert_series, handle_nan,
                              read_config, validate_lengths_match,
                              validate_string_flag)
from ._utils import (temp_plot_theme, save_plot, get_empty_plot,
                     resolve_plotly_palette, DEFAULT_MPL_PLOT_PARAMS,
                     VisualizationResult)

logger = logging.getLogger(__name__)

WRN_MSG_EMPTY_DATA = read_config("messages")["warns"]["DataVisualizer"]["empty_data_f"]
ERR_MSG_ARRAYS_LENS_MISMATCH = read_config(
    "messages")["errors"]["arrays_lens_mismatch_f"]
ERR_MSG_MULTIDIMENSIONAL_DATA = read_config(
    "messages")["errors"]["multidimensional_data_f"]
ERR_MSG_UNSUPPORTED_METHOD = read_config("messages")["errors"]["unsupported_method_f"]
WRN_MSG_CATEGORIES_EXCEEDS_PALETTE_F = read_config("messages")[
    "warns"]["DataVisualizer"]["categories_exceeds_palette_f"]


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
        functions (`plt.bar` & `plt.barh`). This allows overriding any default plotting
        behavior. If not provided, the function internally constructs a dictionary
        from its own relevant parameters. Keys provided in `plot_kws` take precedence
        over internally generated defaults.
    nan_policy : str | Literal['drop', 'raise'], default='drop'
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
    >>> plot = barchart(values, labels, title="Продажи фруктов")
    >>> # plot.figure.show()

    Horizontal Bar Chart with descending sort:

    >>> values_h = [150, 80, 220]
    >>> labels_h = ['Group A', 'Group B', 'Group C']
    >>> plot = barchart(values_h, labels_h, 
    ...                 horizontal=True, 
    ...                 ascending=False,
    ...                 palette='viridis')
    >>> # plot.figure.show()
    """
    params = {
        **DEFAULT_MPL_PLOT_PARAMS,
        "palette": None,
        "opacity": 0.5,
        **kwargs
    }
    plot_kws_merged = {
        "alpha": params["opacity"],
        **params.get("plot_kws", {})}

    series_category = convert_series(category)
    series_value = convert_series(data)

    validate_lengths_match(
        series_category, series_value,
        err_msg=ERR_MSG_ARRAYS_LENS_MISMATCH.format("data", "category"))

    df = pd.DataFrame({"category": series_category,
                       "value": series_value})

    df = handle_nan(
        df, nan_policy=params["nan_policy"],
        supported_policy=("drop", "raise"), data_name="data or category")

    if ascending is not None:
        df = df.sort_values(
         "value", ascending=ascending)

    with temp_plot_theme(palette=params["palette"],
                         style=params["style"]):
        if not df.empty:
            fig, ax = plt.subplots(figsize=params["figsize"])
            if horizontal:
                ax.barh(df["category"], df["value"], **plot_kws_merged)
            else:
                ax.bar(df["category"], df["value"], **plot_kws_merged)
        else:
            fig, ax = get_empty_plot(figsize=params["figsize"])
            warnings.warn(WRN_MSG_EMPTY_DATA.format("barchart"))
        ax.set_title(params["title"])
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        if params["directory"] is not None:
            save_plot(fig, directory=params["directory"],
                      plot_name="barchart", verbose=params["verbose"])
    return VisualizationResult(figure=fig, axes=ax, engine="matplotlib",
                               title=params["title"],
                               width=params["figsize"][0],
                               height=params["figsize"][1],
                               )


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
    plot_kws : dict, optional
        Dictionary of keyword arguments passed directly to the underlying matplotlib
        functions (`plt.pie`). This allows overriding any default plotting
        behavior. If not provided, the function internally constructs a dictionary
        from its own relevant parameters. Keys provided in `plot_kws` take precedence
        over internally generated defaults.
    directory : str, optional
        If provided, the plot will be saved to this directory.
    nan_policy : {"drop", "raise"}, default="drop"
        How to handle NaN values.
    verbose : bool, default=False
        If True, print additional information.

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

    Raises:
    -------
    ValueError
        If input sizes mismatch.
        If invalid autopct method is provided.
    
    Examples
    --------
    >>> import explorica.visualizations as visualizations
    >>> data = [10, 20, 30]
    >>> categories = ["A", "B", "C"]
    >>> result = visualizations.piechart(
    ...     data,
    ...     categories,
    ...     autopct_method="both",
    ...     title="Sample Pie Chart",
    ...     show_legend=True
    ... )
    >>> result.figure.show()  # Show the pie chart
    >>> result.axes  # <matplotlib.axes._subplots.AxesSubplot object at 0x...>
    >>> result.title
    'Sample Pie Chart'
    >>> result.engine
    'matplotlib'
    """
    params = {
        **DEFAULT_MPL_PLOT_PARAMS,
        "show_legend": True,
        "show_labels": True,
        "palette": None,
        **kwargs
    }
    plot_kws_merged = {
        "labels": None,
        "startangle": 90,
        "autopct": None,

        **params.get("plot_kws", {})}
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

    df = handle_nan(
        df, nan_policy=params["nan_policy"],
        supported_policy=("drop", "raise"), data_name="data or category")

    labels = df["category"] if params["show_labels"] else None
    with temp_plot_theme(palette=params["palette"], style=params["style"]):
        if not df.empty:
            # construct labels and autopct if not provided
            plot_kws_merged["labels"] = (labels if plot_kws_merged["labels"] is None
                                         else plot_kws_merged["labels"])
            plot_kws_merged["autopct"] = (_make_autopct(
                df["value"], autopct_method)
                if plot_kws_merged["autopct"] is None
                else plot_kws_merged["autopct"])
            fig, ax = plt.subplots(figsize = params["figsize"])
            wedges, _, _ = ax.pie(
            df["value"],
            **plot_kws_merged)
            if params["show_legend"]:
                ax.legend(wedges, df["category"], loc="center left",
                          bbox_to_anchor=(1, 0.5))
        else:
            fig, ax = get_empty_plot(figsize=params["figsize"])
            warnings.warn(WRN_MSG_EMPTY_DATA.format("piechart"))
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        ax.set_title(params["title"])
        if params["directory"] is not None:
            save_plot(fig, directory=params["directory"],
                      plot_name="piechart", verbose=params["verbose"])
    return VisualizationResult(figure=fig,
                               axes=ax,
                               engine="matplotlib",
                               width=params["figsize"][0],
                               height=params["figsize"][1],
                               title=params["title"])

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
           **kwargs) -> None:
    """
    Display an interactive geographic scatter plot (Mapbox) with optional
    category-based coloring, point scaling, and hover labels.

    This method provides a high-level interface for visualizing spatial data
    using latitude and longitude coordinates. It supports categorical coloring,
    dynamic point sizing, custom hover labels, and Plotly Mapbox styling.

    Parameters
    ----------
    lat : Sequence of float
        Latitude values for each point. Cannot contain null values.
    lon : Sequence of float
        Longitude values for each point. Must match the length of `lat` and
        cannot contain null values.
    category : Sequence, optional
        Categorical labels used to color the points. Must match the length of
        `lat` and `lon`, cannot contain nulls, and determines the number of
        discrete colors in the plot.

    Other Parameters
    ----------------
    hover_name : Sequence, optional
        Labels to show on hover. Must match the length of `lat` and `lon`.
    size : Sequence of float, optional
        Numerical values used to scale point sizes. Must match the length
        of `lat` and `lon`.
    title : str, optional
        Plot title.
    show_legend : bool, default=True
        Whether to display the legend. Relevant only if `category` is provided.
    palette : Sequence of str, optional
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
        over internally generated defaults.
    nan_policy : str, default="drop"
        Policy for handling NaN values in input data. Supports 'drop' 
        (removes rows with NaNs) or 'raise' (raises an error).
    directory : str or Path, optional
        Path to save the figure as HTML.
    verbose : bool, default=False
        Enable logging.

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
        If `lat`, `lon`, or any optional input contain nulls or mismatched
        lengths.
    UserWarning
        If the number of unique categories exceeds the provided palette size.
        If the input data becomes empty after applying `nan_policy`.

    Notes
    -----
    - The plot is saved as an interactive HTML file when `directory` is set.
    - Color resolution is handled internally using `resolve_plotly_palette`.
    - This function is intended for rapid map-based EDA rather than full
      cartographic customization.
    - Supported Plotly templates: https://plotly.com/python/templates/
    - Supported Mapbox styles: https://plotly.com/python/mapbox-layers/

    Examples
    --------
    >>> import explorica.visualizations as visualizations
    >>> latitude = [34.0522, 40.7128, 37.7749, 51.5074]
    >>> longitude = [-118.2437, -74.0060, -122.4194, -0.1278]
    >>> fig = visualizations.mapbox(
    ...     latitude,
    ...     longitude,
    ...     title="Distribution of Cities",
    ...     palette=sns.color_palette("tab10").as_hex()
    ... )
    >>> result.figure.show() # Display the interactive Plotly figure
    >>> result.title # 'Distribution of Cities'
    >>> result.extra_info # Optional metadata dictionary
    """
    params = {"hover_name": None,
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
              **kwargs}

    plot_kws_merged = {
        "opacity": params["opacity"],
        "height": params["height"],
        "width": params["width"],
        "template": params["template"],
        "color_discrete_sequence": params["palette"],
        "zoom": params["zoom"],
        **(params["plot_kws"] or {})}

    lat_series, lon_series, category_series = (
        convert_series(lat), convert_series(lon), convert_series(category))
    params["hover_name"] = convert_series(params["hover_name"])
    params["size"] = convert_series(params["size"])

    validate_lengths_match(lat_series, lon_series,
                            err_msg=ERR_MSG_ARRAYS_LENS_MISMATCH.format("lat", "lon"))
    df = pd.DataFrame({"lat": convert_series(lat),
                       "lon": convert_series(lon)})

    for key, value in {"category": category_series,
                       "hover_name": params["hover_name"],
                       "size": params["size"]}.items():
        if value.empty:
            continue
        validate_lengths_match(
            lat_series, value,
            err_msg=ERR_MSG_ARRAYS_LENS_MISMATCH.format("lat and lon", key))
        df[key] = value
    df = handle_nan(df, nan_policy=params["nan_policy"],
                        supported_policy=("drop", "raise"), data_name="input sequences")
    unique_categories = df["category"].nunique() if "category" in df else 1
    if "category" in df:
        plot_kws_merged["color_discrete_sequence"] = resolve_plotly_palette(
            params["palette"])
        if len(plot_kws_merged["color_discrete_sequence"]) < unique_categories:
            warnings.warn(WRN_MSG_CATEGORIES_EXCEEDS_PALETTE_F.format(
                unique_categories, len(plot_kws_merged["color_discrete_sequence"])),
                UserWarning)
    if not df.empty:
        fig = px.scatter_map(
            lat=df["lat"],
            lon=df["lon"],
            color=df["category"] if "category" in df else None,
            hover_name=df["hover_name"] if "hover_name" in df else None,
            size=df["size"] if "size" in df else None,
            **plot_kws_merged,)
        fig.update_layout(mapbox_style=params["map_style"],
                          showlegend=params["show_legend"])
    else:
        fig = get_empty_plot(figsize=(params["width"], params["height"]),
                             engine="plotly")
        warnings.warn(WRN_MSG_EMPTY_DATA.format("mapbox"))
    if params["title"] is not None:
        fig.update_layout(title=params["title"], title_x=0.5, title_y=0.98,
                          title_font_size=22, margin={"r":0,"t":40,"l":0,"b":0})
    else:
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    if params["directory"] is not None:
        save_plot(fig, directory=params["directory"],
                  plot_name="mapbox", verbose=params["verbose"], engine="plotly")
    return VisualizationResult(figure=fig, engine="plotly",
                               width=params["width"], height=params["height"],
                               title=params["title"])
