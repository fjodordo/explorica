"""
This module contains the DataVisualizer class for visualizing different types of data.
It includes methods for creating various types of plots such as distplots, boxplots, and more.

Modules:
    - DataVisualizer: Class for visualizing data with methods like distplot, heatmap, etc.
"""
from typing import Iterable, Callable, Optional, Sequence
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


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
    style = "whitegrid"
    palette = sns.color_palette("Set2")
    colormap = "coolwarm"

    def __init__(self):
        pass

    @classmethod
    def set_theme(cls,
                  palette: Optional[str | Sequence]=None,
                  style: Optional[str]=None,
                  colormap: Optional[str]=None
                  ) -> None:
        """
        Set the default theme parameters for visualizations, 
        including color palette, plot style and colormap.

        This method updates the class-level attributes `palette`, `style` 
        and `colormap`, which are used 
        by other plotting methods in the class. It accepts either the name of a seaborn 
        palette or a custom sequence of valid matplotlib color specifications (e.g., hex codes,
        RGB tuples, or named colors).
        The plot style and colormap can also be set using one of seaborn's predefined styles.

        Parameters
        ----------
        palette : str or sequence of colors, optional
            A seaborn palette name (e.g. "Set2", "muted") or a custom sequence of color
            values. Custom values must be valid matplotlib color specifications.
        
        style : str, optional
            A seaborn style name (e.g. "whitegrid", "dark", "ticks"). If not provided, the
            current style remains unchanged.
        
        colormap : str, optional
            A seaborn colormap name (e.g., 'coolwarm', 'viridis', 'plasma') If not provided, the
            current colormap remains unchanged
        """
        if palette is not None:
            try:
                cls.palette = sns.color_palette(palette)
            except ValueError as e:
                raise ValueError(
                    "Palette must contain valid color names, hex codes, or RGB tuples.") from e
        if style is not None:
            cls.style = style
        if colormap is not None:
            cls.colormap = colormap

    @classmethod
    def distplot(cls, series: pd.Series, bins: int = 30, kde: bool = True, title: str = ""):
        """
        Plots a histogram with optional KDE (kernel density estimate).

        Parameters
        ----------
        series : pd.Series
            Numeric series to plot.
        bins : int
            Number of bins in the histogram.
        kde : bool
            Whether to include a KDE curve.
        title : str
            Plot title.
        """
        with sns.axes_style(cls.style), sns.color_palette(cls.palette):
            sns.histplot(series.dropna(), bins=bins, kde=kde)
            plt.title(title)
            plt.xlabel(series.name)
            plt.ylabel("Frequency")
            plt.show()

    @classmethod
    def boxplot(cls, series: pd.Series, title: str = ""):
        """
        Draws a boxplot for a numeric variable.

        Parameters
        ----------
        series : pd.Series
            Numeric series to plot.
        title : str
            Plot title.
        """
        with sns.axes_style(cls.style), sns.color_palette(cls.palette):
            sns.boxplot(x=series.dropna())
            plt.title(title)
            plt.xlabel(series.name)
            plt.show()

    @classmethod
    def hexbin(cls,
               x: pd.Series,
               y: pd.Series,
               gridsize: int = 30,
               colormap: str = "viridis",
               title: str = ""):
        """
        Creates a hexbin plot for two numeric variables to visualize density.

        Parameters
        ----------
        x : pd.Series
            First numeric variable (X-axis).
        y : pd.Series
            Second numeric variable (Y-axis).
        gridsize : int
            Size of hexagon bins.
        colormap : str
            Color map.
        title : str
            Plot title.
        """
        if colormap is None:
            colormap = cls.colormap
        with sns.axes_style(cls.style):
            plt.hexbin(x.dropna(), y.dropna(), gridsize=gridsize, cmap=colormap)
            plt.title(title)
            plt.xlabel(x.name)
            plt.ylabel(y.name)
            plt.colorbar(label="Counts")
            plt.show()

    @classmethod
    def heatmap(cls,
                dataframe: pd.DataFrame,
                annot: bool = True,
                title: str = "",
                colormap: str = None):
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
        if colormap is None:
            colormap = cls.colormap
        sns.heatmap(dataframe, annot=annot, fmt=".2f", cmap=colormap, square=True)
        plt.title(title)
        plt.show()

    @classmethod
    def piechart(cls,
                 categories: pd.Series,
                 values: pd.Series,
                 autopct_method: str = "value",
                 title: str = "",
                 show_legend: bool = True,
                 show_labels: bool = True
                 ) -> None:
        """
        Draws a pie chart based on categorical and corresponding numerical data.

        Parameters:
        -----------
        categories : pd.Series
            A categorical series representing the pie chart segments.
        values : pd.Series
            A numerical series representing the sizes of the segments.
        autopct_method : str, default="value"
            Determines how the values are displayed on the pie chart.
            Supported options: "percent", "value", "both".
        title : str, optional
            Title of the pie chart.
        show_legend : bool, default=True
            Whether to display a legend.
        show_labels : bool, default=True
            Whether to display category labels directly on the chart.

        Raises:
        -------
        ValueError
            If input sizes mismatch.
            If either 'categories' or 'values' contains null values.
            If invalid autopct method is provided.
        """

        supported_autopct_method = {"percent", "value", "both"}

        # Parameter validation
        if autopct_method not in supported_autopct_method:
            raise ValueError(f"Unsupported autopct '{autopct_method}'. "
                             f"Choose from: {supported_autopct_method}")

        if categories.isnull().any() or values.isnull().any():
            raise ValueError("The input 'categories' or 'values' contains null values. "
                             "Please clean or impute missing data before visualization.")

        if categories.size != values.size:
            raise ValueError("The length of 'categories' must match the length of 'values'")

        labels = categories if show_labels else None
        with sns.axes_style(cls.style), sns.color_palette(cls.palette):
            wedges, _, _ = plt.pie(
            values,
            labels=labels,
            autopct=DataVisualizer._make_autopct(values, autopct_method),
            startangle=90)

            if show_legend:
                plt.legend(wedges, categories, loc="center left", bbox_to_anchor=(1, 0.5))
            if title:
                plt.title(title)

            plt.show()

    @classmethod
    def barchart(cls,
                 categories: pd.Series,
                 values: pd.Series,
                 ascending: bool = None,
                 horizontal: bool = False,
                 title: str = ""
                 ) -> None:
        """
        Plots a bar chart using the given category and value series.

        Parameters:
        -----------
        categories : pd.Series
            A series containing categorical labels.
        values : pd.Series
            A series containing numeric values associated with each category.
        ascending : bool, optional
            If True or False, sorts the bars by value in ascending or descending order respectively. 
            If None (default), no sorting is applied and original order is preserved.
        horizontal : bool
            If True, plots a horizontal bar chart instead of a vertical one.
        title : str
            Title of the chart. Defaults to an empty string.

        Raises:
        -------
        ValueError
            If input sizes mismatch.
            If either 'categories' or 'values' contains null values.
        """
        # Parameter validation
        if categories.isnull().any() or values.isnull().any():
            raise ValueError("The input 'categories' or 'values' contains null values. "
                         "Please clean or impute missing data before visualization.")

        if categories.size != values.size:
            raise ValueError("The length of 'categories' must match the length of 'values'.")

        df = pd.DataFrame({"category": categories, "value": values})
        if ascending is not None:
            df = df.sort_values(
             "value", ascending=ascending)
        with sns.axes_style(cls.style), sns.color_palette(cls.palette):
            if horizontal:
                plt.barh(df["category"], df["value"])
            else:
                plt.bar(df["category"], df["value"])
            plt.title(title)

            plt.show()

    @classmethod
    def scatterplot(cls,
        x: pd.Series,
        y: pd.Series,
        category: pd.Series = None,
        trend_line: str = None,
        custom_function: Optional[Callable[[float], float]] = None,
        palette: Iterable = None,
        opacity: float = 1.0,
        show_legend: bool = False,
        title: str = "",
        x_label: str = "",
        y_label: str = ""
        ) -> None:
        """
        Draws a scatterplot with optional categorical coloring and trend line fitting.

        Supported trend line's methods
        ----------
        - 'linear':       y = a * x + b
        - 'binomial':     y = a * x² + b * x + c
        - 'exp':          y = b * exp(a * x)
        - 'ln':           y = a * ln(x) + b
        - 'hyperbolic':   y = a / x + b
        - 'power':        y = a * xᵇ
        - 'custom':       y = your function

        Parameters:
        -----------
        x : pd.Series
            Numerical values for the x-axis. Must not contain nulls.
        y : pd.Series
            Numerical values for the y-axis. Must not contain nulls.
        category : pd.Series, optional
            Categorical labels for each point. If provided, must match the length of `x` and `y`,
            and contain no nulls. Points will be colored by category.
        trend_line : str, optional
            Specifies the type of trend line to fit. See supported methods above.
            If not provided, no trend line will be shown.
        custom_function : Callable[[float], float], optional
            Custom function to be used for fitting when `trend_line="custom"`.
        palette : Iterable[str], optional
            Iterable of color values (e.g., hex codes or named colors). If not provided,
            the default class palette will be used. If the number of unique categories exceeds
            the number of colors, the palette will repeat and a warning will be raised.
        opacity : float, default=1.0
            Opacity level of points (from 0.0 to 1.0).
        show_legend : bool, default=False
            Whether to display a legend (only works if `category` is provided).
        title : str, default=""
            Title for the plot.
        x_label : str, default=""
            Label for the x-axis. If not provided, `x.name` will be used.
        y_label : str, default=""
            Label for the y-axis. If not provided, `y.name` will be used.

        Raises:
        -------
        ValueError:
            If input sizes mismatch.
            If input contains null values.
            If an unsupported `trend_line` method is provided.
            If `trend_line` is "custom" but `custom_function` is not supplied.
            If `category` is provided but its length does not match `x` and `y`.
            If `category` contains null values.

        Warnings:
        -------
        UserWarning
            If the number of unique categories exceeds the palette length.
            colors may repeat in the plot.
            If the trend line fitting fails (e.g., model does not converge);
            the trend line will not be displayed.

        Notes:
        ------
        - For some trend types, `x` values are automatically filtered to avoid mathematical errors.
            For example:
                - `exp`: x values above ~350 are excluded to avoid numerical overflow.
                - `ln`: x <= 0 is excluded.
                - `hyperbolic`, `power`: x == 0 are excluded.
        - Trend line is fitted using `scipy.optimize.curve_fit` and 
          drawn smoothly over the filtered domain.
        - If the fit fails due to invalid data or curve_fit issues, no trend line will be shown.
        - The trend line is drawn using the next color in the palette,
          after the last category color.
        """
        models = {"linear": lambda x, a, b: a * x + b,
                  "binomial": lambda x, a, b, c: a * x**2 + b * x + c,
                   "exp": lambda x, a, b: b * np.exp(a * x),
                   "ln": lambda x, a, b: a * np.log(x) + b,
                   "hyperbolic": lambda x, a, b: a / x + b,
                   "power": lambda x, a, b: a * x ** b,
                   "custom": custom_function}

        def filter_x_by_domain(x: pd.Series,
                               y: pd.Series,
                               func_type: str) -> pd.Series:
            if func_type == "ln":
                y = y[x > 0]
                x = x[x > 0]
            elif func_type in {"hyperbolic", "power"}:
                y = y[x != 0]
                x = x[x != 0]
            elif func_type == "exp":
                y = x[x <= 350]
                x = x[x <= 350]
            return x, y

        # Parameter validation
        if x.isnull().any() or y.isnull().any():
            raise ValueError("Input 'x' or 'y' contains null values.")

        if len(x) != len(y):
            raise ValueError("Length of 'x' and 'y' must match.")

        if trend_line is not None and trend_line not in models:
            available = "{'" + "', '".join(models.keys()) + "'}"
            raise ValueError(f"Unsupported method '{trend_line}'. Choose from: {available}")

        if trend_line == "custom" and custom_function is None:
            raise ValueError("Custom function must be provided when method='custom'")

        if palette is None:
            palette = cls.palette[:]

        with sns.axes_style(cls.style):
            if category is not None:
                if len(category) != len(x):
                    raise ValueError("Length of 'category' must match 'x' and 'y'")
                if category.isnull().any():
                    raise ValueError("Input 'category' contains null values.")

                unique_categories = category.unique()
                n_unique_categories = category.nunique()
                if n_unique_categories > len(palette):
                    warnings.warn(f"Number of categories ({n_unique_categories}) "
                                f"exceeds the palette size ({len(palette)}). Colors may repeat.",
                                UserWarning)

                colors = {cat: palette[i % len(palette)] for i, cat in enumerate(unique_categories)}

                for cat in unique_categories:
                    plt.scatter(
                        x[category == cat],
                        y[category == cat],
                        label=cat if show_legend else None,
                        alpha=opacity,
                        color=colors[cat]
                    )
                trend_line_color = palette[n_unique_categories%len(palette)]
            else:
                plt.scatter(x, y, alpha=opacity, color=palette[0])
                trend_line_color = palette[1 % len(palette)]
            if trend_line:
                if trend_line == "custom":
                    model_x = np.linspace(x.min(), x.max(), 1000)
                    model_values = models[trend_line](model_x)
                    plt.plot(model_x, model_values, color=trend_line_color)
                else:
                    model_x, model_y = filter_x_by_domain(x, y, trend_line)
                    try:
                        coeffs = curve_fit(models[trend_line], model_x, model_y, maxfev=10000)[0]
                        model_x = np.linspace(model_x.min(), model_x.max(), 1000)
                        model_values = models[trend_line](model_x, *coeffs)
                        plt.plot(model_x, model_values, color=trend_line_color)
                    except RuntimeError as e:
                        warnings.warn(f"Could not fit '{trend_line}' trend: {e}", UserWarning)

            if category is not None and show_legend:
                plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")

            # None < pd.Series.name < "x/y_label" argument priority
            if x_label:
                plt.xlabel(x_label)
            else:
                plt.xlabel(x.name)
            if y_label:
                plt.ylabel(y_label)
            else:
                plt.ylabel(y.name)
            plt.title(title)

            plt.show()

    @classmethod
    def mapbox(cls,
               lat: Sequence[float],
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
            colors = cls.palette.as_hex()

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

    @staticmethod
    def _make_autopct(values: pd.Series, method: str):
        """
        Internal helper to format piechart percentage labels.
        """
        def formatter(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            if method == "percent":
                return f"{pct:.1f}%"
            elif method == "value":
                return f"{val}"
            else:
                return f"{pct:.1f}%\n({val})"
        return formatter
