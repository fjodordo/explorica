"""
This module contains the DataVisualizer class for visualizing different types of data.
It includes methods for creating various types of plots such as distplots, boxplots, and more.

Modules:
    - DataVisualizer: Class for visualizing data with methods like distplot, heatmap, etc.
"""
from typing import Iterable
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DataVisualizer:
    """
    A utility class for quick and consistent data visualizations using seaborn, matplotlib and
    plotly
 
    This class simplifies the creation of common plots with pre-configured styles and layout 
    options, helping to keep jupyter notebooks clean, readable, and standardized.
    """
    def __init__(self, style: str = "whitegrid", pallete: str = "Set2"):
        """
        Initializes the visualization environment with seaborn styles.

        Parameters
        ----------
        style : str
            Seaborn style to apply globally (default: 'whitegrid').
        palette : str
            Color palette to use for plots (default: 'Set2').
        """
        sns.set_style(style)
        sns.set_palette(pallete)
        self.palette = sns.color_palette()
        plt.rcParams.update({
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.figsize': (8, 5)
        })

    def distplot(self, series: pd.Series, bins: int = 30, kde: bool = True, title: str = ""):
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
        sns.histplot(series.dropna(), bins=bins, kde=kde)
        plt.title(title)
        plt.xlabel(series.name)
        plt.ylabel("Frequency")
        plt.show()

    def boxplot(self, series: pd.Series, title: str = ""):
        """
        Draws a boxplot for a numeric variable.

        Parameters
        ----------
        series : pd.Series
            Numeric series to plot.
        title : str
            Plot title.
        """
        sns.boxplot(x=series.dropna())
        plt.title(title)
        plt.xlabel(series.name)
        return plt

    def hexbin(self, x: pd.Series, y: pd.Series, gridsize: int = 30, cmap: str = "viridis",
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
        cmap : str
            Color map.
        title : str
            Plot title.
        """
        plt.hexbin(x.dropna(), y.dropna(), gridsize=gridsize, cmap=cmap)
        plt.title(title)
        plt.xlabel(x.name)
        plt.ylabel(y.name)
        plt.colorbar(label="Counts")
        plt.show()

    def heatmap(self, dataframe: pd.DataFrame, annot: bool = True, title: str = ""):
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
        """
        sns.heatmap(dataframe, annot=annot, fmt=".2f", cmap="coolwarm", square=True)
        plt.title(title)
        plt.show()

    def piechart(self,
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

        wedges, _, _ = plt.pie(
        values,
        labels=labels,
        autopct=self._make_autopct(values, autopct_method),
        startangle=90)

        if show_legend:
            plt.legend(wedges, categories, loc="center left", bbox_to_anchor=(1, 0.5))
        if title:
            plt.title(title)

        plt.show()

    def barchart(self,
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
        if horizontal:
            plt.barh(df["category"], df["value"])
        else:
            plt.bar(df["category"], df["value"])
        plt.title(title)

        plt.show()

    def scatterplot(
        self,
        x: pd.Series,
        y: pd.Series,
        category: pd.Series = None,
        palette: Iterable = None,
        opacity: float = 1.0,
        show_legend: bool = False,
        title: str = "",
        x_label: str = "",
        y_label: str = ""
        ) -> None:
        """
        Draws a scatterplot with optional categorical coloring.

        Parameters:
        -----------
        x : pd.Series
            Numerical values for the x-axis. Must not contain nulls.
        y : pd.Series
            Numerical values for the y-axis. Must not contain nulls.
        category : pd.Series, optional
            Categorical labels for each point. If provided, must match the length of `x` and `y`,
            and contain no nulls. Points will be colored by category.
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
        Warnings:
        If the number of categories exceeds the size of the palette.
        """

        # Parameter validation
        if x.isnull().any() or y.isnull().any():
            raise ValueError("Input 'x' or 'y' contains null values.")

        if len(x) != len(y):
            raise ValueError("Length of 'x' and 'y' must match.")

        if palette is None:
            palette = self.palette[:]

        if category is not None:
            if len(category) != len(x):
                raise ValueError("Length of 'category' must match 'x' and 'y'")
            if category.isnull().any():
                raise ValueError("Input 'category' contains null values.")

            unique_categories = category.unique()
            n_unique_categories = category.nunique()
            if n_unique_categories > len(palette):
                warnings.warn(f"Number of categories ({n_unique_categories}) "
                              f"exceeds the palette size ({len(palette)}). Colors may repeat.")

            colors = {cat: palette[i % len(palette)] for i, cat in enumerate(unique_categories)}

            for cat in unique_categories:
                plt.scatter(
                    x[category == cat],
                    y[category == cat],
                    label=cat if show_legend else None,
                    alpha=opacity,
                    color=colors[cat]
                )
        else:
            plt.scatter(x, y, alpha=opacity, color=palette[0])

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
