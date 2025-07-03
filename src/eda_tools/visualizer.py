"""
This module contains the DataVisualizer class for visualizing different types of data.
It includes methods for creating various types of plots such as distplots, boxplots, and more.

Modules:
    - DataVisualizer: Class for visualizing data with methods like distplot, heatmap, etc.
"""

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
