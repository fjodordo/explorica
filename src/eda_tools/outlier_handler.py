"""
outlier_handler.py

This module defines the OutlierHandler class for identifying, visualizing, and handling outliers
in numerical datasets using standard statistical methods.

Modules:
    - OutlierHandler: Provides methods for detecting, removing, and replacing outliers using 
      the IQR and Z-score approaches. Inherits from DataVisualizer to support optional plotting.
"""
import pandas as pd
import numpy as np
from scipy.stats import zscore
from eda_tools.visualizer import DataVisualizer

class OutlierHandler(DataVisualizer):
    """
    A class for detecting, removing, and replacing outliers in numerical data.

    Inherits from:
        DataVisualizer: Enables optional visualization capabilities such as boxplots.

    This class provides methods to identify outliers using both the Interquartile Range (IQR)
    and Z-score methods, and allows for flexible removal or replacement of these outliers
    using common statistical strategies (mean, median, mode, or a custom value).
    """
    def __init__(self):
        super().__init__()

    def replace_outliers(self, series: pd.Series, method: str = "iqr", strategy: str = "median",
                         custom_value = None) -> pd.Series:
        """
        Replaces outliers in a pandas Series using the specified method and strategy.

        Parameters:
        -----------
        series : pd.Series
            The pandas Series in which outliers will be replaced.
    
        method : str, default 'iqr'
            The method used for outlier detection. Supported methods are:
            - 'iqr' : Uses the IQR (Interquartile Range) method to detect outliers.
            - 'z-score' : Uses the Z-score method to detect outliers.
    
        strategy : str, default 'median'
            The strategy used to replace outliers. Supported strategies are:
            - 'median' : Replace outliers with the median value of the Series.
            - 'mean' : Replace outliers with the mean value of the Series.
            - 'mode' : Replace outliers with the mode value of the Series.
            - 'custom_value' : Replace outliers with a custom specified value. 
              The `custom_value` parameter should be provided in this case.

        custom_value : scalar, optional, default None
            The custom value used to replace outliers when the strategy is 'custom_value'. 
            This value must be provided if 'custom_value' is chosen as the strategy.

        Returns:
        --------
        pd.Series
            A pandas Series with outliers replaced according to the specified method and strategy.

        Raises:
        -------
        ValueError
            If the provided method or strategy is not supported.
        """
        value = None
        outiliers_indexes = None

        # Validate method and strategy
        supported_methods = {"iqr", "z-score"}
        supported_strategies = {"median", "mean", "mode", "custom_value"}

        series_replaced = series.dropna()

        if method not in supported_methods:
            raise ValueError(f"Unsupported method '{method}'. Choose from: {supported_methods}")

        if strategy not in supported_strategies:
            raise ValueError(
                f"Unsupported strategy '{strategy}'. Choose from:{supported_strategies}")

        # Identify outliers based on selected method
        if method == "iqr":
            outiliers_indexes = self.detect_iqr(series).index
        if method == "z-score":
            outiliers_indexes = self.detect_zscore(series).index

        # Assign value based on chosen strategy
        if strategy == "median":
            value = series.median()
        if strategy == "mean":
            value = series.mean()
        if strategy == "mode":
            value = series.mode()[0]
        if strategy == "custom_value":
            if custom_value is None:
                raise ValueError("custom_value must be provided when using " \
                "'custom_value' strategy.")
            value = custom_value

        # Replace outliers with the selected value
        series_replaced[outiliers_indexes] = value

        return series_replaced

    def remove_outliers(self, series: pd.Series, method: str = "iqr"):
        """
        Removes outliers from a given pandas Series using the specified method.

        This method supports two outlier detection techniques:
        1. IQR (Interquartile Range)
        2. Z-score

        Parameters:
        -----------
        series : pd.Series
            The pandas Series from which outliers should be removed.
    
        method : str, default 'iqr'
            The method used for outlier detection. Supported methods are:
            - 'iqr' : Uses the IQR method to detect and remove outliers.
            - 'z-score' : Uses the Z-score method to detect and remove outliers.

        Returns:
        --------
        pd.Series
            A pandas Series with outliers removed based on the chosen method.

        Raises:
        -------
        ValueError
            If the provided method is not supported.
        """
        supported_methods = {"iqr", "z-score"}

        if method not in supported_methods:
            raise ValueError(f"Unsupported method '{method}'. Choose from: {supported_methods}")

        if method == "iqr":
            # Remove outliers using IQR method
            series_cleaned = series.drop(self.detect_iqr(series).index)
            return series_cleaned

        if method == "z-score":
            # Remove outliers using Z-score method
            series_cleaned = series.drop(self.detect_zscore(series).index)
            return series_cleaned

    def detect_iqr(self, series: pd.Series, show_boxplot: bool = False):
        """
        Detects outliers in a numerical series using the Interquartile Range (IQR) method.

        Parameters
        ----------
        series : pd.Series
            A numeric pandas Series to analyze.
        show_boxplot : bool
            If True, displays a boxplot to visualize the outliers.

        Returns
        -------
        pd.Series
            A Series containing only the outlier values that lie outside the IQR bounds.
            The index corresponds to the positions of the outliers in the original Series.

        Notes
        -----
        An outlier is defined as a value below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.
        """
        if not isinstance(series, pd.Series):
            raise TypeError("Expected input to be a pandas Series")
        q1 = np.percentile(series, 25)
        q3 = np.percentile(series, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr * 1.5
        upper_bound = q3 + iqr * 1.5
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        if show_boxplot:
            self.boxplot(series)
        return outliers

    def detect_zscore(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Detects outliers in a numerical series using the Z-score method.

        Parameters
        ----------
        series : pd.Series
            Numeric series to analyze.
        threshold : float, optional
            Z-score threshold for determining outliers (default is 3.0).

        Returns
        -------
        pd.Series
            A Series of values that are considered outliers.
        """
        z_scores = zscore(series.dropna())
        outliers = series[(np.abs(z_scores) > threshold)]
        return outliers
