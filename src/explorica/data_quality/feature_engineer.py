"""
feature_engineer.py

This module defines the FeatureEngineer class, which provides
utility methods for fast and flexible feature transformation.
Includes implementations of frequency encoding, ordinal encoding,
and numeric binning for preprocessing categorical and continuous variables.
"""

from typing import Iterable

import pandas as pd


class FeatureEngineer:
    """
    A utility class for performing common feature engineering tasks on
    categorical and numerical variables.

    Provides methods for:
    - Frequency encoding (with optional normalization)
    - Ordinal encoding based on category order, frequency, or target statistics
    - Binning numerical features using uniform or quantile-based strategies

    Designed to be modular, extensible, and easy to integrate into EDA workflows.
    """

    def __init__(self):
        pass

    def freq_encode(self, series: pd.Series, normalize: bool = True) -> pd.Series:
        """
        Performs frequency encoding on a categorical feature.

        Parameters:
        -----------
            series : pd.Series
                The input categorical column to encode.
            normalize : bool
                If True, encodes as relative frequency (proportion),
                              otherwise as absolute count.

        Returns:
        -----------
            pd.Series
                Series with frequency-encoded values.
        """
        freq = series.value_counts(normalize=normalize)
        return series.map(freq)

    def ordinal_encode(
        self,
        series: pd.Series,
        target: pd.Series = None,
        order: str = "freq",
        ascending: bool = False,
    ) -> pd.Series:
        """
        Encodes categorical values with ordinal integers based on a specified ordering
        logic.

        Parameters:
        -----------
        series : pd.Series
            The categorical feature to be encoded.
        target : pd.Series, optional
            The target variable, required if order is 'target_mean' or 'target_median'.
        order : str, default='freq'
            The ordering strategy to assign integers:
            - 'freq' : Order by category frequency.
            - 'alphabetical' : Order alphabetically.
            - 'target_mean' : Order by the mean of the target variable per category.
            - 'target_median' : Order by the median of the target variable per category.
        ascending : bool, default=False
            Whether to sort in ascending order.

        Returns:
        --------
        pd.Series
            A series with encoded integer values representing categories.

        Raises:
        -------
        ValueError:
            If an unsupported order is passed or if a target variable is required but
            not provided.
            If either 'series' or 'target' (when provided) contains null values.
        """
        supported_order = {"freq", "alphabetical", "target_mean", "target_median"}

        if series.isnull().any():
            raise ValueError(
                "The input 'series' contains null values. "
                "Please clean or impute missing data before encoding."
            )
        if order not in supported_order:
            raise ValueError(
                f"Unsupported sorting '{order}'. Choose from: {supported_order}"
            )

        if order == "freq":
            categories = (
                series.value_counts().sort_values(ascending=ascending).index.unique()
            )
        elif order == "alphabetical":
            categories = sorted(series.unique(), reverse=not ascending)
        elif order in {"target_mean", "target_median"}:
            if target is None:
                raise ValueError("Target must be provided for target-based encoding.")
            if target.isnull().any():
                raise ValueError(
                    "The input 'target' contains null values. "
                    "Please clean or impute missing data before encoding."
                )

            df = pd.DataFrame({"cat": series, "target": target})
            print(df)
            agg_func = "mean" if order == "target_mean" else "median"
            categories = (
                df.groupby("cat")["target"]
                .agg(agg_func)
                .sort_values(ascending=ascending)
                .index
            )

        mapping = {cat: i + 1 for i, cat in enumerate(categories)}
        return series.map(mapping).astype(int)

    def bin_numeric(
        self,
        series: pd.Series,
        bins: int = 5,
        labels: Iterable = None,
        strategy: str = "uniform",
    ) -> pd.Series:
        """
        Bin a numeric series into discrete intervals using either uniform-width
        or quantile-based binning.

        Parameters:
        -----------
        series : pd.Series
            The numeric data to be binned.
        bins : int, default=5
            The number of bins to divide the data into. Must be at least 2.
        labels : iterable, optional
            Optional custom labels for the bins. Must match the number of bins.
        strategy : str, default='uniform'
            Binning strategy to use. One of:
            - 'uniform': Equal-width binning.
            - 'quantile': Quantile-based binning (equal number of observations per bin).

        Returns:
        --------
        pd.Series
            A series of categorical values representing the bin assigned to each
            original value.

        Raises:
        -------
        ValueError
            If strategy is unsupported.
            If labels don't match bin count.
            If bin count is invalid.
        """
        supported_strategies = {"uniform", "quantile"}
        series = series.dropna()
        series_binned = None

        if strategy not in supported_strategies:
            raise ValueError(
                f"""Unsupported strategy '{strategy}'. Choose from: 
                {supported_strategies}"""
            )
        if labels:
            if len(labels) != bins:
                raise ValueError(
                    f"""Length of 'labels' ({len(labels)}) must match the number 
                    of bins ({bins})."""
                )
        if bins < 2:
            raise ValueError("Number of bins must be at least 2.")

        if strategy == "uniform":
            series_binned = pd.cut(series, bins, include_lowest=True, labels=labels)
        if strategy == "quantile":
            series_binned = pd.qcut(series, bins, labels=labels)
        return series_binned
