"""
outlier_handler.py

This module defines the OutlierHandler class for identifying, visualizing, and handling outliers
in numerical datasets using standard statistical methods.

Modules:
    - OutlierHandler: Provides methods for detecting, removing, and replacing outliers using 
      the IQR and Z-score approaches.
"""
from typing import Optional, Sequence, Mapping, Union
from numbers import Number

import pandas as pd
import numpy as np
from scipy.stats import zscore
from explorica import DataVisualizer

class OutlierHandler:
    """
    A utility class for detecting, removing, replacing, and describing outliers 
    and distribution shapes in numerical datasets.

    This class implements common statistical techniques for outlier detection 
    (Interquartile Range and Z-score) and supports flexible handling strategies, 
    including removal or replacement with summary statistics. It also provides 
    methods for computing distribution shape descriptors such as skewness and kurtosis.

    Methods
    -------
    replace_outliers(series: pd.Series, method: str = "iqr", ...)
        Detects outliers using the specified method and replaces them with 
        a chosen statistical measure (mean, median, mode, or custom value). 
        Returns a modified pandas Series.

    remove_outliers(series: pd.Series, method: str = "iqr")
        Detects outliers using the specified method and removes them from the Series.
        Returns a cleaned pandas Series.

    detect_iqr(cls, series: pd.Series, show_boxplot: bool = False)
        Detects outliers using the Interquartile Range (IQR) method.
        Optionally displays a boxplot if `show_boxplot=True`.
        Returns a pandas Series containing only the detected outliers, 
        preserving the original index.

    detect_zscore(series: pd.Series, threshold: float = 3.0)
        Detects outliers using the Z-score method.
        An observation is considered an outlier if its absolute Z-score 
        exceeds the specified threshold.
        Returns a pandas Series containing the detected outliers.

    get_skewness(series: Sequence[Number])
        Computes skewness as m₃ / q³ for the given numeric sequence.
        Returns a float.

    get_kurtosis(series: Sequence[Number])
        Computes excess kurtosis as m₄ / q⁴ − 3 for the given numeric sequence.
        Returns a float.

    describe_distributions(dataset, threshold_skewness: float = 0.25,
                           threshold_kurtosis: float = 0.25, 
                           return_type: str = "dataframe")
        Analyzes multiple numeric distributions, computing skewness, 
        kurtosis, normality flags, and qualitative shape descriptions 
        (e.g., "left-skewed", "high-pitched").
        Accepts a 2D sequence, pandas DataFrame, or mapping of feature names 
        to numeric sequences.
        Returns either a pandas DataFrame or a dictionary, depending on 
        `return_type`.
    """

    dv = DataVisualizer()

    @staticmethod
    def replace_outliers(series: pd.Series,
                         method: Optional[str] = "iqr",
                         strategy: Optional[str] = "median",
                         custom_value: Optional[bool] = None
                         ) -> pd.Series:
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
            outiliers_indexes = OutlierHandler.detect_iqr(series).index
        if method == "z-score":
            outiliers_indexes = OutlierHandler.detect_zscore(series).index

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

    @staticmethod
    def remove_outliers(series: pd.Series,
                        method: Optional[str] = "iqr"
                        ) -> pd.Series:
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
            series_cleaned = series.drop(OutlierHandler.detect_iqr(series).index)
            return series_cleaned

        if method == "z-score":
            # Remove outliers using Z-score method
            series_cleaned = series.drop(OutlierHandler.detect_zscore(series).index)
            return series_cleaned

    @classmethod
    def detect_iqr(cls,
                   series: pd.Series,
                   show_boxplot: Optional[bool] = False):
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
            cls.dv.boxplot(series)
        return outliers

    @staticmethod
    def detect_zscore(series: pd.Series,
                      threshold: Optional[float] = 3.0
                      ) -> pd.Series:
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

    @staticmethod
    def get_skewness(series: Sequence[Number]) -> float:
        """
        Compute the skewness (third standardized moment) of a numeric sequence.

        Parameters
        ----------
        series : Sequence[Number]
            One-dimensional sequence of numeric values.

        Returns
        -------
        float
            Skewness value of the input data. Positive values indicate
            right-skewed distribution, negative values indicate left-skewed.

        Notes
        -----
        The skewness is calculated as:

        .. math::

            \\text{skewness} = \\frac{E[(X - \\mu)^3]}{\\sigma^3}

        where :math:`\\mu` is the mean and :math:`\\sigma` is the standard deviation.
        """
        array = np.array(series, dtype=float)
        mean = array.mean()
        std = array.std()
        m_3 = np.mean((array - mean) ** 3)
        q_3 = std ** 3
        skewness = m_3/q_3
        return skewness

    @staticmethod
    def get_kurtosis(series: Sequence[Number]) -> float:
        """
        Compute the **excess kurtosis** (fourth standardized moment minus 3)
        of a numeric sequence.

        Parameters
        ----------
        series : Sequence[Number]
            One-dimensional sequence of numeric values.

        Returns
        -------
        float
            Excess kurtosis value of the input data.  
            0.0 for normal distribution, positive values indicate
            heavier tails, negative values indicate lighter tails.

        Notes
        -----
        The excess kurtosis is calculated as:

        .. math::

            \\text{kurtosis} = \\frac{E[(X - \\mu)^4]}{\\sigma^4} - 3

        where :math:`\\mu` is the mean and :math:`\\sigma` is the standard deviation.
        """
        array = np.array(series, dtype=float)
        mean = array.mean()
        std = array.std()
        m_4 = np.mean((array - mean) ** 4)
        q_4 = std ** 4
        kurtosis = m_4/q_4 - 3
        return kurtosis

    @staticmethod
    def describe_distributions(dataset: Union[Sequence[Sequence[Number]],
                                              pd.DataFrame,
                                              Mapping[str, Sequence[Number]]],
                               threshold_skewness: Optional[float] = 0.25,
                               threshold_kurtosis: Optional[float] = 0.25,
                               return_type: Optional[str] = "dataframe"
                               ) -> Union[pd.DataFrame | dict]:
        """
        Describe shape (skewness / kurtosis) of one or multiple numeric distributions.

        The function computes skewness and excess kurtosis for each 1-D sequence
        in `dataset` and classifies the distribution shape according to the
        provided absolute thresholds. Distributions whose absolute skewness and
        absolute excess kurtosis are both less than or equal to the corresponding
        thresholds are considered "normal".

        Parameters
        ----------
        dataset : {Sequence[Sequence[Number]], pandas.DataFrame, Mapping[str, Sequence[Number]]}
            Input container with one or more numeric sequences (distributions).
            Supported forms:
            - 2D sequence (e.g. list of lists, list/array of 1D arrays): each inner
              sequence represents one distribution;
            - ``pandas.DataFrame``: each **column** is treated as a separate distribution;
            - ``Mapping`` (e.g. dict, OrderedDict): mapping keys are used as feature
              names and mapping values should be 1D numeric sequences.
            In the Mapping and DataFrame cases the order of returned metrics follows
            the order of mapping keys or DataFrame columns respectively.
            For plain sequences the order follows the sequence order and the resulting
            DataFrame will use a RangeIndex.

        threshold_skewness : float, optional, default=0.25
            Absolute skewness threshold. If ``abs(skewness) <= threshold_skewness``
            the distribution is considered not skewed (with respect to this threshold).

        threshold_kurtosis : float, optional, default=0.25
            Absolute excess kurtosis threshold. If ``abs(kurtosis) <= threshold_kurtosis``
            the distribution is considered not kurtotic (with respect to this threshold).
            Note: this function uses **excess kurtosis** (kurtosis - 3), so a normal
            distribution is approximately 0.

        return_type : {'dataframe', 'dict'}, optional, default='dataframe'
            Output format:
            - ``'dataframe'`` — return a ``pandas.DataFrame`` with columns:
              ``['is_normal', 'desc', 'skewness', 'kurtosis']``. If input was a
              DataFrame or Mapping the index will reflect column names / mapping keys.
            - ``'dict'`` — return a dict with keys ``'is_normal'``, ``'desc'``,
              ``'skewness'``, ``'kurtosis'`` and list-like values in the same order
              as the features.

        Returns
        -------
        pandas.DataFrame or dict
            Either a DataFrame (if ``return_type='dataframe'``) or a dict (if
            ``return_type='dict'``) containing the following entries per feature:
            - ``is_normal`` (bool) — True if both |skewness| and |kurtosis| are
              within thresholds.
            - ``desc`` (str) — human-friendly description, one of:
              ``'normal'``, ``'left-skewed'``, ``'right-skewed'``,
              ``'low-pitched'`` (platykurtic) and/or ``'high-pitched'`` (leptokurtic).
              Multiple descriptors are joined by a comma (e.g. ``'right-skewed, high-pitched'``).
            - ``skewness`` (float) — Fisher skewness (third standardized moment).
            - ``kurtosis`` (float) — **excess** kurtosis (fourth standardized moment minus 3).

        Raises
        ------
        ValueError
            If ``return_type`` is not in ``{'dataframe', 'dict'}``.

        Notes
        -----
        - The function expects numeric, one-dimensional sequences for each distribution.
          If mapping values are heterogeneous (different lengths / non-sequences) the
          behavior may be unexpected — prefer passing a DataFrame or a well-formed Mapping.
        - Threshold checks are **inclusive**: equality to threshold counts as within.
        - For programmatic consumption prefer ``return_type='dataframe'`` (tabular form).
          The ``dict`` form returns lists of values aligned to the feature order (not a
          transposed mapping of feature -> single-structure per feature).

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "x": np.random.normal(size=1000),
        ...     "y": np.random.exponential(size=1000)
        ... })
        >>> DataPreprocessor.describe_distributions(df, threshold_skewness=0.3)
                        is_normal                         desc  skewness  kurtosis
        feature
        x                True                         normal  0.012345  0.023456
        y               False           right-skewed, high-pitched  1.234567  3.456789

        >>> d = DataPreprocessor.describe_distributions(df, return_type='dict')
        >>> list(d.keys())
        ['is_normal', 'desc', 'skewness', 'kurtosis']
        """
        supported_return_types = {"dataframe", "dict"}
        left_skew = "left-skewed"
        right_skew = "right-skewed"
        high_pitch = "high-pitched"
        low_pitch = "low-pitched"
        normal = "normal"

        if return_type not in supported_return_types:
            raise ValueError(f"Unsupported return type '{return_type}',"
                             f"please, choose from {supported_return_types}")

        indexes = None
        # processing of input sequence
        if isinstance(dataset, pd.DataFrame):
            indexes = dataset.columns.to_list()
            dists = dataset.to_numpy().T
        elif isinstance(dataset, Mapping):
            indexes = list(dataset)
            dists = np.array(list(dataset.values()))
        else:
            dists = np.array(dataset)

        # collection of descriptive information
        kurts, skews, descs, is_normal = [], [], [], []
        for array in dists:
            skewness = OutlierHandler.get_skewness(array)
            kurtosis = OutlierHandler.get_kurtosis(array)
            norm = (True if abs(skewness) <= threshold_skewness
                                  and abs(kurtosis) <= threshold_kurtosis else False)
            is_normal.append(norm)
            if norm:
                descs.append(normal)
            else:
                form = []
                if abs(skewness) > threshold_skewness:
                    if skewness < 0:
                        form.append(left_skew)
                    else:
                        form.append(right_skew)
                if abs(kurtosis) > threshold_kurtosis:
                    if kurtosis < 0:
                        form.append(low_pitch)
                    else:
                        form.append(high_pitch)
                descs.append(", ".join(form))
            kurts.append(kurtosis)
            skews.append(skewness)
        describe = {"is_normal": is_normal,
                    "desc": descs,
                    "skewness": skews,
                    "kurtosis": kurts}
        if return_type == "dict":
            return describe
        if return_type == "dataframe":
            result_df = pd.DataFrame({"is_normal": is_normal,
                                  "desc": descs,
                                  "skewness": skews,
                                  "kurtosis": kurts},
                                  index=indexes)
            if indexes is not None:
                result_df.index.name = "feature"
            return result_df
        return None
