"""
Module for detecting outliers in numeric data.

This module contains the DetectionMethods class, which provides
a collection of outlier detection methods that can be applied
to pandas Series or numeric arrays.

Functions
---------
detect_iqr(data, iqr_factor=1.5,
    get_boxplot=False, nan_policy="drop", boxplot_kws=None
)
    Detects outliers in a numerical series using the Interquartile
    Range (IQR) method.
detect_zscore(
    data: Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Sequence[Any]],
    threshold: Optional[float] = 2.0,
)
    Detects outliers in a numerical series using the Z-score method.

Examples
--------
>>> import pandas as pd
>>> from explorica.data_quality.outliers import DetectionMethods
...
>>> df = pd.DataFrame({"x": [1,2,3,4,100]}
>>> # Detect IQR outliers
>>> DetectionMethods.detect_iqr(df)
          x
4  100
dtype: int64
...
>>> # Detect Z-score outliers
>>> DetectionMethods.detect_zscore(df, threshold=2)
          x
4  100
dtype: int64
"""

import warnings
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from explorica.types import VisualizationResult
from explorica.visualizations import boxplot
from explorica._utils import (
    convert_dataframe,
    read_config,
    validate_array_not_contains_nan,
    handle_nan,
)

_errors = read_config("messages")["errors"]
_warns = read_config("messages")["warns"]


def detect_iqr(
    data: (
        Union[Sequence[float] | Sequence[Sequence[float]]] | Mapping[str, Sequence[Any]]
    ),
    iqr_factor: float = 1.5,
    get_boxplot: Optional[bool] = False,
    nan_policy: str = "drop",
    boxplot_kws: dict = None,
) -> Union[
    pd.Series,
    pd.DataFrame,
    tuple[Union[pd.Series, pd.DataFrame], VisualizationResult],
]:
    """
    Detects outliers in a numerical series using the Interquartile
    Range (IQR) method.

    This method identifies values that are significantly lower or higher
    than the typical range of the data. For 1D input, returns a Series of outliers;
    for 2D input, returns a DataFrame where non-outlier positions are NaN.
    Optionally, a boxplot visualization can be generated for the first column
    to visually inspect outliers.

    Parameters
    ----------
    data : Sequence[float]|Sequence[Sequence[float]]
        A numeric sequence (1D) or a sequence of sequences (2D) to analyze.
        Will be converted to a pandas DataFrame internally. Each inner sequence
        is treated as a separate column.
    iqr_factor : float, default 1.5
        Multiplier for the Interquartile Range used to define outlier bounds.
    get_boxplot : bool, optional
        If True, returns a tuple `(outliers, boxplot_figure)` where `boxplot_figure`
        is a `VisualizationResult` for the first column only.
    nan_policy : {"drop", "raise"}, default="drop"
        How to handle NaN values.
    boxplot_kws : dict, optional
        Additional keyword arguments passed to `explorica.visualizations.boxplot`
        (e.g., `color`, `figsize`, `title`). Only applied if `get_boxplot=True`.

    Returns
    -------
    pd.Series or pd.DataFrame or tuple[pd.Series |
        pd.DataFrame, VisualizationResult
    ]
        - Single column input: returns a pandas.Series with outlier values at
          original indices.
        - Multi-column input: returns a pandas.DataFrame with outlier values
          and NaN elsewhere.
        - If `get_boxplot=True`, returns a tuple with outliers and the boxplot
          figure.

    Warns
    -----
    UserWarning
        If any features have constant or nearly constant values,
        as outliers cannot exist in such series.

    Raises
    ------
    ValueError
        If nan_policy='raise' and missing values (NaN/null) are found in the data.
        If `iqr_factor` is negative.

    Notes
    -----
    - An outlier is defined as a value below `Q1 - iqr_factor * IQR` or above
    `Q3 + iqr_factor * IQR`.
    - For 2D inputs, each column is processed independently.
    - The boxplot is always generated only for the first column if
    `get_boxplot=True`.

    Examples
    --------
    >>> import pandas as pd
    >>> import explorica.data_quality as dq
    >>> s = pd.Series([1, 2, 2, 3, 13, 1, 100, 90])
    >>> outliers = dq.detect_iqr(s, iqr_factor=1.5)
    >>> print(outliers)
    6    100
    7     90
    dtype: int64
    >>> # Multi-column DataFrame
    >>> df = pd.DataFrame({"A": [1, 2, 3, 50], "B": [5, 6, 7, 8]})
    >>> outliers_df = dq.detect_iqr(df)
    >>> print(outliers_df)
        A    B
    3  50.0  NaN
    >>> # With boxplot and custom styling
    >>> outliers, plot_result = dq.detect_iqr(s, get_boxplot=True,
    ...     boxplot_kws={"style": "whitegrid", "figsize": (8, 4)})
    >>> plot_result.figure.show()
    """
    df = convert_dataframe(data)
    df = handle_nan(df, nan_policy, supported_policy=("drop", "raise"))
    _validate_zero_variance(df)
    if iqr_factor < 0:
        raise ValueError("iqr_factor must be non negative number.")
    # Compute IQR bounds & detect outliers
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    a = q1 - iqr * iqr_factor
    b = q3 + iqr * iqr_factor
    mask = (df < a) | (df > b)
    outliers = df[mask].dropna(how="all")
    # If input is 1D and only one column, optionally return Series
    if get_boxplot:
        return (
            outliers.squeeze(axis=1),
            boxplot(df.iloc[:, 0], **(boxplot_kws or {})),
        )
    return outliers.squeeze(axis=1)


def detect_zscore(
    data: Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Sequence[Any]],
    threshold: Optional[float] = 2.0,
) -> pd.Series | pd.DataFrame:
    """
    Detects outliers in a numerical series using the Z-score method.

    Parameters
    ----------
    data : Sequence[float] or Sequence[Sequence[float]]
        Input numeric data. Can be a 1D sequence or a 2D structure
        convertible to a pandas DataFrame.
    threshold : float, default=2.5
        Z-score threshold for identifying outliers. Values with an
        absolute Z-score greater than this threshold are considered
        outliers.

    Returns
    -------
    pd.Series or pd.DataFrame
        If the input contains a single feature, returns a Series of
        outlier values. If multiple features are provided, returns a
        DataFrame with NaN for non-outlier positions.

    Warns
    -----
    UserWarning
        If any features have constant or nearly constant values,
        as outliers cannot exist in such series.

    Raises
    ------
    ValueError
        If `threshold` is not positive or the input contains NaN values.
        If the input contains any NaN values.

    Notes
    -----
    The Z-score method identifies outliers based on their standardized
    distance from the mean:
        Z = (x - μ) / σ
    where μ is the mean and σ is the standard deviation.

    Examples
    --------
    >>> import pandas as pd
    >>> import explorica.data_quality as data_quality
    ...
    >>> s = pd.Series([1, 2, 2, 3, 13, 1, 1000, 2, -1000])
    >>> outliers = data_quality.detect_zscore(s, threshold=1)
    >>> print(outliers)
    6    1000
    8   -1000
    dtype: int64
    >>> # Returns a Series with outlier values and their original indices
    """
    df = convert_dataframe(data)
    validate_array_not_contains_nan(
        df, err_msg=_errors["array_contains_nans_f"].format("data")
    )
    if threshold <= 0:
        raise ValueError("threshold must belong to (0, inf]")
    _validate_zero_variance(df)
    mask = np.abs((df - df.mean()) / df.std()) > threshold
    outliers = df[mask].dropna(how="all")
    if outliers.shape[1] == 1:
        return outliers.iloc[:, 0]
    return outliers


def _validate_zero_variance(data: pd.DataFrame, threshold: float = 1.0e-10):
    """
    Checks for features with zero or near-zero variance and emits a warning.

    Parameters
    ----------
    data : pd.DataFrame
        Input numeric data to validate.
    threshold : float, default=1e-10
        Threshold below which variance is considered effectively zero.

    Warns
    -----
    UserWarning
        If any features have constant or nearly constant values,
        as outliers cannot exist in such series.
    """
    near_zero = data.var() < threshold
    if near_zero.any():
        if near_zero.shape[0] == 1:
            cols = data.columns[near_zero].astype("str")[0]
        else:
            cols = ", ".join(list(data.columns[near_zero].astype("str")))
        wrn_msg = _warns["data_quality"]["outliers_on_zero_variance_f"].format(cols)
        warnings.warn(wrn_msg, UserWarning)
