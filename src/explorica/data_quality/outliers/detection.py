import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from explorica import DataVisualizer
from explorica._utils import ConvertUtils as cutils
from explorica._utils import ValidationUtils as vutils
from explorica._utils import read_messages


class DetectionMethods:
    dv = DataVisualizer()
    _warns = read_messages()["warns"]
    _errors = read_messages()["errors"]

    @classmethod
    def detect_iqr(
        cls,
        data: Union[Sequence[float] | Sequence[Sequence[float]]],
        iqr_factor: float = 1.5,
        show_boxplot: Optional[bool] = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Detects outliers in a numerical series using the Interquartile Range
        (IQR) method.

        Parameters
        ----------
        data : Sequence[float]|Sequence[Sequence[float]]
            A numeric sequence (1D) or a sequence of sequences (2D) to analyze.
            Will be converted to a pandas DataFrame internally. Each inner sequence
            is treated as a separate column.
        iqr_factor : float, default 1.5
            Multiplier for the Interquartile Range used to define outlier bounds.
        show_boxplot : bool
            If True, displays a boxplot to visualize the outliers. Currently supports
            only the first column if the input is 2D. Default is False.

        Returns
        -------
        pd.Series or pd.DataFrame
            - If the input contains a single feature (one column),
              returns a pandas.Series with outlier values only;
              the index is the original row index filtered to outlier positions.
            - If the input is 2D (multiple features), returns a pandas.DataFrame
              where cells contain outlier values at their original indices and
              non-outlier cells are ``NaN``.

        Warns
        -----
        UserWarning
            If any features have constant or nearly constant values,
            as outliers cannot exist in such series.

        Raises
        ------
        ValueError
            If the input contains any NaN values.

        Notes
        -----
        - An outlier is defined as a value below
          Q1 - iqr_factor * IQR or above Q3 + iqr_factor * IQR.
        - For 2D inputs, each column is processed independently.
        """
        df = cutils.convert_dataframe(data)

        vutils.validate_array_not_contains_nan(
            df, err_msg=DetectionMethods._errors["array_contains_nans_f"].format("data")
        )
        DetectionMethods._validate_zero_variance(df)

        # Compute IQR bounds & detect outliers
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        a = q1 - iqr * iqr_factor
        b = q3 + iqr * iqr_factor
        mask = (df < a) | (df > b)
        outliers = df[mask].dropna(how="all")

        if show_boxplot:
            cls.dv.boxplot(df.iloc[:, 0])

        # If input is 1D and only one column, optionally return Series
        if outliers.shape[1] == 1:
            return outliers.iloc[:, 0]
        return outliers

    @staticmethod
    def detect_zscore(
        data: Sequence[float] | Sequence[Sequence[float]],
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
        """
        df = cutils.convert_dataframe(data)

        vutils.validate_array_not_contains_nan(
            df, err_msg=DetectionMethods._errors["array_contains_nans_f"].format("data")
        )
        if threshold <= 0:
            raise ValueError("threshold must belong to (0, inf]")
        DetectionMethods._validate_zero_variance(df)

        mask = np.abs((df - df.mean()) / df.std()) > threshold
        outliers = df[mask].dropna(how="all")
        if outliers.shape[1] == 1:
            return outliers.iloc[:, 0]
        return outliers

    @staticmethod
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
                cols = data.columns[near_zero]
            else:
                cols = ", ".join(data.columns[near_zero])
            wrn_msg = DetectionMethods._warns["data_quality"][
                "outliers_on_zero_variance_f"
            ].format(cols)
            warnings.warn(wrn_msg, UserWarning)
