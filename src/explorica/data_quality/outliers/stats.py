"""
Module for statistical metrics and distribution analysis.

This module defines tools for computing standardized statistical moments
(skewness and excess kurtosis) and for describing the shape of numeric
distributions.

Classes
-------
DistributionMetrics
    Provides methods to compute skewness, excess kurtosis, and describe
    the shape of numeric sequences or collections of sequences.

Examples
--------
>>> import pandas as pd
>>> from explorica.data_quality.outliers import DistributionMetrics
>>> df = pd.DataFrame({
...     "x": [1, 2, 3, 4, 5],
...     "y": [2, 2, 2, 2, 2]
... })
>>> print(DistributionMetrics.get_skewness(df, method="general"))
UserWarning: Columns with near-zero variance: ['y']. Their skewness will be set to 0.0.

x    0.0
y    0.0
dtype: float64
"""

import warnings
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from explorica._utils import (
    convert_dataframe,
    read_config,
    validate_array_not_contains_nan,
    validate_string_flag,
)


class DistributionMetrics:
    """
    Provides utilities to compute distribution shape metrics for numeric data.

    This class implements methods for computing skewness, excess kurtosis,
    and for describing the overall shape of one or multiple numeric distributions.
    It supports 1D sequences, 2D sequences, and mappings of feature names to sequences.

    Methods
    -------
    get_skewness(data, method='general')
        Compute the skewness (third standardized moment) of numeric sequences.

    get_kurtosis(data, method='general')
        Compute the excess kurtosis (fourth standardized moment minus 3)
        of numeric sequences.

    describe_distributions(data, threshold_skewness=0.25, threshold_kurtosis=0.25,
                           return_as='dataframe', **kwargs)
        Compute skewness and excess kurtosis for one or more distributions,
        classify each distribution shape, and optionally return a DataFrame or dict
        summarizing the results.

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.data_quality.outliers import DistributionMetrics
    >>> df = pd.DataFrame({
    ...     "x": [1, 2, 3, 4, 5],
    ...     "y": [2, 2, 2, 2, 2]
    ... })
    >>> print(DistributionMetrics.get_skewness(df, method="general"))
    UserWarning: Columns with near-zero variance: ['y'].
    Their skewness will be set to 0.0.

    x    0.0
    y    0.0
    dtype: float64
    """

    _warns = read_config("messages")["warns"]
    _errors = read_config("messages")["errors"]

    @staticmethod
    def get_skewness(
        data: (
            Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Sequence[float]]
        ),
        method: str = "general",
    ) -> float | pd.Series:
        """
        Compute the skewness (third standardized moment) of a numeric sequence.

        Parameters
        ----------
        data : Sequence[float] | Sequence[Sequence[float]] |
               Mapping[str, Sequence[Number]]
            Numeric input data. Can be 1D (sequence of numbers),
            2D (sequence of sequences), or a mapping of column names to sequences.
        method : str, {"general", "sample"}, default "general"
            Method to compute skewness:
            - "general": standard formula skew = m3 / σ**3
            - "sample": corrected for sample size, skew = m3 / (var * n/(n-1))**1.5

        Returns
        -------
        float or pd.Series
            Skewness of input data. Returns a single float if input is 1D or a Series
            of skewness values (one per column) if input is 2D or a mapping.

        Raises
        ------
        ValueError
            If input contains NaNs.
            If provided method is not supported.

        Warnings
        --------
        UserWarning
            If any features have variance < 1e-8.

        Notes
        -----
        For numerical stability, variance close to zero is treated as zero.

        Examples
        --------
        >>> from explorica.data_quality import get_skewness
        ...
        >>> print(get_skewness({"a": [1,2,3], "b": [2,3,4]}, method="sample"))
        a    0.0
        b    0.0
        dtype: float64
        """
        df = convert_dataframe(data).astype(np.float64)
        if method in {"sigma", "population"}:
            method = "general"
        if method == "s":
            method = "sample"

        supported_methods = {"general", "sample"}
        validate_array_not_contains_nan(
            df,
            err_msg=DistributionMetrics._errors["array_contains_nans_f"].format("data"),
        )
        validate_string_flag(
            method,
            supported_methods,
            DistributionMetrics._errors["unsupported_method_f"].format(
                method, supported_methods
            ),
        )

        n = df.shape[0]
        if n == 0:
            return np.nan
        dfree = n - 1 if method == "sample" else n

        mean = np.sum(df, axis=0) / n
        var = np.sum((df - mean) ** 2, axis=0) / dfree
        zero_var = np.isclose(var, 0.0, atol=1e-8)
        if zero_var.any():
            msg = f"""Columns with near-zero variance: {
                list(df.columns[zero_var])}. Their skewness will be set to 0.0."""
            warnings.warn(msg, UserWarning)

        m_3 = np.sum((df - mean) ** 3, axis=0) / n
        q_3 = np.sqrt(var) ** 3
        skewness = pd.Series(dtype=np.float64, index=df.columns)
        skewness[~zero_var] = m_3[~zero_var] / q_3[~zero_var]
        skewness[zero_var] = 0.0
        return skewness.squeeze()

    @staticmethod
    def get_kurtosis(
        data: (
            Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Sequence[float]]
        ),
        method: str = "general",
    ) -> float:
        """
        Compute the **excess kurtosis** (fourth standardized moment minus 3)
        of a numeric sequence.

        Parameters
        ----------
        data : Sequence[float] | Sequence[Sequence[float]] |
               Mapping[str, Sequence[Number]]
            Numeric input data. Can be 1D (sequence of numbers),
            2D (sequence of sequences), or a mapping of column names to sequences.
        method : {"general", "sample"}, default "general"
            Method to compute excess kurtosis:
            - "general": population excess kurtosis, computed as m4 / σ**4 - 3
            - "sample": biased sample excess kurtosis,
                        computed as m4 / (var * n/(n-1))**2 - 3
            Note that this function does not yet implement the unbiased
            Fisher correction for sample kurtosis.

        Returns
        -------
        pd.Series | float
            Excess kurtosis value of the input data.
            0.0 for normal distribution, positive values indicate
            heavier tails, negative values indicate lighter tails.
            If the sample variance is close to zero, the excess
            kurtosis value will be replaced by np.nan

        Raises
        ------
        ValueError
            If input contains NaNs.
            If provided method is not supported.

        Warnings
        --------
        UserWarning
            If any features have variance < 1e-8.
        """
        if method.lower() in {"sigma", "population", "general"}:
            method = "general"
        if method.lower() in {"s", "sample"}:
            method = "sample"
        supported_methods = {"general", "sample"}
        df = convert_dataframe(data).astype(np.float64)

        validate_array_not_contains_nan(
            df,
            err_msg=DistributionMetrics._errors["array_contains_nans_f"].format("data"),
        )
        validate_string_flag(
            method,
            supported_methods,
            DistributionMetrics._errors["unsupported_method_f"].format(
                method, supported_methods
            ),
        )

        n = df.shape[0]
        if n == 0:
            return np.nan
        dfree = n - 1 if method == "sample" else n

        mean = np.sum(df, axis=0) / n
        var = np.sum((df - mean) ** 2, axis=0) / dfree
        zero_var = np.isclose(var, 0.0, atol=1e-8)
        if zero_var.any():
            msg = f"""Columns with near-zero variance: {
                list(df.columns[zero_var])
                }. Their excess kurtosis will be set to np.nan."""
            warnings.warn(msg, UserWarning)
        m_4 = np.sum((df - mean) ** 4, axis=0) / n
        q_4 = var**2
        kurtosis = pd.Series(dtype=np.float64, index=df.columns)
        kurtosis[~zero_var] = m_4[~zero_var] / q_4[~zero_var] - 3
        kurtosis[zero_var] = np.nan
        return kurtosis.squeeze()

    @staticmethod
    def describe_distributions(
        data: Union[
            Sequence[Sequence[float]], pd.DataFrame, Mapping[str, Sequence[float]]
        ],
        threshold_skewness: Optional[float] = 0.25,
        threshold_kurtosis: Optional[float] = 0.25,
        return_as: Optional[str] = "dataframe",
        **kwargs,
    ) -> Union[pd.DataFrame | dict]:
        """
        Describe shape (skewness / kurtosis) of one or multiple numeric distributions.

        The function computes skewness and excess kurtosis for each 1-D sequence
        in `data` and classifies the distribution shape according to the
        provided absolute thresholds. Distributions whose absolute skewness and
        absolute excess kurtosis are both less than or equal to the corresponding
        thresholds are considered "normal".

        Parameters
        ----------
        data : {Sequence[Sequence[Number]],
                   pandas.DataFrame, Mapping[str, Sequence[Number]]}
            Input container with one or more numeric sequences (distributions).
            Supported forms:
            - 2D sequence (e.g. list of lists, list/array of 1D arrays): each inner
              sequence represents one distribution;
            - ``pandas.DataFrame``: each column is treated as a separate
              distribution;
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
            Absolute excess kurtosis threshold.
            If ``abs(kurtosis) <= threshold_kurtosis``
            the distribution is considered not kurtotic (with respect to this
            threshold).
            Note: this function uses **excess kurtosis** (kurtosis - 3), so a normal
            distribution is approximately 0.
        return_as : {'dataframe', 'dict'}, optional, default='dataframe'
            Output format:
            - ``'dataframe'`` — return a ``pandas.DataFrame`` with columns:
              ``['is_normal', 'desc', 'skewness', 'kurtosis']``. If input was a
              DataFrame or Mapping the index will reflect column names / mapping keys.
            - ``'dict'`` — return a dict with keys ``'is_normal'``, ``'desc'``,
              ``'skewness'``, ``'kurtosis'`` and list-like values in the same order
              as the features.
        method_skewness: {general, sample}, default "general"
            Method to compute skewness. It is used in `data_quality.get_skewness`,
            See `data_quality.get_skewness` for full details.
        method_kurtosis: {general, sample}, default "general"
            Method to compute kurtosis. It is used in `data_quality.get_kurtosis`,
            See `data_quality.get_kurtosis` for full details.

        Returns
        -------
        pandas.DataFrame or dict
            Either a DataFrame (if return_as='dataframe`) or a dict (if
            return_as='dict') containing the following entries per feature:
            - ``is_normal`` (int) - 1 if both |skewness| and |kurtosis| are
              within thresholds.
            - ``desc`` (str) - human-friendly description, one of:
              ``'normal'``, ``'left-skewed'``, ``'right-skewed'``,
              ``'low-pitched'`` (platykurtic) and/or ``'high-pitched'`` (leptokurtic).
              Multiple descriptors are joined by a comma (e.g. ``'right-skewed,
              high-pitched'``).
            - skewness (float) - skewness (third standardized moment).
            - kurtosis (float) - excess kurtosis (fourth standardized moment
              minus 3).

        Raises
        ------
        ValueError
            If ``return_as`` is not in ``{'dataframe', 'dict'}``.

        Notes
        -----
        - The function expects numeric, one-dimensional sequences for each
          distribution.
          If mapping values are heterogeneous (different lengths / non-sequences) the
          behavior may be unexpected — prefer passing a DataFrame or a well-formed
          Mapping.
        - Threshold checks are **inclusive**: equality to threshold counts as within.
        - For programmatic consumption prefer ``return_as='dataframe'``
          (tabular form). The ``dict`` form returns lists of values aligned to the
          feature order (not a transposed mapping of feature -> single-structure per
          feature).

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
        x                   1                         normal  0.012345  0.023456
        y                   0           right-skewed, high-pitched  1.234567  3.456789

        >>> d = DataPreprocessor.describe_distributions(df, return_as='dict')
        >>> list(d.keys())
        ['is_normal', 'desc', 'skewness', 'kurtosis']
        """
        params = {
            "method_skewness": "general",
            "method_kurtosis": "general",
            "desc_nan_kurtosis_policy": "extremely-high",
            **kwargs,
        }

        metrics = {
            "skewness": {
                "func": DistributionMetrics.get_skewness,
                "params": {"method": params["method_skewness"]},
            },
            "kurtosis": {
                "func": DistributionMetrics.get_kurtosis,
                "params": {"method": params["method_kurtosis"]},
            },
        }
        supported_return_types = {"dataframe", "dict"}

        labels = {}
        labels["left_skew"] = "left-skewed"
        labels["right_skew"] = "right-skewed"
        labels["high_pitch"] = "high-pitched"
        labels["low_pitch"] = "low-pitched"
        labels["nan_kurt"] = params["desc_nan_kurtosis_policy"]

        if return_as.lower() in {"df", "dataframe"}:
            return_as = "dataframe"
        elif return_as.lower() in {"dict", "dictionary", "mapping"}:
            return_as = "dict"

        def check_is_normal(df: pd.DataFrame):
            condition = (np.abs(df["skewness"]) <= threshold_skewness) & (
                np.abs(df["kurtosis"]) <= threshold_kurtosis
            )
            return pd.Series(np.where(condition, 1, 0), index=df.index)

        def get_describe(df: pd.DataFrame, is_normal: pd.Series) -> pd.Series:
            desc = pd.Series(index=df.index, dtype=str)
            desc[is_normal == 1] = "normal"

            # describe skewness
            desc.loc[df["skewness"] < -threshold_skewness] = labels["left_skew"]
            desc.loc[df["skewness"] > threshold_skewness] = labels["right_skew"]

            # describe excess kurtosis
            desc.loc[
                desc.notna() & (df["kurtosis"] < -threshold_kurtosis)
            ] += f", {labels['low_pitch']}"
            desc.loc[desc.isna() & (df["kurtosis"] < -threshold_kurtosis)] = labels[
                "low_pitch"
            ]

            desc.loc[
                desc.notna() & (df["kurtosis"] > threshold_kurtosis)
            ] += f", {labels['high_pitch']}"
            desc.loc[desc.isna() & (df["kurtosis"] > threshold_kurtosis)] = labels[
                "high_pitch"
            ]
            desc.loc[desc.notna() & df["kurtosis"].isna()] += f", {labels['nan_kurt']}"
            desc.loc[desc.isna() & df["kurtosis"].isna()] = labels["nan_kurt"]
            return desc

        if return_as not in supported_return_types:
            raise ValueError(
                f"Unsupported return type '{return_as}',"
                f"please, choose from {supported_return_types}"
            )

        # processing of input sequence
        df = convert_dataframe(data)

        validate_array_not_contains_nan(
            df, DistributionMetrics._errors["array_contains_nans_f"].format("data")
        )

        # collection of descriptive information
        output = pd.DataFrame(index=df.columns)
        output["skewness"] = metrics["skewness"]["func"](
            df, **metrics["skewness"]["params"]
        )
        output["kurtosis"] = metrics["kurtosis"]["func"](
            df, **metrics["kurtosis"]["params"]
        )
        output["is_normal"] = check_is_normal(output)
        output["desc"] = get_describe(output, output["is_normal"])
        if return_as == "dict":
            return output.to_dict()
        if return_as == "dataframe":
            return output
