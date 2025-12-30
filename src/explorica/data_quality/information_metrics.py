"""
Module for information-theoretic metrics used in data quality assessment.

This module provides utilities for quantifying uncertainty, variability,
and information content in datasets. Currently, it implements Shannon
entropy as a measure of feature uncertainty. Future extensions may include
divergence measures (e.g., KL divergence) and cross-entropy for
comparing distributions.

Classes
-------
InformationMetrics
    Provides static methods for computing information metrics such as
    Shannon entropy.

Notes
-----
- Currently, only Shannon entropy is implemented. Other metrics may be added
  in future releases.

Examples
--------
>>> import pandas as pd
>>> import explorica.data_quality as data_quality
...
>>> data = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [1, 1, 1, 2, 2]})
>>> data_quality.get_entropy(data)
A    2.321928
B    0.970951
dtype: float64
"""

from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from explorica._utils import (
    convert_dataframe,
    convert_from_alias,
    handle_nan,
    read_config,
    validate_string_flag,
    validate_unique_column_names,
)


def get_entropy(
    data: Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Sequence[Any]],
    method: str = "shannon",
    nan_policy: Literal["drop", "raise", "include"] = "drop",
) -> float | pd.Series:
    """
    Computes the Shannon entropy of the input data.

    Shannon entropy is a measure of uncertainty or randomness in a dataset.
    For a single feature, it quantifies how evenly the values are distributed.
    Lower values indicate more predictability (potentially constant or
    quasi-constant features), while higher values indicate more variability or
    diversity.

    Parameters
    ----------
    data : Sequence[float] | Sequence[Sequence[float]] |
           Mapping[str, Sequence[Number]]
        Numeric input data. Can be 1D (sequence of numbers),
        2D (sequence of sequences), or a mapping of column names to sequences.
    method : str, default="shannon"
        Entropy calculation method. Currently, only "shannon" is supported.
        Other methods (e.g., differential entropy) may be added in future releases.
        Entropy is calculated as:

            H(x) = - sum(w_i * log_2(w_i))

        where w_i is the relative frequency of each unique element of the sample x.
    nan_policy : {'drop', 'raise', 'include'}, default='drop'
        Policy for handling NaN values in input data:
        - 'raise' : raise ValueError if any NaNs are present in `data`.
        - 'drop'  : drop rows (axis=0) containing NaNs before computation. This
                    does not drop entire columns.
        - 'include' : treat NaN as a valid value and include them in computations.

    Returns
    -------
    float or pd.Series
        - If input is 1D, returns a float representing the Shannon entropy.
        - If input is 2D or dict, returns a pd.Series indexed by column names.

    Raises
    ------
    ValueError
        - If column names are not unique (in case of dict or DataFrame input)
        - If `method` is not supported

    Notes
    -----
    - NaN values are included in the computation as a distinct category.

    Examples
    --------
    >>> import pandas as pd
    >>> import explorica.data_quality as data_quality
    ...
    >>> data = pd.DataFrame({"A": [1, 1, 2, 2], "B": [1, 2, 3, 4]})
    >>> data_quality.get_entropy(data)
    A    1.0
    B    2.0
    dtype: float64
    ...
    >>> data = [1, 1, 1, 1, 1, 1]
    >>> data_quality.get_entropy(data)
    np.float64(0.0)
    """
    errors = read_config("messages")["errors"]
    df = convert_dataframe(data)
    supported = {"policy": ["drop", "raise", "include"], "methods": ["shannon"]}

    nan_policy = convert_from_alias(nan_policy, supported["policy"])
    df = handle_nan(df, nan_policy, supported["policy"])

    validate_unique_column_names(
        df,
        err_msg=errors["duplicate_keys_f"].format("data", "data"),
    )
    if df.empty:
        return pd.Series([np.nan] * df.shape[1], index=df.columns)

    method = convert_from_alias(method, supported["methods"])

    validate_string_flag(
        method,
        supported["methods"],
        err_msg=errors["unsupported_method_f"].format(method, supported["methods"]),
    )

    entropy = df.apply(
        (
            lambda col: -np.sum(
                (col.value_counts(dropna=False) / df.shape[0])
                * np.log2(col.value_counts(dropna=False) / df.shape[0])
            )
        ),
        axis=0,
    )

    entropy = entropy.replace(np.float64(-0.0), np.float64(0.0))
    entropy = entropy.squeeze()
    return entropy
