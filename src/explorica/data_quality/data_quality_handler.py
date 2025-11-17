"""
Data quality analysis utilities and composite functions.

This module provides high-level functions that combine multiple data quality
checks into unified reports and workflows. These composite functions integrate
functionality from data preprocessing, feature engineering, and outlier detection
submodules to provide comprehensive data quality analysis.

Methods
-------
get_summary(data, include_missing=True, include_constant=True,
            include_categorical=True, return_type='dataframe')
    Compute comprehensive data quality indicators for a dataset.

Notes
-----
These functions build upon the core functionality provided by:
- ``DataPreprocessing``
- ``FeatureEngineering``
- ``outliers``

Examples
--------
Basic data quality summary
~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> import pandas as pd
>>> import explorica.data_quality as data_quality

>>> df = pd.DataFrame({
...     "x1": [1,2,3],
...     "x2": [2,4,6]
... })
>>> summary = data_quality.get_summary(df, return_as="dataframe")
>>> print(summary)
            nans                  duplicates
   count_of_nans pct_of_nans count_of_unique pct_of_unique quasi_constant_pct
x1           0.0         0.0             3.0           1.0             0.3333
x2           0.0         0.0             3.0           1.0             0.3333
...
"""

import json
from typing import Sequence

import numpy as np
import pandas as pd

from explorica._utils import (
    convert_dataframe,
    read_config,
    validate_array_not_contains_nan,
    validate_string_flag,
)
from explorica.interactions import detect_multicollinearity

from .data_preprocessing import DataPreprocessing
from .outliers import DistributionMetrics


def get_summary(
    data: Sequence[Sequence],
    return_as: str = "dataframe",
    auto_round=True,
    round_digits=4,
    **kwargs,
) -> pd.DataFrame | dict:
    """
    Compute a data-quality summary for a dataset.

    This function computes a set of descriptive and data-quality indicators for each
    feature in `data` and returns the result as a pandas DataFrame (multi-index
    columns) or as a JSON-friendly nested dict.

    Parameters
    ----------
    data : Sequence | Sequence[Sequence] |
               Mapping[str, Sequence]
        Input data. Can be 1D,
        2D (sequence of sequences), or a mapping of column names to sequences.
    return_as : {'dataframe', 'dict'}, optional, default='dataframe'
        Output format. If 'dataframe' (default) returns `pandas.DataFrame` with
        columns arranged as MultiIndex (group, metric). If 'dict' or 'json' returns
        a nested Python dict (JSON-friendly) of the form:
        {"group": {"metric": {"feature": value, ...}, ...}, ...}.
    auto_round : bool, optional, default=True
        If True numeric values are rounded for human-friendly output.
    round_digits : int, optional, default=4
        Number of decimal digits used when auto_round=True.
    threshold_vif : float, default=10
        Threshold for the Variance Inflation Factor (VIF). A numeric feature
        is considered multicollinear if its VIF is greater than or equal
        to this value.
    directory : str, optional, default=None
        Path (including filename) where the summary should be saved. If only a
        filename is provided, the file will be created in the current working
    directory. The file extension controls the format and how the summary is
        written:
        - '.xlsx': saved with MultiIndex columns (preserves grouping visually);
        - '.json': saved as a nested dict (group → metric → {feature: value});
        - '.csv': saved as flattened keys joined by ':' (e.g., "stats:mean").

        Example
        -------
        # Save as Excel in current directory
        summary = DataQualityHandler.get_summary(df, directory="summary.xlsx")

        # Save as CSV in a folder
        summary = DataQualityHandler.get_summary(df,
            directory="reports/summary.csv")

    Returns
    -------
    pandas.DataFrame or dict
        If `return_as` in {'dataframe', 'df'} returns a DataFrame with MultiIndex
        columns (level 0 = group, level 1 = metric) and index = feature names.

        If `return_as` in {'dict', 'json'} returns nested dict:

        {
        "nans": {
            "count_of_nans": {"feat1": 0, "feat2": 1, ...},
            "pct_of_nans": {"feat1": 0.0, ...}
        },
        "stats": {
            "mean": {"feat1": 1.234, ...},
            "median": {...},
            "mode": {"feat1": "A", "feat2": "B", ...},
            ...
        },
        "multicollinearity": {
            "VIF": {"feat1": 2.34, "feat2": 5.21, ...},
            "is_multicollinearity": {"feat1": 0, ...}
        },
        ...
        }

    Schema: groups and metrics
    --------------------------
    Brief list of returned metrics (group -> metric) and short meaning:
    - **nans**
        - `count_of_nans` - number of missing values per feature
        - `pct_of_nans` - fraction of missing values per feature (0..1)
    - **duplicates**
        - `count_of_unique` - number of unique values per feature
        - `pct_of_unique` - fraction of unique values per feature (0..1)
        - `quasi_constant_pct` - top value ratio (quasi-constant score)
    - **distribution**
        - `is_normal` - 0/1 binary flag (0/1). Distribution is considered
           approximately normal if both absolute skewness and
           excess kurtosis are ≤ 0.25.
        - `desc` - human-readable qualitative description of the distribution
          shape: "normal" if the above heuristic holds, otherwise one of
          "left-skewed", "right-skewed", "low-pitched", etc., depending on sample
          skewness/kurtosis.
        - `skewness` - sample skewness
        - `kurtosis` - sample excess kurtosis
    - **stats**
        - `mean`, `std`, `median` - central tendency and dispersion
        - `mode` - most frequent value per feature.
          - In DataFrame/Excel outputs: preserved in its original type.
          - In JSON/dict outputs: always converted to string for safe serialization.
        - `count_of_modes` - number of mode values found
    - **multicollinearity**
        - `VIF` - Variance Inflation Factor per numeric feature (numeric-only).
        - `is_multicollinearity` - 0/1 flag per feature, equal to 1 if
          the feature’s VIF ≥ `threshold_vif` (default=10), else 0.

    Notes
    -----
    - VIF is computed for numeric features only. Non-numeric features will not have
      VIF values;
    - When saving to JSON/nested dict, numeric values (e.g. NumPy scalars) are cast
      to native Python int/float, and non-numeric values (e.g. `mode`) are stored
      as strings to guarantee JSON safety.
    - In DataFrame/Excel outputs, original types are preserved.
    - In CSV outputs, values are stringified implicitly by pandas
      during `to_csv()`.
    - CSV output uses flattened column names joined by ':' - this improves
      portability
      but loses MultiIndex structure. To read flattened CSV back and restore
      MultiIndex use: `cols = df.columns.str.split(':', expand=True)` then set
      `df.columns = pd.MultiIndex.from_frame(cols)`.

    Raises
    ------
    ValueError
    - If invalid `return_as` values.

    Examples
    --------
    Minimal usage (DataFrame return)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    >>> import pandas as pd
    >>> from explorica.data_quality import DataQualityHandler

    >>> df = pd.DataFrame({
    ...     "x1": [1,2,3],
    ...     "x2": [2,4,6]
    ... })
    >>> summary = DataQualityHandler.get_summary(df, return_as="dataframe")
    >>> print(summary)
                nans                  duplicates
    count_of_nans pct_of_nans count_of_unique pct_of_unique quasi_constant_pct
    x1           0.0         0.0             3.0           1.0             0.3333
    x2           0.0         0.0             3.0           1.0             0.3333
    ...


    Usage with save to json:
    >>> import os
    >>> import pandas as pd
    >>> from explorica.data_quality import DataQualityHandler

    >>> df = pd.DataFrame({
    ...     "x1": [1,2,3],
    ...     "x2": [2,4,6]
    ... })
    >>> summary = DataQualityHandler.get_summary(df, return_as="df",
            directory="summary.json")
    >>> print(os.listdir("."))
        ['summary.json', 'main.py']
    """
    params = {"directory": None, "threshold_vif": 10, **kwargs}
    errors = read_config("messages")["errors"]

    def save(directory):
        save_format = directory.split(".")[-1]
        if save_format == "xlsx":
            pd.DataFrame(summary_formats["tuple_dict"]).to_excel(
                directory, engine="openpyxl"
            )
        if save_format == "csv":
            summary_formats["flatten_dict"] = {
                ":".join(k): v for k, v in summary_formats["tuple_dict"].items()
            }
            pd.DataFrame(summary_formats["flatten_dict"]).to_csv(directory)
        if save_format == "json":
            if summary_formats["nested_dict"] is None:
                summary_formats["nested_dict"] = _get_nested_dict(
                    summary_formats["tuple_dict"], not_number_columns
                )
            with open(directory, "w", encoding="utf-8") as f:
                json.dump(
                    summary_formats["nested_dict"], f, ensure_ascii=False, indent=4
                )

    not_number_columns = {
        ("distribution", "is_normal"),
        ("distribution", "desc"),
        ("stats", "mode"),
    }

    validate_string_flag(
        return_as.lower(),
        {"dataframe", "df", "dict", "json"},
        err_msg=errors["unsupported_method_f"].format(return_as, {"dataframe", "dict"}),
    )

    dfs = {"full": convert_dataframe(data)}
    if 0 in dfs["full"].shape:
        raise ValueError("The input 'dataset' is empty. Please provide non-empty data.")
    validate_array_not_contains_nan(
        dfs["full"],
        err_msg=errors["array_contains_nans_f"].format("dataset"),
    )
    dfs["number"] = dfs["full"].select_dtypes("number")

    reports = {}
    reports["nans"] = (
        DataPreprocessing.get_missing(dfs["full"]).astype("float64").to_dict()
    )
    reports["distributions"] = DistributionMetrics.describe_distributions(dfs["number"])
    reports["distributions"]["is_normal"] = reports["distributions"][
        "is_normal"
    ].astype("float64")
    reports["distributions"] = reports["distributions"].to_dict()
    reports["multicollinearity"] = detect_multicollinearity(
        dfs["number"],
        method="vif",
        return_as="dict",
        variance_inflation_threshold=params["threshold_vif"],
    )
    reports["mode"] = dfs["full"].mode().iloc[0]
    if pd.api.types.is_numeric_dtype(reports["mode"]):
        reports["mode"] = reports["mode"].astype("float64")
    else:
        reports["mode"] = reports["mode"].astype("object")
    summary_formats = {}
    summary_formats["nested_dict"] = None
    summary_formats["flatten_dict"] = None
    summary_formats["tuple_dict"] = {
        ("nans", "count_of_nans"): reports["nans"]["count_of_nans"],
        ("nans", "pct_of_nans"): reports["nans"]["pct_of_nans"],
        ("duplicates", "count_of_unique"): dfs["full"]
        .nunique()
        .astype("float64")
        .to_dict(),
        ("duplicates", "pct_of_unique"): (
            dfs["full"].nunique() / dfs["full"].shape[0]
        ).to_dict(),
        ("duplicates", "quasi_constant_pct"): DataPreprocessing.get_constant_features(
            dfs["full"]
        )["top_value_ratio"].to_dict(),
        ("distribution", "is_normal"): reports["distributions"]["is_normal"],
        ("distribution", "desc"): reports["distributions"]["desc"],
        ("distribution", "skewness"): reports["distributions"]["skewness"],
        ("distribution", "kurtosis"): reports["distributions"]["kurtosis"],
        ("stats", "mean"): dfs["number"].mean().to_dict(),
        ("stats", "std"): dfs["number"].std().to_dict(),
        ("stats", "median"): dfs["number"].median().to_dict(),
        ("stats", "mode"): reports["mode"].to_dict(),
        ("stats", "count_of_modes"): dfs["full"]
        .mode()
        .count()
        .astype("float64")
        .to_dict(),
        ("multicollinearity", "is_multicollinearity"): reports["multicollinearity"][
            "multicollinearity"
        ],
        ("multicollinearity", "VIF"): reports["multicollinearity"]["VIF"],
    }

    if auto_round:
        for key in list(summary_formats["tuple_dict"].copy()):
            if key in not_number_columns:
                continue
            for feat in list(summary_formats["tuple_dict"][key].copy()):
                summary_formats["tuple_dict"][key][feat] = np.round(
                    summary_formats["tuple_dict"][key][feat], round_digits
                )
    if return_as in {"json", "dict", "mapping"}:
        summary_formats["nested_dict"] = _get_nested_dict(
            summary_formats["tuple_dict"], not_number_columns
        )
        result = summary_formats["nested_dict"]
    else:
        result = pd.DataFrame(summary_formats["tuple_dict"])

    if params["directory"] is not None:
        save(params["directory"])

    return result


def _get_nested_dict(d: dict, not_number_columns: Sequence = None):
    """
    transforms flat-dict to serializable nested dict.
    not_number_columns: Columns that will be converted to str for
    serializability
    """
    nested_dict = {}
    for key, value in d.items():
        if key[0] not in nested_dict:
            nested_dict[key[0]] = {}
        serializable_value = value.copy()
        if key not in not_number_columns:
            for feature, num in value.items():
                serializable_value[feature] = float(num)
        elif key == ("stats", "mode"):
            for feature, mode in value.items():
                serializable_value[feature] = str(mode)
        nested_dict[key[0]][key[1]] = serializable_value
    return nested_dict
