"""
The ``data_quality_handler`` module provides a unified facade for handling
data preprocessing, feature engineering, and outlier detection tasks.
It serves as a single entry point to access methods from different
submodules of the ``explorica.data_quality`` package.

This design allows for both modular usage (importing specific classes)
and simplified usage through the facade.

Available classes
-----------------
DataPreprocessor
    Provides functionality for handling missing values, managing categories,
    and detecting constant features.

FeatureEngineer
    Includes encoding methods, binning utilities, and other feature
    transformation techniques.

OutlierHandler
    Provides detection and handling of outliers, as well as statistical
    distribution descriptions.

Available functions

- DataPreprocessor:
    - ``check_columns_uniqueness``
    - ``get_missing``
    - ``drop_missing``
    - ``get_constant_features``
    - ``get_categories``
    - ``set_categories``

- FeatureEngineer:
    - ``freq_encode``
    - ``ordinal_encode``
    - ``bin_numeric``

- OutlierHandler:
    - ``replace_outliers``
    - ``remove_outliers``
    - ``detect_iqr``
    - ``detect_zscore``
    - ``get_skewness``
    - ``get_kurtosis``
    - ``describe_distributions``

Examples
--------
>>> from seaborn import load_dataset
>>> from explorica.data_quality import describe_distributions

>>> df = load_dataset("titanic")

>>> df = df.select_dtypes("number").dropna()
>>> print(describe_distributions(df))

| feature  | is_normal | desc                          | skewness | kurtosis  |
|----------|-----------|-------------------------------|----------|-----------|
| survived | False     | right-skewed, low-pitched     | 0.382140 | -1.853969 |
| pclass   | False     | left-skewed, low-pitched      | -0.467558| -1.418028 |
| age      | False     | right-skewed                  | 0.388290 | 0.168637  |
| sibsp    | False     | right-skewed, high-pitched    | 2.514280 | 6.987321  |
| parch    | False     | right-skewed, high-pitched    | 2.613409 | 8.782859  |
| fare     | False     | right-skewed, high-pitched    | 4.643848 | 30.699725 |

Notes
-----
This is a high-level interface. For more complex or specialized
workflows, use the underlying classes directly:

- ``DataPreprocessor``
- ``FeatureEngineer``
- ``OutlierHandler``
"""

import json
from typing import Sequence

import numpy as np
import pandas as pd

from explorica._utils import ConvertUtils as cutils
from explorica._utils import ValidationUtils as vutils
from explorica._utils import read_messages
from explorica.interactions import detect_multicollinearity

from .data_preprocessor import DataPreprocessor as dp
from .feature_engineer import FeatureEngineer as fe
from .outlier_handler import OutlierHandler as oh


class DataQualityHandler:
    _warns = read_messages()["warns"]
    _errors = read_messages()["errors"]

    @staticmethod
    def get_summary(
        dataset: Sequence[Sequence],
        return_as: str = "dataframe",
        auto_round=True,
        round_digits=4,
        **kwargs,
    ) -> pd.DataFrame | dict:
        """
        Compute a data-quality summary for a dataset.

        This function computes a set of descriptive and data-quality indicators for each
        feature in `dataset` and returns the result as a pandas DataFrame (multi-index
        columns) or as a JSON-friendly nested dict.

        Parameters
        ----------
        dataset : Sequence[Sequence]
            Input dataset. Supported input types: list/tuple, numpy.ndarray, deque,
            dict-of-sequences, pandas.Series, pandas.DataFrame. Internally converted
            to pandas.DataFrame (see `cutils.convert_dataframe`).
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
        Minimal usage (DataFrame return):

        >>> import pandas as pd
        >>> from explorica.data_quality import DataQualityHandler

        >>> df = pd.DataFrame({
        ...     "x1": [1,2,3],
        ...     "x2": [2,4,6]
        ... })
        >>> summary = DataQualityHandler.get_summary(df, return_as="dataframe")
        >>> print(summary)

            | nans                        | duplicates                      | ... |
            |-----------------------------|---------------------------------|-----|
            | count_of_nans | pct_of_nans | count_of_unique | pct_of_unique | ... |
            |---------------|-------------|-----------------|---------------|-----|
         x1 | 0             | 0.0         | 3               | 1.0           | ... |
         x2 | 0             | 0.0         | 3               | 1.0           | ... |


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
                    summary_formats["nested_dict"] = (
                        DataQualityHandler._get_nested_dict(
                            summary_formats["tuple_dict"], not_number_columns
                        )
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

        vutils.validate_string_flag(
            return_as.lower(),
            {"dataframe", "df", "dict", "json"},
            err_msg=DataQualityHandler._errors["usupported_method_f"].format(
                return_as, {"dataframe", "dict"}
            ),
        )

        dfs = {"full": cutils.convert_dataframe(dataset)}
        if 0 in dfs["full"].shape:
            raise ValueError(
                "The input 'dataset' is empty. Please provide non-empty data."
            )
        vutils.validate_array_not_contains_nan(
            dfs["full"],
            err_msg=DataQualityHandler._errors["array_contains_nans_f"].format(
                "dataset"
            ),
        )
        dfs["number"] = dfs["full"].select_dtypes("number")

        reports = {}
        reports["nans"] = get_missing(dfs["full"]).astype("float64").to_dict()
        reports["distributions"] = describe_distributions(dfs["number"])
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
            ("duplicates", "quasi_constant_pct"): get_constant_features(dfs["full"])[
                "top_value_ratio"
            ].to_dict(),
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
            summary_formats["nested_dict"] = DataQualityHandler._get_nested_dict(
                summary_formats["tuple_dict"], not_number_columns
            )
            result = summary_formats["nested_dict"]
        else:
            result = pd.DataFrame(summary_formats["tuple_dict"])

        if params["directory"] is not None:
            save(params["directory"])

        return result

    @staticmethod
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


get_summary = DataQualityHandler.get_summary

data_preprocessing = dp
check_columns_uniqueness = dp.check_columns_uniqueness
get_missing = dp.get_missing
drop_missing = dp.drop_missing
get_constant_features = dp.get_constant_features
get_categories = dp.get_categories
set_categories = dp.set_categories

feature_engineering = fe
freq_encode = fe.freq_encode
ordinal_encode = fe.ordinal_encode
bin_numeric = fe.bin_numeric

outliers = oh
replace_outliers = oh.replace_outliers
remove_outliers = oh.remove_outliers
detect_iqr = oh.detect_iqr
detect_zscore = oh.detect_zscore
get_skewness = oh.get_skewness
get_kurtosis = oh.get_kurtosis
describe_distributions = oh.describe_distributions
