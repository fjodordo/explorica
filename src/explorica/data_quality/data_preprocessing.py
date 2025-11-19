"""
Data preprocessing utilities for exploratory data analysis (EDA).

This module provides the `DataPreprocessor` class, which contains static methods
for inspecting and cleaning datasets before analysis or modeling.
The available tools include:
- Detection of duplicate column combinations
- Missing value analysis and selective removal
- Detection of constant and quasi-constant features
- Identification of categorical features
- Conversion of suitable columns to `pandas.Categorical` type for memory optimization

Notes
-----
All methods are implemented as `@staticmethod`, so the class does not maintain any
state.
"""

import logging
from numbers import Number
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from explorica._utils import (
    convert_dataframe,
    convert_from_alias,
    convert_params_for_keys,
    handle_nan,
    natural_number,
    read_config,
    temp_log_level,
    validate_string_flag,
    validate_unique_column_names,
)

from .information_metrics import get_entropy

logger = logging.getLogger(__name__)


class DataPreprocessing:
    """
    A collection of static methods for common data preprocessing tasks.

    This class provides utility functions for dataset inspection, cleaning, and
    optimization, especially useful in the exploratory data analysis (EDA) stage.

    Methods
    -------
    check_columns_uniqueness(dataset: pd.DataFrame,
                             max_combination_size: int, ...) -> pd.DataFrame
        Check for duplicate rows across all combinations of features up to a specified
        size. Useful for identifying unique feature sets or repeated patterns.

    get_missing(dataset: pd.DataFrame) -> pd.DataFrame
        Return the number and proportion of missing (NaN) values per column.

    drop_missing(dataset: pd.DataFrame, threshold_pct: float = 0.01,
        threshold_abs: int = None, return_report: bool = False) -> pd.DataFrame
        Drop rows with missing values only in columns where
        the NaN ratio does not exceed the given threshold.

    get_constant_features(dataset: pd.DataFrame, quasi_constant_threshold: float,
        include_nan: bool = False) -> pd.DataFrame
        Identify constant and quasi-constant features
        based on the frequency of the most common value.
        Returns a DataFrame with columns: `is_constant` and `top_value_ratio`.

    get_categories(dataset: pd.DataFrame, threshold, ...) -> pd.DataFrame
        Identify columns that can be considered categorical based on the number of
        unique values and optional inclusion of numerical, boolean, and datetime
        columns.

    set_categories(dataset: pd.DataFrame, threshold, ...) -> pd.DataFrame
        Convert identified categorical columns to `pandas.Categorical` dtype.
        This can significantly reduce memory usage and may improve performance
        when working with repeated values.
    """

    _errors = read_config("messages")["errors"]

    @staticmethod
    def get_missing(
        data: Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Sequence[Any]],
        ascending=None,
        round_digits=None,
    ):
        """
        Calculate the number and percentage of missing (NaN) values for each column.

        Parameters
        ----------
        data : Sequence[float] | Sequence[Sequence[float]] |
               Mapping[str, Sequence[Number]]
            Numeric input data. Can be 1D (sequence of numbers),
            2D (sequence of sequences), or a mapping of column names to sequences.
        ascending : bool, optional
            If specified, sorts the result by the ``count_of_nans`` column.
            - If True, sorts in ascending order (fewest missing values first).
            - If False, sorts in descending order (most missing values first).
            - If None (default), no sorting is performed.
        round_digits : int, optional
            Number of decimal places to round the ``pct_of_nans`` values to.
            - Must be a non-negative integer (``x >= 0``).
            - If ``None`` (default), no rounding is applied.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the following columns:
            - `count_of_nans` : int
            Number of NaN values in each column.
            - `pct_of_nans` : float
            Proportion of NaN values in each column (0.0 to 1.0).

        Raises
        ------
        ValueError
            If `data` has keys and they are not unique.
            If `round_digits` is not a non-negative integer.

        Notes
        -----
        - The `pct_of_nans` values are calculated as the fraction of missing values
        relative to the total number of rows in the dataset.
        - Useful for quickly identifying columns with high proportions of missing data
        before applying data cleaning or imputation.

        Examples
        --------
        >>> import pandas as pd
        >>> import explorica.data_quality as data_quality
        ...
        >>> df = pd.DataFrame({"A": [1, 2, pd.NA, np.nan, 5, 6, 7],
        ...                    "B": [7, None, 5, 4, 3, 2, 1]})
        >>> print(data_quality.get_missing(df, round_digits=4))
           count_of_nans  pct_of_nans
        A              2       0.2857
        B              1       0.1429
        """
        df = convert_dataframe(data)
        validate_unique_column_names(
            df,
            err_msg=DataPreprocessing._errors["duplicate_keys_f"].format(
                "data", "data"
            ),
        )

        nan_count = df.isna().sum()
        nan_ratio = df.isna().mean()
        missing_values = pd.DataFrame(
            {"count_of_nans": nan_count, "pct_of_nans": nan_ratio},
            index=nan_count.index,
        )
        missing_values["count_of_nans"] = missing_values["count_of_nans"].astype(int)
        missing_values["pct_of_nans"] = missing_values["pct_of_nans"].astype(float)

        if ascending is not None:
            missing_values = missing_values.sort_values(
                by="count_of_nans", ascending=ascending
            )

        if round_digits is not None:
            if not float(round_digits).is_integer() or round_digits < 0:
                raise ValueError("'round_digits' must be a non-negative integer")
            round_digits = int(round_digits)
            missing_values["pct_of_nans"] = np.round(
                missing_values["pct_of_nans"], round_digits
            )

        return missing_values

    @staticmethod
    def drop_missing(
        data: Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Sequence[Any]],
        axis: Optional[int] = 0,
        threshold_pct: Optional[float] = 0.05,
        threshold_abs: Optional[int] = None,
        verbose: Optional[bool] = False,
    ) -> pd.DataFrame:
        """
        Drops rows or columns containing NaNs according to a specified threshold.

        This function removes rows (axis=0) or columns (axis=1) that contain NaN values
        in columns whose proportion of missing values is below (`axis=0`) or above
        (`axis=1`) the specified threshold. Threshold can be specified as a proportion
        (`threshold_pct`) or an absolute number (`threshold_abs`). Absolute threshold,
        if provided, overrides the proportion threshold.

        Parameters
        ----------
        data : Sequence[float] | Sequence[Sequence[float]] |
               Mapping[str, Sequence[Number]]
            Numeric input data. Can be 1D (sequence of numbers),
            2D (sequence of sequences), or a mapping of column names to sequences.
        axis : int, optional, default=0
            Axis along which to remove NaNs:
            - 0 : drop rows with NaNs in columns under the threshold,
            - 1 : drop columns with NaNs above the threshold.
        threshold_pct : float, optional, default=0.05
            The maximum allowed proportion of NaNs for a feature to be retained.
            - When `axis=0` (row-wise deletion): rows are removed
              if the proportion of NaNs
              in their columns exceeds this threshold.
            - When `axis=1` (column-wise deletion): columns are removed if the
              proportion of NaNs exceeds this threshold.
              Ignored if `threshold_abs` is provided.
        threshold_abs : int, optional
            The maximum allowed absolute number of NaNs for a feature to be retained.
            - When `axis=0`: rows are removed if the number of NaNs per column
              exceeds this threshold.
            - When `axis=1`: columns are removed if the number of NaNs
              exceeds this threshold.
            Overrides `threshold_pct` if provided.
        verbose : bool, optional, default=False
            If True, logs detailed information about the operation including:
            - number of rows or columns removed,
            - columns affected,
            - original and resulting DataFrame shape.

        Returns
        -------
        pd.DataFrame
            DataFrame after dropping rows or columns according to the threshold.

        Raises
        ------
        ValueError
            If `data` has keys and they are not unique.
            If `threshold_abs` is not a non-negative integer.
            If `threshold_abs` is greater than 'data' length
            If `threshold_pct` not in [0, 1]
            If `axis` is not 0 or 1.

        Examples
        --------
        >>> import pandas as pd
        >>> import explorica.data_quality as data_quality

        >>> df = pd.DataFrame({"A": [1,2,3,4,5,np.nan],
        ...                    "B": [1,2,3,4,5,6],
        ...                    "C": [np.nan, 2, np.nan, np.nan, np.nan, np.nan]})
        >>> # only removes rows if NaN is less than 2 per feature
        >>> print(data_quality.drop_missing(df, axis=0, threshold_abs=2))
            A  B    C
        0  1.0  1  NaN
        1  2.0  2  2.0
        2  3.0  3  NaN
        3  4.0  4  NaN
        4  5.0  5  NaN

        >>> # only removes columns if more than 20% of values per feature are NaN
        >>> print(data_quality.drop_missing(df, axis=1, threshold_pct=0.2))
            A  B
        0  1.0  1
        1  2.0  2
        2  3.0  3
        3  4.0  4
        4  5.0  5
        5  NaN  6
        """
        df = convert_dataframe(data)
        validate_unique_column_names(
            df, DataPreprocessing._errors["duplicate_keys_f"].format("data", "data")
        )
        if axis not in (0, 1):
            raise ValueError(f"axis must be 0 or 1, provided value is {axis}")

        # Compute threshold
        if threshold_abs is not None:
            if not (float(threshold_abs).is_integer() and threshold_abs >= 0):
                raise ValueError("'threshold_abs' must be a non-negative integer")
            if threshold_abs > df.shape[0]:
                raise ValueError(
                    (
                        "The specified 'threshold_abs' is greater than "
                        "'data' length. filtering cannot be performed"
                    )
                )
            threshold = threshold_abs / np.max([df.shape[0], 1])
        else:
            if threshold_pct < 0 or threshold_pct > 1:
                raise ValueError("'threshold_pct' must be non negative")
            threshold = threshold_pct

        result_df = df.copy()
        nans = DataPreprocessing.get_missing(result_df)
        nan_pct = nans["pct_of_nans"]
        if axis == 0:
            # Identify columns where NaN proportion is under the threshold
            cols_to_remove = nan_pct[
                (nan_pct < threshold) & (nan_pct != 0)
            ].index.to_list()
            # Drop rows with NaNs in those columns
            result_df.dropna(subset=cols_to_remove, inplace=True)
            before = df.shape[0]
            after = result_df.shape[0]
            log_msg = {
                "function": "drop_missing",
                "axis": 0,
                "threshold": threshold,
                "removed_rows": before - after,
                "affected_columns": cols_to_remove,
                "length": f"{before}->{after}",
            }
            if verbose:
                with temp_log_level(logger, logging.INFO):
                    logger.info(log_msg)
        elif axis == 1:
            # Identify columns where NaN proportion is above the threshold
            cols_to_remove = nan_pct[
                (nan_pct > threshold) & (nan_pct != 0)
            ].index.to_list()
            # Drop columns with NaNs in those columns
            cols = result_df.columns[~(result_df.columns.isin(cols_to_remove))]
            result_df = result_df[cols]
            before = df.shape[1]
            after = result_df.shape[1]
            log_msg = {
                "function": "drop_missing",
                "axis": 1,
                "threshold": threshold,
                "removed_columns": before - after,
                "affected_columns": cols_to_remove,
                "width": f"{before}->{after}",
            }
            if verbose:
                with temp_log_level(logger, logging.INFO):
                    logger.info(log_msg)

        return result_df

    @staticmethod
    def get_constant_features(
        data: Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Sequence[Any]],
        method: str = "top_value_ratio",
        threshold: Optional[float] = 1.0,
        nan_policy: str | Literal["drop", "raise", "include"] = "drop",
    ) -> pd.DataFrame:
        """
        Identifies constant and quasi-constant features in the dataset.

        Parameters
        ----------
        data : Sequence[float] | Sequence[Sequence[float]] |
               Mapping[str, Sequence[Number]]
            Numeric input data. Can be 1D (sequence of numbers),
            2D (sequence of sequences), or a mapping of column names to sequences.
        method: str, default 'top_value_ratio'
            Metric used to detect constant features:
            - "top_value_ratio": proportion of the most frequent value.
            - "non_uniqueness": 1 - number of unique values / total count.
            - "entropy": Shannon entropy of the feature.
        nan_policy : str | Literal['drop', 'raise', 'include', 'drop_columns'],
                     default='drop'
            Policy for handling NaN values in input data:
            - 'raise' : raise ValueError if any NaNs are present in `data`.
            - 'drop'  : drop rows (axis=0) containing NaNs before computation. This
                        does **not** drop entire columns.
            - 'include' : treat NaN as a valid value and include them in computations.
        threshold : float, default=1.0
            Non-negative threshold value in the range [0, +∞).
            Decision boundary for each method:
            - For "top_value_ratio" or "non_uniqueness":
              values >= threshold are flagged constant.
            - For "entropy": values <= threshold are flagged constant.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by column names with:
            - 'is_const': bool flag if column is (quasi-)constant
            - 'top_value_ratio': proportion of the most frequent value

        Raises
        ------
        ValueError
            If an unsupported `method` or `nan_policy` is provided.
            If input contains duplicate column names.
            If `threshold is negative.`

        Examples
        --------
        Basic usage
        ~~~~~~~~~~~~
        Demonstrates a simple use case with the default ``top_value_ratio`` method,
        which identifies constant or quasi-constant features based on the most frequent
        value ratio.

        >>> import pandas as pd
        >>> import numpy as np
        >>> import explorica.data_quality as data_quality
        ...
        >>> data = [[1, 3, 3, 3, 3, 6], [1, 2, 3, 4, 5, 5]]
        >>> print(data_quality.get_constant_features(
        ...     data, method="top_value", threshold=0.5))
        top_value_ratio  is_const
        0         0.666667       1.0
        1         0.333333       0.0

        Entropy-based threshold interpretation
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Illustrates how an entropy threshold can be interpreted as a fraction of the
        maximum information capacity (in bits) for each feature. This approach allows
        defining thresholds relative to the diversity of feature values.

        >>> import pandas as pd
        >>> import numpy as np
        >>> import explorica.data_quality as data_quality
        ...
        >>> data = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 6],
        ...                      "B": [0, 0, 0, 0, 0, 1, 1]})
        >>> thresh = 0.7
        >>> thresh_bits_dim = thresh * np.log2(data.nunique())
        >>> print(thresh_bits_dim)
        A    1.809474
        B    0.700000
        dtype: float64
        ...
        >>> data_quality.get_constant_features(
        ...     data, method="entropy", threshold = thresh_bits_dim.mean())
            entropy	is_const
        A	2.521641	0.0
        B	0.863121	1.0
        """
        if not isinstance(threshold, Number):
            raise TypeError("'threshold' must be a numeric type (int or float).")
        if threshold < 0:
            raise ValueError(
                f"Invalid value for 'threshold': {threshold}. "
                "Expected a non-negative float in the range [0, +inf)."
            )

        df = convert_dataframe(data)
        validate_unique_column_names(
            df, DataPreprocessing._errors["duplicate_keys_f"].format("data", "data")
        )

        supported_methods = {
            "top_value_ratio": lambda df: df.apply(
                (lambda col: col.value_counts(dropna=False).max() / df.shape[0]), axis=0
            ),
            "non_uniqueness": lambda df: 1 - df.nunique(dropna=False) / df.shape[0],
            "entropy": get_entropy,
        }

        df = handle_nan(df, nan_policy, supported_policy={"drop", "raise", "include"})
        method = convert_from_alias(method, supported_methods)

        validate_string_flag(
            method,
            supported_methods,
            DataPreprocessing._errors["unsupported_method_f"].format(
                method, set(supported_methods)
            ),
        )
        if df.empty:
            return pd.DataFrame(np.nan, columns=[method, "is_const"], index=df.columns)

        report = pd.DataFrame(pd.Series(supported_methods[method](df), name="metric"))
        if method in ("top_value_ratio", "non_uniqueness"):
            report.loc[report["metric"] >= threshold, "is_const"] = 1
        elif method == "entropy":
            report.loc[report["metric"] <= threshold, "is_const"] = 1
        report["is_const"] = report["is_const"].fillna(0)
        report = report.rename(
            {"metric": method},
            axis=1,
        )

        return report

    @staticmethod
    def get_categorical_features(
        data: Sequence[Any] | Sequence[Sequence[Any]] | Mapping[str, Sequence[Any]],
        threshold: Optional[int | Sequence[int] | Mapping[str, int]] = 30,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Identify categorical features in a DataFrame based on unique value counts
        and optionally on data type filters.

        Parameters
        ----------
        data : Sequence[Any] | Sequence[Sequence[Any]] |
               Mapping[str, Sequence[Any]]
            Input data. Can be 1D (sequence of numbers),
            2D (sequence of sequences), or a mapping of column names to sequences.
        threshold : int, Sequence[int] or Mapping[str, int], optional, default=30
            Maximum number of unique values allowed for a column to be considered
            categorical. If a mapping is provided, values are applied per column.
            Scalar values are broadcast to all columns;
            sequences or mappings are aligned by column name.
        sign_bin : bool, default=False
            If True, append an `is_binary` column to the result,
            marking columns with exactly two unique values.
        sign_const : bool, default=False
            If True, append an `is_constant` column to the result,
            marking columns with only one unique value.
        include_number : bool, default=False
            Include numeric (`number`) columns that satisfy the `threshold`
        include_int : bool, default=False
            Include integer (`int`) columns that satisfy the `threshold`.
        include_str : bool, default=False
            Include string (`object`) columns that satisfy the `threshold`.
        include_bool : bool, default=False
            Include boolean columns.
        include_datetime : bool, default=False
            Include datetime columns.
        include_bin : bool, default=False
            Include binary columns (exactly two unique values).
        include_const : bool, default=False
            Include constant columns (exactly one unique value).
        include_all : bool, default=False
            Disable dtype filtering; only `threshold` is applied.
        include : Iterable[str], default={"object"}
            Explicit set of dtype aliases to include
            (e.g. `{"object", "number"}` or `{"int", "bin"}`).
            The parameter has the highest priority among inclusion rules:
            1. Explicit `include` argument (user-defined)
            2. Flag parameters (e.g., `include_int`, `include_str`, etc.)
            3. Default value `{"object"}`
            If `include` is provided directly, all flags are ignored.
        nan_policy : str | Literal['drop', 'raise', 'include'],
                     default='drop'
            Policy for handling NaN values in input data:
            - 'raise' : raise ValueError if any NaNs are present in `data`.
            - 'drop'  : drop rows (axis=0) containing NaNs before computation. This
                        does **not** drop entire columns.
            - 'include' : treat NaN as a valid value and include them in computations.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by column names with:
            - `categories_count` : number of unique values in each column
            - `is_category` : flag for categorical columns
            - `is_binary` : (optional) flag for binary columns
            - `is_constant` : (optional) flag for constant columns

        Raises
        ------
        ValueError
            If input data contains duplicate column names or invalid
            `nan_policy`.
        TypeError
            If `threshold` is not scalar, list, or mapping convertible
            to per-column limits.

        Notes
        -----
        - The function supports combined filtering:
          first by unique value count (`threshold`), then by dtype matching.
        - Internal helper functions `_filter_standard_dtypes`, `_filter_bin_const`
          and `_filter_categories` provide modular filtering logic.
        - The original data are **not modified**.
        - Compatible with `get_constant_features` for constant detection.


        Examples
        --------
        >>> import pandas as pd
        >>> import seaborn as sns
        >>> import explorica.data_quality as data_quality
        ...
        >>> df = sns.load_dataset("titanic")
        >>> # marks as a category string and integer columns
        >>> # with 4 or fewer unique objects
        >>> print(data_quality.get_categorical_features(
        ...     df, threshold=4, include={"str", "int"}))
                    categories_count  is_category
        survived                    2            1
        pclass                      3            1
        sex                         2            1
        age                        63            0
        sibsp                       4            0
        parch                       4            0
        fare                       93            0
        embarked                    3            1
        class                       3            0
        who                         3            1
        adult_male                  2            0
        deck                        7            0
        embark_town                 3            1
        alive                       2            1
        alone                       2            0


        >>> df["constant_feature"] = 0
        >>> # additionally signs binary and constant features
        >>> print(data_quality.get_categorical_features(
        ...     df, threshold=10, sign_bin=True, sign_const=True))
                        categories_count  is_category  is_binary  is_constant
        survived                         2            0          1            0
        pclass                           3            0          0            0
        sex                              2            1          1            0
        age                             63            0          0            0
        sibsp                            4            0          0            0
        parch                            4            0          0            0
        fare                            93            0          0            0
        embarked                         3            1          0            0
        ...
        embark_town                      3            1          0            0
        alive                            2            1          1            0
        alone                            2            0          1            0
        constant_feature                 1            0          0            1
        """

        params = {
            "nan_policy": "drop",
            "include_number": False,
            "include_int": False,
            "include_bin": False,
            "include_const": False,
            "include_datetime": False,
            "include_str": False,
            "include_bool": False,
            "include_all": False,
            "include": {},
            "sign_bin": False,
            "sign_const": False,
            **kwargs,
        }

        # preprocess input params
        if params["include"] is None:
            params["include"] = set()
        else:
            params["include"] = set(params["include"])
        if len(params["include"]) == 0:
            mapper = {
                "include_number": {"number"},
                "include_int": {"int"},
                "include_bin": {"bin"},
                "include_datetime": {"datetime64"},
                "include_str": {"object"},
                "include_bool": {"bool"},
                "include_const": {"const"},
            }
            for include_flag, include_dtype in mapper.items():
                if params[include_flag]:
                    params["include"] = params["include"] | include_dtype

        # default values
        if len(params["include"]) == 0:
            params["include"] = {"object"}

        for elem in params["include"]:
            if not isinstance(elem, str):
                raise TypeError(
                    f"Invalid types in 'include' parameter."
                    f"Expected strings, but got: {elem}."
                )

        def _filter_standard_dtypes(df, included_dtypes):
            """
            Select columns of standard pandas dtypes matching `included_dtypes`.

            Parameters
            ----------
            df : pd.DataFrame
                Input dataframe.
            included_dtypes : set
                Target dtype names to include (e.g. {"number", "object"}).

            Returns
            -------
            Index
                Columns of matching dtypes.
            """
            include = {
                "number",
                "int",
                "bool",
                "object",
                "datetime64",
            } & included_dtypes
            if len(include) == 0:
                return []
            return list(df.select_dtypes(include=include).columns)

        def _filter_bin_const(df):
            """
            Identify binary and constant columns in the dataframe.

            Parameters
            ----------
            df : pd.DataFrame
                Input dataframe.

            Returns
            -------
            dict
                - `bin_cols`: columns with exactly two unique values.
                - `const_cols`: columns with exactly one unique value.
            """
            result = {"bin_cols": [], "const_cols": []}
            result["bin_cols"].extend(
                [col for col in df.columns if df[col].nunique() == 2]
            )
            result["const_cols"].extend(
                [col for col in df.columns if df[col].nunique() == 1]
            )
            return result

        def _filter_categories(df, threshold, included_dtypes, **kwargs):
            """
            Apply combined filtering by threshold and dtype.

            Parameters
            ----------
            df : pd.DataFrame
                Input dataframe.
            threshold : dict[str, int]
                Mapping of column → unique value limit.
            included_dtypes : Iterable
                Dtypes to include in categorical detection.
            skip_dtypes_filter : bool, optional
                If True, skip dtype filtering.
            sign_bin : bool, optional
                If True, include `is_binary` flag in the result.
            sign_const : bool, optional
                If True, include `is_constant` flag in the result.

            Returns
            -------
            pd.DataFrame
                A dataframe with flags:
                - optionally `is_binary` and `is_constant`
            """
            params = {
                "skip_dtypes_filter": False,
                "sign_bin": False,
                "sign_const": False,
                **kwargs,
            }

            # threshold filter
            filtered_by_thresh = []
            for col in df.columns:

                if df[col].nunique(dropna=False) <= threshold[col]:
                    filtered_by_thresh.append(col)

            # dtypes filter
            bin_const_cols = _filter_bin_const(df)
            if params["skip_dtypes_filter"]:
                filtered_by_dtypes = df.columns
            else:
                filtered_by_dtypes = []
                filtered_by_dtypes.extend(_filter_standard_dtypes(df, included_dtypes))
                if "bin" in included_dtypes:
                    filtered_by_dtypes.extend(bin_const_cols["bin_cols"])
                if "const" in included_dtypes:
                    filtered_by_dtypes.extend(bin_const_cols["const_cols"])
            filtered = set(filtered_by_dtypes) & set(filtered_by_thresh)

            is_category = {col: 1 if col in filtered else 0 for col in df.columns}
            result = pd.DataFrame({"is_category": is_category})
            if params["sign_bin"]:
                result["is_binary"] = {
                    col: 1 if col in bin_const_cols["bin_cols"] else 0
                    for col in df.columns
                }
            if params["sign_const"]:
                result["is_constant"] = {
                    col: 1 if col in bin_const_cols["const_cols"] else 0
                    for col in df.columns
                }
            return result

        df = convert_dataframe(data)
        validate_unique_column_names(
            df, err_msg=DataPreprocessing._errors["duplicate_keys_f"]
        )
        df = handle_nan(df, params["nan_policy"], {"drop", "raise", "include"})

        # threshold: scalar -> broadcast to all columns,
        # dict/list -> mapped by column name
        threshold = convert_params_for_keys(
            threshold, df.columns, "threshold", validate_dtype_as=natural_number
        )
        include = {
            convert_from_alias(dtype, path="get_categorical_features")
            for dtype in params["include"]
        }
        for elem in include:
            validate_string_flag(
                elem,
                {"object", "number", "int", "datetime64", "bin", "bool", "const"},
                err_msg=(
                    f"Unsupported data type flag '{elem}' in 'include'. "
                    f"Choose from: 'object', 'number', 'int', 'datetime64',"
                    f"'bin', 'bool', 'const'."
                ),
            )

        result = pd.DataFrame({"categories_count": df.nunique(dropna=False)})
        result = pd.concat(
            [
                result,
                _filter_categories(
                    df,
                    threshold,
                    include,
                    skip_dtypes_filter=params["include_all"],
                    sign_bin=params["sign_bin"],
                    sign_const=params["sign_const"],
                ),
            ],
            axis=1,
        )

        return result

    @staticmethod
    def set_categorical(
        data: Sequence[Any] | Sequence[Sequence[Any]] | Mapping[str, Sequence[Any]],
        threshold: Optional[int | Sequence[int] | Mapping[str, int]] = 30,
        nan_policy: str | Literal["drop", "raise", "include"] = "drop",
        verbose: Optional[bool] = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Convert eligible columns to Pandas `category` dtype for memory optimization
        and improved performance in certain operations.

        Parameters
        ----------
        data : Sequence[float] | Sequence[Sequence[float]] |
               Mapping[str, Sequence[Number]]
            Input data. Can be 1D,
            2D (sequence of sequences), or a mapping of column names to sequences.
        threshold : int, Sequence[int] or Mapping[str, int], optional, default=30
            Maximum number of unique values allowed for a column to be considered
            categorical. If a mapping is provided, values are applied per column.
            Scalar values are broadcast to all columns;
            sequences or mappings are aligned by column name.
        include_number : bool, default=False
            Include numeric (`number`) columns that satisfy the `threshold`
        include_int : bool, default=False
            Include integer (`int`) columns that satisfy the `threshold`.
        include_str : bool, default=False
            Include string (`object`) columns that satisfy the `threshold`.
        include_bool : bool, default=False
            Include boolean columns.
        include_datetime : bool, default=False
            Include datetime columns.
        include_bin : bool, default=False
            Include binary columns (exactly two unique values).
        include_const : bool, default=False
            Include constant columns (exactly one unique value).
        include_all : bool, default=False
            Disable dtype filtering; only `threshold` is applied.
        include : Iterable[str], default={"object"}
            Explicit set of dtype aliases to include
            (e.g. `{"object", "number"}` or `{"int", "bin"}`).
            The parameter has the highest priority among inclusion rules:
            1. Explicit `include` argument (user-defined)
            2. Flag parameters (e.g., `include_int`, `include_str`, etc.)
            3. Default value `{"object"}`
            If `include` is provided directly, all flags are ignored.
        nan_policy : str | Literal['drop', 'raise', 'include'],
                     default='drop'
            Policy for handling NaN values in input data:
            - 'raise' : raise ValueError if any NaNs are present in `data`.
            - 'drop'  : drop rows (axis=0) containing NaNs before computation. This
                        does **not** drop entire columns.
            - 'include' : treat NaN as a valid value and include them in computations.
        verbose : bool, optional, default=False
            If True, logs detailed information about the operation including:
            - count and names of affected columns.

        Returns
        -------
        pd.DataFrame
            A copy of the original DataFrame with selected columns converted to
            `category` dtype.

        Raises
        ------
        Exception
            Propagates exceptions from `get_categorical_features` for parameter
            validation errors. See `get_categorical_features` documentation for
            specific error conditions.

        Notes
        -----
        - Converting to `category` can significantly reduce memory usage,
        especially for string/object columns with many repeated values.
        - `category` stores integer codes (`int8`/`int16`) and a category mapping,
        making comparisons and filtering faster than for `object` dtype.
        - For numeric columns, memory savings may be smaller, but grouping and filtering
        can still be faster.
        - The original DataFrame is not modified — a copy is returned.

        Examples
        --------
        Basic usage example
        ~~~~~~~~~~~~~~~~~~~
        >>> import pandas as pd
        >>> import explorica.data_quality as data_quality
        ...
        >>> df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8],
        ...                    "B": ["A", "A", "B", "C", "A", "B", "C", "A"],
        ...                    "C": [1, 0, 1, 0, 1, 1, 1, 1]})
        >>> df = data_quality.set_categorical(df, include_bin=True, include_str=True)
        >>> df.info()
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 8 entries, 0 to 7
        Data columns (total 3 columns):
        #   Column  Non-Null Count  Dtype
        ---  ------  --------------  -----
        0   A       8 non-null      int64
        1   B       8 non-null      category
        2   C       8 non-null      category
        dtypes: category(2), int64(1)
        memory usage: 464.0 bytes

        Memory usage reducing example
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        >>> df = pd.read_csv('titanic.csv')
        >>> df.info()
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 891 entries, 0 to 890
        Data columns (total 12 columns):
        #   Column       Non-Null Count  Dtype
        ---  ------       --------------  -----
        0   PassengerId  891 non-null    int64
        1   Survived     891 non-null    int64
        2   Pclass       891 non-null    int64
        3   Name         891 non-null    object
        4   Sex          891 non-null    object
        5   Age          714 non-null    float64
        6   SibSp        891 non-null    int64
        7   Parch        891 non-null    int64
        8   Ticket       891 non-null    object
        9   Fare         891 non-null    float64
        10  Cabin        204 non-null    object
        11  Embarked     889 non-null    object
        dtypes: float64(2), int64(5), object(5)
        memory usage: 83.7+ KB
        >>> data_quality.set_categorical(df).info()
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 891 entries, 0 to 890
        Data columns (total 12 columns):
        #   Column       Non-Null Count  Dtype
        ---  ------       --------------  -----
        0   PassengerId  891 non-null    int64
        1   Survived     891 non-null    int64
        2   Pclass       891 non-null    int64
        3   Name         891 non-null    object
        4   Sex          891 non-null    category
        5   Age          714 non-null    float64
        6   SibSp        891 non-null    int64
        7   Parch        891 non-null    int64
        8   Ticket       891 non-null    object
        9   Fare         891 non-null    float64
        10  Cabin        204 non-null    object
        11  Embarked     889 non-null    category
        dtypes: category(2), float64(2), int64(5), object(3)
        memory usage: 71.7+ KB
        """
        include_kwargs = {
            "include_number": kwargs.get("include_number"),
            "include_int": kwargs.get("include_int"),
            "include_bin": kwargs.get("include_bin"),
            "include_const": kwargs.get("include_const"),
            "include_datetime": kwargs.get("include_datetime"),
            "include_str": kwargs.get("include_str"),
            "include_bool": kwargs.get("include_bool"),
            "include_all": kwargs.get("include_all"),
            "include": kwargs.get("include"),
        }

        df = convert_dataframe(data)

        categorical_features = DataPreprocessing.get_categorical_features(
            df, threshold=threshold, nan_policy=nan_policy, **include_kwargs
        )

        columns_to_set = categorical_features[
            categorical_features["is_category"] == 1
        ].index
        df[columns_to_set] = df[columns_to_set].astype("category")

        if verbose:
            log_msg = {
                "function": "set_categorical",
                "converted_columns": list(columns_to_set),
                "conversion_count": len(columns_to_set),
            }
            with temp_log_level(logger, logging.INFO):
                logger.info(log_msg)

        return df
