"""
Module for handling outliers in numerical datasets.

This module defines the `HandlingMethods` class, which provides utility
methods to detect, remove, and replace outliers in number sequences using
common statistical techniques such as the Interquartile Range (IQR) and
Z-score methods.

Classes
-------
HandlingMethods
    Provides methods for:
    - Removing outliers from a sequences or mappings.
    - Replacing outliers with summary statistics or custom values.

Examples
--------
>>> import pandas as pd
>>> from explorica.data_quality.outliers import HandlingMethods
...
>>> df = pd.DataFrame([2, 1, 5, 4, 4, 3, 500, 9, 2, 10])
>>> outliers = HandlingMethods.remove_outliers(df, detection_method="iqr")
>>> print(outliers)
0     2
1     1
2     5
3     4
4     4
5     3
7     9
8     2
9    10
dtype: int64
"""

import warnings
from typing import Any, Callable, Hashable, Mapping, Optional, Sequence

import pandas as pd

from explorica._utils import (
    convert_dataframe,
    read_config,
    validate_array_not_contains_nan,
    validate_string_flag,
)
from explorica.data_quality._utils import Replacers

from .detection import DetectionMethods


class HandlingMethods:
    """
    A utility class for detecting and handling outliers in numerical datasets.

    This class provides methods for identifying outliers using standard
    statistical techniques — specifically the Interquartile Range (IQR)
    and Z-score approaches — and for either removing or replacing them.
    Both methods support recursive detection for iterative cleaning of
    distributions.

    Methods
    -------
    replace_outliers(data: Sequence | Mapping[str, Sequence],
                     detection_method: str = "iqr", strategy: str = "median",
                     recursive: bool = False, **kwargs)
        Detects outliers in the input data using the specified method
        and replaces them according to the chosen replacement strategy
        (e.g., median, mean, mode, or a custom value).
        Returns a modified pandas Series or DataFrame.

    remove_outliers(data: Sequence | Sequence[Sequence],
                    subset: Sequence[str] = None,
                    detection_method: str = "iqr",
                    recursive: bool = False, **kwargs)
        Detects and removes outliers in the provided data using the chosen
        detection technique (IQR or Z-score). Supports optional recursive
        removal until no outliers remain.
        Returns a cleaned pandas Series or DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.data_quality.outliers import HandlingMethods
    ...
    >>> df = pd.DataFrame([2, 1, 5, 4, 4, 3, 500, 9, 2, 10])
    >>> outliers = HandlingMethods.remove_outliers(df, detection_method="iqr")
    >>> print(outliers)
    0     2
    1     1
    2     5
    3     4
    4     4
    5     3
    7     9
    8     2
    9    10
    dtype: int64
    """

    _warns = read_config("messages")["warns"]
    _errors = read_config("messages")["errors"]

    @staticmethod
    def replace_outliers(
        data: Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Sequence[Any]],
        detection_method: Optional[str] = "iqr",
        strategy: Optional[str] = "median",
        recursive: Optional[bool] = False,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """
        Replaces outliers in sequences or mappings according
        to the specified detection method and replacement strategy.

        Parameters
        ----------
        data : Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Sequence[Any]]
            Input data to process. Can be:
            - 1D sequence -> returns pd.Series
            - 2D sequence -> returns pd.DataFrame
            - Mapping of column names to sequences -> returns pd.DataFrame

        detection_method : str, default 'iqr'
            Method to detect outliers. Supported options:
            - 'iqr' : Interquartile Range method
            - 'zscore' : Z-score method

        strategy : str, default 'median'
            Method to replace detected outliers. Supported options:
            - 'median' : replace with median of the column
            - 'mean' : replace with mean of the column
            - 'mode' : replace with mode of the column
            - 'random' : replace with a random value sampled from the non-outlier values
            - 'custom' : replace with a user-provided value (see `custom_value`)

        custom_value : scalar, optional, default None
            Value to use when strategy='custom'. Must be provided in this case.

        random_state : int, optional, default None
            Seed for random number generator used in 'random' replacement strategy.
            Ensures reproducible replacements.

        recursive : bool, default False
            If True, replaces outliers repeatedly until no outliers remain.
            Ignored if `iters` is specified.

        iters : int, optional
            Number of iterations to replace outliers. Must be a positive integer.
            If specified, `recursive` is ignored.

        subset : Sequence[str], default None
            Features subset by column names.
            If specified, `i_subset` is ignored.

        i_subset : Sequence[int], default None
            Features subset by column positions (like iloc).
            Used only if `subset` is None.

        zscore_threshold : float, default 2.0
            Threshold in units of standard deviations for Z-score detection.
            Z-values beyond this threshold are considered outliers.
            Has effect only if `detection_method='zscore'`.
            If set, it overrides the `"threshold"` key in `zscore_kws`.

        iqr_factor : float, default 1.5
            Used in iqr detection. Multiplier for the Interquartile Range
            used to define outlier bounds. Has effect only if `detection_method='iqr'`
            If set, it overrides the `"iqr_factor"` key in `iqr_kws`.

        zscore_kws : dict, optional
            Additional keyword arguments passed to `data_quality.detect_zscore`.
            See `Outliers.detect_zscore` for full details.

        iqr_kws : dict, optional
            Additional keyword arguments passed to `data_quality.detect_iqr`.
            See `data_quality.detect_iqr` for full details.

        Returns
        -------
        pd.Series or pd.DataFrame
            Object of same shape as input with outliers replaced.
            - Returns pd.Series if input is 1D or if the DataFrame has only one column.
            - Returns pd.DataFrame otherwise.
            Replacement values respect original data types: integers are rounded
            automatically if replacement value is float.

        Raises
        ------
        ValueError
            If input data contains NaN values
            If the provided `detection_method` or `strategy` is not supported
            If `iters` is not a positive integer.
            If strategy='custom' and custom_value is not provided.

        Examples
        --------
        >>> import pandas as pd
        >>> import explorica.data_quality as dq
        ...
        >>> data = pd.DataFrame({
        ...     "feature_1": [1.0, 2.4, 1.6, 12, 1.2, 501.1, 0.6],
        ...     "feature_2": [10, 11, 9, 12, 10, 11, 500]
        ...     })
        >>> print(dq.replace_outliers(data, detection_method="iqr", strategy="mean"))
            feature_1  feature_2
        0    1.000000         10
        1    2.400000         11
        2    1.600000          9
        3   12.000000         12
        4    1.200000         10
        5    3.133333         11
        6    0.600000         10
        """
        df = convert_dataframe(data)

        params = {
            "iters": None,
            "replace_mode": "any",
            "subset": None,
            "i_subset": None,
            "custom_value": None,
            "random_state": None,
            "zscore_threshold": 2.0,
            "iqr_factor": 1.5,
            "zscore_kws": {"threshold": 2.0},
            "iqr_kws": {"iqr_factor": 1.5},
            **kwargs,
        }
        params = HandlingMethods._preproccess_params(
            params,
            mapper={
                "zscore_threshold": {
                    "default_value": 2.0,
                    "path": ("zscore_kws", "threshold"),
                },
                "iqr_factor": {"default_value": 1.5, "path": ("iqr_kws", "iqr_factor")},
            },
        )
        methods = {
            "detection": {
                "iqr": {
                    "func": DetectionMethods.detect_iqr,
                    "params": params["iqr_kws"],
                },
                "zscore": {
                    "func": DetectionMethods.detect_zscore,
                    "params": params["zscore_kws"],
                },
            },
            "fill": {
                "mean": {"func": Replacers.replace_mct, "params": {"measure": "mean"}},
                "median": {
                    "func": Replacers.replace_mct,
                    "params": {"measure": "median"},
                },
                "mode": {"func": Replacers.replace_mct, "params": {"measure": "mode"}},
                "random": {
                    "func": Replacers.replace_random,
                    "params": {"seed": params["random_state"]},
                },
                "custom": {
                    "func": Replacers.replace,
                    "params": {"value": params["custom_value"]},
                },
            },
        }

        def get_columns_for_replace(df: pd.DataFrame) -> list[str]:
            cols = None
            if params["subset"] is not None:
                cols = params["subset"]
            elif params["i_subset"] is not None:
                cols = [df.columns[i] for i in [*params["i_subset"]]]
            else:
                cols = list(df.columns)
            return cols

        def replace(
            df: pd.DataFrame,
            outliers: pd.DataFrame,
            columns_to_replace: list[str],
            replace_method: dict[str, Callable[..., Any]],
        ) -> pd.DataFrame:
            replaced = df.copy()
            for i in columns_to_replace:
                replaced[i] = replace_method["func"](
                    df[i], outliers[i].dropna().index, **replace_method["params"]
                )
            return replaced

        # Validate method and strategy
        validate_string_flag(
            detection_method,
            methods["detection"],
            HandlingMethods._errors["unsupported_method_f"].format(
                detection_method, methods["detection"]
            ),
        )
        validate_string_flag(
            strategy,
            methods["fill"],
            HandlingMethods._errors["unsupported_method_f"].format(
                strategy, methods["fill"]
            ),
        )
        validate_array_not_contains_nan(
            df, err_msg=HandlingMethods._errors["array_contains_nans_f"].format("data")
        )

        if (params["iters"] is not None) and (
            not float(params["iters"]).is_integer() or params["iters"] <= 0
        ):
            raise ValueError("'iters' must be a positive integer")

        if (strategy == "custom") and (params["custom_value"] is None):
            raise ValueError("Strategy is 'custom', but 'custom_value' is not provided")

        # Replace outliers with the selected strategy
        replaced = df.copy()
        columns_to_replace = get_columns_for_replace(replaced)
        if params["iters"] is not None:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", category=UserWarning)
                for i in range(params["iters"]):
                    outliers = methods["detection"][detection_method]["func"](
                        replaced[columns_to_replace],
                        **methods["detection"][detection_method]["params"],
                    )
                    # outliers may be pd.Series
                    outliers = pd.DataFrame(outliers)

                    # Stop if no outliers remain
                    if outliers.shape[0] == 0:
                        break
                    # Apply replacement
                    replaced = replace(
                        replaced,
                        outliers,
                        columns_to_replace,
                        methods["fill"][strategy],
                    )
            HandlingMethods._show_variance_warning_once(w)
        elif recursive:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", category=UserWarning)
                while True:
                    outliers = methods["detection"][detection_method]["func"](
                        replaced[columns_to_replace],
                        **methods["detection"][detection_method]["params"],
                    )
                    # outliers may be pd.Series
                    outliers = pd.DataFrame(outliers)
                    # Stop if no outliers remain
                    if outliers.shape[0] == 0:
                        break
                    # Apply replacement
                    replaced = replace(
                        replaced,
                        outliers,
                        columns_to_replace,
                        methods["fill"][strategy],
                    )
                HandlingMethods._show_variance_warning_once(w)
        else:
            outliers = methods["detection"][detection_method]["func"](
                replaced[columns_to_replace],
                **methods["detection"][detection_method]["params"],
            )
            # outliers may be pd.Series
            outliers = pd.DataFrame(outliers)
            replaced = replace(
                replaced, outliers, columns_to_replace, methods["fill"][strategy]
            )
        if replaced.shape[1] == 1:
            replaced = replaced.iloc[:, 0]
        return replaced

    @staticmethod
    def remove_outliers(
        data: Sequence[float] | Sequence[Sequence[float]],
        subset: Optional[Sequence[str]] = None,
        detection_method: Optional[str] = "iqr",
        recursive: Optional[bool] = False,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """
        Remove outliers from a given sequence of numerical data.

        This method supports two outlier detection techniques:
        1. IQR (Interquartile Range)
        2. Z-score

        Outliers can be removed in three modes:
        - Single removal (default)
        - Iterative removal (`iters` > 0)
        - Recursive removal until no outliers remain (`recursive=True`)

        Parameters:
        -----------
        data : Sequence[float] or Sequence[Sequence[float]]
            Input data from which outliers should be removed. Can be a list,
            NumPy array, pandas Series, DataFrame, etc.

        subset : Sequence[str], default None
            Features subset by column names.
            If specified, `i_subset` is ignored.

        i_subset : Sequence[int], default None
            Features subset by column positions (like iloc).
            Used only if `subset` is None.

        detection_method : str, default 'iqr'
            Method used for outlier detection. Supported methods are:
            - 'iqr' : Interquartile Range method
            - 'zscore' : Z-score method

        recursive : bool, default False
            If True, removes outliers repeatedly until no outliers remain.
            Ignored if `iters` is specified.

        iters : int, optional
            Number of iterations to remove outliers. Must be a positive integer.
            If specified, `recursive` is ignored.

        remove_mode : {'any', 'all'}, default 'any'
            Defines how to treat multi-column outliers:
            - 'any': remove a row if any feature in subset is an outlier
            - 'all': remove a row only if all features in subset are outliers"

        zscore_threshold : float, default 2.0
            Threshold in units of standard deviations for Z-score detection.
            Z-values beyond this threshold are considered outliers.
            Has effect only if `detection_method='zscore'`.
            If set, it overrides the `"threshold"` key in `zscore_kws`.

        zscore_kws : dict, default {"threshold": 2.0}
            Dictionary of additional parameters to pass to `Outliers.detect_zscore`.
            Can be used to customize detection behavior.
            Has effect only if `detection_method='zscore'`.

        Returns
        -------
        pd.Series or pd.DataFrame
            Cleaned data with outliers removed. Returns a Series if the
            input has a single column, otherwise returns a DataFrame.

        Raises
        ------
        ValueError
            If input data contains NaN values
            If the provided `detection_method` or `remove_mode` is not supported
            If `iters` is not a positive integer.
        """
        df = convert_dataframe(data)

        params = {
            "iters": None,
            "remove_mode": "any",
            "i_subset": None,
            "zscore_threshold": 2.0,
            "zscore_kws": {"threshold": 2.0},
            "iqr_kws": {},
            **kwargs,
        }
        params = HandlingMethods._preproccess_params(
            params,
            {
                "zscore_threshold": {
                    "path": ("zscore_kws", "threshold"),
                    "default_value": 2.0,
                }
            },
        )

        def rm(
            df: pd.DataFrame,
            detection_method,
            remove_mode,
        ) -> pd.DataFrame:
            cols = None
            if subset is not None:
                cols = subset
            elif params["i_subset"] is not None:
                cols = [df.columns[i] for i in [*params["i_subset"]]]
            else:
                cols = df.columns
            outliers = detection_method["func"](df[cols], **detection_method["params"])
            if remove_mode == "all":
                outliers.dropna(inplace=True)
            return df.drop(outliers.index)

        supported_methods = {
            "iqr": {"func": DetectionMethods.detect_iqr, "params": params["iqr_kws"]},
            "zscore": {
                "func": DetectionMethods.detect_zscore,
                "params": params["zscore_kws"],
            },
        }

        # input args validation
        validate_array_not_contains_nan(
            df, err_msg=HandlingMethods._errors["array_contains_nans_f"].format("data")
        )
        validate_string_flag(
            detection_method,
            list(supported_methods),
            err_msg=HandlingMethods._errors["unsupported_method_f"].format(
                detection_method, list(supported_methods)
            ),
        )
        validate_string_flag(
            params["remove_mode"],
            {"any", "all"},
            err_msg=HandlingMethods._errors["unsupported_method_f"].format(
                params["remove_mode"], {"any", "all"}
            ),
        )
        if (params["iters"] is not None) and (
            not float(params["iters"]).is_integer() or params["iters"] <= 0
        ):
            raise ValueError("'iters' must be a positive integer")

        # removal outliers by iterations
        if params["iters"] is not None:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", category=UserWarning)
                for i in range(int(params["iters"])):
                    df = rm(
                        df, supported_methods[detection_method], params["remove_mode"]
                    )
            HandlingMethods._show_variance_warning_once(w)
        # removal outliers until they are completely absent
        elif recursive:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", category=UserWarning)
                while True:
                    length = df.shape[0]
                    df = rm(
                        df, supported_methods[detection_method], params["remove_mode"]
                    )
                    prev_length = df.shape[0]
                    if length == prev_length:
                        break
            HandlingMethods._show_variance_warning_once(w)
        # simple removal outliers
        else:
            df = rm(df, supported_methods[detection_method], params["remove_mode"])

        # returns pd.Series if 'cleaned_df' contains 1 column
        if df.shape[1] == 1:
            return df.iloc[:, 0]
        return df

    @staticmethod
    def _show_variance_warning_once(w: list):
        variance_warned = False

        for warn in w:
            msg = str(warn.message)
            if "zero or very small variance" in msg:
                # zero variance warn shows once
                if not variance_warned:
                    variance_warned = True
                    warnings.warn(warn.message, warn.category)
                continue
            # other is shown unchanged
            warnings.warn(warn.message, warn.category)

    @staticmethod
    def _preproccess_params(
        params: dict,
        mapper: dict,
    ):
        """
        Update nested parameters based on user-friendly aliases.

        Parameters
        ----------
        params : dict
            Full parameter dictionary containing both internal
            and user-facing keys.
        mapper : dict
            Mapping of alias -> {
                "default_value": Any,
                "path": tuple[Hashable, ...]
            }

        Returns
        -------
        dict
            Updated parameter dictionary.
        """

        def set_by_path(d: dict, path: tuple[Hashable, ...], value: Any) -> None:
            """Set a value deep in a nested dict by tuple path."""
            target = d
            for key in path[:-1]:
                target = target.setdefault(key, {})
            target[path[-1]] = value

        preprocessed_params = params.copy()
        for alias in mapper:
            if mapper[alias]["default_value"] != params[alias]:
                set_by_path(preprocessed_params, mapper[alias]["path"], params[alias])
        return preprocessed_params
