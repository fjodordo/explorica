"""
Data-quality summary utilities.

This module provides the `get_summary` function, which computes
a full data-quality summary for a dataset, including metrics for
missing values, duplicates, distribution, basic statistics, and
multicollinearity. The summary can be returned as a pandas DataFrame
with MultiIndex columns or as a JSON-serializable nested dictionary.

Functions
---------
get_summary(data, return_as='dataframe', auto_round=True, round_digits=4, **kwargs)
    Compute a data-quality summary for a dataset.
    Supports saving the summary to CSV, Excel, or JSON, and can return
    either a pandas DataFrame or a JSON-friendly nested dictionary.

Notes
-----
- The module contains internal helper functions for `get_summary`,
  which are not intended for standalone use.
- Saved JSON outputs are fully serializable, with NaNs converted
  to None and non-numeric metrics (like mode) converted to strings.

Examples
--------
>>> import explorica.data_quality as dq

# Minimal usage (DataFrame output)
>>> import pandas as pd

>>> df = pd.DataFrame({
...     "x1": [1, 2, 3],
...     "x2": [2, 4, 6]
... })
>>> summary = dq.get_summary(df, return_as="dataframe")
>>> summary
section          nans               duplicates  ...
metric    count_of_nans pct_of_nans count_of_unique ...
x1                0.0         0.0              3 ...
x2                0.0         0.0              3 ...
...

# Saving summary as JSON (nested dict, JSON-friendly)
>>> summary_dict = dq.get_summary(
...     df,
...     return_as="dict",
...     directory="summary.json"
... )
>>> # The JSON file summary.json is saved in the current directory.
>>> summary_dict
{
    "nans": {"count_of_nans": {"x1": 0, "x2": 0},
             "pct_of_nans": {"x1": 0.0, "x2": 0.0}},
    "duplicates": {"count_of_unique": {...}, ...},
    ...
}

# Verbose logging (optional)
>>> summary_verbose = get_summary(df, verbose=True)
# verbose=True will log computation steps but does not affect returned object
>>> summary_verbose
...
"""

import json
from typing import Sequence
from pathlib import Path
from contextlib import nullcontext
import logging

import numpy as np
import pandas as pd

from .._utils import (
    convert_dataframe,
    read_config,
    validate_string_flag,
    handle_nan,
    temp_log_level,
)
from ..interactions import detect_multicollinearity

from .data_preprocessing import get_missing, get_constant_features
from .outliers import describe_distributions

logger = logging.getLogger(__name__)


def get_summary(
    data: Sequence[Sequence],
    return_as: str = "dataframe",
    auto_round=True,
    round_digits=4,
    **kwargs,
) -> pd.DataFrame | dict:
    """
    Compute a data-quality summary for a dataset.

    The summary includes metrics for missing values, duplicates, distribution,
    basic statistics, and multicollinearity. The result can be returned as a
    pandas DataFrame with MultiIndex columns or as a JSON-serializable nested dict.

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

    Other parameters
    ----------------
    nan_policy : {'drop', 'raise', 'include'}, default='drop'
        How to handle missing values during computations.
        - 'drop' – missing values are removed before computing metrics.
        - 'raise' – missing values cause an exception.
        - 'include' – missing values are kept only for categorical metrics
        (e.g. quasi_constant_pct, mode), where NaN can be treated as a category.
        For numerical metrics, NaN values are still dropped,
        as their interpretation remains undefined.
    threshold_vif : float, default=10
        VIF threshold for multicollinearity.
    directory : str, optional
        Path to save the summary. Supports:
        - '.csv': saved as a CSV file with MultiIndex columns (section, metric).
        Can be reopened via `pd.read_csv(directory, header=[0, 1], index_col=0)`.
        - '.xlsx': saved as an Excel file with MultiIndex columns,
        preserving grouping visually.
        - '.json': saved as a JSON file using nested dict format
        `{section: {metric: {feature: value}}}`, fully JSON-serializable.
        If a file already exists, can raise FileExistsError (unless overwrite=True),
        or PermissionError if access is denied.
    overwrite: bool, default=True
        Whether to overwrite an existing file when saving the summary.
        - True (default): existing files will be overwritten without error.
        - False: if the target file already exists, a FileExistsError is raised.
    verbose : bool, optional
        Enable info-level logging.

    Returns
    -------
    pd.DataFrame or dict
        if `return_as` is 'dataframe' or 'df':
            pd.DataFrame with MultiIndex columns (section, metric),
            index = feature names.
        if `return_as` is 'dict' or 'mapping':
            Nested dict of the form {section: {metric: {feature: value}}},
            JSON-friendly with NaNs converted to None and non-numeric values
            serialized as strings.
        Sections and metrics included:
        - nans
            - count_of_nans: number of missing values per feature
            - pct_of_nans: fraction of missing values per feature (0..1)
        - duplicates
            - count_of_unique: number of unique values per feature
            - pct_of_unique: fraction of unique values per feature (0..1)
            - quasi_constant_pct: top value ratio (quasi-constant score)
        - distribution
            - is_normal: 0/1 flag, distribution approximately normal
              if |skew| ≤ 0.25 and |excess kurtosis| ≤ 0.25
            - desc: qualitative description of distribution shape
              ("normal", "left-skewed", "right-skewed", etc.)
            - skewness: sample skewness
            - kurtosis: sample excess kurtosis
        - stats
            - mean: mean value per feature
            - std: standard deviation per feature
            - median: median value per feature
            - mode: most frequent value per feature (original type)
            - count_of_modes: number of mode values found per feature
        - multicollinearity
            - VIF: Variance Inflation Factor (numeric features only)
            - is_multicollinearity: 0/1 flag if VIF ≥ threshold_vif (default=10)

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
        If invalid `return_as` values.
        If `data` contains NaNs and nan_policy is 'raise'

    Examples
    --------
    >>> import explorica.data_quality as dq

    # Minimal usage (DataFrame output)
    >>> import pandas as pd

    >>> df = pd.DataFrame({
    ...     "x1": [1, 2, 3],
    ...     "x2": [2, 4, 6]
    ... })
    >>> summary = dq.get_summary(df, return_as="dataframe")
    >>> summary
    section          nans               duplicates  ...
    metric    count_of_nans pct_of_nans count_of_unique ...
    x1                0.0         0.0              3 ...
    x2                0.0         0.0              3 ...
    ...

    # Saving summary as JSON (nested dict, JSON-friendly)
    >>> summary_dict = dq.get_summary(
    ...     df,
    ...     return_as="dict",
    ...     directory="summary.json"
    ... )
    >>> # The JSON file summary.json is saved in the current directory.
    >>> summary_dict
    {
        "nans": {"count_of_nans": {"x1": 0, "x2": 0},
                 "pct_of_nans": {"x1": 0.0, "x2": 0.0}},
        "duplicates": {"count_of_unique": {...}, ...},
        ...
    }

    # Verbose logging (optional)
    >>> summary_verbose = get_summary(df, verbose=True)
    # verbose=True will log computation steps but does not affect returned object
    >>> summary_verbose
    ...
    """
    params = {
        "directory": None,
        "threshold_vif": 10,
        "overwrite": True,
        "nan_policy": "drop",
        "verbose": False,
        **kwargs,
    }
    params["nan_policy"] = params["nan_policy"].lower()
    errors = read_config("messages")["errors"]
    not_number_columns = {
        ("distribution", "is_normal"),
        ("distribution", "desc"),
        ("stats", "mode"),
    }
    if params["verbose"]:
        context = temp_log_level(logger, level=logging.INFO)
    else:
        context = nullcontext()
    with context:
        validate_string_flag(
            return_as.lower(),
            {"dataframe", "df", "dict", "json"},
            err_msg=errors["unsupported_method_f"].format(
                return_as, {"dataframe", "dict"}
            ),
        )
        df = convert_dataframe(data)
        logger.info("Data-quality summary started for %d features.", df.shape[1])
        report = _compute_nans(data=df)
        report = pd.concat(
            [report, _compute_duplicates(df, nan_policy=params["nan_policy"])], axis=1
        )
        report = pd.concat(
            [report, _compute_distribution(df, nan_policy=params["nan_policy"])], axis=1
        )
        report = pd.concat(
            [report, _compute_stats(df, nan_policy=params["nan_policy"])], axis=1
        )
        report = pd.concat(
            [report, _compute_multicollinearity(df, nan_policy=params["nan_policy"])],
            axis=1,
        )

        if auto_round:
            report = report.round(round_digits)
        if params["directory"] is not None:
            _save_summary(
                report,
                directory=params["directory"],
                not_number_columns=not_number_columns,
                overwrite=True,
            )
        logger.log(logging.INFO, "Returning summary as %s.", return_as)
    if return_as in {"json", "dict", "mapping"}:
        return _get_nested_dict(report.to_dict(), not_number_columns)
    return report


def _normalize_nan_policy(policy: str) -> str:
    """
    Normalize the nan_policy for get_summary function.

    This function does not validate the policy against allowed values.
    Its purpose is to handle the special case where 'include' should be
    interpreted as 'drop' for numeric metrics. Other values are returned as-is
    and will be validated later by `handle_nan`.

    Parameters
    ----------
    policy : str
        Input policy name provided by the user.

    Returns
    -------
    str
        Returns 'drop' if `policy` is 'include' (case-insensitive), else returns
        the original `policy`.
    """
    if policy.lower() == "include":
        return "drop"
    return policy


def _save_summary(
    data: pd.DataFrame,
    directory: str = ".",
    not_number_columns=(),
    data_name: str = "summary",
    overwrite: bool = True,
):
    """
    Save a summary DataFrame to CSV, Excel, or JSON for get_summary function.

    Parameters
    ----------
    data : pd.DataFrame
        Summary DataFrame with MultiIndex columns.
    directory : str, default='.'
        Path or filename to save the summary. Suffix determines format:
        - '.csv' → CSV
        - '.xlsx' → Excel
        - '.json' → JSON (nested dict)
    not_number_columns : sequence of tuples, optional
        Columns to treat as non-numeric (converted to str for JSON serializability).
    data_name : str, default='summary'
        Default name used when `directory` is a folder and no filename provided.

    Raises
    ------
    FileExistsError
        Raised if the target file exists and overwrite=False, to prevent
        unintentional data loss.
    PermissionError
        Raised if the current process lacks permission to write to the
        specified directory or file.
    ValueError
        Raised if the file format suffix is unsupported (not csv, xlsx, or json).
    Exception
        Any other unexpected error encountered during saving is re-raised
        after logging.

    Notes
    -----
    - CSV/Excel preserve numeric types in DataFrame.
    - JSON output uses _get_nested_dict to convert the DataFrame to a serializable
      format.
    - The function logs important events such as automatic directory creation
      or errors encountered during saving, for debugging and audit purposes.
    """
    try:
        directory = Path(directory)
        if not directory.suffix:
            directory = directory / f"{data_name}.csv"
        save_format = directory.suffix[1:]

        if directory.exists() and not overwrite:
            logger.error(
                "Attempted to save %s to existing path "
                "without 'overwrite=True'. Path: %s",
                data_name,
                directory,
            )
            raise FileExistsError(
                (
                    f"Attempted to save {data_name} to existing path "
                    f"without 'overwrite=True'. Path: {directory}"
                )
            )
        path = directory.parent
        if not path.exists():
            path.mkdir(parents=True)
            logger.warning("Directory '%s' was created automatically.", path)
        if save_format == "xlsx":
            data.to_excel(directory)
        elif save_format == "csv":
            data.to_csv(directory)
        elif save_format == "json":
            dict_report = _get_nested_dict(data.to_dict(), not_number_columns)
            with open(directory, "w", encoding="utf-8") as f:
                json.dump(dict_report, f, ensure_ascii=False, indent=4)
        else:
            raise ValueError(
                (
                    f"Unsupported file format ({save_format}) for {data_name},"
                    f"please provide csv, xlsx or json"
                )
            )
    except PermissionError as e:
        logger.error("Permission denied saving '%s' to %s: %s", data_name, directory, e)
        raise
    except Exception as e:
        logger.error("Failed to save '%s' to %s: %s", data_name, directory, e)
        raise


def _get_nested_dict(d: dict, not_number_columns: Sequence = None):
    """
    Transform a flat dict from DataFrame.to_dict() to a nested, JSON-serializable dict
    for get_summary function.

    Parameters
    ----------
    d : dict
        Flat dict where keys are tuples (section, metric) and values are dicts
        {feature: value}.
    not_number_columns : sequence of tuples, optional
        Columns to convert values to str for JSON serializability
        (e.g., ('stats', 'mode')).

    Returns
    -------
    dict
        Nested dict of the form {section: {metric: {feature: value}}}, with:
        - NaNs replaced by None
        - non-numeric columns converted to str
    """
    nested_dict = {}
    for key, value in d.items():
        if key[0] not in nested_dict:
            nested_dict[key[0]] = {}
        metric = value.copy()
        if key not in not_number_columns:
            for feature, num in value.items():
                metric[feature] = float(num)
        elif key == ("stats", "mode"):
            for feature, mode in value.items():
                metric[feature] = str(mode)
        nested_dict[key[0]][key[1]] = _handle_nan_for_json(metric)
    return nested_dict


def _handle_nan_for_json(arg: dict):
    """
    Replace NaN values with None in a dictionary for JSON serialization.

    Parameters
    ----------
    arg : dict
        Dictionary of feature:value pairs (may include NaN).

    Returns
    -------
    dict
        Copy of input dict with NaNs replaced by None.
    """
    handled_data = arg.copy()
    for key, value in arg.items():
        if pd.isna(value):
            handled_data[key] = None
    return handled_data


def _designate_section(data: pd.DataFrame, section_name: str):
    """
    Assign section name to columns for get_summary function.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    section_name : str
        Section name for MultiIndex columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex columns: (section_name, original column names)
    """
    signed = data.copy()
    signed.columns = pd.MultiIndex.from_product([[section_name], signed.columns])
    return signed


def _compute_nans(data: pd.DataFrame, section_name: str = "nans") -> pd.DataFrame:
    """
    Compute metrics for missing values for get_summary function.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    section_name : str, default='nans'
        Section name for MultiIndex columns.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame (section_name, metric), with metrics:
        - count_of_nans: number of missing values per feature
        - pct_of_nans: fraction of missing values per feature (0..1)
    """
    report = get_missing(data)
    report = _designate_section(report, section_name)
    logger.log(
        logging.INFO, "Section '%s' computed. nan_policy is not used.", section_name
    )
    return report


def _compute_duplicates(
    data: pd.DataFrame,
    quasi_constant_method: str = "top_value_ratio",
    section_name: str = "duplicates",
    nan_policy: str = "drop",
) -> pd.DataFrame:
    """
    Compute metrics for duplicates and quasi-constant features for get_summary
    function.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    quasi_constant_method : str, default='top_value_ratio'
        Method to compute quasi-constant score.
    section_name : str, default='duplicates'
        Section name for MultiIndex columns.
    nan_policy : {'drop', 'raise', 'include'}, default='drop'
        How to handle missing values.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame (section_name, metric), with metrics:
        - count_of_unique: number of unique values per feature
        - pct_of_unique: fraction of unique values per feature (0..1)
        - quasi_constant_pct: top value ratio
    """
    report = pd.DataFrame(
        {"count_of_unique": [], "pct_of_unique": [], "quasi_constant_pct": []}
    )
    if data.empty:
        report = _designate_section(report, section_name)
        logger.log(
            logging.INFO, "Section '%s' computed. nan_policy is not used.", section_name
        )
        return report
    df = handle_nan(data, nan_policy, supported_policy=("drop", "raise", "include"))
    nunique = df.nunique(dropna=False)
    report["count_of_unique"] = nunique
    report["pct_of_unique"] = nunique / df.shape[0]
    report["quasi_constant_pct"] = get_constant_features(df, nan_policy=nan_policy)[
        quasi_constant_method
    ]
    logger.log(
        logging.INFO,
        "Section '%s' computed. Used 'nan_policy': '%s' for %s metrics",
        section_name,
        nan_policy,
        {*report.columns},
    )
    report = _designate_section(report, section_name)
    return report


def _compute_distribution(
    data: pd.DataFrame,
    section_name: str = "distribution",
    nan_policy: str = "drop",
) -> pd.DataFrame:
    """
    Compute distribution metrics for get_summary function.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    section_name : str, default='distribution'
        Section name for MultiIndex columns.
    nan_policy : {'drop', 'raise', 'include'}, default='drop'
        How to handle missing values for numerical metrics.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame (section_name, metric), with metrics:
        - is_normal: 0/1 flag if distribution approximately normal
        - desc: qualitative description of distribution shape
        - skewness: sample skewness
        - kurtosis: sample excess kurtosis
    """
    report = pd.DataFrame({"is_normal": [], "desc": [], "skewness": [], "kurtosis": []})
    data_num = data.select_dtypes("number")
    if data_num.empty:
        logger.log(
            logging.INFO, "Section '%s' computed. nan_policy is not used.", section_name
        )
        report = _designate_section(report, section_name)
        return report
    nan_policy_normalized = _normalize_nan_policy(nan_policy)
    data_num = handle_nan(
        data_num, nan_policy_normalized, supported_policy=("drop", "raise")
    )
    description_sk = describe_distributions(data_num)
    report["is_normal"] = description_sk["is_normal"]
    report["desc"] = description_sk["desc"]
    report["skewness"] = description_sk["skewness"]
    report["kurtosis"] = description_sk["kurtosis"]
    logger.log(
        logging.INFO,
        "Section '%s' computed. Used 'nan_policy': '%s' for %s metrics",
        section_name,
        nan_policy,
        {*report.columns},
    )
    report = _designate_section(report, section_name)
    return report


def _compute_stats(
    data: pd.DataFrame,
    section_name: str = "stats",
    nan_policy: str = "drop",
) -> pd.DataFrame:
    """
    Compute basic statistics for get_summary function.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    section_name : str, default='stats'
        Section name for MultiIndex columns.
    nan_policy : {'drop', 'raise', 'include'}, default='drop'
        How to handle missing values for categorical metrics (mode, count_of_modes).
        Numerical metrics always use drop or raise.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame (section_name, metric), with metrics:
        - mean, std, median: numerical metrics
        - mode: most frequent value per feature
        - count_of_modes: number of mode values per feature
    """
    report = pd.DataFrame(
        {"mean": [], "std": [], "median": [], "mode": [], "count_of_modes": []}
    )
    df = handle_nan(data, nan_policy, supported_policy=("drop", "raise", "include"))
    if df.empty:
        logger.log(
            logging.INFO, "Section '%s' computed. nan_policy is not used.", section_name
        )
        report = _designate_section(report, section_name)
        return report
    # compute categorical metrics
    report["mode"] = df.mode().iloc[0]
    report["count_of_modes"] = df.mode(dropna=False).count()
    # normilize policy for num metrics computing
    nan_policy_normalized = _normalize_nan_policy(nan_policy)
    df_num = handle_nan(
        data.select_dtypes("number"),
        nan_policy_normalized,
        supported_policy=("drop", "raise"),
    )
    # compute num metrics
    report["mean"] = np.mean(df_num, axis=0)
    report["std"] = np.std(df_num, axis=0)
    report["median"] = pd.Series(np.median(df_num, axis=0), index=df_num.columns)
    logger.log(
        logging.INFO,
        (
            "Section '%s' computed. Used 'nan_policy':"
            "'%s' for %s metrics, '%s' for %s metrics."
        ),
        section_name,
        nan_policy,
        {"mode", "count of modes"},
        nan_policy_normalized,
        {"mean", "std", "median"},
    )
    report = _designate_section(report, section_name)
    return report


def _compute_multicollinearity(
    data: pd.DataFrame,
    section_name: str = "multicollinearity",
    nan_policy: str = "drop",
) -> pd.DataFrame:
    """
    Compute multicollinearity metrics for get_summary function.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    section_name : str, default='multicollinearity'
        Section name for MultiIndex columns.
    nan_policy : {'drop', 'raise', 'include'}, default='drop'
        How to handle missing values for numerical metrics.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame (section_name, metric), with metrics:
        - is_multicollinearity: 0/1 flag if VIF ≥ threshold
        - VIF: Variance Inflation Factor (numeric features only)
    """
    report = pd.DataFrame({"is_multicollinearity": [], "VIF": []})
    data_num = data.select_dtypes("number")
    if data_num.empty:
        logger.log(
            logging.INFO, "Section '%s' computed. nan_policy is not used.", section_name
        )
        report = _designate_section(report, section_name)
        return report
    nan_policy = _normalize_nan_policy(nan_policy)
    data_num = handle_nan(data_num, nan_policy, supported_policy=("drop", "raise"))
    description_multicol = detect_multicollinearity(numeric_features=data_num)
    report["is_multicollinearity"] = description_multicol["multicollinearity"]
    report["VIF"] = description_multicol["VIF"]
    logger.log(
        logging.INFO,
        "Section '%s' computed. Used 'nan_policy': '%s' for %s metrics.",
        section_name,
        nan_policy,
        {*report.columns},
    )
    report = _designate_section(report, section_name)
    return report
