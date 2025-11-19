"""
Data validation and integrity checking utilities.

This module provides functions for validating data structures, value constraints,
and dimensional consistency across the framework. Includes checks for supported
values, non-None existence, length matching, NaN detection, and column uniqueness.

Methods
-------
validate_string_flag(arg, supported_values, err_msg)
    Validate that a string flag is among a set of supported values.
validate_at_least_one_exist(values, err_msg)
    Ensure that at least one element in an iterable is not ``None``.
validate_lenghts_match(array1, array2, err_msg, n_dim=2)
    Check that two sequences have matching lengths along the given
    dimensional index.
validate_array_not_contains_nan(array, err_msg)
    Validate that a sequence contains no ``NaN`` values when converted
    to a DataFrame or Series.
validate_unique_column_names(dataset, err_msg)
    Ensure that a ``pandas.DataFrame`` has unique column labels.

Examples
--------
>>> import explorica._utils as utils

>>> arg = "invalid_method"
>>> supported_methods = {"Pearson", "Spearman"}

>>> utils.validate_string_flag(arg,
...                            supported_values=supported_methods,
...                            err_msg=(f"Unsupported correlation method '{arg}'. "
...                                     f"Please, choose from {supported_methods}."))
ValueError: Unsupported correlation method 'invalid_method'.
            Please, choose from {'Pearson', 'Spearman'}.
"""

from typing import Iterable, Sequence

import pandas as pd

from .conversion import convert_dataframe


def validate_string_flag(
    arg: str, supported_values: Iterable[str], err_msg: str
) -> None:
    """
    Validate a string flag against a set of supported values.

    Checks whether the provided string `arg` exists in the given
    iterable of `supported_values`. If not, raises a ValueError
    with the class-defined error message template
    `err_message_unsupported_method`.

    This method centralizes flag validation logic to improve
    code readability and consistency across the framework.

    Parameters
    ----------
    arg : str
        The string flag to validate.
    supported_values : Iterable[str]
        An iterable containing all supported flag values. Can be
        a list, tuple, set, or any iterable type supporting the
        `in` operator.
    err_msg : str
        The error message used in the raised ``ValueError`` if validation fails.


    Raises
    ------
    ValueError
        If `arg` is not found in `supported_values`.

    Returns
    -------
    None
        This method is for validation only and does not return a value.

    Examples
    --------
    >>> validate_string_flag(
        "A", {"A", "B", "C"}, "Method 'A' not in supported methods")
    >>> validate_string_flag(
        "D", {"A", "B", "C"}, "Method 'D' not in supported methods")
    Traceback (most recent call last):
        ...
    ValueError: Method 'D' not in supported methods
    """
    if arg not in supported_values:
        raise ValueError(err_msg)


def validate_at_least_one_exist(values: Iterable, err_msg: str) -> None:
    """
    Validate that at least one element in the iterable is not ``None``.

    Iterates over the input iterable and checks whether there exists
    at least one non-``None`` value. If all elements are ``None``,
    raises a ``ValueError`` with the provided error message.

    Parameters
    ----------
    values : Iterable
        Iterable of elements to check.
    err_msg : str
        Error message to be used in the raised ``ValueError`` if
        validation fails.

    Raises
    ------
    ValueError
        If all elements in ``values`` are ``None``.

    Returns
    -------
    None
        This method is for validation only and does not return a value.

    Examples
    --------
    >>> validate_at_least_one_exist(
            [None, 5, None], "At least one must exist")
    >>> validate_at_least_one_exist(
            [None, None], "At least one must exist")
    Traceback (most recent call last):
        ...
    ValueError: At least one must exist
    """
    show_err = True
    for i in values:
        if i is not None:
            show_err = False
            break
    if show_err:
        raise ValueError(err_msg)


def validate_lengths_match(array1: Sequence, array2: Sequence, err_msg: str) -> None:
    """
    Validate that two arrays (or sequences) have matching number of rows
    in column-based representation.

    The arrays are internally converted to pandas DataFrames using
    ``convert_dataframe``, which follows a **column-based processing**
    approach. Each nested sequence is treated as a separate feature/column,
    not as rows. This means the length comparison is performed on the
    resulting DataFrame's row count (``.shape[0]``).

    Parameters
    ----------
    array1 : Sequence
        First input array-like structure.
    array2 : Sequence
        Second input array-like structure.
    err_msg : str
        Error message used in the raised ``ValueError`` if validation
        fails.

    Raises
    ------
    ValueError
        If the arrays have mismatched lengths.

    Notes
    -----
    **Column-based Processing:**
    This function uses column-based data representation, where each nested
    sequence is treated as a separate column/feature. For example:

    Input: [[1, 0, 1, 1, 1], [9, 1, 2, 3, 4]]
    Becomes DataFrame:
         0  1
      0  1  9
      1  0  1
      2  1  2
      3  1  3
      4  1  4

    Examples
    --------
    >>> # Two arrays with 3 features each, each feature has 2 observations
    >>> validate_lengths_match([[1, 2], [3, 4], [5, 6]],
    ...                        [[7, 8], [9, 10], [11, 12]],
    ...                        "Row counts must match")

    >>> # Single feature with 5 observations vs two features with 5 observations
    >>> validate_lengths_match([1, 0, 1, 1, 1],
    ...                        [[9, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    ...                        "Row counts must match")

    >>> # This will raise an error - 3 observations vs 2 observations
    >>> validate_lengths_match([1, 2, 3], [4, 5], "Row counts must match")
    Traceback (most recent call last):
        ...
    ValueError: Row counts must match
    """
    len1 = convert_dataframe(array1).shape[0]
    len2 = convert_dataframe(array2).shape[0]
    if len1 != len2:
        raise ValueError(err_msg)


def validate_array_not_contains_nan(array: Sequence, err_msg: str) -> None:
    """
    Validate that an array-like object does not contain NaN values
    in its first two dimensions.

    The input is internally converted to a pandas ``DataFrame`` using
    ``convert_dataframe``. The validation checks for the
    presence of missing values (``NaN``) at the level of DataFrame
    cells or Series elements. If any missing value is found, a
    ``ValueError`` is raised.

    Note that this validation does not descend recursively into
    nested objects (e.g., lists or arrays stored inside a DataFrame
    cell). Only the top-level DataFrame/Series structure is checked.

    Parameters
    ----------
    array : Sequence
        Array-like object to validate. Can be a sequence, NumPy array,
        pandas Series, or DataFrame.
    err_msg : str
        Error message used in the raised ``ValueError`` if validation
        fails.

    Raises
    ------
    ValueError
        If the array contains at least one ``NaN`` value in its
        first two dimensions.

    Returns
    -------
    None
        This method is for validation only and does not return a value.

    Examples
    --------
    >>> validate_array_not_contains_nan(
            [1, 2, 3], "Array must not contain NaN")
    >>> validate_array_not_contains_nan(
            [1, float("nan"), 3], "Array must not contain NaN")
    Traceback (most recent call last):
        ...
    ValueError: Array must not contain NaN

    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2], "b": [3, None]})
    >>> validate_array_not_contains_nan(df, "NaN found")
    Traceback (most recent call last):
        ...
    ValueError: NaN found
    """
    condition = convert_dataframe(array).isna().values.any()
    if condition:
        raise ValueError(err_msg)


def validate_unique_column_names(dataset: pd.DataFrame, err_msg: str) -> None:
    """
    Validate that a DataFrame has unique column names.

    This check ensures that all column labels in the provided
    ``pandas.DataFrame`` are unique. Duplicate column names may
    arise, for example, after a merge or concatenation operation.
    If duplicates are detected, a ``ValueError`` is raised.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Input DataFrame whose column labels are to be validated.
    err_msg : str
        Error message used in the raised ``ValueError`` if validation
        fails.

    Raises
    ------
    ValueError
        If the DataFrame contains duplicate column names.

    Returns
    -------
    None
        This method is for validation only and does not return a value.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> validate_unique_column_names(
            df, "Duplicate columns not allowed")

    >>> df_dup = pd.DataFrame([[1, 2]], columns=["a", "a"])
    >>> validate_unique_column_names(
            df_dup, "Duplicate columns not allowed")
    Traceback (most recent call last):
        ...
    ValueError: Duplicate columns not allowed
    """
    columns = dataset.columns
    if len(columns) != len(set(list(columns))):
        raise ValueError(err_msg)
