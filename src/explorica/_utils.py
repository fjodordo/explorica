"""
Utility module providing helper functions and classes for data validation,
conversion, and configuration management within the Explorica framework.

This module is intended for internal use and supports core operations
needed by the public-facing facades such as `InteractionAnalyzer`. It
centralizes common utilities for input validation, type conversion, and
loading of messages (errors and warnings).

Main components
---------------
read_messages
    Loads error and warning messages from a configuration file
    and returns them as a dictionary. Used for consistent error reporting
    across the framework.

ValidationUtils
    A collection of static utility methods for validating inputs and
    raising exceptions when preconditions are not met. Methods include:
    - validate_string_flag: Ensures a string value is in a set of supported flags.
    - validate_at_least_one_exist: Checks that at least one element in an iterable
      is not None.
    - validate_lengths_match: Verifies that sequences have matching lengths.
    - validate_array_not_contains_nan: Checks arrays for NaN or null values.
    - validate_unique_column_names: Ensures that DataFrame column names are unique.

ConvertUtils
    A collection of static utility methods for converting data between
    different types. Methods include:
    - convert_numpy: Converts input to a NumPy array.
    - convert_dataframe: Converts input to a pandas DataFrame.

Notes
-----
- This module is designed for internal use only and is not part of the public API.
- All methods are either standalone functions or static methods; no
  instance of these classes is typically required.
"""

import json
import pathlib
from collections import deque
from functools import lru_cache
from typing import Iterable, Mapping, Sequence, Union

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def read_messages() -> dict:
    """
    Load the framework's error and warning messages from the configuration file.

    This function reads the JSON file located at `config/messages.json` relative
    to this module and returns its content as a dictionary. The result is cached
    after the first load to avoid repeated file I/O.

    Returns
    -------
    dict
        A dictionary containing error and warning messages used throughout the
        framework. Typically structured with keys such as `"errors"` and `"warns"`.

    Notes
    -----
    - The cache ensures that subsequent calls return the same dictionary without
      re-reading the file.
    """
    path = pathlib.Path(__file__).resolve().parent / "config/messages.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ValidationUtils:
    """
    Internal validation utilities for framework consistency checks.

    This class provides a set of static helper functions used across
    the framework to validate input arguments and intermediate results.
    Each method performs a specific check and raises ``ValueError`` if
    the condition is not satisfied.

    The utilities are not part of the public API and are intended for
    internal use only.

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
    """

    @staticmethod
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
        >>> ValidationUtils.validate_string_flag(
            "A", {"A", "B", "C"}, "Method 'A' not in supported methods")
        >>> ValidationUtils.validate_string_flag(
            "D", {"A", "B", "C"}, "Method 'D' not in supported methods")
        Traceback (most recent call last):
            ...
        ValueError: Method 'D' not in supported methods
        """
        if arg not in supported_values:
            raise ValueError(err_msg)

    @staticmethod
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
        >>> ValidationUtils.validate_at_least_one_exist(
                [None, 5, None], "At least one must exist")
        >>> ValidationUtils.validate_at_least_one_exist(
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

    @staticmethod
    def validate_lenghts_match(
        array1: Sequence, array2: Sequence, err_msg: str
    ) -> None:
        """
        Validate that two arrays (or sequences) have matching lengths
        along the specified dimension.

        For one-dimensional sequences, length is determined using
        ``shape[0]``. For higher-dimensional arrays (e.g., matrices,
        DataFrame-like structures), length is determined using
        ``shape[1]``, corresponding to the number of elements in the
        first nested dimension (e.g., number of rows in a column).

        The arrays are internally converted to NumPy arrays using
        ``ConvertUtils.convert_numpy`` before comparison. If the lengths
        do not match, a ``ValueError`` is raised.

        Parameters
        ----------
        array1 : Sequence
            First input array-like structure.
        array2 : Sequence
            Second input array-like structure.
        err_msg : str
            Error message used in the raised ``ValueError`` if validation
            fails.
        n_dim : int, default=2
            Dimensionality of the arrays.

        Raises
        ------
        ValueError
            If the arrays have mismatched lengths along the specified
            dimension.

        Returns
        -------
        None
            This method is for validation only and does not return a value.

        Examples
        --------
        >>> ValidationUtils.validate_lenghts_match(
                [1, 2, 3], [4, 5, 6], "Lengths must match", n_dim=1)
        >>> ValidationUtils.validate_lenghts_match(
                [[1, 2], [3, 4]], [[5, 6], [7, 8]], "Lengths must match")
        >>> ValidationUtils.validate_lenghts_match(
                [1, 2, 3], [4, 5], "Lengths must match", n_dim=1)
        Traceback (most recent call last):
            ...
        ValueError: Lengths must match
        """
        index = 0
        len1 = ConvertUtils.convert_dataframe(array1).shape[index]
        len2 = ConvertUtils.convert_dataframe(array2).shape[index]
        if len1 != len2:
            raise ValueError(err_msg)

    @staticmethod
    def validate_array_not_contains_nan(array: Sequence, err_msg: str) -> None:
        """
        Validate that an array-like object does not contain NaN values
        in its first two dimensions.

        The input is internally converted to a pandas ``DataFrame`` using
        ``ConvertUtils.convert_dataframe``. The validation checks for the
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
        >>> ValidationUtils.validate_array_not_contains_nan(
                [1, 2, 3], "Array must not contain NaN")
        >>> ValidationUtils.validate_array_not_contains_nan(
                [1, float("nan"), 3], "Array must not contain NaN")
        Traceback (most recent call last):
            ...
        ValueError: Array must not contain NaN

        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2], "b": [3, None]})
        >>> ValidationUtils.validate_array_not_contains_nan(df, "NaN found")
        Traceback (most recent call last):
            ...
        ValueError: NaN found
        """
        condition = ConvertUtils.convert_dataframe(array).isna().values.any()
        if condition:
            raise ValueError(err_msg)

    @staticmethod
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
        >>> ValidationUtils.validate_unique_column_names(
                df, "Duplicate columns not allowed")

        >>> df_dup = pd.DataFrame([[1, 2]], columns=["a", "a"])
        >>> ValidationUtils.validate_unique_column_names(
                df_dup, "Duplicate columns not allowed")
        Traceback (most recent call last):
            ...
        ValueError: Duplicate columns not allowed
        """
        columns = dataset.columns
        if len(columns) != len(set(list(columns))):
            raise ValueError(err_msg)


class ConvertUtils:
    """
    Utility class for converting between various sequence types,
    NumPy arrays, and pandas DataFrames.

    The utilities are not part of the public API and are intended for
    internal use only.

    Methods
    -------
    convert_numpy(dataset)
        Convert an input dataset to a NumPy ndarray. The input can be a
        pandas DataFrame, NumPy array, nested lists, or other sequence
        types. If the input is a DataFrame, it is converted to a NumPy
        array with columns becoming rows.

    convert_dataframe(dataset)
        Convert an input dataset to a pandas DataFrame. The input can be
        a NumPy array, nested lists, or other sequence types. Nested
        sequences are interpreted as columns (pandas Series) of the
        resulting DataFrame.

    Notes
    -----
    - These conversions assume that the input data is rectangular;
      irregular nested sequences may raise errors.
    """

    supported_inputs = {
        "iterable": {list, tuple, np.ndarray, deque, pd.Series},
        "iterable_by_key": {pd.DataFrame, dict},
    }

    @staticmethod
    def convert_dict(dataset: Union[Sequence[Sequence] | Mapping]) -> dict:
        """
        Convert various sequence types to a dictionary representation.

        The conversion logic handles different dimensionalities and data structures,
        providing a consistent dictionary interface for heterogeneous input data.

        Parameters
        ----------
        dataset: Union[numpy.ndarray, list, tuple, collections.deque, dict,
                 pandas.Series, pandas.DataFrame]
            Input data to convert. Supported types include:
            - None: returns empty dictionary
            - 1D sequences (list, tuple, np.ndarray, etc.): converted to {0: sequence}
            - 2D sequences: converted to {index: subsequence} for each row
            - Dict-like objects: returned as-is (with copy consideration)
            - Iterables with keys: converted using available keys

        Returns
        -------
        dict
            Dictionary representation of the input data. Structure depends on input
            type:
            - None → {}
            - 1D sequence → {0: original_sequence}
            - 2D sequence → {index: row_data} for each row
            - Dict-like → same structure as input

        Examples
        --------
        >>> ConvertUtils.convert_dict(None)
        {}

        >>> ConvertUtils.convert_dict([1, 2, 3])
        {0: [1, 2, 3]}

        >>> ConvertUtils.convert_dict([[1, 2], [3, 4]])
        {0: [1, 2], 1: [3, 4]}

        >>> ConvertUtils.convert_dict({'a': [1, 2], 'b': [3, 4]})
        {'a': [1, 2], 'b': [3, 4]}

        Notes
        -----
        - For deque and other collections.abc.Sequence types, conversion follows
        the same pattern as lists and tuples
        - Empty sequences return empty dictionaries
        - String inputs are treated as 1D sequences of characters
        """
        # Handle None and empty inputs
        if dataset is None:
            return {}
        try:
            seq_type = ConvertUtils._extract_array_type(dataset)
        except IndexError as e:
            if str(e) in {
                "list index out of range",
                "index 0 is out of bounds for axis 0 with size 0",
            }:
                return {}
            raise e

        # Handle already dictionary-like objects
        if isinstance(dataset, (dict, Mapping)):
            return dict(dataset)  # Return copy to avoid mutation issues

        # Handle 1D sequences
        if seq_type == "1dimensional":
            return {0: list(dataset)}

        # Handle 2D+ sequences and iterables

        dictionary = {}

        if seq_type == "iterable":
            for key, arr in enumerate(dataset):
                dictionary[key] = arr

        elif seq_type == "iterable_by_key":
            for key in dataset:
                dictionary[key] = dataset[key]

        return dictionary

    @staticmethod
    def convert_numpy(dataset: Union[Sequence[Sequence] | Mapping]) -> np.ndarray:
        """
        Convert an input dataset to a NumPy array with optional transposition.

        Behavior depends on the input type:
        - If the input is a pandas DataFrame, it is converted to a NumPy array
        and transposed so that each original column becomes a row.
        - If the input is a mapping (e.g., dict of lists or arrays), the mapping
        values are extracted and converted to a NumPy array without transposition.
        - Otherwise, the input is directly converted to a NumPy array without
        transposition.

        Parameters
        ----------
        dataset : Union[numpy.ndarray, list, tuple, collections.deque, dict,
                  pandas.Series, pandas.DataFrame]
        The dataset to convert. These input types are supported.

        Returns
        -------
        numpy.ndarray
            Converted dataset as a NumPy array. For DataFrames, the result is
            transposed. For mappings, the values are used as array rows.
        """
        dictionary = ConvertUtils.convert_dict(dataset)
        result = np.array(list(dictionary.values()))
        return result

    @staticmethod
    def convert_dataframe(dataset: Union[Sequence[Sequence] | Mapping]) -> pd.DataFrame:
        """
        Convert an input dataset to a pandas DataFrame with optional transposition.

        Behavior depends on the input type:
        - If the input is a pandas DataFrame, it is returned unchanged.
        - If the input is a pandas Series, it is converted into a one-column DataFrame
        without transposition.
        - If the input is a mapping (e.g., dict of lists or arrays), it is converted
        into a DataFrame without transposition (keys become column names).
        - If the input is an array-like object (NumPy array, list, tuple, deque),
        it is converted to a DataFrame and **transposed**, so that each nested
        sequence is interpreted as a column.


        Parameters
        ----------
        dataset : Union[numpy.ndarray, list, tuple, collections.deque, dict,
                  pandas.Series, pandas.DataFrame]
        The dataset to convert. These input types are supported.

        Returns
        -------
        pandas.DataFrame
            Converted dataset as a pandas DataFrame. For DataFrames, Series, and
            mappings, the result is not transposed. For array-like inputs, the
            result is transposed.
        """
        dictionary = ConvertUtils.convert_dict(dataset)
        result = pd.DataFrame(dictionary)
        return result

    @staticmethod
    def _extract_array_type(array):
        def is_dim1(x):
            for dtype in [
                *ConvertUtils.supported_inputs["iterable"],
                *ConvertUtils.supported_inputs["iterable_by_key"],
            ]:
                if isinstance(x, dtype):
                    return False
            return True

        seq_type = None

        for dtype in ConvertUtils.supported_inputs["iterable_by_key"]:
            if isinstance(array, dtype):
                first_obj = array[list(array)[0]]
                if is_dim1(first_obj):
                    seq_type = "1dimensional"
                else:
                    seq_type = "iterable_by_key"
                break
        if seq_type is None:
            for dtype in ConvertUtils.supported_inputs["iterable"]:
                if isinstance(array, dtype):
                    first_obj = array[0]
                    if is_dim1(first_obj):
                        seq_type = "1dimensional"
                    else:
                        seq_type = "iterable"
        if seq_type is None:
            seq_type = "1dimensional"

        return seq_type
