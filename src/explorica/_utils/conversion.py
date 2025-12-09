"""
Conversion utilities for data transformation and standardization.

This module provides low-level conversion functions for transforming data
between different formats and standardizing input parameters across the
framework. The functions handle common data conversion patterns including
array transformations, parameter normalization, and alias resolution.

Methods
-------
convert_numpy(data)
    Convert an input data to a NumPy array with optional transposition.
convert_dataframe(data)
    Convert an input data to a pandas DataFrame with optional transposition.
convert_series(data)
    Convert an input data to a pandas Series.
convert_from_alias(arg, default_values, path)
    Convert a string alias into its canonical (default) configuration value.
convert_params_for_keys(arg, keys, data_name, validate_dtype_as)
    Converts an input parameter to a dictionary keyed by `keys`.

Notes
-----
- All conversion functions are designed to be robust to various input types
- Functions return copies of data rather than modifying in-place
- For columnar data processing, use convert_dataframe()

Examples
--------
>>> from explorica._utils import convert_dataframe

>>> data = [[1, 2, 3, 4, 6, 7],
...         ["A", "A", "B", "A", "B", "C"],
...         [0.1, 2.2, 1.2, 0.4, 2.3, 2.0]]
>>> convert_dataframe(data)
   0  1    2
0  1  A  0.1
1  2  A  2.2
2  3  B  1.2
3  4  A  0.4
4  6  B  2.3
5  7  C  2.0
"""

from collections import deque
from numbers import Number
from typing import Any, Hashable, Iterable, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from .readers import read_config

ERR_MSG_MULTIDIMENSIONAL_DATA = read_config("messages")["errors"][
    "multidimensional_data_f"
]


@staticmethod
def convert_params_for_keys(
    arg: Any,
    keys: Sequence[Hashable],
    data_name: str = "data",
    validate_dtype_as: type = None,
) -> dict:
    """
    Converts an input parameter to a dictionary keyed by `keys`.

    This utility standardizes heterogeneous parameter inputs
    (such as scalars, lists, Series, or DataFrames) into a unified
    dictionary format `{key: value}`. It supports broadcasting of a
    single value to all keys, copying mappings, and expanding
    sequences or tabular data.

    Parameters
    ----------
    arg : Any
        The input object to be converted. Supported types:
        - ``Number`` or ``str`` - broadcast to all keys
        - ``Mapping`` or ``dict`` - shallow copy of key-value pairs
        - ``pandas.DataFrame`` - each column becomes a key
        - ``pandas.Series`` - treated like a sequence;
          elements are mapped to `keys` by position.
        - ``Sequence`` - must have the same length as `keys`
    keys : Sequence of Hashable
        The target keys for the resulting dictionary.
    data_name : str, default='data'
        The name of the input parameter, used for error reporting.
    validate_dtype_as : type, optional
        If provided, validates that all values in the resulting dictionary
        are instances of this type. Useful for ensuring data consistency
        when working with mixed-type inputs like `[1, 2, "A"]`.

    Returns
    -------
    dict
        A dictionary {key: value} for each key in `keys`.

    Raises
    ------
    ValueError
        If the input does not conform to expected shapes or keys.
        If `keys` contains not unique values.
        If `validate_dtype_as` is provided and any value has incorrect type.
    TypeError
        If the type of `arg` is unsupported (not one of the listed types).

    Examples
    --------
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> from explorica._utils import convert_params_for_keys
    ...
    >>> threshold = 0.7
    >>> df = sns.load_dataset("titanic")
    >>> columns = df.columns
    ...
    >>> print(convert_params_for_keys(threshold, columns))
    {'survived': 0.7,
    'pclass': 0.7,
    'sex': 0.7,
    'age': 0.7,
    'sibsp': 0.7,
    'parch': 0.7,
    'fare': 0.7,
    'embarked': 0.7,
    'class': 0.7,
    'who': 0.7,
    'adult_male': 0.7,
    'deck': 0.7,
    'embark_town': 0.7,
    'alive': 0.7,
    'alone': 0.7}
    """
    if len(keys) != len(set(keys)):
        raise ValueError("`keys` must be unique.")
    if isinstance(arg, (Number, str)):
        dynamic = {key: arg for key in keys}
    elif isinstance(arg, (Mapping, dict)):
        if set(arg) != set(keys):
            raise ValueError(
                f"Keys mismatch: expected {set(keys)}, got {set(arg)} "
                f"in '{data_name}'."
            )
        dynamic = arg.copy()
    elif isinstance(arg, pd.DataFrame):
        if set(arg.columns) != set(keys):
            raise ValueError(
                f"Keys mismatch: expected {set(keys)}, got {set(arg.columns)} "
                f"in '{data_name}'."
            )
        dynamic = {col: arg[col].to_list() for col in arg.columns}
    elif isinstance(arg, pd.Series):
        if arg.shape[0] != len(keys):
            raise ValueError(
                f"Number of elements in '{data_name}' ({arg.shape[0]}) "
                f"does not match number of keys ({len(keys)})."
            )
        dynamic = {keys[i]: elem for i, elem in enumerate(arg.to_list())}
    elif isinstance(arg, Sequence):
        if len(arg) != len(keys):
            raise ValueError(
                f"Number of elements in '{data_name}' ({len(arg)}) "
                f"does not match number of keys ({len(keys)})."
            )
        dynamic = {key: arg[i] for i, key in enumerate(keys)}
    else:
        raise TypeError(
            f"Unsupported type {type(arg).__name__} for '{data_name}'. "
            "Expected one of: Number, str, Mapping, dict, DataFrame, Series, Sequence."
        )
    if validate_dtype_as is not None:
        _validate_dict_values_dtype(dynamic, validate_dtype_as, data_name=data_name)
    return dynamic


@staticmethod
def convert_numpy(data: Sequence[Sequence]) -> np.ndarray:
    """
    Convert an input data to a NumPy array with optional transposition.

    Behavior depends on the input type:
    - If the input is a pandas DataFrame, it is converted to a NumPy array
    and transposed so that each original column becomes a row.
    - If the input is a mapping (e.g., dict of lists or arrays), the mapping
    values are extracted and converted to a NumPy array without transposition.
    - Otherwise, the input is directly converted to a NumPy array without
    transposition.

    Parameters
    ----------
    data : Sequence[Sequence] or Mapping or pandas.DataFrame
        The data to convert. Can be a pandas DataFrame, a mapping (such
        as a dictionary), a nested list, a NumPy array, or any sequence of
        sequences.

    Returns
    -------
    numpy.ndarray
        Converted data as a NumPy array. For DataFrames, the result is
        transposed. For mappings, the values are used as array rows.
    """
    dictionary = convert_dict(data)
    result = np.array(list(dictionary.values()))
    return result


@staticmethod
def convert_dataframe(data: Sequence[Sequence]) -> pd.DataFrame:
    """
    Convert an input data to a pandas DataFrame with optional transposition.

    If the input is a pandas DataFrame, it is returned as-is.
    If the input is a mapping (e.g., dict of lists), it is converted
    directly to a DataFrame. Otherwise, the input is converted to a DataFrame
    and transposed so that each original row becomes a column (i.e., each
    nested sequence is interpreted as a column).

    Parameters
    ----------
    data : Sequence[Sequence] or Mapping or pandas.DataFrame
        The data to convert. Can be a pandas DataFrame, a mapping (such
        as a dictionary), a nested list, a NumPy array, or any sequence of
        sequences.

    Returns
    -------
    pandas.DataFrame
        Converted data as a pandas DataFrame. For non-mapping, non-DataFrame
        inputs, the result is transposed.


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
    """
    dictionary = convert_dict(data)
    result = pd.DataFrame(dictionary)
    return result


def convert_series(data: Union[Sequence[Any] | Mapping]) -> pd.Series:
    """
    Validate and normalize input data into a single-dimensional Pandas Series.

    This function ensures that the input data represents a univariate sample ready for
    statistical processing or visualization. It handles common inputs
    like lists, NumPy arrays, dictionaries, and Pandas structures.

    Parameters
    ----------
    data : array-like, dict
        The input data structure. This may be a 1D sequence (list, np.ndarray,
        pd.Series) or a dictionary/DataFrame containing a single feature.

    Returns
    -------
    pd.Series
        A validated, one-dimensional data series. Returns an empty Series
        if the input data is empty, ensuring pipeline robustness.

    Raises
    ------
    ValueError
        If the input structure contains more than one dimension or feature.
        This is typically raised for dictionaries or DataFrames with multiple
        keys/columns.

    Notes
    -----
    The function relies on an internal 'convert_dict' utility to standardize
    various inputs (e.g., NumPy arrays, lists, Series) into a dictionary
    format before validating dimensionality.

    Examples
    --------
    >>> result = convert_series({"0": [1, 2, 3]})
    >>> print(result.name)
    0

    >>> # Example of unacceptable input violating the dimensionality constraint:
    >>> convert_series({'a': [1, 2], 'b': [3, 4]})
    Traceback (most recent call last):
        ...
    ValueError: Input data is multidimensional and has 2 columns.
    """
    dictionary = convert_dict(data)
    if len(dictionary) > 1:
        raise ValueError(ERR_MSG_MULTIDIMENSIONAL_DATA.format(len(dictionary)))
    if len(dictionary) == 0:
        return pd.Series([])
    return pd.Series(list(dictionary.values())[0], name=list(dictionary)[0])


def convert_dict(
    dataset: Union[Sequence[Sequence] | Mapping], data_name="data"
) -> dict:
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
    >>> from explorica._utils import convert_dict
    >>> convert_dict(None)
    {}

    >>> convert_dict([1, 2, 3])
    {0: [1, 2, 3]}

    >>> convert_dict([[1, 2], [3, 4]])
    {0: [1, 2], 1: [3, 4]}

    >>> convert_dict({'a': [1, 2], 'b': [3, 4]})
    {'a': [1, 2], 'b': [3, 4]}

    Notes
    -----
    - For deque and other collections.abc.Sequence types, conversion follows
    the same pattern as lists and tuples
    - Empty sequences return empty dictionaries
    - String inputs are treated as 1D sequences of characters
    """

    def get_name_if_exists(dataset):
        name = 0
        if isinstance(dataset, pd.Series):
            name = dataset.name if dataset.name is not None else 0
        elif isinstance(dataset, pd.DataFrame):
            name = dataset.columns[0]
        elif isinstance(dataset, dict):
            name = dataset.keys[0]
        return name

    # Handle None and empty inputs
    if dataset is None:
        return {}
    try:
        seq_type = _extract_array_type(dataset)
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
        name = get_name_if_exists(dataset)
        return {name: list(dataset)}

        # Handle 2D+ sequences and iterables

    dictionary = {}

    if seq_type == "iterable":
        for key, arr in enumerate(dataset):
            dictionary[key] = arr

    elif seq_type == "iterable_by_key":
        if len(set(dataset)) != len(list(dataset)):
            raise ValueError(
                read_config("messages")["errors"]["duplicate_keys_f"].format(
                    data_name, data_name
                )
            )
        for key in dataset:
            dictionary[key] = dataset[key]
    return dictionary


def convert_from_alias(arg: str, default_values: Iterable = None, path: str = "global"):
    """
    Convert a string alias into its canonical (default) configuration value.

    The function maps short or alternative forms of names to the
    corresponding default value, using the alias configuration file.
    This allows flexible usage of user input or configuration keywords
    without breaking consistency across Explorica.

    Parameters
    ----------
    arg : str
        Input string to convert. The function is case-insensitive.
    default_values : Iterable, optional
        Subset of default values to restrict the search domain.
        If ``None`` (default), lookup is performed across the entire
        alias set for the specified path.
    path : str, default='global'
        Section name in the alias configuration. Each section defines
        its own mapping between canonical names and their aliases.
        For example: ``"get_categorical_features"``.

    Returns
    -------
    str
        Canonical name corresponding to the alias.
        If no matching alias is found, returns the input argument unchanged.

    Raises
    ------
    KeyError
        If the specified alias section ``path`` does not exist in the configuration.

    Examples
    --------
    >>> from explorica._utils import convert_from_alias
    ...
    >>> shortname = "freq"
    >>> print(convert_from_alias(shortname))
    'frequency'
    ...
    >>> dtype_shortname = "num"
    >>> print(convert_from_alias(dtype_shortname, path="get_categorical_features
    ... "))
    'number'
    """
    alias_dict = read_config("aliases")

    if path not in alias_dict:
        raise KeyError(f"Aliases path '{path}' not found in configuration.")

    arg_lower = arg.lower()
    if default_values is None:
        for default_value, aliases in alias_dict[path].items():
            if arg_lower in aliases:
                return default_value
    else:
        for default_value in default_values:
            if default_value in alias_dict[path]:
                if arg_lower in alias_dict[path][default_value]:
                    return default_value
    return arg


def _validate_dict_values_dtype(
    data: dict, dtype_class: type, err_msg: str = None, data_name: str = "data"
):
    """
    Validate that all values in a dictionary are of the specified data type.

    This function checks that every value in the input dictionary is an instance
    of the given `dtype_class`. If any value has an incorrect type, a `ValueError`
    is raised with a descriptive error message.

    Parameters
    ----------
    data : dict
        The dictionary to validate. Keys can be of any type, but all values
        must be instances of `dtype_class`.
    dtype_class : type
        The expected class/type for all values in the dictionary.
        Can be a built-in type (e.g., `int`, `str`, `list`) or a custom class.
    err_msg : str, optional
        Custom error message to use when validation fails. If not provided,
        a default message is generated.
    data_name : str, default="data"
        Default name of the data parameter to use in error messages for better context.

    Raises
    ------
    ValueError
        If any value in the dictionary is not an instance of `dtype_class`.
        The error message indicates the first offending key and its actual type.

    Examples
    --------
    >>> from explorica._utils.conversion import _validate_dict_values_dtype
    >>> try:
    >>>     _validate_dict_values_dtype({"A": 1, "B": 2}, int)
    >>> except ValueError as e:
    >>>     print(e)
    None
    >>> try:
    >>>     _validate_dict_values_dtype({"A": 1, "B": 2.0}, int)
    >>> except ValueError as e:
    >>>     print(e)
    Invalid data type 'float' in input 'data'. Please provide 'int' data type

    Notes
    -----
    - Validation stops at the first encountered error and raises immediately.
    - For checking against multiple types, use a tuple: `(int, float)`
    """
    for values in data.values():
        if not isinstance(values, dtype_class):
            if err_msg is None:
                err_msg = read_config("messages")["errors"]["invalid_dtype_in_input"]
                err_msg = err_msg.format(
                    type(values).__name__, data_name, dtype_class.__name__
                )
            raise ValueError(err_msg)


def _extract_array_type(array):
    supported_inputs = {
        "iterable": {list, tuple, np.ndarray, deque, pd.Series},
        "iterable_by_key": {pd.DataFrame, dict},
    }

    def is_dim1(x):
        for dtype in [
            *supported_inputs["iterable"],
            *supported_inputs["iterable_by_key"],
        ]:
            if isinstance(x, dtype):
                return False
        return True

    seq_type = None

    for dtype in supported_inputs["iterable_by_key"]:
        if isinstance(array, dtype):
            first_obj = array[list(array)[0]]
            if is_dim1(first_obj):
                seq_type = "1dimensional"
            else:
                seq_type = "iterable_by_key"
            break
    if seq_type is None:
        for dtype in supported_inputs["iterable"]:
            if isinstance(array, dtype):
                if isinstance(array, pd.Series):
                    seq_type = "1dimensional"
                else:
                    first_obj = array[0]
                    if is_dim1(first_obj):
                        seq_type = "1dimensional"
                    else:
                        seq_type = "iterable"
    if seq_type is None:
        seq_type = "1dimensional"
    return seq_type
