"""
Internal utilities for the Explorica framework.

This module provides low-level utilities for data conversion, validation,
and common framework operations. These are internal APIs and may change
without notice.

Methods
-------
convert_dataframe(dataset)
    Convert an input dataset to a pandas DataFrame with optional transposition.
convert_series(data)
    Convert an input data to a pandas Series.
convert_numpy(dataset)
    Convert an input dataset to a NumPy array with optional transposition.
convert_dict(dataset, data_name)
    Convert various sequence types to a dictionary representation.
convert_params_for_keys(arg, keys, data_name, validate_dtype_as)
    Standardize parameters to dictionary format
convert_from_alias(arg, default_values, path)
    Convert a string alias into its canonical (default) configuration value.
validate_array_not_contains_nan(array, err_msg)
    Validate that an array-like object does not contain NaN values
    in its first two dimensions.
validate_at_least_one_exist(values, err_msg)
    Validate that at least one element in the iterable is not ``None``.
validate_lengths_match(array1, array2, err_msg, n_dim)
    Validate that two arrays (or sequences) have matching lengths
    along the specified dimension.
validate_string_flag(arg, supported_values, err_msg)
    Validate a string flag against a set of supported values.
validate_unique_column_names(dataset, err_msg)
    Validate that a DataFrame has unique column names.
natural_number
    Type descriptor for natural number validation
handle_nan(data, nan_policy, supported_policy, is_dataframe, data_name)
    Handles NaN values in a dataset according to a specified policy.
temp_log_level(logger, level)
    Temporarily sets the logging level of a logger within a context.
read_config(name)
    Read and cache JSON configuration files.
convert_filepath(path, default_filename)
    Ensure a given path always points to a file,
    appending a default filename if necessary.
validate_path(path, overwrite_check=True, dir_exists_check=True,
              have_permissions_check=True)
    Validate a filesystem path with optional checks.
enable_io_logs(logger)
    Decorator for I/O functions to automatically log PermissionError, FileExistsError,
    and other unexpected exceptions.

Notes
-----
- These utilities are for internal framework use only
- APIs may change between versions without deprecation warnings
- Use public APIs from main modules for stable functionality
"""

from .conversion import (
    convert_dataframe,
    convert_series,
    convert_from_alias,
    convert_numpy,
    convert_dict,
    convert_params_for_keys,
)
from .helpers import handle_nan, temp_log_level
from .readers import read_config
from .types import natural_number, NaturalNumber
from .io import enable_io_logs, validate_path, convert_filepath
from .validation import (
    validate_array_not_contains_nan,
    validate_at_least_one_exist,
    validate_lengths_match,
    validate_string_flag,
    validate_unique_column_names,
)

__all__ = [
    "convert_dataframe",
    "convert_series",
    "convert_numpy",
    "convert_dict",
    "convert_params_for_keys",
    "convert_from_alias",
    "validate_array_not_contains_nan",
    "validate_at_least_one_exist",
    "validate_lengths_match",
    "validate_string_flag",
    "validate_unique_column_names",
    "natural_number",
    "NaturalNumber",
    "handle_nan",
    "temp_log_level",
    "read_config",
    "enable_io_logs",
    "validate_path",
    "convert_filepath",
]
