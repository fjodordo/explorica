"""
General-purpose utilities and temporary helpers.

This module contains miscellaneous utility functions that don't yet have
a dedicated home in the framework. These include NaN value handling,
logging utilities, and other general-purpose helpers that may be
categorized into specific modules as the framework evolves.

Methods
-------
handle_nan(data, nan_policy, supported_policy, is_dataframe, data_name)
    Handles NaN values in a dataset according to a specified policy.
temp_log_level(logger, level)
    Temporarily sets the logging level of a logger within a context.

Notes
-----
- Functions in this module are considered temporary and may be moved to
  more specific modules in future versions

Examples
--------
>>> from explorica._utils import helpers
>>> import numpy as np

>>> data = [1.0, np.nan, 3.0, np.nan]
>>> helpers.handle_nan(data, is_dataframe=False, nan_policy='drop')
     0
0  1.0
2  3.0
"""

from contextlib import contextmanager
from typing import Iterable, Literal, Mapping, Sequence

import pandas as pd

from .conversion import convert_dataframe, convert_from_alias
from .readers import read_config
from .validation import validate_array_not_contains_nan, validate_string_flag


def handle_nan(
    data: pd.DataFrame | Sequence | Mapping,
    nan_policy: Literal["drop", "raise", "include", "drop_columns"],
    supported_policy: Iterable[str] = ("drop", "raise"),
    is_dataframe: bool = True,
    data_name: str = "data",
):
    """
    Handles NaN values in a dataset according to a specified policy.

    Parameters
    ----------
    data : pd.DataFrame, Sequence, or Mapping
        Input data to process. Will be converted to a DataFrame if not already.
    nan_policy : {'drop', 'raise', 'include', 'drop_columns'}, default='drop'
        Policy for handling NaN values:
        - 'drop': drop rows with NaNs.
        - 'drop_columns': drop columns with NaNs.
        - 'raise': raise ValueError if NaNs are present.
        - 'include': treat NaNs as valid values (do nothing).
    supported_policy : Iterable[str], default={'drop', 'raise'}
        List of nan_policy values that are allowed in the current context.
        If `nan_policy` is not in this set, a ValueError is raised.
        This allows restricting the set of applicable NaN-handling strategies
        for a particular function or workflow.
    is_dataframe : bool, default=True
        Whether the input is already a DataFrame. If False, converts input to
        DataFrame.
    data_name : str, default='data'
        Name of the dataset (used in error messages).

    Returns
    -------
    pd.DataFrame
        DataFrame with NaNs handled according to the policy.

    Raises
    ------
    ValueError
        If `nan_policy` is 'raise' and NaNs are present in the data.
        If `nan_policy` not in `supported_policy`.
    """
    if not is_dataframe:
        df = convert_dataframe(data)
    else:
        df = data.copy()
    nan_policy = convert_from_alias(nan_policy, supported_policy)
    validate_string_flag(
        nan_policy,
        supported_policy,
        err_msg=(
            f"Unsupported nan_policy: '{nan_policy}'. "
            f"Choose from: {supported_policy}"
        ),
    )

    if nan_policy == "drop":
        df = df.dropna(axis=0)
    elif nan_policy == "drop_columns":
        df = df.dropna(axis=1)
    elif nan_policy == "raise":
        error_messages = read_config("messages")["errors"]
        # will raise valueError if necessary
        validate_array_not_contains_nan(
            df, err_msg=error_messages["array_contains_nans_f"].format(data_name)
        )
    # 'include' -> do nothing
    return df


@contextmanager
def temp_log_level(logger, level):
    """
    Temporarily sets the logging level of a logger within a context.

    Parameters
    ----------
    logger : logging.Logger
        The logger whose level will be temporarily changed.
    level : int
        The logging level to set (e.g., logging.INFO, logging.DEBUG).

    Usage
    -----
    >>> import logging
    >>> logger = logging.getLogger("my_logger")
    >>> with temp_log_level(logger, logging.INFO):
    ...     logger.info("This will be shown if logger level was lower before")
    ...
    # After the context, logger level is restored to its original value.

    Notes
    -----
    This is a simple utility context manager intended for temporary
    adjustment of logger levels. After exiting the context, the
    original log level is always restored, even if an exception occurs.
    """
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)
