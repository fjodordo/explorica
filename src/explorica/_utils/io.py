"""
Utilities for robust file path handling, validation, and PDF I/O.

This module provides helper functions for managing filesystem paths, validating them
against common conditions, and saving PDF content safely. It includes utilities
to ensure a path points to a file, validate overwrites and permissions, and log
I/O errors consistently. These tools are intended for internal use in Explorica.

Functions
---------
convert_filepath(path, default_filename)
    Ensure a given path always points to a file,
    appending a default filename if necessary.
validate_path(path, overwrite_check=True, dir_exists_check=True,
              have_permissions_check=True)
    Validate a filesystem path with optional checks for:
        - Existing files (overwrite check)
        - Directory existence (warn if directory does not exist)
        - Write permissions
enable_io_logs(logger)
    Decorator for I/O functions to automatically log PermissionError, FileExistsError,
    and other unexpected exceptions.

Notes
-----
- Most functions do not perform actual I/O themselves, but
  are used to standardize and safeguard file operations.
- `enable_io_logs` can be applied to any I/O function to log errors consistently
  without modifying function behavior.
- Designed to be composable and reusable for higher-level operations within
  Explorica.

Examples
--------
>>> from explorica._utils.io import convert_filepath, validate_path

# Convert a path to always include a filename
>>> file_path = convert_filepath("reports", "report.pdf")
>>> print(file_path)
PosixPath('reports/report.pdf')

# Validate path before saving
>>> validate_path(file_path, overwrite_check=True)
"""

import os
import re
import logging
import warnings
from pathlib import Path
from typing import Callable
import functools

logger = logging.getLogger(__name__)


def convert_filepath(path: str | Path, default_filename: str) -> Path:
    """
    Convert a given path into a full file path with a specified filename.

    This utility ensures that the returned `Path` object always points to a
    file. If the input `path` is a directory, the `default_filename` will be
    appended. If the input is already a file path (ends with a suffix), it is
    returned unchanged.

    Parameters
    ----------
    path : str or Path
        The input path, which can be either a directory or a file path.
    default_filename : str
        The filename to append if `path` is a directory (has no suffix).

    Returns
    -------
    Path
        A `Path` object pointing to a file. If `path` was a directory, the
        `default_filename` is appended. If `path` was already a file path,
        it is returned as-is.

    Notes
    -----
    - This function does not perform any filesystem operations; it does not
      create directories or files.
    - Designed to standardize path handling in functions that save files,
      ensuring there is always a concrete file path to write to.

    Examples
    --------
    >>> convert_filepath("output", "report.pdf")
    PosixPath('output/report.pdf')

    >>> convert_filepath("output/report.pdf", "ignored.pdf")
    PosixPath('output/report.pdf')
    """
    path_pl = Path(path)
    if path_pl.suffix == "":
        return path_pl / default_filename
    return path_pl


def validate_path(
    path: str | Path,
    overwrite_check: bool = True,
    dir_exists_check: bool = True,
    have_permissions_check: bool = True,
):
    """
    Validate a filesystem path before performing IO operations.

    This function performs a set of pre-checks on a given path to ensure
    that subsequent file operations (like saving a PDF) will not fail
    due to common filesystem issues.

    Parameters
    ----------
    path : str or Path
        The filesystem path to validate. Can be a full file path or a directory.
        - If a file path is provided, the directory containing the file will be checked.
        - If a directory path is provided, the directory itself will be checked.
    overwrite_check : bool, default=True
        If True, raises a `FileExistsError` when the path already exists.
        Prevents accidental overwriting of files.
    dir_exists_check : bool, default=True
        If True, issues a `UserWarning` when the directory does not exist.
        Intended as a heads-up that the directory will be created automatically
        during subsequent IO operations.
    have_permissions_check : bool, default=True
        If True, raises a `PermissionError` when the path is not writable.
        The function checks permissions on the first existing parent directory
        in the path. This ensures a possibility to create a file or
        directory at the specified location. It does not check permissions
        on non-existent directories themselves. If no existing parent directory
        can be found (for example, if the path is relative like
        './some/nonexistent/path' and none of the parents exist),
        a `ValueError` is raised indicating that permissions cannot be verified.

    Raises
    ------
    FileExistsError
        If `overwrite_check` is True and the file or directory already exists.
    PermissionError
        If `have_permissions_check` is True and the directory is not writable.
    ValueError
        If `have_permissions_check` is True but no existing parent directory
        can be found to verify permissions (e.g., a relative path with all
        non-existent ancestors).

    Warns
    -----
    UserWarning
        If `dir_exists_check` is True and the directory does not exist.

    Notes
    -----
    - This function does not create directories or modify the filesystem.
      It only validates the state of the path.
    - Designed to be used as a pre-check before actual file IO operations,
      such as saving PDF reports.

    Examples
    --------
    >>> validate_path("output/report.pdf")  # Raises warning if 'output/' does not exist
    >>> validate_path("output/report.pdf", overwrite_check=False)  # Allows overwriting
    >>> validate_path("output/")  # Validates a directory path
    """
    path_pl = Path(path)
    directory = path_pl.parent if path_pl.suffix else path_pl

    if overwrite_check and path_pl.exists():
        raise FileExistsError(f"Path '{path}' already exists.")
    if dir_exists_check and not directory.exists():
        wmsg = (
            f"Directory '{directory}' does not exist."
            "It will be created automatically."
        )
        warnings.warn(wmsg)
    if have_permissions_check:
        existing_path = path_pl
        while True:
            if existing_path.exists():
                break
            if existing_path.parent == existing_path:
                raise ValueError(
                    f"Unable to verify permissions to the specified path: {path}"
                    "Try to provide an absolute path."
                )
            existing_path = existing_path.parent
        if not os.access(existing_path, os.W_OK):
            raise PermissionError(f"No write permissions for path '{existing_path}'")


def enable_io_logs(io_logger: logging.Logger = None) -> Callable:
    """
    Decorator factory for logging I/O errors with a specified logger.

    Wraps an I/O function to catch common filesystem exceptions, log them,
    and re-raise. Intended for internal use in Explorica modules handling
    filesystem or other I/O operations.

    Parameters
    ----------
    io_logger : logging.Logger, optional
        Logger instance to emit error messages through. If not provided,
        defaults to the logger of this module (`explorica._utils.io`).
        Using a specific logger allows error messages to appear under the
        module of the calling code rather than the utility module.

    Returns
    -------
    Callable
        A decorator to wrap I/O functions.

    Notes
    -----
    Exceptions
    - Catches and logs:
        - ``PermissionError``
        - ``FileExistsError``
        - any other unexpected ``Exception``
    - After logging, all exceptions are re-raised unchanged.
    Warnings
    - Captures warnings emitted during function execution.
    - Selected Explorica-generated I/O warnings (currently:
      warnings about non-existing directories that will be created)
      are logged using ``logger.warning``.
    - All captured warnings are re-emitted via ``warnings.warn`` so that:
        - user code,
        - test frameworks,
        - and warning filters
      continue to observe them normally.
    This ensures that the decorator does **not swallow warnings** and does
    not alter warning semantics, only mirrors selected ones into logs.

    Examples
    --------
    # Using outer logger
    >>> import logging
    >>> logger = logging.getLogger("explorica.reports.renderers")
    >>> @enable_io_logs(logger)
    ... def _save_pdf(pdf_bytes, path):
    ...     with open(path, "wb") as f:
    ...         f.write(pdf_bytes)

    # Using default logger from explorica._utils.io
    >>> @enable_io_logs()
    ... def _save_pdf_default(pdf_bytes, path):
    ...     with open(path, "wb") as f:
    ...         f.write(pdf_bytes)
    """
    if io_logger is None:
        io_logger = logger

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            caught_warnings = []
            try:
                warns_to_log_regex = {
                    r"Directory '.*' does not exist. It will be created automatically."
                }
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    result = fn(*args, **kwargs)

                return result
            except PermissionError as e:
                logger.error("Permission denied in %s: %s", fn.__name__, e)
                raise
            except FileExistsError as e:
                logger.error("File already exists in %s: %s", fn.__name__, e)
                raise
            except Exception as e:
                logger.error("Unexpected IO error in %s: %s", fn.__name__, e)
                raise
            finally:
                for w in caught_warnings:
                    msg = str(w.message)
                    for regex in warns_to_log_regex:
                        if re.search(regex, msg):
                            logger.warning(
                                "Captured warning in %s: %s", fn.__name__, msg
                            )
                    warnings.warn(
                        w.message,
                        category=w.category,
                        stacklevel=2,
                    )

        return wrapper

    return decorator
