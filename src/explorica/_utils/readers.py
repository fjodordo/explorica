"""
Configuration and file reading utilities.

This module provides utilities for reading JSON configuration files
used throughout the framework. All functions include caching to
improve performance when the same files are accessed multiple times.

Methods
-------
read_config
    Read and cache JSON configuration files from the framework's config directory.

Notes
-----
- All functions use LRU caching to avoid repeated file I/O

Examples
--------
>>> from explorica._utils import read_config

>>> read_config("messages")
{'errors': {'unsupported_method_f': "Unsupported method '{}'. Choose from: {}.",
  'unsupported_method': 'Unsupported method choosed.',
  ...
"""

import json
import pathlib
from functools import lru_cache


@lru_cache(maxsize=2)
def read_config(name) -> dict:
    """
    Read and cache JSON configuration files.

    This function reads JSON files from the framework's `config/` directory
    and caches the results to avoid repeated file system access. The cache
    can hold up to 2 different configurations simultaneously.

    Parameters
    ----------
    name : str
        The name of the configuration file (without .json extension).
        File is located at `config/{name}.json` relative to the package root.

    Returns
    -------
    dict
        The parsed JSON content of the configuration file.

    Raises
    ------
    FileNotFoundError
        If the requested configuration file does not exist.

    Examples
    --------
    >>> from explorica._utils import read_config

    >>> read_config("messages")
    {'errors': {'unsupported_method_f': "Unsupported method '{}'. Choose from: {}.",
    'unsupported_method': 'Unsupported method choosed.',

    Notes
    -----
    - The cache size is set to 2 because Explorica currently uses 2 configuration
      files
    """
    path = pathlib.Path(__file__).resolve().parent.parent / f"config/{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
