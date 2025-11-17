"""
Module for feature engineering on numeric and categorical data.

This module provides utilities for fast and flexible feature
transformation. It is focused on common preprocessing tasks:
frequency encoding, ordinal encoding and discretization (binning)
of continuous variables. Implementations accept pandas Series/DataFrame,
NumPy arrays, Python sequences and mappings (dict-like inputs).

The main entrypoint is the :class:`EncodeMethods` class which groups
static helper methods intended for use in EDA pipelines and model
preprocessing steps.

Classes
-------
EncodeMethods
    Collection of static methods for feature encoding and discretization:
    - frequency encoding of categorical features (absolute/relative counts),
    - ordinal encoding by frequency / alphabetical order / target statistics,
    - discretization of continuous variables using uniform or quantile bins
      with flexible labeling options.

Examples
--------
>>> import pandas as pd
>>> import explorica.data_quality as data_quality
...
>>> df = pd.DataFrame({
...      "color": ["red", "blue", "red", "green", "blue", "red"],
...      "shape": ["circle", "square", "circle", "triangle", "square", "circle"]
... })
>>> encoded = data_quality.freq_encode(df, round_digits=4)
>>> print(encoded)
    color   shape
0  0.5000  0.5000
1  0.3333  0.3333
2  0.5000  0.5000
3  0.1667  0.1667
4  0.3333  0.3333
5  0.5000  0.5000
"""

import warnings
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

from explorica._utils import (
    convert_dataframe,
    convert_from_alias,
    read_config,
    validate_array_not_contains_nan,
    validate_lengths_match,
    validate_string_flag,
)


class EncodeMethods:
    """
    A utility class for performing feature encoding and discretization on
    categorical and numerical variables.

    This class provides static methods for transforming feature values into
    encoded or discretized representations suitable for statistical analysis
    and machine learning preprocessing.

    Designed to be modular, extensible, and easily integrated into exploratory
    data analysis (EDA) and data-cleaning workflows.

    Methods
    -------
    freq_encode(data, axis=0, normalize=True, round_digits=None)
        Perform frequency encoding on categorical features.
        Each category is replaced by its relative or absolute frequency.

    ordinal_encode(data, axis=0, order_method='frequency', order_ascending=False,
                   **kwargs)
        Encode categorical values as integer ranks according to a specified ordering
        rule. Supports ordering by frequency, alphabetical order, or by target
        statistics (mean, median, or mode of another variable).

    discretize_continuous(data, bins=None, binning_method='uniform',
                          intervals='pandas')
        Discretize continuous numeric features into categorical bins.
        Supports uniform-width and quantile-based binning, as well as
        customizable labels.

    Examples
    --------
    >>> import pandas as pd
    >>> import explorica.data_quality as data_quality
    ...
    >>> df = pd.DataFrame({
    ...      "color": ["red", "blue", "red", "green", "blue", "red"],
    ...      "shape": ["circle", "square", "circle", "triangle", "square", "circle"]
    ... })
    >>> encoded = data_quality.freq_encode(df, round_digits=4)
    >>> print(encoded)
        color   shape
    0  0.5000  0.5000
    1  0.3333  0.3333
    2  0.5000  0.5000
    3  0.1667  0.1667
    4  0.3333  0.3333
    5  0.5000  0.5000
    """

    _warns = read_config("messages")["warns"]
    _errors = read_config("messages")["errors"]
    _aliases = read_config("aliases")

    @staticmethod
    def freq_encode(
        data: Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Sequence[Any]],
        axis: int = 0,
        normalize: bool = True,
        round_digits: int = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Performs frequency encoding on a categorical feature(s).

        Parameters
        ----------
        data : Sequence[float] | Sequence[Sequence[float]] |
               Mapping[str, Sequence[Number]]
            Numeric input data. Can be 1D (sequence of numbers),
            2D (sequence of sequences), or a mapping of column names to sequences.
        axis: int, {0, 1}, default 0
            Applicable only if input is 2D:
            - 0: encode each column independently (column-wise), returns pd.DataFrame.
            - 1: encode each row based on the combination of column values (row-wise),
                returns pd.Series.
            Ignored if input is 1D.
        normalize : bool
            If True, encodes as relative frequency (proportion),
            otherwise as absolute count.
        round_digits : int, optional
            Number of decimal digits to round the encoded frequencies to.
            Applicable only when `normalize=True`.
            If `None` (default), no rounding is performed.
            Must be a non-negative integer.

        Returns
        -------
        pd.Series or pd.DataFrame
            Frequency-encoded feature(s). Returns Series for row-wise encoding or 1D
            input, DataFrame for column-wise encoding of multiple features.

        Raises
        ------
        ValueError
            If input contains NaNs.
            If `axis` is not 0 or 1.
            If `round_digits` is negative or not an integer.
        """
        df = convert_dataframe(data)

        validate_array_not_contains_nan(
            df, EncodeMethods._errors["array_contains_nans_f"].format("data")
        )
        if axis not in (0, 1):
            raise ValueError(f"axis must be 0 or 1, provided value is {axis}")
        if (round_digits is not None) and (
            not float(round_digits).is_integer() or round_digits <= 0
        ):
            raise ValueError("'round_digits' must be a positive integer")

        # row wise
        if axis == 1:
            freqs = df.value_counts(normalize=normalize)
            freqs_map = dict(freqs.items())
            tuples = [tuple(row) for row in df.to_numpy()]
            freq = pd.Series(
                [freqs_map.get(t) for t in tuples], index=df.index, name="frequency"
            )

        # column wise
        if axis == 0:
            freq = pd.DataFrame(index=df.index)
            for col in df.columns:
                counts = df[col].value_counts(normalize=normalize)
                freq[col] = df[col].map(counts)
            freq = freq.squeeze()

        if normalize and round_digits is not None:
            freq = np.round(freq, round_digits)
        return freq

    @staticmethod
    def ordinal_encode(
        data: Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Sequence[Any]],
        axis: int = 0,
        order_method: str = "frequency",
        order_ascending: bool = False,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """
        Encode categorical values with ordinal integers
        based on a specified ordering rule.

        This method converts categorical data into integer-encoded representations
        according to the chosen ordering strategy. Supported strategies include
        frequency-based, alphabetical, or target-based orderings (using mean, median,
        or mode of a provided reference variable).

        Parameters
        ----------
        data : Sequence[float] | Sequence[Sequence[float]] |
               Mapping[str, Sequence[Number]]
            Numeric input data. Can be 1D (sequence of numbers),
            2D (sequence of sequences), or a mapping of column names to sequences.
        axis: int, {0, 1}, default 0
            Applicable only if input is 2D:
            - 0: encode each column independently (column-wise).
                 Returns a DataFrame if multiple columns are provided.
            - 1: encode each row based on the combination of column values (row-wise).
                 Always returns a Series.
            Ignored if input is 1D.
        order_method : {"frequency", "alphabetical", "mean", "median", "mode"},
                       default "frequency"
            Ordering rule to determine integer assignment:
            - "frequency" : order by category frequency.
            - "alphabetical" : order alphabetically by category label.
            - "mean" : order by the mean of corresponding `order_by` values per
                       category.
            - "median" : order by the median of corresponding `order_by` values per
                         category.
            - "mode" : order by the most frequent corresponding `order_by` value per
                       category.
        order_ascending : bool, default False
            Whether to assign integers in ascending (True) or descending (False) order.
        order_by : Sequence, pandas.Series, pandas.DataFrame, or Mapping, optional
            Numerical data to use for computing
            central tendency measures (mean, median, mode).
            Required when `order_method` is one of {"mean", "median", "mode"}.
            Must be aligned by shape with `data`.
        offset : int, default 0
            The starting integer value for encoding categories.
            Each group label is incremented by this offset, so setting
            `offset=1` makes encoded values start from 1 instead of 0.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Encoded data, where each unique category or category combination
            is replaced by an integer reflecting its relative order:
            - If ``axis=1``, returns a Series with encoded values per row.
            - If ``axis=0`` and multiple columns were passed, returns a DataFrame
              where each column is encoded independently.
            - If ``axis=0`` and a single column was passed, returns a Series.

        Raises
        ------
        ValueError:
            If input contains NaNs.
            If the provided `order_method` is not supported.
            If `order_by` is missing when required.
            If `data` and `order_by` have mismatched lengths.

        Notes
        -----
        - When ``order_method`` is ``"mode"``, if multiple modes exist within a group,
          the first encountered mode is used for ordering
          (tie-breaking is deterministic).

        Examples
        --------
        >>> import pandas as pd
        >>> import explorica.data_quality as data_quality

        >>> df = pd.DataFrame({"category_1": ["A", "B", "C", "A", "A", "B", "A"]})
        >>> data_quality.ordinal_encode(
        ...     df, order_method="abc", order_ascending=True, offset=1)
        0    1
        1    2
        2    3
        3    1
        4    1
        5    2
        6    1
        Name: category_1, dtype: int64
        """
        params = {"offset": 0, "order_by": None, **kwargs}

        df = convert_dataframe(data)
        df_order_by = convert_dataframe(params["order_by"])

        def encode(
            data: pd.DataFrame,
            method,
            asc=False,
            order_by: pd.DataFrame = None,
            offset: int = 0,
        ) -> pd.Series:
            def sort_groups_by_central_measure() -> pd.MultiIndex:
                reindexed_order_by = order_by.copy()
                reindexed_order_by.index = data.index
                agg_data = pd.concat([reindexed_order_by, data], axis=1)
                agg_data = agg_data.groupby([*data.columns])
                if method == "mean":
                    agg_data = agg_data.mean()
                elif method == "median":
                    agg_data = agg_data.median()
                elif method == "mode":
                    agg_data = agg_data.agg(lambda x: x.mode().iloc[0])
                agg_data = agg_data.sort_values(by=[*agg_data.columns], ascending=asc)
                ordered_keys = agg_data.index
                return ordered_keys

            get_order_methods = {
                "frequency": lambda: data.value_counts()
                .sort_values(ascending=asc)
                .index,
                "alphabetical": lambda: data.value_counts()
                .sort_index(ascending=asc)
                .index,
                "mean": sort_groups_by_central_measure,
                "median": sort_groups_by_central_measure,
                "mode": sort_groups_by_central_measure,
            }
            ordered_keys = get_order_methods[method]()

            mapper = {tuple(key): num for num, key in enumerate(ordered_keys)}
            combinations = [tuple(row) for row in data.to_numpy()]
            output = pd.Series(
                [mapper.get(t) + offset for t in combinations], index=data.index
            )
            return output

        order_method = convert_from_alias(
            order_method, {"frequency", "alphabetical", "mean", "median", "mode"}
        )

        supported_order = {"frequency", "alphabetical", "mean", "median", "mode"}

        validate_array_not_contains_nan(
            df, EncodeMethods._errors["array_contains_nans_f"].format("data")
        )
        validate_string_flag(
            order_method,
            supported_order,
            EncodeMethods._errors["unsupported_method_f"].format(
                order_method, supported_order
            ),
        )
        if order_method in {"mean", "median", "mode"}:
            if params["order_by"] is None:
                raise ValueError(
                    "'order_by' must be provided for central measures ordering"
                )
            validate_array_not_contains_nan(
                df_order_by,
                err_msg=EncodeMethods._errors["array_contains_nans_f"].format(
                    "order_by"
                ),
            )
            validate_lengths_match(
                df,
                df_order_by,
                err_msg=EncodeMethods._errors["arrays_lens_mismatch_f"].format(
                    "data", "order_by"
                ),
            )

        # row wise
        if axis == 1:
            encoded = encode(
                df,
                order_method,
                order_ascending,
                params["order_by"],
                offset=params["offset"],
            )

        # column wise
        elif axis == 0:
            encoded = pd.DataFrame(index=df.index)
            for col in df.columns:
                encoded[col] = encode(
                    df[[col]],
                    order_method,
                    order_ascending,
                    params["order_by"],
                    params["offset"],
                )
            encoded = encoded.squeeze()
            if isinstance(encoded, pd.Series):
                encoded.name = None
        return encoded

    @staticmethod
    def discretize_continuous(
        data: Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Sequence[Any]],
        bins: int = None,
        binning_method: str = "uniform",
        intervals: str | Sequence = "pandas",
    ) -> pd.Series | pd.DataFrame:
        """
        Discretize continuous numeric data into categorical intervals.

        Parameters
        ----------
        data : Sequence[float] | Sequence[Sequence[float]] |
               Mapping[str, Sequence[Number]]
            Numeric input data. Can be 1D (sequence of numbers),
            2D (sequence of sequences), or a mapping of column names to sequences.

        bins : int | Sequence[int] | Mapping[str, int], default=None
            Number of discrete bins (intervals) to split each numeric feature into.

            - **int** — applies the same number of bins to all columns.
              Example: ``bins=5`` is equivalent to
              ``bins={'col1': 5, 'col2': 5, ..., 'colN': 5}``.

            - **Sequence[int]** — specifies an individual number of bins for each
              column, in order of appearance. The sequence length must match
              the number of columns.
              Example: ``bins=[3, 2, 3]`` ≡ ``bins={'col1': 3, 'col2': 2, 'col3': 3}``.

            - **Mapping[str, int]** — explicit per-column specification.
              Keys must exactly match the column names present in ``data``.
              Missing or extra keys will raise ``KeyError``.

            The number of bins must be a **positive integer** for every column.

            If not provided, the number of bins is automatically estimated
            using **Sturges' formula**:

                k = 1 + 3.322log_10(n)

            where :`n` is the number of samples per column.

            **Priority of bin determination:**
                1. If ``intervals`` is a sequence of custom labels, its length
                   defines the number of bins (even if ``bins`` is specified).
                2. Otherwise, ``bins`` is used as provided.
                3. If neither ``bins`` nor ``intervals`` defines the bin count,
                   the Sturges' rule is applied.

            Example
            -------
            >>> data = {'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 40, 50]}
            >>> discretize_continuous(data, bins={'x': 3, 'y': 2})
                            x              y
            0  (0.995, 2.333]  (9.959, 30.0]
            1  (0.995, 2.333]  (9.959, 30.0]
            2  (2.333, 3.667]  (9.959, 30.0]
            3    (3.667, 5.0]   (30.0, 50.0]
            4    (3.667, 5.0]   (30.0, 50.0]
        binning_method
            Binning method to use. One of:
            - 'uniform': Equal-width binning.
            - 'quantile': Quantile-based binning (equal number of observations per bin).
        intervals : str | Sequence, default="pandas"
            Defines how the bins are labeled or represented.

            - **str** — predefined string flags:
              - `"pandas"`: returns pandas.Interval objects for each bin (default).
              - `"string"`: returns string representations of intervals, e.g., "(a, b]".

            - **Sequence** — custom labels:
              - 1D sequence: same labels applied to all columns.
                Example: ``intervals=["A", "B", "C"]`` ≡
                ``labels={"col1": ["A","B","C"], "col2": ["A","B","C"], ...}``.
              - Nested sequence (list of sequences): one sequence per column,
                order matches columns in ``data``. Example:
                ``intervals=[["A","B","C"], ["X","Y"]]`` ≡
                ``labels={"col1":["A","B","C"], "col2":["X","Y"]}``.
              - Mapping[str, Sequence]: explicitly assign labels per column.
                Keys must match column names; values can be any sequence
                (list, tuple, np.array).
                Example: ``intervals={"col1": (1,2,3), "col2": np.array([10,20])}``.

            **Behavior**
                - When a custom sequence or mapping is provided,
                  it **overrides `bins`**:
                  the number of labels defines the number of bins for each column.
                - Each label sequence **must contain unique values** within its column.
                  Duplicate labels in the same column are not allowed.
                - Mismatched column names in nested sequences or mappings will
                  raise a ``KeyError``.

            Example
            -------
            >>> data = {'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 40, 50]}
            >>> data_quality.discretize_continuous(data,
                    intervals={'x': ["A", "B", "C"], 'y': ["A", "B"]})
               x  y
            0  A  A
            1  A  A
            2  B  A
            3  C  B
            4  C  B

        Returns
        -------
        pd.Series or pd.DataFrame
            Categorical representation of the binned data. Returns a Series for
            a single-column input, or a DataFrame for multi-column input.

        Warns
        -----
        UserWarning
            If the number of bins specified for a feature exceeds the number of its
            unique values. In this case, the number of bins will be automatically
            reduced to ``n_unique - 1`` for the corresponding column.
            A warning message will inform the user of this adjustment.

        Raises
        ------
        ValueError
            If `binning_method` or `intervals` is unsupported.
            If `intervals` is sequence and don't match bin count.
            If `data` contains NaN values.
            If `intervals` is sequence and contains NaN values.
            If `bins` is negative or not an integer.
        KeyError
            If the keys of `bins` or `intervals` (when provided as a mapping)
            do not match the column names of the input data.
            Also raised if `intervals` or `bins` are provided as sequences whose
            lengths do not correspond to the number of input features.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        ...
        >>> import explorica.data_quality as data_quality
        ...
        >>> df = pd.DataFrame({"f1": np.linspace(0, 1000, 100),
        ...                    "f2": np.linspace(0, 2150, 100)})
        >>> encoded = data_quality.discretize_continuous(df, bins=[10, 15])
        >>> print(encoded)
        f1                  f2
        0   (-1.001, 100.0]   (-2.151, 143.333]
        1   (-1.001, 100.0]   (-2.151, 143.333]
        2   (-1.001, 100.0]   (-2.151, 143.333]
        3   (-1.001, 100.0]   (-2.151, 143.333]
        4   (-1.001, 100.0]   (-2.151, 143.333]
        ..              ...                 ...
        95  (900.0, 1000.0]  (2006.667, 2150.0]
        96  (900.0, 1000.0]  (2006.667, 2150.0]
        97  (900.0, 1000.0]  (2006.667, 2150.0]
        98  (900.0, 1000.0]  (2006.667, 2150.0]
        99  (900.0, 1000.0]  (2006.667, 2150.0]

        [100 rows x 2 columns]
        """

        def encode(
            data: pd.DataFrame,
            bins: dict,
            labels: dict[str, list],
            interval_format: str,
            how: str = "uniform",
        ):
            encoded = pd.DataFrame([])
            if how == "uniform":
                for col in data.columns:
                    labels_by_col = labels[col] if labels is not None else None
                    encoded[col] = pd.cut(
                        data[col],
                        bins=bins[col],
                        labels=labels_by_col,
                        include_lowest=True,
                    )
            elif how == "quantile":
                for col in data.columns:
                    if data[col].nunique() < bins[col]:
                        bins_by_col = data[col].nunique() - 1
                        if bins_by_col == 0:
                            wmsg = (
                                f"Number of unique values for '{col}' column less than"
                                f"number of bins. Column cannot be split."
                            )
                        else:
                            wmsg = (
                                f"Number of unique values for '{col}' "
                                "column less than number of bins. "
                                f"Column will be split into {bins_by_col} groups."
                            )
                        warnings.warn(wmsg, UserWarning)
                    else:
                        bins_by_col = bins[col]
                    labels_by_col = labels[col] if labels is not None else None
                    encoded[col] = pd.qcut(
                        data[col], q=bins_by_col, labels=labels_by_col
                    )
            if interval_format == "string":
                encoded = encoded.astype(str)
            encoded = encoded.squeeze(axis=1)
            return encoded

        df = convert_dataframe(data)

        supported = {
            "binning_methods": {"uniform", "quantile"},
            "interval_formats": {"pandas", "string", "custom"},
        }

        params = EncodeMethods._discretize_continuous_transform_input_parameters(
            df, bins, intervals, binning_method, supported
        )

        interval_format = params["interval_format"]
        binning_method = params["binning_method"]
        labels, bins = params["labels"], params["bins"]

        EncodeMethods._discretize_continuous_validate_input_parameters(
            df,
            bins,
            interval_format=interval_format,
            binning_method=binning_method,
            supported=supported,
            labels=labels,
        )
        encoded = encode(df, bins, labels, interval_format, how=binning_method)
        return encoded

    @staticmethod
    def _discretize_continuous_validate_input_parameters(
        data: pd.DataFrame,
        bins: dict[Hashable, int | np.integer],
        labels: dict[Hashable, list],
        supported: dict,
        **kwargs,
    ):
        """
        Internal helper for `data_quality.discretize_continuous`.

        Validates all input parameters before discretization step.
        Ensures data integrity and consistency of configuration parameters.

        Checks:
        1. `data` or `labels` does not contain NaN values.
        2. Each value in `bins` is a positive integer.
        3. `interval_format` is one of the supported formats.
        4. `binning_method` is one of the supported methods.

        Raises
        ------
        ValueError
            If any of the validation rules are violated.
        """
        interval_format = kwargs.get("interval_format")
        binning_method = kwargs.get("binning_method")
        validate_string_flag(
            interval_format,
            supported["interval_formats"],
            EncodeMethods._errors["unsupported_method_f"].format(
                interval_format, supported["interval_formats"]
            ),
        )

        validate_string_flag(
            binning_method,
            supported["binning_methods"],
            EncodeMethods._errors["unsupported_method_f"].format(
                binning_method, supported["binning_methods"]
            ),
        )

        validate_array_not_contains_nan(
            data, EncodeMethods._errors["array_contains_nans_f"].format("data")
        )
        if labels is not None:
            for col in list(labels):
                if any(pd.isna(labels[col])):
                    raise ValueError(
                        EncodeMethods._errors["array_contains_nans_f"].format(
                            "intervals"
                        )
                    )

        for bins_number in bins.values():
            if not isinstance(bins_number, (np.integer, int)) or bins_number <= 0:
                raise ValueError(
                    EncodeMethods._errors["is_not_positive_integer_f"].format(
                        "bins (or each element of bins)"
                    )
                )

    @staticmethod
    def _discretize_continuous_transform_input_parameters(
        data: pd.DataFrame,
        bins: int | list[int] | dict[list[int]],
        intervals: str | Sequence,
        binning_method: str,
        supported: dict,
    ):
        """
        Internal helper for `data_quality.discretize_continuous`.

        Transforms user input parameters into standardized internal form.

        Performs:
        1. **bins**
        - Converts `bins` to `{column_name: int}`.
        - If `bins` is None, computes optimal number of bins (Sturges formula).
        - If `labels` are provided, overrides `bins` by their length per column.

        2. **intervals**
        - If `intervals` is a string → treated as interval format
          (e.g., "pandas", "str").
        - If `intervals` is a list/dict → treated as user-provided labels per column.

        3. **interval_format** and **binning_method**
        - Normalized via aliases from `supported`.

        Returns
        -------
        dict
            Dictionary with standardized parameters:
            {
                "interval_format": str,
                "binning_method": str,
                "labels": dict[str, list[str]] | None,
                "bins": dict[str, int]
            }
        """

        def transform_labels(data: pd.DataFrame, labels):
            if labels is None:
                return None

            if isinstance(labels, pd.Series):
                labels2d = {col: labels.to_list() for col in data.columns}
            elif isinstance(labels, pd.DataFrame):
                labels2d = {col: labels[col].to_list() for col in labels.columns}
            elif isinstance(labels, Mapping):
                labels2d = dict(labels)
            elif (
                (isinstance(labels, Sequence))
                and (not isinstance(labels, str))
                or (isinstance(labels, np.ndarray))
            ):
                if (
                    (isinstance(labels[0], Sequence))
                    and (not isinstance(labels[0], str))
                    or (isinstance(labels[0], np.ndarray))
                ):
                    if len(labels) != len(data.columns):
                        print(type(labels[0]))
                        raise KeyError(
                            f"intervals length {len(labels)} must match number "
                            f"of data columns {len(data.columns)}."
                        )
                    labels2d = {
                        col: list(lbls) for col, lbls in zip(data.columns, labels)
                    }
                else:
                    # 1D sequence applied to all columns
                    labels2d = {col: list(labels) for col in data.columns}
            else:
                raise TypeError(f"Unsupported type for labels: {type(labels)}")

            if set(labels2d) != set(data.columns):
                raise KeyError("`intervals` keys must match `data` columns or indeces")

            return labels2d

        def transform_bins(data: pd.DataFrame, bins: int | list | dict, labels: dict):
            if labels is not None:
                bins = {col: len(labels[col]) for col in data.columns}
            elif bins is not None:
                if isinstance(bins, int):
                    bins = {col: bins for col in data.columns}
                elif isinstance(bins, list):
                    if len(bins) != data.shape[1]:
                        raise KeyError(
                            f"Number of features in 'data' ({data.shape[1]}) must "
                            f"match the number of bins ({len(bins)})."
                        )
                    bins = {col: bins[i] for i, col in enumerate(data.columns)}
                elif isinstance(bins, dict):
                    pass
            else:
                bins = {
                    col: int(np.round(1 + 3.3222 * np.log10(data.shape[0])))
                    for col in data.columns
                }
            if set(bins) != set(data.columns):
                raise KeyError("`bins` keys must match `data` keys or indeces")
            return bins

        interval_format = intervals if isinstance(intervals, str) else "custom"
        # labels is list or dict of lists
        labels = intervals if not isinstance(intervals, str) else None
        labels2d = transform_labels(data, labels)
        bins = transform_bins(data, bins, labels2d)
        interval_format = convert_from_alias(
            interval_format, supported["interval_formats"]
        )
        binning_method = convert_from_alias(
            binning_method, supported["binning_methods"]
        )
        output = {
            "interval_format": interval_format,
            "binning_method": binning_method,
            "labels": labels2d,
            "bins": bins,
        }
        return output
