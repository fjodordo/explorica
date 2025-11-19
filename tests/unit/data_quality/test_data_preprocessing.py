import logging
import json

import pytest
import pandas as pd
import numpy as np


import explorica.data_quality as data_quality

# tests for data_quality.get_missing()

@pytest.mark.parametrize(
    "data, round_digits, ascending",
    [
        # 1D sequence
        ([1, 2, np.nan, 4], 2, True),
        ([1, 2, np.nan, 4], 2.0, False),
        ([1, 2, np.nan, 4], np.int64(3), None),
        ([1, 2, np.nan, 4], np.float32(1), np.bool_(True)),

        # 2D nested list
        ([[1, 2], [3, 4], [np.nan, 6]], np.int32(0), None),

        # dict of lists
        ({"a": [1, 2, np.nan], "b": [4, 5, 6]}, np.float64(2), np.bool_(False)),

        # DataFrame
        (pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, 6]}), None, np.bool_(True)),
    ]
)
def test_get_missing_different_sequences_and_dtypes(data, round_digits, ascending):
    report = data_quality.get_missing(data, ascending=ascending, round_digits=round_digits)
    assert report is not None



@pytest.mark.parametrize("data, expected, kwargs",
[
    (
        pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8, np.nan],
                    "B": [None, pd.NA, 5, 4, 3, 2, 1, 0, -1]}),
        # default pct_of_nans format is 6 digits after floating point.
        pd.DataFrame({"count_of_nans": [1, 2],
                    "pct_of_nans": [0.111111, 0.222222]},
                    index=["A", "B"]),
        {"round_digits": None, "ascending": None}
    ),
    (
        pd.DataFrame({"A": [pd.NA, pd.NaT, np.nan, None, np.float64("nan")]}),
        pd.DataFrame({"count_of_nans": [5], "pct_of_nans": [1.0]}, index=["A"]),
        {"round_digits": None, "ascending": None}

    ),
    (
        pd.Series([1, 2, 3, 4]),
        pd.DataFrame({"count_of_nans": [0], "pct_of_nans": [0.0]}, index=[0]),
        {}
    ),
    (
        pd.DataFrame({
        "dates": [pd.Timestamp("2025-01-01"), pd.NaT, pd.Timestamp("2025-01-03")]
        }),
        pd.DataFrame({"count_of_nans": [1], "pct_of_nans": [0.333333]}, index=["dates"]),
        {}
    ),
    (
        pd.DataFrame({"A": [], "B": []}),
        pd.DataFrame({"count_of_nans": [0, 0], "pct_of_nans": [np.float64("nan"), np.float64("nan")]}, index=["A", "B"]),
        {}
    ),
    (
        pd.DataFrame({"A": [1, None, pd.NA, "x", np.nan]}),
        pd.DataFrame({"count_of_nans": [3], "pct_of_nans": [0.6]}, index=["A"]),
        {}
    ),
    (
        pd.DataFrame({"A": [1,2,3], "B": [4,5,6]}),
        pd.DataFrame({"count_of_nans": [0,0], "pct_of_nans": [0.0,0.0]}, index=["A","B"]),
        {}
    ),
    (
        [[1, 2], [None, 3], [np.nan, 4]],
        pd.DataFrame({"count_of_nans": [0, 1, 1], "pct_of_nans": [0.0, 0.5, 0.5]}, index=[0,1,2]),
        {}
    ),
    (
        pd.DataFrame({
            "A": [1, np.nan, 3, np.nan],   # 2 NaN -> count=2
            "B": [np.nan, np.nan, np.nan, np.nan],  # 4 NaN -> count=4
            "C": [1, 2, 3, 4]              # 0 NaN -> count=0
        }),
        pd.DataFrame({
            "count_of_nans": [0, 2, 4],
            "pct_of_nans": [0.0, 0.5, 1.0]
        }, index=["C", "A", "B"]),
        {"ascending": True, "round_digits": 1}
    ),
    (
        pd.DataFrame({
            "A": [1, np.nan, 3, np.nan],
            "B": [np.nan, np.nan, np.nan, np.nan],
            "C": [1, 2, 3, 4]
        }),
        pd.DataFrame({
            "count_of_nans": [4, 2, 0],
            "pct_of_nans": [1.0, 0.5, 0.0]
        }, index=["B", "A", "C"]),
        {"ascending": False, "round_digits": 1}
    )
])
def test_get_missing_deterministic(data, expected, kwargs):
    report = data_quality.get_missing(data, **kwargs)
    pd.testing.assert_frame_equal(report, expected)

def test_get_missing_empty_input():
    report = data_quality.get_missing(pd.DataFrame([]))
    expected = pd.DataFrame(columns=["count_of_nans", "pct_of_nans"])
    expected["count_of_nans"] = expected["count_of_nans"].astype(np.int64)
    expected["pct_of_nans"] = expected["pct_of_nans"].astype(np.float64)
    pd.testing.assert_frame_equal(report, expected, check_dtype=True)

def test_get_missing_not_unique_col_names():
    data = pd.DataFrame({"A": [1], "B": [2], "C": [2], "D": [2], "E": [2]})
    data.columns = [None] * 5
    with pytest.raises(ValueError, match="Duplicate keys detected in"):
        data_quality.get_missing(data)
    
@pytest.mark.parametrize("round_digits", [-1, 1.4, -1.3])
def test_get_missing_unacceptable_round_digits(round_digits):
    data = pd.DataFrame({"A": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="'round_digits' must be a non-negative integer"):
        data_quality.get_missing(data, round_digits=round_digits)

# tests for data_quality.drop_missing()


@pytest.mark.parametrize("data", [

    # DataFrame with np.nan
    pd.DataFrame({"A": [1, 2, 3, np.nan, 5], "B": [5, 6, 7, 8, 9]}),

    # DataFrame with pd.NA
    pd.DataFrame({"A": [1, 2, pd.NA, 4, 5], "B": [10, 20, 30, 40, 50]}),

    # DataFrame with None
    pd.DataFrame({"A": [1, None, 3, 4, 5], "B": [5, 6, 7, 8, 9]}),

    # DataFrame with float('nan')
    pd.DataFrame({"A": [1, 2, float('nan'), 4, 5], "B": [10, 20, 30, 40, 50]}),

    # DataFrame with pd.NaT (datetime column)
    pd.DataFrame({"A": [pd.Timestamp("2025-01-01"), pd.NaT, pd.Timestamp("2025-01-03")],
                  "B": [1, 2, 3]}),

    # numpy array with np.nan
    np.array([1, 2, 3, np.nan, 5]),

    # list without NaNs
    [1, 2, 3, 4, 5],

    # dict mapping with np.nan
    {"col1": [1, np.nan, 3], "col2": [4, 5, 6]},

    # dict mapping without NaNs
    {"col1": [1, 2, 3], "col2": [4, 5, 6]},
])
def test_drop_missing_different_sequences_and_dtypes(data):
    removed = data_quality.drop_missing(data)
    assert removed is not None
    assert isinstance(removed, pd.DataFrame)

def test_drop_missing_empty_input():
    removed = data_quality.drop_missing(pd.DataFrame([]))
    assert removed.empty
    removed = data_quality.drop_missing(pd.DataFrame([], columns=["a"]))
    assert removed.empty
    assert set(removed.columns) == {"a"}


@pytest.mark.parametrize("data, expected, kwargs",
[
    (
        pd.DataFrame({"A": [1,2,3,4,5,6,7,np.nan]}),
        {"function": "drop_missing",
         "threshold": 0.25,
         "removed_rows": 1,
         "axis": 0,
         "affected_columns": ["A"],
         "length" : "8->7"},
         {"axis": 0, "threshold_abs": 2}
    ),
    (
        pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [1, 2, 3, 4]}),
        {"function": "drop_missing",
         "threshold": 0.3,
         "removed_rows": 1,
         "axis": 0,
         "affected_columns": ["A"],
         "length": "4->3"},
        {"axis": 0, "threshold_pct": 0.3}
    ),
    (
        pd.DataFrame({"A": [1, np.nan, np.nan, 4],
                      "B": [1, 2, 3, 4],
                      "C": [np.nan, np.nan, np.nan, np.nan]}),
        {"function": "drop_missing",
         "threshold": 0.5,
         "removed_columns": 1,
         "axis": 1,
         "affected_columns": ["C"],
         "width": "3->2"},
        {"axis": 1, "threshold_pct": 0.5}
    ),
    (
        pd.DataFrame({"A": [1, np.nan, 3, 4, np.nan],
                      "B": [1, 2, 3, 4, 5],
                      "C": [np.nan, np.nan, np.nan, np.nan, np.nan]}),
        {"function": "drop_missing",
         "threshold": 0.4,   # threshold_abs=2, total=5 => 2/5=0.4
         "removed_columns": 1,
         "axis": 1,
         "affected_columns": ["C"],
         "width": "3->2"},
        {"axis": 1, "threshold_abs": 2}
    ),
])
def test_drop_missing_verbose(data, expected, kwargs, caplog):
    with caplog.at_level(logging.INFO):
        data_quality.drop_missing(data, verbose=True, **kwargs)
    assert len(caplog.records) > 0
    log_record = caplog.records[-1].msg
    log_dict = log_record if isinstance(log_record, dict) else json.loads(log_record)
    for key, value in expected.items():
        assert log_dict[key] == value

def test_drop_missing_full_nan_axis1():
    data = pd.DataFrame([np.nan] * 10)
    removed = data_quality.drop_missing(data, axis=1)
    assert removed.empty
    assert len(removed.columns) == 0

@pytest.mark.skip(reason="Performance test - too heavy for CI environment")
def test_drop_missing_large_sequence():
    data = [1] * 10_000_000
    nan_indeces = [0, 100, 1000, 1001, 250000, 321000]
    for i in nan_indeces:
        data[i] = np.nan
    removed = data_quality.drop_missing(data, axis=0)
    assert len(data) - removed.shape[0] == 6

@pytest.mark.parametrize("axis", [0, 1])
def test_drop_missing_zero_abs(axis):
    """
    Test that threshold_abs=0 when:
    - axis = 0 - does not remove anything.
    - axis = 1 - remove anything that contains NaNs.
    """
    df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, 6]})
    result = data_quality.drop_missing(df, axis=axis, threshold_abs=0)
    if axis == 0:
        assert result.shape == df.shape
    elif axis == 1:
        assert set(result.columns) == {"B"}
    
def test_drop_missing_threshold_abs_more_than_sample_size():
    """Verify that a ValueError is raised when threshold_abs exceeds number of rows."""
    df = pd.DataFrame({"A": [1, np.nan], "B": [1, 2]})
    with pytest.raises(ValueError):
        data_quality.drop_missing(df, axis=0, threshold_abs=5)

@pytest.mark.parametrize("invalid_pct", [-0.1, 1.5])
def test_drop_missing_threshold_pct_invalid_domain(invalid_pct):
    """ValueError should be raised for invalid threshold_pct values."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    with pytest.raises(ValueError):
        data_quality.drop_missing(df, axis=0, threshold_pct=invalid_pct)

@pytest.mark.parametrize("invalid_axis", [-1, 2, 42])
def test_drop_missing_invalid_axis(invalid_axis):
    """Verify that invalid axis raises a ValueError."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    with pytest.raises(ValueError):
        data_quality.drop_missing(df, axis=invalid_axis)

def test_drop_missing_threshold_abs_overrides_pct(caplog):
    """Check that threshold_abs takes precedence over threshold_pct."""
    df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, np.nan]})
    with caplog.at_level(logging.INFO):
        data_quality.drop_missing(df, axis=0, threshold_abs=1, threshold_pct=0, verbose=True)
    # 'threshold' in logs must be 1/3, not 0.0
    log_record = caplog.records[-1].msg
    log_dict = log_record if isinstance(log_record, dict) else json.loads(log_record)
    assert log_dict["threshold"] == 1/3

def test_drop_missing_df_with_duplicate_columns():
    """ValueError should be raised if DataFrame has duplicate column names."""
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "A"])
    with pytest.raises(ValueError):
        data_quality.drop_missing(df)

def test_drop_missing_turned_off_verbose(caplog):
    """Ensure that no logs are produced when verbose is False."""
    df = pd.DataFrame({"A": [1, np.nan, 3]})
    with caplog.at_level(logging.INFO):
        data_quality.drop_missing(df, verbose=False)
    assert len(caplog.records) == 0

def test_drop_missing_multiindex_df():
    tuples = [("X", "a"), ("X", "b"), ("Y", "c")]
    index = pd.MultiIndex.from_tuples([("row1", 0), ("row2", 1), ("row3", 2)])
    df = pd.DataFrame([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]], index=index, columns=tuples)
    result = data_quality.drop_missing(df, axis=0, threshold_pct=0.5)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 3)
    assert set(result.index) == {("row3", 2)}

@pytest.mark.parametrize("data, expected, kwargs",
[
    # ------------------------------
    # 1. Single column, 1 NaN, threshold_abs=1
    # ------------------------------
    (
        pd.DataFrame([0, 1, 2, 3, 4, 5, pd.NA]),
        pd.DataFrame([0, 1, 2, 3, 4, 5, pd.NA]),
        {"threshold_abs": 1, "axis": 0}
    ),

    # ------------------------------
    # 2. Multiple columns, some NaNs, axis=0, threshold_abs
    # ------------------------------
    (
        pd.DataFrame({"A": [1, 2, np.nan, 4],
                      "B": [5, np.nan, 7, 8]}),
        pd.DataFrame({"A": [1, 2, np.nan, 4],
                      "B": [5, np.nan, 7, 8]}),
        {"threshold_abs": 1, "axis": 0}
    ),

    # ------------------------------
    # 3. axis=0, threshold_pct
    # ------------------------------
    (
        pd.DataFrame({"A": [1, np.nan, 3, 4],
                      "B": [np.nan, 2, 3, 4]}),
        pd.DataFrame({"A": [3.0, 4.0],
                      "B": [3.0, 4.0]}, index=[2, 3]),
        {"threshold_pct": 0.3, "axis": 0}
    ),

    # ------------------------------
    # 4. axis=1, threshold_abs
    # ------------------------------
    (
        pd.DataFrame({"A": [1, 2, np.nan, 4],
                      "B": [1, 2, 3, 4],
                      "C": [np.nan, np.nan, np.nan, np.nan]}),
        pd.DataFrame({"A": [1, 2, np.nan, 4],
                      "B": [1, 2, 3, 4]}),
        {"threshold_abs": 1, "axis": 1}
    ),

    # ------------------------------
    # 5. axis=1, threshold_pct
    # ------------------------------
    (
        pd.DataFrame({"A": [np.nan, 2, 3, 4],
                      "B": [1, np.nan, 3, 4],
                      "C": [np.nan, np.nan, np.nan, np.nan]}),
        pd.DataFrame({"A": [np.nan, 2, 3, 4],
                      "B": [1, np.nan, 3, 4]}),
        {"threshold_pct": 0.5, "axis": 1}
    ),

    # ------------------------------
    # 6. No NaNs, axis=0
    # ------------------------------
    (
        pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        {"threshold_abs": 1, "axis": 0}
    ),

    # ------------------------------
    # 7. No NaNs, axis=1
    # ------------------------------
    (
        pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        {"threshold_pct": 0.1, "axis": 1}
    ),

    # ------------------------------
    # 8. Fully NaN column, axis=1
    # ------------------------------
    (
        pd.DataFrame({"A": [np.nan, np.nan, np.nan], "B": [1, 2, 3]}),
        pd.DataFrame({"B": [1, 2, 3]}),
        {"threshold_pct": 0.5, "axis": 1}
    ),

])
def test_drop_missing_deterministic(data, expected, kwargs):
    removed = data_quality.drop_missing(data, **kwargs)
    pd.testing.assert_frame_equal(removed, expected)

# tests for data_quality.get_constant_features()


def test_get_constant_features_input_contains_nans():
    data = [1, 2, np.nan, 3]
    with pytest.raises(ValueError, match="contains null values"):
        data_quality.get_constant_features(data, nan_policy='raise')
    report_drop = data_quality.get_constant_features(data, nan_policy='drop')
    report_include = data_quality.get_constant_features(data, nan_policy='include')
    assert (report_drop["top_value_ratio"] > report_include["top_value_ratio"]).all()


@pytest.mark.parametrize("data, kwargs",
[
    ([1, 2, 3, 4, 5], {"method": "top_value_ratio"}),  # list
    ((1, 2, 3, 4, 5), {"method": "top_value_ratio"}),  # tuple
    (np.array([1, 2, 3, 4, 5]), {"method": "top_value_ratio"}),  # numpy array
    (pd.Series([1, 2, 3, 4, 5]), {"method": "top_value_ratio"}),  # pandas Series

    ([[1, 1, 1], [2, 2, 3], [4, 4, 4]], {"method": "top_value_ratio"}),  # list of lists
    (np.array([[1, 1, 1], [2, 2, 3], [4, 4, 4]]), {"method": "top_value_ratio"}),  # numpy 2D
    (pd.DataFrame([[1, 1, 1], [2, 2, 3], [4, 4, 4]]), {"method": "top_value_ratio"}),  # DataFrame (no column names)

    ({"A": [1, 1, 1], "B": [1, 2, 3]}, {"method": "top_value_ratio"}),  # dict of lists
    ({"X": np.arange(5), "Y": np.random.randint(0, 3, size=5)}, {"method": "top_value_ratio"}),  # dict of arrays
    (pd.DataFrame({"A": [1, 2, 3], "B": [3, 3, 3]}), {"method": "top_value_ratio"}),  # proper DataFrame

    (pd.DataFrame({"A": [1, np.nan, 3], "B": [np.nan, np.nan, 3]}),
     {"method": "top_value_ratio", "nan_policy": "include"}),
])
def test_get_constant_features_different_sequences_and_dtypes(data, kwargs):
    result = data_quality.get_constant_features(data, **kwargs)
    assert result is not None
    assert set(result.columns) == {"top_value_ratio", "is_const"}

def test_get_constant_features_empty_input():
    empty_data = []
    result = data_quality.get_constant_features(empty_data, method="top_value")
    assert result.empty

    empty_df = pd.DataFrame([], columns=["A", "B"])
    result = data_quality.get_constant_features(empty_df)
    assert set(result.index) == {"A", "B"}
    assert set(result.columns) == {"top_value_ratio", "is_const"}
    assert "is_const" in result.columns


@pytest.mark.parametrize("data, method, threshold, expected_is_const", [
    # --- top_value_ratio ---
    ([1, 1, 1, 1, 1], "top_value_ratio", 0.8, [1.0]),

    ([1, 1, 1, 2, 2], "top_value_ratio", 0.5, [1.0]),

    ([1, 2, 3, 4, 5], "top_value_ratio", 0.6, [0.0]),

    ([1, 2, 3, 4], "non_uniqueness", 0.2, [0.0]),

    ([1, 1, 2, 2, 3, 4], "non_uniqueness", 0.5, [0.0]),

    # --- entropy ---
    (pd.DataFrame({"A": [5, 5, 5, 5]}), "entropy", 0.5, [1.0]),

    (pd.DataFrame({"A": [0, 0, 1, 1]}), "entropy", 0.8, [0.0]),

    (pd.DataFrame({"A": [1, 1, 2, 2, 3, 3]}), "entropy", 0.5, [0.0]),

    (pd.DataFrame({"A": [1, 1, 1, 1, 2, 3]}), "entropy", 1.0, [0.0]),

    (pd.DataFrame({
        "A": [1, 1, 1, 1, 1, 1],        # const
        "B": [1, 2, 3, 4, 5, 6],        # varying
        "C": [1, 1, 2, 2, 3, 3]         # moderate entropy
     }),
     "entropy", 0.8, [1.0, 0.0, 0.0]),

    # --- edge case ---
    (pd.DataFrame({"A": [1, 1, np.nan, np.nan]}), "entropy", 1.5, [1.0]),
])
def test_get_constant_features_example_based(data, method, threshold, expected_is_const):
    result = data_quality.get_constant_features(data, method=method, threshold=threshold)
    assert all(result["is_const"].values == expected_is_const)


def test_get_constant_features_invalid_method():
    data = [1,2,3,4]
    with pytest.raises(ValueError, match="Unsupported"):
        data_quality.get_constant_features(data, method="missing_method")


def test_get_constant_features_invalid_threshold():
    data = [1,2,3,4]
    with pytest.raises(ValueError):
        # threshold must be a number, not negative or wrong type
        data_quality.get_constant_features(data, threshold=-1)
    with pytest.raises(TypeError):
        data_quality.get_constant_features(data, threshold="invalid")


# tests for data_quality.get_constant_features()

def test_get_categorical_features_input_contains_nans():
    data = pd.DataFrame({
        "A": [1, 2, np.nan, 4],
        "B": [np.nan, 2, 3, 4]
    })
    with pytest.raises(ValueError, match="contains null values"):
        data_quality.get_categorical_features(data, nan_policy="raise")
    report_without_nans = data_quality.get_categorical_features(data, nan_policy="drop")
    assert report_without_nans is not None
    # expected output:
    # |     | categories_count | is_category |
    # | --- | ---------------- | ----------- |
    # | A   | 2                | 0           |
    # | B   | 2                | 0           |
    assert (report_without_nans["categories_count"] == 2).all()

    report_with_nans = data_quality.get_categorical_features(data, nan_policy="include")
    assert report_with_nans is not None
    # expected output:
    # |     | categories_count | is_category |
    # | --- | ---------------- | ----------- |
    # | A   | 4                | 0           |
    # | B   | 4                | 0           |
    assert (report_with_nans["categories_count"] == 4).all()

# It is necessary to test behavior with NaNs

@pytest.mark.parametrize("data, kwargs", [
    # list of ints
    ([1, 2, 3, 4, 5], {}),
    # list of lists
    ([[1, 2, 3], [4, 5, 6]], {}),
    # dict of lists
    ({"a": [1, 1, 1], "b": [2, 2, 3]}, {}),
    # tuple of ints
    ((1, 2, 3, 4), {}),
    # tuple of tuples
    (((1, 2), (3, 4)), {}),
    # numpy array 1D
    (np.array([1, 2, 3, 4]), {}),
    # numpy array 2D
    (np.array([[1, 2], [3, 4]]), {}),
    # pandas Series
    (pd.Series([1, 2, 2, 3]), {}),
    # pandas DataFrame
    (pd.DataFrame({"A": [1, 1, 2], "B": [3, 4, 4]}), {}),
])
def test_get_categorical_features_different_sequences_and_dtypes(data, kwargs):
    result = data_quality.get_categorical_features(data, **kwargs)
    assert result is not None
    assert isinstance(result, pd.DataFrame)

def test_get_categorical_features_empty_input():
    empty_data = []
    result = data_quality.get_categorical_features(empty_data)
    assert result.empty
    assert set(result.columns).issuperset({"is_category", "categories_count"})
    result = data_quality.get_categorical_features(empty_data, sign_bin=True, sign_const=True)
    assert result.empty
    assert set(result.columns).issuperset({"is_category", "categories_count",
                                           "is_binary", "is_constant"})

def test_get_categorical_features_threshold_lengths_mismatch():
    df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6]})
    threshold_list = [10]
    with pytest.raises(ValueError, match="'threshold'"):
        data_quality.get_categorical_features(df, threshold=threshold_list)

    threshold_dict = {"A": 10, "C": 5}
    with pytest.raises(ValueError, match="'threshold'"):
        data_quality.get_categorical_features(df, threshold=threshold_dict)

@pytest.mark.parametrize("include, include_flags, expected_is_category", 
    [
        ({"number"}, {}, [1, 1, 0, 0, 0]),
        ({"int", "str"}, {}, [0, 1, 1, 0, 0]),
        ({"object"}, {}, [0, 0, 1, 0, 0]),
        ({"bool"}, {}, [0, 0, 0, 1, 0]),
        ({"datetime"}, {}, [0, 0, 0, 0, 1]),
        ({"number", "object"}, {}, [1, 1, 1, 0, 0]),
        (set(), {}, [0, 0, 1, 0, 0]),
        
        ({}, {"include_number": True}, [1, 1, 0, 0, 0]),
        ({}, {"include_int": True}, [0, 1, 0, 0, 0]),
        ({}, {"include_str": True}, [0, 0, 1, 0, 0]),
        ({}, {"include_bool": True}, [0, 0, 0, 1, 0]),
        ({}, {"include_datetime": True}, [0, 0, 0, 0, 1]),
        ({}, {"include_bin": True}, [0, 0, 0, 1, 0]),
        ({}, {"include_const": True}, [0, 0, 0, 0, 0]),
        
        ({}, {"include_number": True, "include_str": True}, [1, 1, 1, 0, 0]),
        ({}, {"include_int": True, "include_bool": True}, [0, 1, 0, 1, 0]),
        ({}, {"include_all": True}, [1, 1, 1, 1, 1]),
    
        ({"number"}, {"include_str": True}, [1, 1, 0, 0, 0]),
        ({"str"}, {"include_number": True, "include_bool": True}, [0, 0, 1, 0, 0]),
        
        (None, {}, [0, 0, 1, 0, 0]),
        ({}, {}, [0, 0, 1, 0, 0]),
        
        ({"bin"}, {}, [0, 0, 0, 1, 0]),
        ({"constant"}, {}, [0, 0, 0, 0, 0]),
    ])
def test_get_categorical_features_include_behavior(include, include_flags, expected_is_category):
    # df with various data types
    data = pd.DataFrame({
        "float_col": [1.1, 2.2, 3.3, 4.4],  # number
        "int_col": [1, 2, 2, 3],            # int (number)
        "str_col": ["a", "b", "b", "c"],    # object/str  
        "bool_col": [True, False, True, False],  # bool (binary)
        "date_col": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-03"]),
    })
    result = data_quality.get_categorical_features(data, threshold=999, include=include, **include_flags)
    assert (result["is_category"] == expected_is_category).all()

@pytest.mark.parametrize(
    "threshold, include, expected_is_category, error_type", 
    [
        # 1. Scalar threshold (broadcast to all columns)
        (2, {"include_all": True}, [1, 0], None),
        (3, {"include_all": True}, [1, 0], None),
        (4, {"include_all": True}, [1, 1], None),
        (1, {"include_all": True}, [0, 0], None),
        (0, {"include_all": True}, [0, 0], ValueError),  # edge case: 0 threshold
        
        # 2. Sequence threshold (aligned by column order)
        ([2, 4], {"include_all": True}, [1, 1], None),  # A <= 2, B <= 4
        ([3, 2], {"include_all": True}, [1, 0], None),  # A <= 3, B > 2
        ([1, 5], {"include_all": True}, [0, 1], None),  # A > 1, B <= 5
        
        # 3. Mapping threshold (by column name)
        ({"A": 2, "B": 4}, {"include_all": True}, [1, 1], None),
        ({"A": 3, "B": 2}, {"include_all": True}, [1, 0], None),
        ({"B": 5}, {"include_all": True}, [0, 1], ValueError),
        ({"A": 1, "C": 5}, {"include_all": True}, [0, 0], ValueError),
        
        # 4. Edge cases with different data types
        (2.5, {"include_all": True}, [1, 0], ValueError),  # float threshold
        (2, {"include_int": True}, [1, 0], None),
        
        # 5. Error cases
        ("invalid", {"include_all": True}, None, ValueError),
        ([2], {"include_all": True}, None, ValueError),
        ([2, 3, 4], {"include_all": True}, None, ValueError),
        
        # 6. Empty and zero cases
        ([], {"include_all": True}, None, ValueError),
        ({}, {"include_all": True}, [0, 0], ValueError),
        
        # 7. Special threshold values
        (np.int32(2), {"include_all": True}, [1, 0], None),  # numpy scalar
        (pd.Series([2, 4])[0], {"include_all": True}, [1, 0], None),  # pandas scalar
    ]
)
def test_get_categorical_features_threshold_behavior(threshold, include, expected_is_category, error_type):
    """Test different threshold types and their behavior."""
    data = pd.DataFrame({
        "A": [1, 1, 2, 2],  # 2 unique values
        "B": [1, 2, 3, 4],  # 4 unique values
    })
    
    if error_type is not None:
        with pytest.raises(error_type):
            data_quality.get_categorical_features(data, threshold=threshold, **include)
    else:
        result = data_quality.get_categorical_features(data, threshold=threshold, **include)
        assert (result["is_category"].values == expected_is_category).all()


def test_get_categorical_features_invalid_input():
    data = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError):
        data_quality.get_categorical_features(data, nan_policy="unsupported_policy")
    with pytest.raises(ValueError):
        data_quality.get_categorical_features(data, threshold={"A": "invalid"})

def test_get_categorical_features_sign_bin_sign_const_flags():
    data = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6],
                         "B": [0, 1, 0, 1, 1, 0],
                         "C": [4, 4, 4, 4, 4, 4]})
    report_with_binary = data_quality.get_categorical_features(data, sign_bin = True)
    assert set(report_with_binary.columns) == {"categories_count", "is_category", "is_binary"}
    assert report_with_binary.loc["B", "is_binary"] == 1
    report_with_const = data_quality.get_categorical_features(data, sign_const = True)
    assert set(report_with_const.columns) == {"categories_count", "is_category", "is_constant"}
    assert report_with_const.loc["C", "is_constant"] == 1
    both = data_quality.get_categorical_features(data, sign_bin = True, sign_const = True)
    assert set(both.columns) == {"categories_count", "is_category", "is_constant", "is_binary"}

@pytest.mark.parametrize(
    "data, expected, kwargs",
    [
        # 1. Basic test with different dtypes
        (
            pd.DataFrame({
                "ints": [1, 2, 2, 3, 3],
                "floats": [1.1, 2.2, 3.3, 4.4, 5.5],
                "strings": ["a", "b", "c", "a", "b"],
                "bools": [True, False, True, False, True]
            }),
            pd.DataFrame({
                "categories_count": [3, 5, 3, 2],
                "is_category": [1, 0, 1, 1]
            }, index=["ints", "floats", "strings", "bools"]),
            {"threshold": 4, "include_all": True}
        ),
        
        # 2. Filtering by threshold
        (
            pd.DataFrame({
                "low_card":    [1, 1, 2, 2, 2, 2],      # 2 unique
                "medium_card": [1, 2, 3, 4, 4, 4],    # 4 unique  
                "high_card":   [1, 2, 3, 4, 5, 6] # 6 unique
            }),
            pd.DataFrame({
                "categories_count": [2, 4, 6],
                "is_category": [1, 1, 0]
            }, index=["low_card", "medium_card", "high_card"]),
            {"threshold": 5, "include_all": True}
        ),
        
        # 3. Filtering by data types (include_* flags)
        (
            pd.DataFrame({
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True]
            }),
            pd.DataFrame({
                "categories_count": [3, 3, 3, 2],
                "is_category": [1, 0, 1, 1]
            }, index=["int_col", "float_col", "str_col", "bool_col"]),
            {"threshold": 10, "include_int": True, "include_str": True, "include_bool": True}
        ),
        
        # 4. Binary и constant detection
        (
            pd.DataFrame({
                "binary": [1, 0, 1, 0],
                "constant": [5, 5, 5, 5],
                "normal": [1, 2, 3, 4]
            }),
            pd.DataFrame({
                "categories_count": [2, 1, 4],
                "is_category": [1, 1, 0],
                "is_binary": [1, 0, 0],
                "is_constant": [0, 1, 0]
            }, index=["binary", "constant", "normal"]),
            {"threshold": 3, "include_all": True, "sign_bin": True, "sign_const": True}
        ),
        
        # 5. NaN handling with different nan_policy
        (
            pd.DataFrame({
                "with_nan": [1, 2, np.nan, 1, 2],
                "without_nan": [1, 2, 3, 4, 5]
            }),
            pd.DataFrame({
                "categories_count": [3, 5],
                "is_category": [1, 0]
            }, index=["with_nan", "without_nan"]),
            {"threshold": 4, "include_all": True, "nan_policy": "include"}
        ),
        
        # 6. Sequence input (not DataFrame)
        (
            [[1, 1, 2], [3, 4, 5], ["a", "b", "a"]],
            pd.DataFrame({
                "categories_count": [2, 3, 2],
                "is_category": [1, 0, 1]
            }, index=[0, 1, 2]),
            {"threshold": 2, "include_all": True}
        ),
        
        # 7. Dict input with custom names
        (
            {"feature_A": [1, 2, 1, 2], "feature_B": [1, 2, 3, 4]},
            pd.DataFrame({
                "categories_count": [2, 4],
                "is_category": [1, 0]
            }, index=["feature_A", "feature_B"]),
            {"threshold": 3, "include_all": True}
        ),
        
        # 8. Per-column threshold (mapping)
        (
            pd.DataFrame({
                "col1": [1, 1, 2, 2, 2, 2],
                "col2": [1, 2, 3, 4, 4, 4],
                "col3": [1, 2, 3, 4, 5, 6]
            }),
            pd.DataFrame({
                "categories_count": [2, 4, 6],
                "is_category": [1, 0, 1]
            }, index=["col1", "col2", "col3"]),
            {"threshold": {"col1": 2, "col2": 3, "col3": 10}, "include_all": True}
        ),
        
        # 9. Combined filtration (threshold + dtype)
        (
            pd.DataFrame({
                "low_card_int":  [1, 2, 1, 2, 2],
                "high_card_int": [1, 2, 3, 4, 5],
                "low_card_str":  ["a", "b", "a", "b", "b"],
                "high_card_str": ["a", "b", "c", "d", "e"]
            }),
            pd.DataFrame({
                "categories_count": [2, 5, 2, 5],
                "is_category": [1, 0, 1, 0]
            }, index=["low_card_int", "high_card_int", "low_card_str", "high_card_str"]),
            {"threshold": 4, "include_int": True, "include_str": True}
        ),
        
        # 10. Edge case - empty data
        (
            pd.DataFrame(),
            pd.DataFrame({
                "categories_count": [],
                "is_category": []
            }, index=[]),
            {"include_all": True}
        ),
    ]
)
def test_get_categorical_features_example_based(data, expected, kwargs):
    """Test core functionality with specific input-output examples."""
    result = data_quality.get_categorical_features(data, **kwargs)
    
    assert set(result.columns) == set(expected.columns)
    
    result = result[expected.columns]

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)

# tests for data_quality.set_categorical()

def test_set_categorical_featutes_verbose_output(caplog):
    df = pd.DataFrame({"A": [0, 0, 0, 0, 0, 0, 1],
                       "B": [1, 2, 3, 4, 5, 6, 7]})
    with caplog.at_level(logging.INFO):
        data_quality.set_categorical(df, verbose=True, include_all = True, threshold = 4)
    assert len(caplog.records) > 0
    log_record = caplog.records[-1].msg
    log_dict = log_record if isinstance(log_record, dict) else json.loads(log_record)
    assert log_dict["function"] == "set_categorical"
    assert log_dict["converted_columns"] == ["A"]
    assert log_dict["conversion_count"] == 1

def test_set_categorical_input_contains_nans():
    data = pd.Series([1,2,3,4,5,np.nan])
    with pytest.raises(ValueError, match="contains null values"):
        data_quality.set_categorical(data, nan_policy="raise")
    without_nans =  data_quality.set_categorical(data, threshold=5,
                                                 include_all = True, nan_policy="drop")
    with_nans =  data_quality.set_categorical(data, threshold=5,
                                              include_all = True, nan_policy="include")
    assert without_nans.iloc[:, 0].dtype.name == "category"
    assert with_nans.iloc[:, 0].dtype.name != "category"

def test_set_categorical_empty_input():
    data = []
    setted = data_quality.set_categorical(data)
    assert setted.empty
    pd.testing.assert_frame_equal(setted, pd.DataFrame([]))

@pytest.mark.parametrize("data, threshold, kwargs, error_type",
[
    (pd.DataFrame({"A": [0, 0, 0, 0, 0]}), {"B": 9}, {}, ValueError),
    (pd.DataFrame({"A": [0, 0, 0, 0, 0]}), 3, {}, None),
    (pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}), 3, {}, None),
    (pd.DataFrame({"A": [1, 1, 1], "B": [1, 2, 3]}), 2, {"include_all": True}, None),
    (pd.DataFrame({"A": [1, 2, 3]}), 0, {}, ValueError),  # edge case: zero threshold
    (pd.DataFrame({"A": [1, 2, 3]}), 1.5, {}, ValueError),  # float threshold (округление)
    (pd.DataFrame({"A": [1, 2, 3], "B": [1, 1, 2]}), [2, 3], {}, None),
    (pd.DataFrame({"A": [1, 2, 3], "B": [1, 1, 2], "C": [1, 2, 3]}), [2, 3, 4], {}, None),
    (pd.DataFrame({"A": [1, 2, 3]}), [2], {}, None),
    (pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}), [2], {}, ValueError),
    (pd.DataFrame({"A": [1, 2, 3]}), [2, 3], {}, ValueError),
    (pd.DataFrame({}), "WASD", {}, None),
    
    # MAPPING thresholds
    (pd.DataFrame({"A": [1, 2, 3], "B": [1, 1, 2]}), {"A": 2, "B": 3}, {}, None),
    (pd.DataFrame({"A": [1, 2, 3], "B": [1, 1, 2]}), {"A": 2}, {}, ValueError),
    (pd.DataFrame({"A": [1, 2, 3], "B": [1, 1, 2]}), {"C": 5}, {}, ValueError),
    
    (pd.DataFrame({"A": [1, 2, 3]}), {"A": "invalid"}, {}, ValueError),
    (pd.DataFrame({"A": [1, 2, 3]}), {"A": -1}, {}, ValueError),
    
    (pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]}), 
     {"A": 2, "B": 4}, 
     {"include_str": True}, 
     None),  # mapping + include
    
    (pd.DataFrame({"A": [1, 2, 3], "B": [1, 1, 2]}), 
     [3, 2], 
     {"include_int": True}, 
     None),  # sequence + include
    
    # EDGE cases
    (pd.DataFrame(), 3, {}, None),
    (pd.DataFrame({"A": []}), 2, {}, None),
    (pd.DataFrame({"A": [1], "B": [2]}), [1, 1], {}, None),

])
def test_set_categorical_threshold_behavior(data, threshold, kwargs, error_type):
    if error_type is not None:
        with pytest.raises(error_type):
            data_quality.set_categorical(data, threshold=threshold, **kwargs)
    else:
        result = data_quality.set_categorical(data, threshold=threshold, **kwargs)
        assert result is not None
        assert isinstance(result, pd.DataFrame)

def test_set_categorical_original_df_does_not_mutate():
    data = pd.DataFrame([0, 0, 0, 0, 0, 0, 0])
    original = data.copy()
    data_quality.set_categorical(data, threshold=1, include_number=True)
    pd.testing.assert_frame_equal(data, original)


