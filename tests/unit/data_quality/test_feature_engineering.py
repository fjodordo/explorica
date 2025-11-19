import re

import pytest
import pandas as pd
import numpy as np

import explorica.data_quality as data_quality

# consts

DF_WITH_NAN = pd.DataFrame([[1, 2, 3, 4],
                            [1, 2, 3, None]])
DFS_DIFFERENT_LENGTH = [pd.DataFrame({"A": [1, 2, 3, 4, 5, 6],
                                      "B": [6, 5, 4, 3, 2, 1]}),
                        pd.DataFrame({"A": [1, 2, 3]})]

DATA_SEQUENCES = [
    # 1D list
    [1, 2, 2, 3, 3, 3, 4],

    # 1D tuple
    (1, 2, 2, 3, 3, 3, 4),

    # 1D numpy array
    np.array([1, 2, 2, 3, 3, 3, 4]),

    # 2D nested list
    [[1, 2], [2, 2], [3, 1], [1, 1]],

    # 2D numpy array
    np.array([[1, 2], [2, 2], [3, 1], [1, 1]]),

    # dict with sequences
    {"num": [1, 2, 2, 3], "cat": ["x", "y", "y", "z"]},

    # pandas Series
    pd.Series([1, 2, 2, 3, 3]),

    # pandas DataFrame
    pd.DataFrame({"num": [1, 2, 2, 3], "cat": ["x", "y", "y", "z"]}),
]


DATA_WITH_NAN = [
    # 1D list
    [1, 2, np.nan, 4, None, 6, pd.NA],

    # 1D tuple
    (1, np.nan, 3, None, 5, pd.NA),

    # 1D numpy array
    np.array([1, 2, np.nan, 4, np.nan]),

    # 2D nested list
    [[1, np.nan], [None, 2], [3, 4], [pd.NA, 5]],

    # 2D numpy array
    np.array([[1, np.nan], [3, 4], [np.nan, 6]]),

    # dict with sequences
    {"num": [1, 2, np.nan, 4], "cat": ["x", None, "y", pd.NA]},

    # pandas Series
    pd.Series([1, 2, np.nan, 4, pd.NA, None]),

    # pandas DataFrame
    pd.DataFrame({
        "num": [1, np.nan, 3, None, 5],
        "cat": ["a", "b", None, pd.NA, "c"]
    }),
]

# tests for data_quality.freq_encode()

def test_freq_encode_input_contains_nans():
    with pytest.raises(ValueError, match="contains null values"):
        data_quality.freq_encode(DF_WITH_NAN)

@pytest.mark.parametrize("data", DATA_SEQUENCES)
def test_freq_encode_different_sequences_and_dtypes(data):
    encoded = data_quality.freq_encode(data)
    assert encoded is not None

def test_freq_encode_deterministic():
    # 1D Series
    s = pd.Series(["a", "b", "b", "c", "a", "a"], name="data_row")
    expected_series = pd.Series([0.5, 0.3333, 0.3333, 0.1667, 0.5, 0.5], name="data_row")
    
    encoded = data_quality.freq_encode(s, normalize=True, round_digits=4)
    pd.testing.assert_series_equal(encoded, expected_series)

    # 2D DataFrame, column-wise
    df = pd.DataFrame({
        "A": ["a", "b", "b", "c", "a", "a"],
        "B": [1, 3, 2, 3, 1, 1]
    })
    expected_df = pd.DataFrame({
        "A": [3, 2, 2, 1, 3, 3],
        "B": [3, 2, 1, 2, 3, 3]
    })
    
    encoded_df = data_quality.freq_encode(df, axis=0, normalize=False)
    pd.testing.assert_frame_equal(encoded_df, expected_df)

    # 2D DataFrame, row-wise
    df = pd.DataFrame({
        "A": [1, 2, 2, 1, 3, 2, 1],
        "B": ["x", "y", "y", "x", "z", "y", "x"],
        "C": [True, False, False, True, True, False, True]
    })

    encoded = data_quality.freq_encode(df, axis=1, normalize=False)
    expected = pd.Series([3, 3, 3, 3, 1, 3, 3], name="frequency")

# tests for data_quality.ordinal_encode()

def test_ordinal_encode_input_contains_nans():
    with pytest.raises(ValueError, match="contains null values"):
        data_quality.ordinal_encode(DF_WITH_NAN)

@pytest.mark.parametrize("data", DATA_SEQUENCES)
def test_ordinal_encode_different_sequences_and_dtypes(data):
    encoded = data_quality.ordinal_encode(data)
    assert encoded is not None

def test_ordinal_encode_frequency():
    data = pd.Series(["a", "b", "a", "c", "a", "b"])
    expected_ascending = pd.Series([2, 1, 2, 0, 2, 1])
    expected_descending = pd.Series([0, 1, 0, 2, 0, 1])
    encoded_ascending = data_quality.ordinal_encode(data, order_method="freq", order_ascending=True)
    encoded_descending = data_quality.ordinal_encode(data, order_method="freq", order_ascending=False)
    pd.testing.assert_series_equal(expected_ascending, encoded_ascending)
    pd.testing.assert_series_equal(expected_descending, encoded_descending)

def test_ordinal_encode_aplhabetical():
    data = pd.Series(["aa", "a", "ab", "aa", "c", "aa", "ab"])
    expected_ascending = pd.Series([2, 1, 3, 2, 4, 2, 3])
    expected_descending = pd.Series([3, 4, 2, 3, 1, 3, 2])
    encoded_ascending = data_quality.ordinal_encode(data, order_method="abc", order_ascending=True, offset=1)
    encoded_descending = data_quality.ordinal_encode(data, order_method="abc", order_ascending=False, offset=1)
    pd.testing.assert_series_equal(expected_ascending, encoded_ascending)
    pd.testing.assert_series_equal(expected_descending, encoded_descending)

def test_ordinal_encode_ct_measures():
    data = pd.DataFrame({
        "feature": ["A", "A", "B", "B", "C", "C"]
    })
    
    # order_by — числовые значения для расчета центральных мер
    order_by = pd.DataFrame({
        "target": [10, 20, 5, 15, 30, 25]
    })
    
    # Central measures by groups:
    # A: mean=15, median=15, mode=10
    # B: mean=10, median=10, mode=5
    # C: mean=27.5, median=27.5, mode=25

    expected_mean_asc = pd.Series([1, 1, 0, 0, 2, 2])
    encoded_mean_asc = data_quality.ordinal_encode(
        data, axis=1, order_method="mean", order_ascending=True, order_by=order_by
    )
    pd.testing.assert_series_equal(expected_mean_asc, encoded_mean_asc)

    expected_median_desc = pd.Series([1, 1, 2, 2, 0, 0])
    encoded_median_desc = data_quality.ordinal_encode(
        data, axis=1, order_method="median", order_ascending=False, order_by=order_by
    )
    pd.testing.assert_series_equal(expected_median_desc, encoded_median_desc)

    expected_mode_asc = pd.Series([1, 1, 0, 0, 2, 2])
    encoded_mode_asc = data_quality.ordinal_encode(
        data, axis=1, order_method="mode", order_ascending=True, order_by=order_by
    )
    pd.testing.assert_series_equal(expected_mode_asc, encoded_mode_asc)

def test_ordinal_encode_different_axis():
    df = pd.DataFrame({
        "A": ["x", "y", "x", "z"],
        "B": ["a", "a", "b", "b"]
    })
    expected_axis0 = pd.DataFrame({
        "A": [0, 1, 0, 2],
        "B": [0, 0, 1, 1]
    })
    encoded_axis0 = data_quality.ordinal_encode(df, order_method="abc", order_ascending=True, axis=0)
    pd.testing.assert_frame_equal(expected_axis0, encoded_axis0)
    expected_axis1 = pd.Series([0, 2, 1, 3], name=None)
    encoded_axis1 = data_quality.ordinal_encode(df, order_method="abc", order_ascending=True, axis=1)
    pd.testing.assert_series_equal(expected_axis1, encoded_axis1)
        

def test_ordinal_encode_empty_order_by():
    with pytest.raises(ValueError, match="'order_by' must be provided"):
        data_quality.ordinal_encode(DATA_SEQUENCES[0], order_method="avg",
                                    order_by=None)

def test_ordinal_encode_lenghts_mismatch():
    with pytest.raises(ValueError, match="must match length"):
        data_quality.ordinal_encode(DFS_DIFFERENT_LENGTH[0], order_method="median",
                                    order_by=DFS_DIFFERENT_LENGTH[1])
    
# tests for data_quality.discretize_continuous()

@pytest.mark.parametrize("data, intervals, expected_labels", [
    (
            {"x": [1,2,3], "y": [4,5,6]},
            ["A", "B", "C"],
            pd.DataFrame({"x": ["A","B","C"], "y": ["A","B","C"]})
    ),
    (
            {"x": [1,2,3], "y": [4,5,6]},
            [["A","B","C"], ["D","E","F"]],
            pd.DataFrame({"x": ["A","B","C"], "y": ["D","E","F"]})
    ),
    (
            pd.DataFrame({"x": [1,2,3], "y": [4,5,6]}),
            pd.Series(["A","B","C"]),
            pd.DataFrame({"x": ["A","B","C"], "y": ["A","B","C"]})
    ),
    (
            pd.DataFrame({"x": [1,2,3], "y": [4,5,6]}),
            pd.DataFrame({"x": ["A","B","C"], "y": ["D","E","F"]}),
            pd.DataFrame({"x": ["A","B","C"], "y": ["D","E","F"]})
    ),
    (
            pd.DataFrame({"x": [1,2,3], "y": [4,5,6]}),
            {"x": ["A","B","C"], "y": ["D","E","F"]},
            pd.DataFrame({"x": ["A","B","C"], "y": ["D","E","F"]})
    ),
    (
            pd.DataFrame({"x": [1,2,3], "y": [4,5,6]}),
            ("A","B","C"),
            pd.DataFrame({"x": ["A","B","C"], "y": ["A","B","C"]})
    )
])
def test_discretize_continuous_different_interval_dtypes(data, intervals, expected_labels):
    encoded = data_quality.discretize_continuous(data, intervals=intervals).astype("str")
    pd.testing.assert_frame_equal(encoded, expected_labels,)

@pytest.mark.parametrize("data", DATA_WITH_NAN)
def test_discretize_continuous_data_contains_nans(data):
    with pytest.raises(ValueError, match="The input 'data' contains null values."):
        data_quality.discretize_continuous(data)

@pytest.mark.parametrize("intervals", [
    # 1D list
    ["A", np.nan, "B", "C"],
    
    # 1D tuple
    ("A", None, "B", "C"),
    
    # 1D numpy array
    np.array(["A", "B", pd.NA, "C"]),
    
    # 2D nested list
    [["A", "B", np.nan], ["X", "Y", "Z"]],
    [["A", "B", "C"], ["X", None, "Z"]],
    
    # dict of sequences
    {"A": ["A", "B", np.nan], "B": ["X", "Y", "Z"]},
    {"A": ["A", "B", "C"], "B": ["X", pd.NA, "Z"]},

    # pandas Series (1D case)
    pd.Series(["A", None, "B", "C"]),

    # pandas DataFrame (2D case)
    pd.DataFrame({
        "A": ["A", "B", "C", None],
        "B": ["X", "Y", np.nan, "Z"]
    }),
])
def test_discretize_continuous_intervals_contains_nans(intervals):
    data = pd.DataFrame({"A": [1,2,3,4,5,6],
                         "B": [5,4,3,2,1,0]})
    with pytest.raises(ValueError,
                       match="The input 'intervals' contains null values."):
        data_quality.discretize_continuous(data, intervals=intervals)
    

@pytest.mark.parametrize("data", DATA_SEQUENCES[:-3])
def test_discretize_continuous_different_sequences_and_dtypes(data):
    encoded = data_quality.discretize_continuous(data)
    assert encoded is not None

@pytest.mark.parametrize("data, bins, intervals, err_msg_fragment",
[

        (
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}),
            [2, 3],
            "pandas",
            "match the number of bins",
        ),
        (
            pd.DataFrame({"A": [1, 2, 3]}),
            [2, 3, 4],
            "pandas",
            "match the number of bins",
        ),
        (
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            None,
            [["x", "y", "z"]],
            "intervals length",
        ),
        (
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            None,
            {"C": ["x", "y", "z"]},
            "`intervals` keys must match",
        ),
        (
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            None,
            {"A": ["low", "high"], "C": ["x", "y", "z"]},
            "`intervals` keys must match",
        ),
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            {"A": 3, "B": 4},
            "pandas",
            "`bins` keys must match `data",
        ),
        (
            np.array([[1, 2], [3, 4]]),
            None,
            {"A": ["x", "y"], "B": ["z", "w"]},
            "`intervals` keys must match",
        ),
        (
            pd.DataFrame({"A": [1, 2, 3], "B": [3, 4, 5]}),
            [2],
            [["low", "mid", "high"], ["x", "y"], ["z"]],
            "match the number of bins|intervals length",
        ),
])
def test_discretize_continuous_lengths_mismatch(data, bins, intervals, err_msg_fragment):
    with pytest.raises(KeyError, match=err_msg_fragment):
        data_quality.discretize_continuous(data, bins=bins, intervals=intervals)


@pytest.mark.parametrize("const_array", [[0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 1, 0, 0, 0]])
def test_discretize_continuous_quantile_with_const(const_array):
    df = pd.DataFrame({"A": const_array,
                       "B": [1, 2, 3, 4, 5, 6, 7, 8]})
    with pytest.warns(UserWarning, match="Number of unique values"):
        data_quality.discretize_continuous(df, binning_method="quantile", bins=3)

def test_discretize_continuous_deterministic1():
    data = pd.Series([0,1,2,50,51,52,101,102,103])
    discretized = data_quality.discretize_continuous(data, bins=3, binning_method="uniform")
    assert all(isinstance(v, pd.Interval) for v in discretized)

    assert all([discretized.iloc[0 + i] == discretized.iloc[1 + i] == discretized.iloc[2 + i]] for i in [0, 3, 6])

    widths = [v.right - v.left for v in discretized]
    assert np.std(widths) < 0.1
    
    assert discretized.nunique() == 3

def test_discretize_continuous_deterministic2():
    data = pd.Series([0, 10, 20, 30, 40, 50])
    discretized = data_quality.discretize_continuous(data, bins=3,
                                                     binning_method="uniform")
    assert all(isinstance(v, pd.Interval) for v in discretized)

    assert all([discretized.iloc[0 + i] == discretized.iloc[1 + i]] for i in [0, 2, 4])

    widths = [v.right - v.left for v in discretized]
    assert np.std(widths) < 0.1
    
    assert discretized.nunique() == 3

def test_discretize_continuous_deterministic3():
    # quantile-based binning method test
    data = [1,2,3,4,5,6,7,8,9]
    discretized = data_quality.discretize_continuous(data, bins=3, binning_method="quantile")
    
    assert all([discretized.iloc[0 + i] == discretized.iloc[1 + i] == discretized.iloc[2 + i]] for i in [0, 3, 6])

    assert discretized.nunique() == 3

    order = [cat for cat in discretized.dtype.categories]
    assert np.isclose(order[0].left, 1, atol=1e-2)
    assert np.isclose(order[-1].right, 9, atol=1e-2)

def test_discretize_continuous_deterministic3():
    # 2d input test
    data = pd.DataFrame({
    "A": [0,1,2,3,4,5,6],
    "B": [100,200,300,400,500,600,700]
        })
    bins = [3, 2]
    discretized = data_quality.discretize_continuous(data, bins=bins)

    assert discretized["A"].nunique() == 3
    assert discretized["B"].nunique() == 2
    
    # len = 7 and bins = 3 for 'A' column, intervals_by_bin = 7/3
    # 2 < 7/3 < 3
    assert ((discretized["A"].value_counts(
        ) == 2) | (discretized["A"].value_counts() == 3)).all()
    
    # len = 7 and bins = 2 for 'B' column, intervals_by_bin = 7/2
    # 3 < 3.5 < 4
    assert ((discretized["B"].value_counts(
        ) == 3) | (discretized["B"].value_counts() == 4)).all()

def test_discretize_continuous_deterministic4():
    # test if `intervals` is Sequence
    data = [1,2,3,4,5,6]
    intervals = ["low", "mid", "high"]
    discretized = data_quality.discretize_continuous(
        data, intervals=intervals)
    
    assert discretized.to_list() == ['low', 'low', 'mid', 'mid', 'high', 'high']

def test_discretize_continuous_deterministic5():
    data = pd.DataFrame({"A":[1,2,3,4], "B":[10,20,30,40]})
    intervals = [["X","Y"], ["M","N","O","P"]]
    expected = pd.DataFrame({"A": pd.Categorical(categories=(["X", "Y"]), values=["X", "X", "Y", "Y"], ordered=True),
    "B": pd.Categorical(categories=(["M","N","O","P"]), values=["M","N","O","P"], ordered=True)})

    discretized = data_quality.discretize_continuous(
        data, intervals=intervals)
    pd.testing.assert_frame_equal(discretized, expected)

def test_discretize_continuous_string_output():
    data = pd.Series([0, 1, 2, 3, 4, 5])
    discretized = data_quality.discretize_continuous(data, bins=3, intervals="string")
    assert all(isinstance(v, str) for v in discretized)

    pattern = r"^\([0-9\.\-eE]+,\s*[0-9\.\-eE]+\]$"
    assert all(re.match(pattern, v) for v in discretized), \
        "All intervals must match format '(left, right]'"

    unwanted_output = discretized.copy()
    unwanted_output.iloc[0] = "pd.Interval(left = 0, right=1)"
    assert not (all(re.match(pattern, v) for v in unwanted_output))

