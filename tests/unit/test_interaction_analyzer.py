from datetime import datetime
from itertools import chain
from collections import deque

import pytest
import pandas as pd
import numpy as np
from explorica import InteractionAnalyzer as ia

# tests for InteractionAnalyzer.corr_matrix()

def test_corr_matrix_unsupported_method():
    df = pd.DataFrame({1: [1, 2, 3], 2: [1, 3, 4], 3: [3, 2, 1]})
    with pytest.raises(ValueError, match="Unsupported method"):
        ia.corr_matrix(df, method="unsupported_method_name")

@pytest.mark.parametrize(
    "method, dataset, groups",
    [
        (
            "pearson",
            [[1, 2, 3], [4, 5, 6]],
            None
        ),
        (
            "pearson",
            np.array([[1, 2, 3], [4, 5, 6]]).T,
            None
        ),
        (
            "pearson",
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            None
        ),
        (
            "multiple",
            {"a": [1, 2, 3], "b": [4, 5, 1], "c": [5, 5, 6]},
            None
        ),
        (
            "exp",
            pd.DataFrame({"a": [1.0, 2.5, 3.1], "b": [4.0, 5.0, 6.0]}),
            None
        ),
        (
            "cramer_v",
            pd.DataFrame({"cat1": ["A", "B", "C"], "cat2": ["X", "Y", "X"]}),
            None
        ),
        (
            "eta",
            pd.DataFrame({"val": [1.2, 2.3, 3.1, 4.5], "val2": [10, 20, 10, 30]}),
            [["A", "A", "B", "B"]]
        ),
    ]
)
def test_corr_matrix_different_sequences_and_dtypes(method, dataset, groups):
    result = ia.corr_matrix(dataset, method=method, groups=groups)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty

@pytest.mark.parametrize("method", ["pearson", "cramer_v", "eta", "multiple", "exp"])
@pytest.mark.parametrize("dataset", [
    [[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]],              
    [[1.0, 2.0, pd.NA], [4.0, 5.0, 6.0]],               
    [[1.0, None, 3.0], [4.0, 5.0, 6.0]],                
    pd.DataFrame({"x": [1.0, 2.0, np.nan], "y": [4, 5, 6]}),
])
def test_corr_matrix_input_contains_nan(method, dataset):
    if method == "eta":
        groups = [["A", "B", "C"]]
    else:
        groups = None
    with pytest.raises(ValueError, match="null values"):
        ia.corr_matrix(dataset, method=method, groups=groups)

# tests for InteractionAnalyzer.corr_matrix_corr_index()

def test_corr_matrix_corr_index_unsupported_method():
    df = pd.DataFrame({1: [1, 2, 3], 2: [1, 3, 4], 3: [3, 2, 1]})
    with pytest.raises(ValueError, match="Unsupported method"):
        ia.corr_matrix_corr_index(df, method="unsupported_method_name")

@pytest.mark.parametrize("method", ["exp", "binomial", "ln", "hyperbolic", "power"])
@pytest.mark.parametrize("dataset", [
    # nested lists
    [[1, 2, 3], [4, 5, 6]],
    # numpy array (2D)
    np.array([[1, 2, 3], [4, 5, 6]]),
    # pandas DataFrame
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
    # dict of lists
    {"a": [1, 2, 3], "b": [4, 5, 6]},
    # with float + int mix
    pd.DataFrame({"a": [1.0, 2.5, 3.1], "b": [4, 5, 6]}),
    # with string columns
    pd.DataFrame({"col1": [1, 2, 3], "col2": [10, 20, 30]}),
])
def test_corr_matrix_corr_index_different_sequences_and_dtypes(method, dataset):
    result = ia.corr_matrix_corr_index(dataset, method=method)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == result.shape[1]  # square matrix
    assert not result.empty

@pytest.mark.parametrize("dataset", [
    [[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]],              # np.nan in list
    [[1.0, 2.0, pd.NA], [4.0, 5.0, 6.0]],               # pd.NA in list
    [[1.0, None, 3.0], [4.0, 5.0, 6.0]],                # None in list
    pd.DataFrame({"x": [1.0, 2.0, np.nan], "y": [4, 5, 6]}),  # DataFrame with NaN
])
def test_corr_matrix_corr_index_input_contains_nan(dataset):
    with pytest.raises(ValueError, match="null values"):
        ia.corr_matrix_corr_index(dataset)

# tests for InteractionAnalyzer.corr_matrix_multiple()

def test_corr_matrix_multiple_determined():
    """test with determined answer ≈ 0.9659"""
    expectation = 0.9659
    dataset = pd.DataFrame({"feat1": (4, 6.2, 6.1, 5.4, 5.9, 6, 5.6, 5.2),
                            "feat2": (12, 9, 8, 14, 15, 11, 10, 15),
                            "feat3": (42, 107, 100, 60, 78, 79, 90, 54)})
    result = ia.corr_matrix_multiple(dataset)
    result = result[result["target"] == "feat3"]["corr_coef"].iloc[0]
    assert np.isclose(result, expectation, atol=0.0001)


def test_corr_matrix_multiple_multicollinearity_case():
    """we're expecting to catch UserWarning"""
    dataset = pd.DataFrame({"feat1": (1, 2, 3), "feat2": (2, 4, 6), "feat3": (0, 9, 1)})
    with pytest.warns(UserWarning):
        ia.corr_matrix_multiple(dataset)

@pytest.mark.parametrize(
    "dataset",
    [
        [[1, 2, 3], [4, 2, 6], [7, 8, 8]],
        [[1, "a", 3], [4, "b", 6], [7, "c", 9]],
        {"col1": [1, 2, 3], "col2": [1, 1, 1], "col3": [3, 2, 2]},
        np.array([[1, 2, 4], [4, 5, 6], [8, 7, 9]]),
        np.array([[1.0, 3, 3], [4.0, 5, 6], [7.5, 8, 9]]),
        pd.DataFrame({"a": [1, 2, 2], "b": [5, 5, 6], "c": [7, 7, 7]}),
        pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [7, 8, 10],  "d": [6, 8, 9]}),
        pd.DataFrame({"only": [1, 2, 3]}),
        [[1], [2], [3]],
    ]
)
def test_corr_matrix_multiple_different_sequences_and_dtypes(dataset):
    result = ia.corr_matrix_multiple(dataset)
    assert result is not None

@pytest.mark.parametrize(
    "dataset",
    [
        np.array([[1, 2, np.nan], [4, 5, 6], [3, 2, 3]]),
        pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6], "c": [5, 5, 5]}),
        pd.DataFrame({"a": [1, pd.NA, 3], "b": [4, 5, 6], "c": [2, 5, 6]}),
        [[1, 2, None], [4, 5, 6], [4, 5, 2]],
        [[1, 2, np.nan], [None, 5, 6], [4, 5, 6]],
        pd.DataFrame({"a": [1, np.nan, pd.NA], "b": [4, 5, 6], "c": [5, 5, 5]}),
    ]
)
def test_corr_matrix_multiple_input_contains_nan(dataset):
    with pytest.raises(ValueError):
        ia.corr_matrix_multiple(dataset)

# tests for InteractionAnalyzer.corr_matrix_eta()

def test_corr_matrix_eta_lenghts_mismatch():
    dataset = pd.DataFrame({"num_feat": [0, 1, 2, 3, 4]})
    categories =  pd.DataFrame({"cat_feat": ["A", "B", "C"]})
    with pytest.raises(ValueError):
        ia.corr_matrix_eta(dataset, categories)

@pytest.mark.parametrize(
    "dataset, categories, expected",
    [
        # Case 1: perfect separation (two numeric × two categorical)
        (
        
            pd.DataFrame({
                "x1": [1, 1, 2, 2],
                "x2": [10, 10, 20, 20],
            }),
            pd.DataFrame({
                "cat1": ["a", "a", "b", "b"],
                "cat2": ["u", "u", "v", "v"],
            }),
            pd.DataFrame(
                [[1.0, 1.0],
                 [1.0, 1.0]],
                index=["x1", "x2"],
                columns=["cat1", "cat2"],
            ),
        ),

        # Case 2: no association (numeric increases regardless of categories)
        (
            pd.DataFrame({
                "x1": [1, 2, 3, 4],
                "x2": [5, 6, 7, 8]
            }),
            pd.DataFrame({
                "cat": ["a", "a", "a", "a"]
            }),
            # matrix 2×1: both eta ≈ 0
            pd.DataFrame(
                [[0.0],
                 [0.0]],
                index=["x1", "x2"],
                columns=["cat"]
            )
        ),
    ]
)
def test_corr_matrix_eta_determined(dataset, categories, expected):
    result = ia.corr_matrix_eta(dataset, categories)
    pd.testing.assert_frame_equal(result.round(5), expected.round(5))


@pytest.mark.parametrize(
    "dataset, categories",
    [
        # list of lists (nested python lists)
        ([[1, 2, 3], [4, 5, 6]],
         [["a", "b", "a"], ["b", "a", "b"]]),

        # numpy arrays
        (np.array([[1, 2, 3], [4, 5, 6]]),
         np.array([["a", "b", "a"], ["b", "a", "b"]])),

        # pandas DataFrame with ints + string categories
        (pd.DataFrame({"x": [1, 2, 3]}),
         pd.DataFrame({"cat": ["low", "mid", "high"]})),

        # floats + int categories
        (pd.DataFrame({"x": [0.1, 0.2, 0.3]}),
         pd.DataFrame({"cat": [1, 2, 1]})),

        # booleans as categories
        (pd.DataFrame({"x": [10, 20, 30]}),
         pd.DataFrame({"flag": [True, False, True]})),

        # dictionary of lists (will be converted to DataFrame inside)
        ({"num": [1, 2, 3]}, {"grp": ["a", "a", "b"]}),

        # tuple of tuples
        (((1, 2, 3), (4, 5, 6)),
         (("g1", "g2", "g1"), ("g2", "g1", "g2"))),
    ]
)
def test_corr_matrix_eta_different_sequences_and_dtypes(dataset, categories):
    result = ia.corr_matrix_eta(dataset, categories)
    assert result is not None



@pytest.mark.parametrize(
    "dataset, categories",
    [
        # numeric NaNs
        (pd.DataFrame({"x": [1, 2, np.nan]}),
         pd.DataFrame({"g": ["a", "b", "a"]})),

        # categorical NaNs
        (pd.DataFrame({"x": [1, 2, 3]}),
         pd.DataFrame({"g": ["a", np.nan, "b"]})),

        # both sides with NaNs
        (pd.DataFrame({"x": [1, np.nan, 3]}),
         pd.DataFrame({"g": ["a", pd.NA, "b"]})),

        # None instead of NaN
        (pd.DataFrame({"x": [1, None, 3]}),
         pd.DataFrame({"g": ["a", "b", "a"]})),
    ]
)
def test_corr_matrix_eta_input_contains_nan(dataset, categories):
    with pytest.raises(ValueError):
        ia.corr_matrix_eta(dataset, categories)

# tests for InteractionAnalyzer.corr_matrix_cramer_v()

def test_corr_matrix_cramer_v_0x0():
    result = ia.corr_matrix_cramer_v(pd.DataFrame())
    assert result is not None

@pytest.mark.parametrize(
    "dataset",
    [
        # NaN in numeric column
        pd.DataFrame({"A": [1, 2, np.nan], "B": [1, 2, 3]}),

        # NaN in string column
        pd.DataFrame({"A": ["x", None, "y"], "B": ["a", "b", "c"]}),

        # NaN in categorical column
        pd.DataFrame({
            "A": pd.Series(["low", "medium", pd.NA], dtype="category"),
            "B": ["x", "y", "z"]
        }),

        # Nested list with None
        [["cat", "dog"], ["dog", None], ["cat", "dog"]],

        # Numpy array with np.nan
        np.array([[1, 2], [3, pd.NA], [1, 4]]),
    ]
)
def test_corr_matrix_cramer_v_input_contains_nan(dataset):
    with pytest.raises(ValueError):
        ia.corr_matrix_cramer_v(dataset)

@pytest.mark.parametrize(
    "dataset",
    [
        # DataFrame with integers (treated as categories)
        pd.DataFrame({"A": [1, 2, 3], "B": [1, 1, 2]}),

        # DataFrame with strings
        pd.DataFrame({"A": ["cat", "dog", "dog"], "B": ["yes", "no", "yes"]}),

        # DataFrame with pandas Categorical dtype
        pd.DataFrame({
            "A": pd.Series(["low", "medium", "high"], dtype="category"),
            "B": pd.Series(["x", "y", "x"], dtype="category")
        }),

        # DataFrame with datetime (treated as categories)
        pd.DataFrame({
            "A": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "B": ["a", "b", "a"]
        }),

        # Numpy array of ints
        np.array([[1, 2, 3], [1, 2, 4], [2, 3, 4]]),

        # Nested list of strings
        [["apple", "red"], ["apple", "green"], ["pear", "green"]],

        # Single column DataFrame
        pd.DataFrame({"A": ["x", "y", "z"]}),
    ]
)
def test_corr_matrix_cramer_v_different_sequences_and_dtypes(dataset):
    print(dataset)
    result = ia.corr_matrix_cramer_v(dataset)
    assert result is not None

# tests for InteractionAnalyzer.corr_matrix_linear()

@pytest.mark.parametrize(
    "dataset, method, expectation",
    [
        (
            pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]}),
            "pearson",
            1.0
        ),
        (
            pd.DataFrame({"x": [1, 2, 3, 4], "y": [-1, -2, -3, -4]}),
            "pearson",
            -1.0
        ),
        (
            pd.DataFrame({"x": [1, 0, 1, 0], "y": [1, 1, 0, 0]}),
            "spearman",
            0.0
        ),
    ]
)
def test_corr_matrix_linear_determined(dataset, method, expectation):
    matrix = ia.corr_matrix_linear(dataset, method=method)
    corr_value = matrix.loc["x", "y"]
    assert np.isclose(corr_value, expectation, atol=1e-8)

def test_corr_matrix_linear_unsupported_method():
    df = pd.DataFrame({1: [1, 2, 3], 2: [1, 3, 4], 3: [3, 2, 1]})
    with pytest.raises(ValueError):
        ia.corr_matrix_linear(df, method="unsupported_method_name")

@pytest.mark.parametrize(
    "dataset",
    [
        pd.DataFrame({1: [1, 2, 3], 2: [1, 3, 4], 3: [3, np.nan, 1]}),
        np.array([[1, 2, 3], [1, None, 5], [2, 3, 4]], dtype=object),
        pd.DataFrame({"a": [1, pd.NA, 3], "b": [4, 5, 6]}),
        pd.DataFrame({"x": [1.0, 2.0, float("nan")], "y": [4.0, 5.0, 6.0]}),
        np.array([[1, 2, np.nan], [4, 5, 6], [7, 8, 9]]),
    ]
)
def test_corr_matrix_linear_input_contains_nan(dataset):
    with pytest.raises(ValueError):
        ia.corr_matrix_linear(dataset, method="pearson")

def test_corr_matrix_linear_different_dtypes():
    dataset = pd.DataFrame({"numeric1":[1, 2, 3, 4],
                            "numeric_and_string": [1, "b", 3, "a"],
                            "numeric2": [1, 2, 3, 4],
                            "datetime": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-04")]})
    result = ia.corr_matrix_linear(dataset)
    assert result.shape == (2, 2)

@pytest.mark.parametrize(
    "dataset",
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        pd.DataFrame({"a": [1, 4, 7], "b": [2, 5, 8], "c": [3, 6, 9]}),
        pd.Series([1, 2, 3, 4]),
        pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.0, 5.0, 6.0]}),
    ]
)
def test_corr_matrix_linear_different_sequences(dataset):
    result = ia.corr_matrix_linear(dataset)
    assert result is not None

# tests for InteractionAnalyzer.corr_index()

@pytest.mark.skip(reason="Bug planned to fix: Handling different types of Sequences")
@pytest.mark.parametrize(
    "x, y",
    [
        # list[int]
        ([1, 2, 3], [4, 5, 6]),
        # list[float]
        ([1.0, 2.0, 3.0], [4.5, 5.5, 6.5]),
        # numpy.ndarray
        (np.array([1, 2, 3]), np.array([4, 5, 6])),
        # pandas.Series
        (pd.Series([1, 2, 3]), pd.Series([4, 5, 6])),
        # tuple
        ((1, 2, 3), (4, 5, 6)),
        # deque
        (deque([1, 2, 3]), deque([4, 5, 6])),
        # mixed types (int + float)
        ([1, 2.5, 3], [4.1, 5, 6.7]),
    ]
)
def test_corr_index_different_sequences_and_dtypes(x, y):
    result = ia.corr_index(x, y, method="exp")
    assert 0 <= result <= 1

@pytest.mark.parametrize("y, method", [
    (pd.Series([2.9 * np.exp(i * 2) for i in range(1, 11)]), "exp"),
    (pd.Series([-2 * i + 31 for i in range(1, 11)]), "linear"),
    (pd.Series([3 * i**2 + 2 * i + 13 for i in range(1, 11)]), "binomial"),
    #  x^-1 & log(x) very sensitive to normalization [1e-13, 1], so they don't work
    # Bug planned to fix
    # (pd.Series([5 * np.log(i) + 15 for i in range(1, 11)]), "ln"),
    # (pd.Series([1/i + 14 for i in range(1, 11)]), "hyperbolic"), 
    (pd.Series([2 * i * 3 for i in range(1, 11)]), "power")
])
def test_corr_index_full_dependence(y, method):
    x = pd.Series([i for i in range(1, 11)])
    result = ia.corr_index(x, y, method)
    assert np.isclose(result, 1.00, atol=0.01)

# tests for InteractionAnalyzer.eta_squared()

def test_eta_squared_lengths_mismatch():
    x = [1, 2, 3, 4]
    y = ["A", "A", "B", "B", "C"]
    with pytest.raises(ValueError):
        ia.eta_squared(x, y)

@pytest.mark.parametrize(
    "x, y",
    [
        ([1, 2, 3, None], ["u", "v", None, "v"]),
        ([3, np.nan, 1, 0], ["u", "v", "u", np.nan]),
        (pd.Series([3, 4, 1, 1]), pd.Series([1, 2, pd.NA, 2])),
        ([4, 1, 0, None], ["u", "v", 2, np.nan]),
        ([None, None, None], [np.nan, np.nan, np.nan]),
    ]
)
def test_cramer_v_contains_nan(x, y):
    with pytest.raises(ValueError):
        ia.eta_squared(x, y)

@pytest.mark.skip(reason="Bug planned to fix: NaN handling")
@pytest.mark.parametrize("x, y", [
    ([1, 2, np.nan, 4], ["A", "B", "C", "D"]),
    ([1, 2, 3, 4], ["A", np.nan, "C", "D"])
])
def test_eta_squared_input_contains_nan(x, y):
    with pytest.raises(ValueError):
        ia.eta_squared(x, y)

@pytest.mark.parametrize("x, y", [
    ([1, 2, 3, 4], ["A", "B", "C"]),
    ([1, 2, 3], ["A", "B", "C", "D"])])
def test_eta_squared_sequence_lenghts_mismatch(x, y):
    with pytest.raises(ValueError):
        ia.eta_squared(x, y)

@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    "x_data, y_data",
    [
        # simple int / string
        ([3, 2, 2, 4, 2], ["X", "Y", "X", "Y", "X"]),
        # np.int64 / pd.Categorical
        (np.array([1, 3, 2, 1, 3], dtype=np.int64),
         pd.Categorical(["A", "B", "A", "B", "A"])),
        # np.float / string
        (np.array([1.1, 2.2, 1.1, 3.3], dtype=np.float64),
         ["cat", "dog", "cat", "mouse"]),
        # int / pd.Timestamp
        ([5, 7, 5, 8], pd.to_datetime([
            "2025-08-10", "2025-08-11", "2025-08-10", "2025-08-12"
        ])),
        # float / pd.Timestamp
        ([0.1, 0.2, 0.3, 0.1],
         pd.to_datetime([
            "2025-08-10", "2025-08-10", "2025-08-11", "2025-08-12"
         ])),
        # int & float / string
        ([1, 2.0, 3, 2.5], ["red", "blue", "red", "green"]),
    ]
)
def test_eta_squared_accepts_different_sequences_and_dtypes(x_data, y_data):
    result = ia.eta_squared(x_data, y_data)
    assert 0.0 <= result <= 1.0

def test_eta_squared_determined():
    """test with determined answer ≈ 0.6693"""
    ans = 0.6693
    data_counts = [
    (47, "A", 3),
    (47, "B", 1),
    (59, "A", 2),
    (59, "B", 3),
    (71, "B", 11),
    (71, "C", 1),
    (83, "B", 2),
    (83, "C", 4),
    (95, "C", 3),
]
    data = list(chain.from_iterable(
        [[x_val,y_val]] * count for x_val, y_val, count in data_counts))

    x = [i[0] for i in data]
    y = [i[1] for i in data]
    result = ia.eta_squared(x, y)
    assert np.isclose(result, ans, atol=0.01)

@pytest.mark.parametrize(
    "x, y", [([1, 1, 2, 2, 9, 9, 9], ["const"] * 7),
              ([1] * 7, ["A", "B", "C", "B", "A", "A", "B"]),
              ([1] * 7, ["const"] * 7),
              ([1, 2] * 3, ["A", "A", "B", "B", "C", "C"])])
def test_eta_squared_full_independence(x, y):
    result = ia.eta_squared(x, y)
    assert np.isclose(result, 0.0, atol=0.01)


@pytest.mark.parametrize(
    "x, y", [([1, 1, 2, 2, 9, 9, 9], ["A", "A", "B", "B", "C", "C", "C"]),
             ([10, 10, 20, 20, 30, 30],
              ["red", "red", "green", "green", "blue", "blue"]),
    (
        [1, 1, 2, 2, 3, 3],
        [
            pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-01"),
            pd.Timestamp("2020-03-01"), pd.Timestamp("2020-03-01"),
        ]
    )
])
def test_eta_squared_full_dependence(x, y):
    result = ia.eta_squared(x, y)
    assert np.isclose(result, 1.0, atol=0.01)

# tests for InteractionAnalyzer.cramer_v

def test_cramer_v_lengths_mismatch():
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        ia.cramer_v(x, y)

@pytest.mark.parametrize(
    "x, y",
    [
        (["a", None, "a", "b"], ["u", "v", None, "v"]),
        (["a", np.nan, "b", "b"], ["u", "v", "u", np.nan]),
        (pd.Series(["a", pd.NA, "b", "b"]), pd.Series([1, 2, pd.NA, 2])),
        (["a", 1, "b", None], ["u", "v", 2, np.nan]),
        ([None, None, None], [np.nan, np.nan, np.nan]),
    ]
)
def test_cramer_v_contains_nan(x, y):
    with pytest.raises(ValueError):
        ia.cramer_v(x, y)
    

@pytest.mark.skip(reason="Borderline cases." \
"`bias_correction` can return negative sqrt " \
"argument on some samples. bug planned to fix")
@pytest.mark.parametrize("x_data, y_data", [
    (("A", "B", "B"), ("X", "Y", "X")),  # tuple of str
    ([1, 2, 2], [10, 20, 10]),  # int list
    ([1.1, 2.2, 2.2], [10.0, 20.0, 10.0]),  # float list
    ([np.int64(1), np.int64(2), np.int64(2)], ["a", "b", "a"]),  # numpy int
    ([np.float64(1.5), np.float64(2.5), np.float64(2.5)], ["a", "b", "a"]),  # numpy float
    ([pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-02")],
     ["X", "Y", "X"]),  # pandas timestamp
    ([datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 2)],
     ["X", "Y", "X"]),  # python datetime
    (pd.Categorical(["cat", "dog", "dog"]), ["yes", "no", "yes"]),  # categorical
    (np.array(["A", "B", "B"]), np.array(["X", "Y", "X"])),  # numpy array of str
    (pd.Series(["A", "B", "B"]), pd.Series(["X", "Y", "X"])),  # pandas Series
])
def test_cramer_v_accepts_different_sequence_types(x_data, y_data):
    result = ia.cramer_v(x_data, y_data, bias_correction=False)
    assert 0.0 <= result <= 1.0
    result_corrected = ia.cramer_v(x_data, y_data, bias_correction=True)
    assert 0.0 <= result_corrected <= 1.0


def test_cramer_v_full_dependence():
    # small samples (without bias correction)
    x1, y1 = ["A", "A", "B", "B"], ["X", "X", "Y", "Y"]
    result = ia.cramer_v(x1, y1, bias_correction=False)
    assert np.isclose(result, 1.0, atol=0.05)

    # small samples (with bias correction)
    # borderline case, a strong downward shift is expected
    result = ia.cramer_v(x1, y1, bias_correction=True)
    assert result < 1.0 - 0.1

    # big samples (without bias correction)
    # an overestimated value is expected
    x2 = np.random.choice([1, 2, 3, 4], size=1000)
    y2 = x2.copy()
    result = ia.cramer_v(x2, y2, bias_correction=False)
    assert np.isclose(result, 1.0, atol=1e-10)

    # big samples (with bias correction)
    result = ia.cramer_v(x2, y2, bias_correction=True)
    assert np.isclose(result, 1.0, atol=0.05)

@pytest.mark.skip(reason="bias_correction can produce negative sqrt argument " \
                         "on small samples, bug planned to fix")
def test_cramer_v_full_independence():
    x1, y1 = ["A", "B", "A", "B"], ["X", "X", "Y", "Y"]
    result = ia.cramer_v(x1, y1, bias_correction=True)
    assert result == 0.0
    result = ia.cramer_v(x1, y1, bias_correction=False)
    assert result == 0.0

def test_cramer_v_determined():
    """
    |    | Y1 | Y2 | Y3 |
    | -- | -- | -- | -- |
    | X1 | 10 | 5  | 5  |
    | X2 | 2  | 8  | 4  |
    | X3 | 3  | 7  | 6  |
    
    N = 50
    𝜒^2 ≈ 7.1801
    k = min(r - 1, c - 1) = 2

    Without correction: 
    V ≈ sqrt(7.1801 / (50 * 2)) ≈ 0.2680

    ϕ^2 = 𝜒^2/N ≈ 0.1436
    ϕ^2_corr = max(0, ϕ^2 - (r-1)(c-1)/(N-1)) ≈ max(0, 0.1436 - 4/49) ≈ 0.0620
    r_corr = r - (r-1)^2/(N-1) ≈ 1.9796
    c_corr = c - (c-1)^2/(N-1) ≈ 1.9796
    k_corr = min(r_corr, c_corr) ≈ 1.9796

    With correction:
    V = sqrt(ϕ^2/k_corr) ≈ sqrt(0.1436/1.9796) ≈ 0.1769
    """
    v = 0.2680
    v_corrected = 0.1769

    x = (
    ["X1"] * 10 + ["X1"] * 5 + ["X1"] * 5 +
    ["X2"] * 2  + ["X2"] * 8 + ["X2"] * 4 +
    ["X3"] * 3  + ["X3"] * 7 + ["X3"] * 6
    )

    y = (
    ["Y1"] * 10 + ["Y2"] * 5 + ["Y3"] * 5 +
    ["Y1"] * 2  + ["Y2"] * 8 + ["Y3"] * 4 +
    ["Y1"] * 3  + ["Y2"] * 7 + ["Y3"] * 6)

    result = ia.cramer_v(x, y, bias_correction=False)
    assert np.isclose(v, result, atol=0.002)
    result_corrected = ia.cramer_v(x, y, bias_correction=True)
    assert np.isclose(v_corrected, result_corrected, atol=0.002)

def test_cramer_v_yates_correction():
    # result with yates correction must be lower than without it for 2x2 confussion matrix
    x = ["A", "B", "B", "B"]
    y = ["X", "Y", "X", "Y"]
    result_no_corr = ia.cramer_v(x, y, yates_correction=False, bias_correction=False)
    result_with_corr = ia.cramer_v(x, y, yates_correction=True, bias_correction=False)
    assert result_with_corr < result_no_corr

def test_cramer_v_constant_feature():
    # with at least 1 constant feature cramer_v should return 0.0
    x = ["A"] * 10
    y = ["X", "Y"] * 5
    result = ia.cramer_v(x, y)
    assert result == 0.0
    y = ["B"] * 10
    result = ia.cramer_v(x, y)
    assert result == 0.0
