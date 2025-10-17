import json
import warnings

import pytest
from unittest.mock import patch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import explorica.data_quality as data_quality


DATA_SEQUENCES = [
    # === 1D structures ===
    [1, 2, 3, 4, 100],                        # Python list
    (1, 2, 3, 4, 100),                        # Tuple
    np.array([1, 2, 3, 4, 100]),              # NumPy 1D array
    pd.Series([1, 2, 3, 4, 100]),             # pandas Series

    # === 1D edge variants ===
    range(1, 6),                              # range object
    (x for x in [1, 2, 3, 4, 100]),           # generator
    np.array([1., 2., 3., 4., 100.], dtype=np.float64),  # explicit dtype
    pd.Index([1, 2, 3, 4, 100]),              # pandas Index

    # === 2D structures ===
    [[1, 2, 3], [4, 5, 100]],                 # list of lists
    tuple(zip([1, 4], [2, 5], [3, 100])),     # tuple of tuples (transposed)
    np.array([[1, 2, 3], [4, 5, 100]]),       # 2D numpy array
    pd.DataFrame({
        "A": [1, 2, 3, 4, 100],
        "B": [2, 3, 4, 5, 200]
    }),                                       # pandas DataFrame

    # === 2D exotic types ===
    np.matrix([[1, 2, 3], [4, 5, 100]]),      # legacy numpy matrix
    pd.DataFrame(np.array([[1, 2], [3, 100]]), columns=["x", "y"]),  # DataFrame from ndarray
    pd.DataFrame.from_records(
        [(1, 2, 3), (4, 5, 100)], columns=["a", "b", "c"]
    ),                                        # DataFrame from records
]

DETECT_METHODS = ["iqr", "zscore"]

DF_WITH_NAN = pd.DataFrame({"A": [0, 1, 2, 3],
                       "B": [0, 1, 2, None]})

# contract tests

@pytest.mark.parametrize("f", [data_quality.detect_iqr, data_quality.detect_zscore])
def test_detect_outliers_warns_on_zero_variance(f):
    df = pd.DataFrame({
        "A": [5, 5, 5, 5, 5],
        "B": [1, 2, 3, 4, 5]
    })
    with pytest.warns(UserWarning, match="zero or very small variance"):
        f(df)


# tests for Outliers.detect_iqr()

def test_detect_iqr_input_contains_nans():
    with pytest.raises(ValueError):
        data_quality.detect_iqr(DF_WITH_NAN)

@pytest.mark.parametrize("data", DATA_SEQUENCES)
def test_detect_iqr_different_sequences_and_dtypes(data):
    outliers = data_quality.detect_iqr(data)
    assert outliers is not None

def test_detect_iqr_deterministic():
    data = [30, 35, 40, 45, 50, 55, 60, 80, 90, 100, 120, 200, 250, 300]
    outliers = data_quality.detect_iqr(data).reset_index(drop=True)
    expected = pd.Series([250.0, 300.0])
    assert outliers.equals(expected), (
        f"Expected outliers {expected.tolist()}, got {outliers.tolist()}")

@pytest.mark.filterwarnings("ignore:.*vert:PendingDeprecationWarning:seaborn.categorical")
def test_detect_iqr_boxplot_call():
    data = [1, 2, 3, 4, 5, 100]

    with patch.object(plt, "show") as mock_show:
        data_quality.detect_iqr(data, show_boxplot=True)

    mock_show.assert_called_once()

# tests for Outliers.detect_zscore()

def test_detect_zscore_input_contains_nans():
    with pytest.raises(ValueError):
        data_quality.detect_zscore(DF_WITH_NAN)

@pytest.mark.parametrize("data", DATA_SEQUENCES)
def test_detect_zscore_different_sequences_and_dtypes(data):
    outliers = data_quality.detect_zscore(data)
    assert outliers is not None

def test_detect_zscore_deterministic():
    data = [5, 2, 4.5, 4, 3, 2, 6, 20, 9, 2.5, 3.5, 4.75, 6.5, 2.5, 8, 1]
    outliers = data_quality.detect_zscore(data).reset_index(drop=True)
    expected = pd.Series([20.0])
    assert outliers.equals(expected), (
        f"Expected outliers {expected.tolist()}, got {outliers.tolist()}")

# tests for Outliers.remove_outliers()

def test_remove_outliers_input_contains_nans():
    with pytest.raises(ValueError):
        data_quality.remove_outliers(DF_WITH_NAN)

@pytest.mark.parametrize("data", DATA_SEQUENCES)
def test_remove_outliers_different_sequences_and_dtypes(data):
    cleaned_df = data_quality.remove_outliers(data)
    assert cleaned_df is not None

@pytest.mark.parametrize("detect_method", DETECT_METHODS)
@pytest.mark.parametrize("remove_mode", ["any", "all"])
def test_remove_outliers_deterministic(detect_method, remove_mode):
    data = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5, 100, 6],
        "feature_2": [10, 11, 9, 12, 10, 11, 300]
    })
    # Expected result depending on remove_mode:
    # 'any' - remove row if any feature is an outlier
    # 'all' - remove row only if all features are outliers
    if remove_mode == "any":
        expected = data.drop([5, 6]).reset_index(drop=True)
    else:  # remove_mode == "all"
        expected = data.copy().reset_index(drop=True)

    # 1) Check deterministic outlier removal
    result = data_quality.remove_outliers(data, detection_method=detect_method, remove_mode=remove_mode)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
    # 2) Check iterative removal: result is the same after multiple iterations
    result_repeat = data_quality.remove_outliers(
        data, detection_method=detect_method, remove_mode=remove_mode, iters=15)
    pd.testing.assert_frame_equal(result, result_repeat)
    # 3) Check input data remains unchanged
    assert len(data) == 7
    assert data.iloc[5, 0] == 100 and data.iloc[6, 1] == 300

def test_remove_outliers_expected_warnings():
    """Expected warnings are 1 UserWarning"""
    df = pd.DataFrame({"A": [0, 1, 2, 3],
                       "B": [0, 1, 2, 4],
                       "const": [5, 5, 5, 5]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=UserWarning)
        data_quality.remove_outliers(df)
    assert len(w) == 1
    assert "zero or very small variance" in str(w[0].message)

# tests for Outliers.replace_outliers()

def test_replace_outliers_input_contains_nans():
    with pytest.raises(ValueError, match="contains null values"):
        data_quality.replace_outliers(DF_WITH_NAN)

@pytest.mark.parametrize("data", DATA_SEQUENCES)
def test_different_sequences_and_dtypes(data):
    replaced = data_quality.replace_outliers(data)
    assert replaced is not None


@pytest.mark.parametrize(
    "data, detection_method, strategy, kwargs, expected",
    [
        # === 1D sequence, IQR, mean ===
        ([1.0, 2.4, 1.6, 12, 1.2, 501.1, 0.6],
         "iqr", "mean", {},
         pd.Series([1.0, 2.4, 1.6, 12.0, 1.2, 3.1333, 0.6])),
        
        # === 1D sequence, IQR, median ===
        ([1.0, 2.4, 1.6, 12, 1.2, 501.1, 0.6],
         "iqr", "median", {},
         pd.Series([1.0, 2.4, 1.6, 12.0, 1.2, 1.4, 0.6])),
        
        # === 1D sequence, IQR, mode ===
        ([1, 2, 2, 12, 1, 501, 0],
         "iqr", "mode", {},
         pd.Series([1, 2, 2, 12, 1, 1, 0])),
        
        # === 1D sequence, IQR, custom ===
        ([1.0, 2.4, 1.6, 12, 1.2, 501.1, 0.6],
         "iqr", "custom", {"custom_value": 999},
         pd.Series([1.0, 2.4, 1.6, 12.0, 1.2, 999, 0.6])),

        # === Mapping input, IQR, median ===
        ({"f1": [1, 2, 100, 3], "f2": [10, 20, 30, 500]},
         "iqr", "median", {},
         pd.DataFrame({"f1": [1, 2, 2, 3], "f2": [10, 20, 30, 20]})),
    ]
)
def test_replace_outliers_deterministic(data, detection_method, strategy, kwargs, expected):
    replaced = data_quality.replace_outliers(data, detection_method, strategy, **kwargs)
    assert np.round(replaced, 4).equals(expected)

@pytest.mark.parametrize(
    "data, detection_method, strategy, kwargs, expected",
    [
        ([1.0, 2.4, 1.6, 12, 1.2, 501.1, 0.6],
         "iqr", "median", {"iqr_factor": 3},
         pd.Series([1.0, 2.4, 1.6, 1.3, 1.2, 1.4, 0.6])),

         ([10, 11, 9, 10, 200, 8, 10],
         "zscore", "mean", {},
         pd.Series([10, 11, 9, 10, 10, 8, 10])),
         
         ([1, 2, 3, 100, 50, 4, 5, 5],
         "zscore", "mode", {},
         pd.Series([1, 2, 3, 5, 5, 4, 5, 5])),
    ])
def test_replace_outliers_deterministic_recursive(data, detection_method, strategy, kwargs, expected):
    replaced = data_quality.replace_outliers(data, detection_method, strategy, recursive=True, **kwargs)
    assert np.round(replaced, 4).equals(expected)

def test_replace_outliers_deterministic_multiple_iters():
    """
    Replacing the median with a low IQR factor will make the feature constant
    over an unlimited number of iterations.
    The test verifies that this scenario is prevented.
    """
    data = [1, 2, 3, 4, 5, 6, 7, 8, 10, 500, 1200, 1, 33, 303, 3]
    not_expected = pd.Series([1] * 15)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="features (.*) have zero or very small variance")
        replaced = data_quality.replace_outliers(data, "iqr", "median", iters=10, iqr_factor = 0.5)
    assert not replaced.equals(not_expected)

@pytest.mark.timeout(5)
def test_replace_outliers_large_number_of_iters():
    data = [1, 2, 3, 4, 1000, 50]
    data_quality.replace_outliers(data, detection_method="zscore", 
                                  iters=int(10e20))

def test_replace_outliers_expected_warnings():
    """Expected warnings are 1 UserWarning"""
    df = pd.DataFrame({"A": [0, 1, 2, 3],
                       "B": [0, 1, 2, 4],
                       "const": [5, 5, 5, 5]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=UserWarning)
        data_quality.replace_outliers(df)
    assert len(w) == 1
    assert "zero or very small variance" in str(w[0].message)

# tests for Outliers.get_skewness()

def test_get_skewness_input_contains_nans():
    with pytest.raises(ValueError, match="contains null values"):
        data_quality.get_skewness(DF_WITH_NAN)

def test_get_skewness_unsupported_method():
    with pytest.raises(ValueError, match="Unsupported method"):
        data_quality.get_skewness(DATA_SEQUENCES[0], method="this is unsupported method")

@pytest.mark.parametrize("data", DATA_SEQUENCES)
def test_get_skewness_different_sequences_and_dtypes(data):
    calculated = data_quality.get_skewness(data)
    assert calculated is not None

@pytest.mark.parametrize(
    "data, method, expected",
    [
        ([1, 2, 3, 4, 5, 6], "sample", 0.0),
        ([1, 2, 3, 4, 100], "general", 1.498),
        ([1, 2, 3, 4, 100], "sample", 1.072),
        ([-100, 1, 1, 1, 1], "general", -1.5),
        ([-100, 1, 1, 1, 1], "sample", -1.073)
    ]
)
def test_get_skewness_deterministic(data, method, expected):
    calculated = data_quality.get_skewness(data, method)
    assert np.isclose(calculated, expected, atol=1e-3)

def test_get_skewness_zero_var():
    data = {"a": [1, 1, 1, 1, 1.000000001], "b": [1, 2, 3, 4, 5]}
    with pytest.warns(UserWarning, match="Columns with near-zero variance"):
        calculated = data_quality.get_skewness(data)
    assert isinstance(calculated, pd.Series)

def test_get_skewness_empty_input():
    data = pd.DataFrame([])
    result = data_quality.get_skewness(data)
    assert np.isnan(result)

def test_get_skewness_large_numbers():
    arr = np.array([1e10, 1e10+1, 1e10+2, 1e10+3, 1e10+4])
    calculated = data_quality.get_skewness(arr)
    assert np.isclose(calculated, 0.0, atol=1e-3)

# tests for Outliers.get_kurtosis()

def test_get_kurtosis_input_contains_nans():
    with pytest.raises(ValueError, match="contains null values"):
        data_quality.get_kurtosis(DF_WITH_NAN)

def test_get_kurtosis_unsupported_method():
    with pytest.raises(ValueError, match="Unsupported method"):
        data_quality.get_kurtosis(DATA_SEQUENCES[0], method="this is unsupported method")

@pytest.mark.parametrize("data", DATA_SEQUENCES)
def test_get_kurtosis_different_sequences_and_dtypes(data):
    calculated = data_quality.get_kurtosis(data)
    assert calculated is not None

@pytest.mark.parametrize(
    "data, method, expected, atol",
    [
        ([1, 2, 3, 4, 100], "general", 0.247, 1e-3),
        ([1, 2, 3, 4, 100], "sample", -0.922, 1e-3),
        ([1, 2, 3, 4, 5], "general", -1.3, 1e-3),
        ([ -6.53055989, -12.40198026,   9.56729396,  -8.46939156,
        -1.94867146,   5.29982783,  18.45382912,  10.35562596,
         6.32760225,   6.44769337], "general", 0.0, 1.5),
         ([-10, -5, 0, 5, 10, 50], "general", 0.486, 1e-3),
         ([-10, -5, 0, 5, 10, 50], "sample", -0.579, 1e-3)
    ]
)
def test_get_kurtosis_deterministic(data, method, expected, atol):
    calculated = data_quality.get_kurtosis(data, method)
    assert np.isclose(calculated, expected, atol=atol)

def test_get_kurtosis_zero_var():
    data = {"a": [1, 1, 1, 1, 1.000000001], "b": [1, 2, 3, 4, 5]}
    with pytest.warns(UserWarning, match="Columns with near-zero variance"):
        calculated = data_quality.get_kurtosis(data)
    assert isinstance(calculated, pd.Series)

def test_get_kurtosis_empty_input():
    data = pd.DataFrame([])
    result = data_quality.get_kurtosis(data)
    assert np.isnan(result)