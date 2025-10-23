import json
import warnings

import pytest
from unittest.mock import patch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import explorica.data_quality as data_quality
from .constants import (DATA_SEQUENCES,
                       DETECT_METHODS,
                       DF_WITH_NAN)

# tests for data_quality.remove_outliers()

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

# tests for data_quality.replace_outliers()

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
