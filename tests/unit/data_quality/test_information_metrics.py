import pytest
import pandas as pd
import numpy as np

import explorica.data_quality as data_quality

# tests for data_quality.get_entropy()

def test_get_entropy_with_nans():
    """Correct calculations with NaNs in data"""
    data_with_nan = pd.DataFrame({"A": [1, 1, np.nan, 2, np.nan]})
    result_drop = data_quality.get_entropy(data_with_nan, method="shannon", nan_policy="drop")
    result_include = data_quality.get_entropy(data_with_nan, method="shannon", nan_policy="include")
    with pytest.raises(ValueError, match="contains null values"):
        data_quality.get_entropy(data_with_nan, method="shannon", nan_policy="raise")
    assert np.isclose(result_drop, 0.9183, atol=1e-3)
    assert np.isclose(result_include, 1.5219, atol=1e-3) 

@pytest.mark.parametrize("data, kwargs", [
    ([1, 2, 3, 4, 5], {}),
    ((1, 2, 3), {}),
    (np.array([1, 2, 3, 4]), {}),
    ({"A": [1, 2, 2], "B": [3, 4, 4]}, {})
])
def test_get_entropy_different_sequences_and_dtypes(data, kwargs):
    result = data_quality.get_entropy(data, **kwargs)
    assert result is not None

def test_get_entropy_empty_input():
    empty_data = []
    result = data_quality.get_entropy(empty_data)
    assert result.empty
    empty_data = pd.DataFrame([], columns=["A", "B"])
    result = data_quality.get_entropy(empty_data)
    assert result.isna().all()
    assert set(result.index) == {"A", "B"}

@pytest.mark.parametrize("data, nested, expected, kwargs", [
    (
        pd.DataFrame([0, 0, 0, 0, 0, 0]), False, np.float64(0.0), {}
    ),
    (
        pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4]}), True,
        pd.Series({"A": 1.0, "B": 2.0}), {}
    ),
    (
        pd.DataFrame([5, 5, 5, 5]), False, np.float64(0.0), {}
    ),
    (
        [1, 2, 3, 4], False, np.float64(2.0), {}
    ),
    (
        (1, 2, 1, 2), False, np.float64(1.0), {}
    ),
    (
        (1, 2, np.nan, np.nan), False, np.float64(1.0), {"nan_policy": "drop"}
    ),
        (
        (1, 2, np.nan, np.nan), False, np.float64(1.5), {"nan_policy": "include"}
    )

])
def test_get_entropy_example_based(data, nested, expected, kwargs):
    result = data_quality.get_entropy(data, **kwargs)
    if nested:
        pd.testing.assert_series_equal(result, expected)
    else:
        assert result == expected

def test_get_entropy_invalid_method():
    data = [1, 2, 3]
    with pytest.raises(ValueError, match="Unsupported method 'unsupported_method'"):
        data_quality.get_entropy(data, method="unsupported_method")