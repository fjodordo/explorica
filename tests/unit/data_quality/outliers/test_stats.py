import pytest
import pandas as pd
import numpy as np

import explorica.data_quality as data_quality
from .constants import (DF_WITH_NAN, 
                        DATA_SEQUENCES,
                        DESCRIBE_DISTRIBUTIONS_STRUCTURE)

# tests for data_quality.get_skewness()

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

# tests for data_quality.get_kurtosis()

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

# tests for data_quality.describe_distributions()

def test_describe_distributions_input_contains_nans():
    with pytest.raises(ValueError, match="contains null values"):
        data_quality.describe_distributions(DF_WITH_NAN)

@pytest.mark.parametrize("data", DATA_SEQUENCES)
def test_describe_distributions_different_sequences_and_dtypes(data):
    report = data_quality.describe_distributions(data)
    assert report is not None

def test_describe_distributions_structure():
    report_as_dict = data_quality.describe_distributions(
        DATA_SEQUENCES[0], return_as="dict")
    report_as_df = data_quality.describe_distributions(
        DATA_SEQUENCES[0], return_as="dataframe")
    assert list(report_as_dict) == DESCRIBE_DISTRIBUTIONS_STRUCTURE
    assert list(report_as_df.columns) == DESCRIBE_DISTRIBUTIONS_STRUCTURE