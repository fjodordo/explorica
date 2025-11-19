import json
import warnings

import pytest
from unittest.mock import patch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import explorica.data_quality as data_quality
from .constants import (DF_WITH_NAN, DATA_SEQUENCES)

# contract tests

@pytest.mark.parametrize("f", [data_quality.detect_iqr, data_quality.detect_zscore])
def test_detect_outliers_warns_on_zero_variance(f):
    df = pd.DataFrame({
        "A": [5, 5, 5, 5, 5],
        "B": [1, 2, 3, 4, 5]
    })
    with pytest.warns(UserWarning, match="zero or very small variance"):
        f(df)


# tests for data_quality.detect_iqr()

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

# tests for data_quality.detect_zscore()

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