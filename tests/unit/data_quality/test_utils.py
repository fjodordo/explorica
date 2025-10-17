import pytest
import pandas as pd
import numpy as np
from explorica.data_quality._utils import Replacers

# tests for data_quality._utils.replace()

def test_replace_deterministic():
    series = pd.Series([1, 2, 3, np.nan, np.nan, 5, 6, 7])
    expected = pd.Series([1, 2, 3, 4, 4, 5, 6, 7], dtype=float)
    result = Replacers.replace(series, to_replace=pd.Index([3, 4]), value=4)
    assert result.equals(expected)

# tests for data_quality._utils.replace_random()

def test_replace_random_deterministic():
    seed = 42
    value_by_seed = 1.0
    series = pd.Series([1, 2, 3, np.nan, np.nan, 5, 6, 7])
    expected = pd.Series([1, 2, 3, value_by_seed, value_by_seed, 5, 6, 7], dtype=float)
    result = Replacers.replace_random(series, to_replace=pd.Index([3, 4]), seed=seed)
    assert result.equals(expected)

# tests for data_quality._utils.replace_mct()

@pytest.mark.parametrize("measure, value", [("mean", 4.0),
                                            ("median", 3.0),
                                            ("mode", 2.0)])
def test_replace_mct_deterministic(measure, value):
    series = pd.Series([1, 2, 2, 3, np.nan, np.nan, 5, 6, 7])
    expected = pd.Series([1, 2, 2, 3, value, value, 5, 6, 7], dtype=float)
    result = Replacers.replace_mct(series, to_replace=pd.Index([4, 5]), measure=measure)
    assert round(result).equals(expected)