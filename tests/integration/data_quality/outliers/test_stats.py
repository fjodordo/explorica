import pytest

import pandas as pd
import numpy as np

import explorica.data_quality as data_quality

# tests for data_quality.describe_distributions()

def test_describe_distributions_deterministic():
    expected = pd.DataFrame({"skewness": [0.0, 0.0, 0.6429],
                             "kurtosis": [-1.25, np.nan, -1.25],
                             "is_normal": [0, 0, 0],
                             "desc": ["low-pitched",
                                      "extremely-high",
                                      "right-skewed, low-pitched"]},
                            index=["ascend", "const", "random"])
    expected = expected.astype({"skewness": np.float64,
                                "kurtosis": np.float64,
                                "is_normal": np.int64,
                                "desc": object})
    df = pd.DataFrame({"ascend": [0, 1, 2, 3, 4, 5, 6],
                       "const": [5, 5, 5, 5, 5, 5, 5],
                       "random": [6, 1, 1, 6, 3, 2, 2]})
    with pytest.warns(UserWarning,
                      match="Columns with near-zero variance") as w:
        report = data_quality.describe_distributions(df)
    assert len(w) == 2, f"Expected 2 warnings, got {len(w)}"
    expected_warn_msgs = ["Columns with near-zero variance: ['const']. Their skewness will be set to 0.0.",
                          "Columns with near-zero variance: ['const']. Their excess kurtosis will be set to np.nan."]
    actual_messages = [str(i.message) for i in w]
    assert pd.Series(actual_messages).isin(expected_warn_msgs).all()
    pd.testing.assert_frame_equal(report, expected, atol=1e-4)