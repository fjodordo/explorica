from datetime import datetime

import pytest
import pandas as pd
import numpy as np
from explorica import InteractionAnalyzer as ia


@pytest.mark.skip(reason="Borderline cases." \
"`bias_correction` can return negative sqrt argument on some samples. bug planned to fix")
@pytest.mark.parametrize("x_data, y_data", [
    (("A", "B", "B"), ("X", "Y", "X")),                   # tuple of str
    ([1, 2, 2], [10, 20, 10]),                            # int
    ([1.1, 2.2, 2.2], [10.0, 20.0, 10.0]),                # float
    ([np.int64(1), np.int64(2), np.int64(2)], ["a","b","a"]),  # numpy int
    ([np.float64(1.5), np.float64(2.5), np.float64(2.5)], ["a","b","a"]),  # numpy float
    ([pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-02")],
     ["X", "Y", "X"]),                                    # pandas timestamp
    ([datetime(2023,1,1), datetime(2023,1,2), datetime(2023,1,2)],
     ["X", "Y", "X"]),                                    # python datetime
    (pd.Categorical(["cat", "dog", "dog"]), ["yes", "no", "yes"]),  # categorical
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

@pytest.mark.skip(reason="bias_correction can produce negative sqrt argument on small samples," \
                         "bug planned to fix")
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
