from datetime import datetime
from itertools import chain
from collections import deque

import pytest
import pandas as pd
import numpy as np
from explorica import InteractionAnalyzer as ia

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
