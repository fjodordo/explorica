import pandas as pd
import pytest
from explorica.types import FeatureAssignment
from explorica.reports.utils import normalize_assignment

# -------------------------------
# Fixtures & helper functions
# -------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "x1": [1, 2, 3, 4],
        "x2": [10, 20, 30, 40],
        "c1": ["a", "b", "a", "b"],
        "y_num": [0, 1, 0, 1],
        "y_cat": ["yes", "no", "yes", "no"]
    })

# -------------------------------
# Tests for normalize_assignment
# -------------------------------

def test_normalize_assignment_basic_features_only(sample_df):
    fa = normalize_assignment(
        sample_df,
        numerical_names=["x1", "x2"],
        categorical_names=["c1"]
    )
    assert fa.numerical_features == ["x1", "x2"]
    assert fa.categorical_features == ["c1"]
    assert fa.numerical_target is None
    assert fa.categorical_target is None

def test_normalize_assignment_target_numerical_heuristic(sample_df):
    fa = normalize_assignment(
        sample_df,
        numerical_names=["x1", "x2"],
        categorical_names=["c1"],
        target_name="y_num",
        categorical_threshold=1  # threshold lower than nunique
    )
    # Since target unique > threshold, should pick numerical_target only
    assert fa.numerical_target == "y_num"
    assert fa.categorical_target is None


def test_normalize_assignment_target_categorical_heuristic(sample_df):
    fa = normalize_assignment(
        sample_df,
        numerical_names=["x1", "x2"],
        categorical_names=["c1"],
        target_name="y_cat",
        categorical_threshold=4  # threshold >= nunique
    )
    # Since nunique <= threshold, should pick categorical_target
    assert fa.categorical_target == "y_cat"
    assert fa.numerical_target is None

def test_normalize_assignment_target_explicit_names(sample_df):
    fa = normalize_assignment(
        sample_df,
        numerical_names=["x1", "x2"],
        categorical_names=["c1"],
        target_name="y_num",
        target_numerical_name="explicit_num",
        target_categorical_name="explicit_cat"
    )
    # Explicit names take priority
    assert fa.numerical_target == "explicit_num"
    assert fa.categorical_target == "explicit_cat"

def test_normalize_assignment_target_none(sample_df):
    fa = normalize_assignment(
        sample_df,
        numerical_names=["x1", "x2"],
        categorical_names=["c1"],
        target_name=None
    )
    # No target provided -> both targets remain None
    assert fa.numerical_target is None
    assert fa.categorical_target is None

