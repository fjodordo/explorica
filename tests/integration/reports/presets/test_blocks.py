import pdb

import numpy as np
import pandas as pd
import pytest

from explorica.reports.presets.blocks import (
    get_ctm_block,
    get_data_shape_block,
    get_data_quality_overview_block,
    get_outliers_block,
    get_distributions_block,
    get_cardinality_block)
from explorica.types import TableResult
from explorica.reports.core.block import Block

# -------------------------------
# Tests for get_cardinality_block
# -------------------------------

def test_cardinality_empty_dataframe():
    df = pd.DataFrame(columns=["a", "b"])
    block = get_cardinality_block(df)
    table = block.block_config.tables[0].table
    assert table["n_unique"].sum() == 0
    assert table["is_unique"].isna().all()
    assert table["is_constant"].isna().all()

def test_cardinality_constant_feature():
    df = pd.DataFrame({"const": [5, 5, 5, 5]})
    block = get_cardinality_block(df)
    table = block.block_config.tables[0].table
    assert table.loc["const", "is_constant"]
    assert not table.loc["const", "is_unique"]
    assert table.loc["const", "n_unique"] == 1
    assert pd.isna(table.loc["const", "entropy (normalized)"])

def test_cardinality_unique_feature():
    df = pd.DataFrame({"unique": [1, 2, 3, 4]})
    block = get_cardinality_block(df)
    table = block.block_config.tables[0].table
    assert table.loc["unique", "is_unique"]
    assert not table.loc["unique", "is_constant"]
    assert table.loc["unique", "n_unique"] == 4
    assert table.loc["unique", "entropy (normalized)"] == 1.0

def test_cardinality_mixed_features():
    df = pd.DataFrame({
        "const": [1, 1, 1, 1],
        "semi": [1, 1, 2, 2],
        "unique": [1, 2, 3, 4]
    })
    block = get_cardinality_block(df)
    table = block.block_config.tables[0].table
    # const
    assert table.loc["const", "is_constant"]
    # quasi-const
    assert 0 < table.loc["semi", "top_value_ratio"] < 1
    # unique
    assert table.loc["unique", "is_unique"]

def test_cardinality_nan_inclusion():
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 2],
        "b": [np.nan, np.nan, np.nan, np.nan]
    })
    block = get_cardinality_block(df, nan_policy="include")
    table = block.block_config.tables[0].table
    # NaN defines as an additional category
    assert table.loc["a", "n_unique"] == 3
    assert table.loc["b", "n_unique"] == 1

# -------------------------------
# Tests for get_distributions_block
# -------------------------------

def test_distributions_block_basic():
    df = pd.DataFrame({
        "normal": np.random.normal(loc=0, scale=1, size=100),
        "skewed": np.random.exponential(scale=1, size=100),
    })

    block = get_distributions_block(df)

    # Check block type
    assert block is not None
    assert hasattr(block, "block_config")

    # Check table
    table = block.block_config.tables[0].table
    assert set(table.columns).issuperset({"skewness", "kurtosis", "is_normal", "desc"})
    assert set(table.index) == {"normal", "skewed"}

    # Check that is_normal is bool
    assert table.loc["normal", "is_normal"] in [True, False]
    assert table.loc["skewed", "is_normal"] in [True, False]

    # Check that visualizations added
    assert len(block.block_config.visualizations) >= 4  # boxplots + distplots for every column

def test_distributions_block_nan_policy_drop():
    df = pd.DataFrame({
        "x": [1, 2, np.nan, 4]
    })
    block = get_distributions_block(df, nan_policy="drop")
    table = block.block_config.tables[0].table
    # The table is calculated based on 3 elements
    assert table.loc["x", "skewness"] == table.loc["x", "skewness"]  # not NaN

def test_distributions_block_nan_policy_raise():
    df = pd.DataFrame({
        "x": [1, 2, np.nan, 4]
    })
    import pytest
    with pytest.raises(ValueError):
        get_distributions_block(df, nan_policy="raise")


# -------------------------------
# Tests for get_outliers_block
# -------------------------------

def test_get_outliers_block_basic():
    df = pd.DataFrame({
        "a": [1, 2, 3, 100],
        "b": [10, 11, 12, 13],
    })

    block = get_outliers_block(df)

    assert isinstance(block, Block)
    assert len(block.block_config.tables) == 1

    table = block.block_config.tables[0].table

    assert set(table.columns) == {"IQR (1.5)", "Z-Score (3.0σ)"}
    assert set(table.index) == {"a", "b"} or set(table.index) == {0, "a", "b"}

    # sanity checks
    assert table.loc["a", "IQR (1.5)"] >= 0
    assert table.loc["a", "Z-Score (3.0σ)"] >= 0

def test_get_outliers_block_single_column():
    df = pd.DataFrame({
        "x": [1, 2, 3, 100]
    })

    block = get_outliers_block(df)
    table = block.block_config.tables[0].table

    
    assert set(table.index) == {"x"} or set(table.index) == {0, "x"}
    assert table.shape == (1, 2)

def test_get_outliers_block_zero_variance_feature():
    df = pd.DataFrame({
        "const": [5, 5, 5, 5, 5]
    })

    block = get_outliers_block(df)
    table = block.block_config.tables[0].table

    # No outliers should be detected for a constant feature
    assert table.loc["const", "IQR (1.5)"] == 0
    assert table.loc["const", "Z-Score (3.0σ)"] == 0

    # Second table: zero / near-zero variance features
    assert len(block.block_config.tables) == 2

    variance_table = block.block_config.tables[1].table

    # Table should list the constant feature
    assert "feature_name" in variance_table.columns
    assert variance_table["feature_name"].tolist() == ["const"]

def test_get_outliers_block_with_nans_drop():
    df = pd.DataFrame({
        "x": [1, 2, np.nan, 4],
        "y": [10, 10, 99, 10],
    })

    block = get_outliers_block(df, nan_policy="drop")
    table = block.block_config.tables[0].table

    assert set(table.index) == {"x", "y"}

    # 'y' column becomes constant after drop
    assert table.loc["y", "IQR (1.5)"] == 0
    assert table.loc["y", "Z-Score (3.0σ)"] == 0


# -------------------------------
# Tests for get_data_quality_overview_block
# -------------------------------

def test_quick_quality_basic_dataframe():
    df = pd.DataFrame({
        "a": [1, 2, 2, 3],
        "b": ["x", "y", "y", "z"]
    })
    block = get_data_quality_overview_block(df)
    
    metrics = {m['name']: m['value'] for m in block.block_config.metrics}
    # Duplicates
    assert metrics["Duplicates rows"] == 1
    assert np.isclose(metrics["Duplicates ratio"], 1/4)
    
    # NaNs
    nans_table = block.block_config.tables[0].table
    assert all(nans_table['nan_count'] == [0, 0])
    assert all(np.isclose(nans_table['nan_ratio'], [0.0, 0.0]))

def test_quick_quality_with_nans():
    df = pd.DataFrame({
        "a": [1, np.nan, 2],
        "b": [None, "y", "z"]
    })
    block = get_data_quality_overview_block(df)
    
    metrics = {m["name"]: m["value"] for m in block.block_config.metrics}
    assert metrics["Duplicates rows"] == 0
    assert metrics["Duplicates ratio"] == 0.0
    
    nans_table = block.block_config.tables[0].table

    # index already represents column names
    assert set(nans_table.index) == {"a", "b"}

    assert nans_table.loc["a", "nan_count"] == 1
    assert np.isclose(nans_table.loc["a", "nan_ratio"], 1 / 3, atol=1e-4)

    assert nans_table.loc["b", "nan_count"] == 1
    assert np.isclose(nans_table.loc["b", "nan_ratio"], 1 / 3, atol=1e-4)


def test_quick_quality_empty_dataframe():
    df = pd.DataFrame(columns=["a", "b"])
    block = get_data_quality_overview_block(df)
    
    metrics = {m['name']: m['value'] for m in block.block_config.metrics}
    assert metrics["Duplicates rows"] == 0
    assert metrics["Duplicates ratio"] is np.nan
    
    nans_table = block.block_config.tables[0].table
    assert nans_table['nan_count'].sum() == 0
    assert nans_table["nan_ratio"].isna().all()


def test_quick_quality_all_duplicates():
    df = pd.DataFrame({
        "a": [1, 1, 1],
        "b": ["x", "x", "x"]
    })
    block = get_data_quality_overview_block(df)
    
    metrics = {m['name']: m['value'] for m in block.block_config.metrics}
    # Only one unique row
    assert metrics["Duplicates rows"] == 2
    assert np.isclose(metrics["Duplicates ratio"], 2/3, atol=1e-4)

# -------------------------------
# Tests for get_data_shape_block
# -------------------------------

def test_get_data_shape_block_basic_dataframe():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"]
    })
    block = get_data_shape_block(df)
    
    # Metrics
    metrics = {m['name']: m['value'] for m in block.block_config.metrics}
    assert metrics["Rows"] == 3
    assert metrics["Columns"] == 2
    assert metrics["Index is positional"] is True

    # Dtypes table
    dtype_table = block.block_config.tables[0].table
    assert set(dtype_table['dtype'].astype("str")) == {"int64", "object"}
    assert dtype_table['n_features'].sum() == 2


def test_get_data_shape_block_dataframe_with_nans_drop():
    df = pd.DataFrame({
        "a": [1, np.nan, 3],
        "b": ["x", "y", None]
    })
    block = get_data_shape_block(df, nan_policy="drop")
    # given drop policy must drop 2 strings
    metrics = {m['name']: m['value'] for m in block.block_config.metrics}
    assert metrics["Rows"] == 1
    assert metrics["Columns"] == 2
    assert metrics["Index is positional"]


def test_get_data_shape_block_dataframe_with_nans_include():
    df = pd.DataFrame({
        "a": [1, np.nan, 3],
        "b": ["x", "y", None]
    })
    block = get_data_shape_block(df, nan_policy="include")
    
    metrics = {m['name']: m['value'] for m in block.block_config.metrics}
    assert metrics["Rows"] == 3
    # Index remains positional, despite the presence of NaNs
    assert metrics["Index is positional"] is True


def test_get_data_shape_block_non_default_index():
    df = pd.DataFrame({
        "a": [1, 2, 3]
    }, index=[10, -1, 30])
    
    block = get_data_shape_block(df)
    metrics = {m['name']: m['value'] for m in block.block_config.metrics}
    # Indices include negative values, so not positional
    assert not metrics["Index is positional"]


def test_get_data_shape_block_mixed_dtypes():
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [0.1, 0.2, 0.3],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True]
    })
    block = get_data_shape_block(df)
    
    dtype_table = block.block_config.tables[0].table
    # Check all types
    expected_dtypes = {"int64", "float64", "object", "bool"}
    assert set(dtype_table['dtype'].astype("str")) == expected_dtypes
    assert dtype_table['n_features'].sum() == 4


# -------------------------------
# Tests for get_ctm_block
# -------------------------------

def test_get_ctm_block_structure():
    data = {'num1': [1, 2, 3], 'num2': [4, 5, 6], 'cat': ['a', 'b', 'a']}
    block = get_ctm_block(data)

    # Check a type of returned object
    assert isinstance(block, Block)
    
    # Check that block contains 3 tables
    assert len(block.block_config.tables) == 3
    for table in block.block_config.tables:
        assert isinstance(table, TableResult)

def test_get_ctm_block_values():
    data = {'num': [1, 2, 3], 'cat': ['x', 'y', 'x']}
    block = get_ctm_block(data, round_digits=2)
   

    # Central tendency measures
    ctm_table = block.block_config.tables[0].table
    assert np.allclose(ctm_table['mean'], [2.0])
    assert np.allclose(ctm_table['median'], [2.0])
    
    # Mode
    mode_table = block.block_config.tables[1].table
    assert mode_table.loc['cat', 'mode'] == 'x'
    
    # Range
    var_table = block.block_config.tables[2].table
    assert np.allclose(var_table['std'], [np.std([1,2,3], ddof=0)], atol=2)
    assert np.allclose(var_table['min'], [1])
    assert np.allclose(var_table['max'], [3])
    assert np.allclose(var_table['range'], [2])

def test_get_ctm_block_nan_handling():
    data = {'num': [1, 2, np.nan], 'cat': ['x', 'y', 'x']}
    
    # nan_policy='drop' must execute
    block_drop = get_ctm_block(data, nan_policy='drop')
    assert isinstance(block_drop, Block)
    
    # nan_policy='raise' must raise ValueError
    with pytest.raises(ValueError):
        get_ctm_block(data, nan_policy='raise')

def test_get_ctm_block_rounding():
    data = {'num': [1.12345, 2.67891, 3.98765]}
    block = get_ctm_block(data, round_digits=2)
    ctm_table = block.block_config.tables[0].table
    assert np.all(ctm_table['mean'].round(2) == [2.60])
