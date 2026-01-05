import pdb

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from explorica.reports.presets.blocks import (
    get_ctm_block,
    get_data_shape_block,
    get_data_quality_overview_block,
    get_outliers_block,
    get_distributions_block,
    get_cardinality_block,
    get_linear_relations_block,
    get_nonlinear_relations_block,)
from explorica.types import TableResult, VisualizationResult
from explorica.reports.core.block import Block

# -------------------------------
# Helper functions & fixtures
# -------------------------------

@pytest.fixture
def small_linear_data():
    df = pd.DataFrame({
        "x1": np.arange(100),
        "x2": np.arange(100) * 2,
        "x3": np.random.randn(100),
    })
    y = pd.Series(np.arange(100), name="target")
    return df, y

@pytest.fixture
def large_linear_data():
    n = 10_000
    df = pd.DataFrame({
        "x1": np.arange(n),
        "x2": np.arange(n) * 2,
        "x3": np.random.randn(n),
    })
    y = pd.Series(np.arange(n), name="target")
    return df, y

def close_visualizations(block: Block):
    for vr in block.block_config.visualizations:
        if vr.engine == "matplotlib":
            plt.close(vr.figure)

# -------------------------------
# Tests for get_nonlinear_relations_block
# -------------------------------

def test_get_nonlinear_relations_block_with_all_data():
    df_num = pd.DataFrame({'x1': [1,2,3], 'x2': [4,5,6]})
    df_cat = pd.DataFrame({'c1': ['a','b','a'], 'c2': ['x','y','x']})
    target = pd.Series(['a','b','a'], name='target')
    
    block = get_nonlinear_relations_block(df_num, df_cat, categorical_target=target)
    try:    
        assert isinstance(block, Block)
        # Check that block contains visualizations and table
        assert len(block.block_config.visualizations) == 2
        assert len(block.block_config.tables) == 1
    finally:
        close_visualizations(block)

def test_get_nonlinear_relations_block_without_categorical_target():
    df_num = pd.DataFrame({'x1': [1,2,3], 'x2': [4,5,6]})
    df_cat = pd.DataFrame({'c1': ['a','b','a'], 'c2': ['x','y','x']})
    
    block = get_nonlinear_relations_block(df_num, df_cat)
    try:
        # Block should be not empty, but table should be missing
        assert isinstance(block, Block)
        assert len(block.block_config.visualizations) == 2  # heatmaps
        assert len(block.block_config.tables) == 0
    finally:
        close_visualizations(block)

def test_get_nonlinear_relations_block_empty_data():
    df_num = pd.DataFrame()
    df_cat = pd.DataFrame()
    
    block = get_nonlinear_relations_block(df_num, df_cat)
    try:
        # Empty block are expected
        assert isinstance(block, Block)
        assert len(block.block_config.visualizations) == 0
        assert len(block.block_config.tables) == 0
    finally:
        close_visualizations(block)

def test_get_nonlinear_relations_block_with_numerical_target_only():
    df_num = pd.DataFrame({'x1': [1,2,3], 'x2': [4,5,6]})
    df_cat = pd.DataFrame({'c1': ['a','b','a'], 'c2': ['x','y','x']})
    num_target = pd.Series([0,1,0], name='target_num')
    
    block = get_nonlinear_relations_block(df_num, df_cat, numerical_target=num_target)
    try:
        assert isinstance(block, Block)
        # The table should not be built, visualizations are still here
        assert len(block.block_config.visualizations) == 2
        assert len(block.block_config.tables) == 0
    finally:
        close_visualizations(block)

# -------------------------------
# Tests for get_linear_relations_block
# -------------------------------

def test_get_linear_relations_block_smoke():
    df = pd.DataFrame({
        "x1": [1, 2, 3, 4],
        "x2": [2, 4, 6, 8],
        "x3": [4, 3, 2, 1],
    })
    y = pd.Series([1, 0, 1, 0], name="target")

    block = get_linear_relations_block(df, y)
    try:
        assert isinstance(block, Block)
        assert block.block_config.title == "Linear relations"

        # 3 scatterplots + 2 corr matrices
        assert len(block.block_config.visualizations) == 5
        # 2 multicoll tables + 1 highest corr table
        assert len(block.block_config.tables) == 3
    finally:
        close_visualizations(block)

def test_get_linear_relations_block_visualizations_smoke():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [4, 3, 2, 1],
    })
    y = pd.Series([0, 1, 0, 1], name="y")

    block = get_linear_relations_block(df, y)
    try:
        visualizations = block.block_config.visualizations

        assert len(visualizations) == 4
        assert all(isinstance(v, VisualizationResult) for v in visualizations)

        # Optional but safe
        assert all(v.engine == "matplotlib" for v in visualizations)
    finally:
        close_visualizations(block)
    
def test_get_linear_relations_block_pairs_table_structure():
    df = pd.DataFrame({
        "x1": [1, 2, 3, 4],
        "x2": [1, 2, 3, 4],
        "x3": [4, 3, 2, 1],
    })
    y = pd.Series([1, 2, 3, 4], name="target")

    block = get_linear_relations_block(df, y)
    try:
        table = block.block_config.tables[0].table

        assert "coef" in table.columns
        assert table.shape[0] <= 5
    finally:
        close_visualizations(block)

def test_get_linear_relations_block_ignores_non_numeric_features():
    df = pd.DataFrame({
        "num": [1, 2, 3, 4],
        "cat": ["a", "b", "c", "d"],
    })
    y = pd.Series([0, 1, 0, 1], name="target")

    block = get_linear_relations_block(df, y)
    try:
        table = block.block_config.tables[0].table
        # 'cat' should be ignored
        assert "cat" not in table["X"].values
    finally:
        close_visualizations(block)

@pytest.mark.parametrize("nan_policy", ["drop", "raise"])
def test_get_linear_relations_block_nan_policy(nan_policy):
    df = pd.DataFrame({
        "x": [1, 2, np.nan, 4],
    })
    y = pd.Series([1, 0, 1, 0], name="y")

    if nan_policy == "raise":
        with pytest.raises(ValueError):
            get_linear_relations_block(df, y, nan_policy=nan_policy)
    else:
        block = get_linear_relations_block(df, y, nan_policy=nan_policy)
        try:
            assert block.block_config.title == "Linear relations"
        finally:
            close_visualizations(block)

def test_linear_relations_adds_multicollinearity_tables(small_linear_data):
    df, y = small_linear_data

    block = get_linear_relations_block(df, y)
    try:

        tables = block.block_config.tables
        titles = [t.title for t in tables]

        assert "Multicollinearity diagnostic (VIF)" in titles
        assert "Multicollinearity diagnostic (highest correlation)" in titles
    finally:
        close_visualizations(block)

def test_multicollinearity_tables_have_expected_columns(small_linear_data):
    df, y = small_linear_data

    block = get_linear_relations_block(df, y)
    try:
        tables = block.block_config.tables

        vif_table = next(
            t for t in tables if "VIF" in t.title
        ).table

        corr_table = next(
            t for t in tables if "diagnostic (highest correlation)" in t.title.lower()
        ).table

        assert "is_multicollinearity" in vif_table.columns
        assert "VIF" in vif_table.columns

        assert "is_multicollinearity" in corr_table.columns
        assert "highest_correlation" in corr_table.columns
    finally:
        close_visualizations(block)

def test_scatterplots_used_for_small_samples(small_linear_data):
    df, y = small_linear_data
    block = get_linear_relations_block(
        df,
        y,
        sample_size_threshold=5000,
    )
    try:

        visualizations = block.block_config.visualizations
        assert any(
            vr.extra_info is not None and vr.extra_info.get("kind") == "scatterplot"
            for vr in visualizations
        ), "Expected at least one scatter visualization"

    finally:
        close_visualizations(block)

def test_hexbins_used_for_large_samples(large_linear_data):
    df, y = large_linear_data

    block = get_linear_relations_block(
        df,
        y,
        sample_size_threshold=5000,
    )
    try:
        visualizations = block.block_config.visualizations
        assert any(
                vr.extra_info is not None and vr.extra_info.get("kind") == "hexbin"
                for vr in visualizations
            ), "Expected at least one scatter visualization"
    finally:
        close_visualizations(block)

def test_visualization_switching_respects_threshold(small_linear_data):
    df, y = small_linear_data

    block = get_linear_relations_block(
        df,
        y,
        sample_size_threshold=10,
    )
    try:
        visualizations = block.block_config.visualizations

        hexbin_plots = [
            v for v in visualizations
            if "hex" in v.title.lower()
        ]

        assert len(hexbin_plots) > 0
    finally:
        close_visualizations(block)


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
