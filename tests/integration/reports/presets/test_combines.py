import pytest
import numpy as np
import pandas as pd
import seaborn as sns

from explorica.reports.presets.data_overview import get_data_overview_report, get_data_overview_blocks
from explorica.reports.presets.data_quality import get_data_quality_blocks, get_data_quality_report
from explorica.reports.presets.interactions import get_interactions_blocks, get_interactions_report
from explorica.reports.presets.eda import get_eda_blocks, get_eda_report
from explorica.reports.core.block import Block, BlockConfig
from explorica.reports.core.report import Report
from explorica.types import FeatureAssignment

# -------------------------------
# Fixtures & helper functions
# -------------------------------

@pytest.fixture
def df_mixed():
    return pd.DataFrame({
        "x1": [1, 2, 3, 4],
        "x2": [10, 20, 30, 40],
        "c1": ["a", "b", "a", "b"],
        "y": [0, 1, 0, 1],
    })

# -------------------------------
# Tests for get_eda_blocks
# -------------------------------

def test_get_eda_blocks_basic_blocks_generation(df_mixed):

    blocks = []
    try:
        blocks = get_eda_blocks(df_mixed)
        # Check that `blocks` are returned as a list of Blocks
        assert isinstance(blocks, list)
        assert all(isinstance(b, Block) for b in blocks)
        # Check availability of main blocks
        titles = [b.block_config.title for b in blocks]
        assert "Data Overview" in titles
        assert "Data Quality" in titles
        assert "Feature Interactions" in titles
    finally:
        for block in blocks:
            block.close_figures()

def test_get_eda_blocks_target_assignment_heuristics():
    df = pd.DataFrame({
        "num_small_unique": [0, 1, 0, 1],
        "num_large_unique": [10, 20, 30, 40],
        "cat": ["a", "b", "a", "b"]
    })
    blocks = []
    try:
        blocks = get_eda_blocks(
            df,
            numerical_names=["num_large_unique"],
            target_name="num_small_unique",
            categorical_threshold=3
        )
        assert len(blocks) > 0
        titles = [b.block_config.title for b in blocks]
        assert "Feature Interactions" in titles
        assert "Linear relations" in titles
        assert "Non-linear relations" in titles
    finally:
        for block in blocks:
            block.close_figures()

def test_get_eda_blocks_nan_policy_include_vs_drop():
    df = pd.DataFrame({
        "x1": [1, 2, None, 4],
        "c1": ["a", "b", None, "b"],
        "y": [0, 1, 0, None]
    })
    blocks_drop = []
    blocks_include = []
    try:
        # nan_policy='drop'
        blocks_drop = get_eda_blocks(df, nan_policy="drop")
        assert all(isinstance(b, Block) for b in blocks_drop)
        # nan_policy='include'
        blocks_include = get_eda_blocks(df, nan_policy="include")
        assert all(isinstance(b, Block) for b in blocks_include)
    finally:
        for block in blocks_drop + blocks_include:
            block.close_figures()

# -------------------------------
# Tests for get_eda_report
# -------------------------------

def test_get_eda_report_smoke(mocker):
    df = pd.DataFrame({"x": [1, 2, 3]})

    fake_blocks = []

    mocked_blocks = mocker.patch(
        "explorica.reports.presets.eda.get_eda_blocks",
        return_value=fake_blocks,
    )

    report = get_eda_report(
        df,
        numerical_names=["x"],
        target_name="x",
        categorical_threshold=42,
        round_digits=7,
        nan_policy="raise",
    )

    try:
        # Report is constructed correctly
        assert isinstance(report, Report)
        assert report.title == "Exploratory Data Analysis Report"
        assert len(report.blocks) == len(fake_blocks)

        # Parameters are forwarded correctly
        mocked_blocks.assert_called_once_with(
            df,
            ["x"],
            None,
            "x",
            categorical_threshold=42,
            round_digits=7,
            nan_policy="raise",
        )
    finally:
        report.close_figures()


def test_get_eda_report_example_based(tmp_path):
    """
    Example-based integration test for the EDA preset.

    This test validates the full EDA report generation pipeline on a fixed,
    real-world dataset (Titanic from seaborn). The goal is to ensure that the
    entire preset - including preprocessing, statistics, visualizations, and
    dependency analysis - remains stable and correct as the codebase evolves.

    Key properties of this test:
    - It is example-based rather than purely unit-based.
    - The input dataset is deterministic.
    - Expected outputs are hard-coded and treated as a ground truth baseline.
    - Numeric expectations were precomputed using external trusted tools or
    prior validated runs.

    What this test verifies:
    - End-to-end report generation (PDF & HTML rendering).
    - Structural integrity of the report (block presence and ordering).
    - Correctness of statistical summaries (mean, median, mode, skewness, etc.).
    - Data quality metrics (NaNs, cardinality, entropy).
    - Outlier detection logic.
    - Linear dependency analysis (correlations, VIF, multicollinearity).
    - Non-linear dependency analysis.
    - Stability of visual outputs (count-based assertions).

    Why hard-coded expected values are acceptable here:
    Since the dataset and computation logic are deterministic, any deviation in
    results likely indicates a regression, numerical instability, or a breaking
    change in methodology. This test therefore acts as a regression guardrail
    for the entire EDA preset.

    This is intentionally a high-cost, high-coverage test and should not replace
    unit tests. Instead, it complements them by ensuring system-level correctness.
    """
    data = sns.load_dataset("titanic")
    report = get_eda_report(data, target_name="survived")
    try:
        # Smoke test PDF rendering
        pdf_bytes = report.render_pdf(path=str(tmp_path), report_name="titanic_eda_report")

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 10_000
        assert (tmp_path / "titanic_eda_report.pdf").exists()

        # Smoke test HTML rendering
        html_str = report.render_html(path=str(tmp_path), report_name="titanic_eda_report")
        assert isinstance(html_str, str)
        assert (tmp_path / "titanic_eda_report.html").exists()

        # Structure checks
        assert report.title == "Exploratory Data Analysis Report"
        structure = [
            "Data Overview",
            "Data quality quick summary",
            "Basic statistics for the dataset",
            "Dataset shape",
            "Data Quality",
            "Cardinality",
            "Distributions",
            "Outliers",
            "Feature Interactions",
            "Linear relations",
            "Non-linear relations",
        ]
        for section in structure:
            assert any(b.block_config.title == section for b in report.blocks), f"Missing block: {section}"

        # Computations checks

        # central tendency measures block
        block_ctm = next(
            b for b in report.blocks if b.block_config.title == "Basic statistics for the dataset"
        )
        mean_expected = pd.Series(
            [0.383838, 2.308642, 29.699118, 0.523008, 0.381594, 32.204208],
            index=["survived", "pclass", "age", "sibsp", "parch", "fare"], name="mean").sort_index()
        mean_result = block_ctm.block_config.tables[0].table["mean"].sort_index()
        assert all(np.isclose(mean_expected, mean_result, atol=1e-4))

        median_expected = pd.Series(
            [0.0, 3.0, 28.0, 0.0, 0.0, 14.4542],
            index=["survived", "pclass", "age", "sibsp", "parch", "fare"], name="median").sort_index()
        median_result = block_ctm.block_config.tables[0].table["median"].sort_index()
        assert all(np.isclose(median_expected, median_result, atol=1e-4))

        mode_expected = pd.Series(
            [0, 3, "male", 24.0, 0, 0, 8.05, "S",
             "Third", "man", True, "C", "Southampton", "no", True],
            index=["survived", "pclass", "sex", "age", "sibsp",
                   "parch", "fare", "embarked", "class", "who", "adult_male",
                   "deck", "embark_town", "alive", "alone"
                   ], name="mode").sort_index()
        mode_result = block_ctm.block_config.tables[1].table["mode"].sort_index()
        assert all(mode_expected == mode_result)

        # dataset shape block
        block_data_shape = next(
            b for b in report.blocks if b.block_config.title == "Dataset shape"
        )
        assert block_data_shape.block_config.metrics[0]["value"] == 891
        assert block_data_shape.block_config.metrics[1]["value"] == 15
        tr = block_data_shape.block_config.tables[0].table
        assert tr[tr["dtype"] == "int64"].iloc[0, 1] == 4

        # data quality overview block
        block_dq_overview = next(
            b for b in report.blocks if b.block_config.title == "Data quality quick summary"
        )
        nan_table = block_dq_overview.block_config.tables[0].table
        nan_table = nan_table[nan_table["nan_count"] > 0]
        assert set(nan_table.index) == {
            "age", "deck", "embarked", "embark_town"}

        # outliers block
        block_outliers = next(
            b for b in report.blocks if b.block_config.title == "Outliers"
        )
        outlier_table = block_outliers.block_config.tables[0].table
        iqr_expected = pd.Series(
            [0, 0, 11, 46, 213, 116],
            index=["survived", "pclass", "age", "sibsp", "parch", "fare"],
            name="IQR (1.5)").sort_index()
        iqr_result = outlier_table["IQR (1.5)"].sort_index()
        assert all(iqr_expected == iqr_result)

        zscrore_expected = pd.Series([0, 0, 2, 30, 15, 20],
                                     index=["survived", "pclass", "age", "sibsp", "parch", "fare"],
                                     name="Z-Score (3.0σ)").sort_index()
        zscore_result = outlier_table["Z-Score (3.0σ)"].sort_index()
        assert all(zscrore_expected == zscore_result)

        # distributions block
        block_distributions = next(
            b for b in report.blocks if b.block_config.title == "Distributions"
        )
        sk_expected = pd.DataFrame({
            "skewness": [0.478523, -0.630548, 0.389108, 3.695352, 2.749117, 4.787317],
            "kurtosis": [-1.775005, -1.280015, 0.178274, 17.880420, 9.778125, 33.398141]
            }, index=['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']).sort_index()
        sk_result = block_distributions.block_config.tables[0].table[["skewness", "kurtosis"]].sort_index()
        pd.testing.assert_frame_equal(sk_expected, sk_result, atol=0.2)
        
        assert len(block_distributions.block_config.visualizations) == 12
        
        # cardinality block
        block_cardinality = next(
            b for b in report.blocks if b.block_config.title == "Cardinality"
        )
        cardinality_table = block_cardinality.block_config.tables[0].table
        assert all(~cardinality_table["is_constant"])
        assert all(~cardinality_table["is_unique"])

        nunique_expected = pd.Series(
            {'survived': np.int64(2),
             'pclass': np.int64(3),
             'sex': np.int64(2),
             'age': np.int64(88),
             'sibsp': np.int64(7),
             'parch': np.int64(7),
             'fare': np.int64(248),
             'embarked': np.int64(3),
             'class': np.int64(3),
             'who': np.int64(3),
             'adult_male': np.int64(2),
             'deck': np.int64(7),
             'embark_town': np.int64(3),
             'alive': np.int64(2),
             'alone': np.int64(2)}, name="n_unique").sort_index()
        nunique_result = cardinality_table["n_unique"].sort_index()
        pd.testing.assert_series_equal(nunique_expected, nunique_result)

        entropy_expected = pd.Series({
            'survived': 0.9607079018756469,
            'pclass': 0.9081107406574219,
            'sex': 0.9362046432498521,
            'age': 0.9035754194307962,
            'sibsp': 0.47680439862432533,
            'parch': 0.4019218837689023,
            'fare': 0.8853899579593114,
            'embarked': 0.6920475086483984,
            'class': 0.9081107406574219,
            'who': 0.8085498105061705,
            'adult_male': 0.969353114218277,
            'deck': 0.8891975341159102,
            'embark_town': 0.6920475086483984,
            'alive': 0.9607079018756469,
            'alone': 0.969353114218277,
        }, name="entropy (normalized)")
        entropy_result = cardinality_table["entropy (normalized)"]
        pd.testing.assert_series_equal(entropy_expected, entropy_result, atol=1e-4)

        # interactions linear block
        interactions_linear_block = next(
            b for b in report.blocks if b.block_config.title == "Linear relations"
        )
        highest_correlations_expected = pd.DataFrame({
            "coef": [-0.360656, -0.359653, 0.339027, 0.268189, 0.156444],
            "method": ["spearman", "pearson", "spearman", "pearson", "spearman"]
        })
        highest_correlations_result = interactions_linear_block.block_config.tables[0].table[["coef", "method"]]

        pd.testing.assert_frame_equal(highest_correlations_expected, highest_correlations_result, atol=1e-1)

        multicol_vif_expected = pd.Series({
            'pclass': 1.706840,
            'age': 1.292372,
            'sibsp': 1.273264,
            'parch': 1.229164,
            'fare': 1.581994,
        }, name="VIF")
        multicol_vif_result = interactions_linear_block.block_config.tables[1].table["VIF"]

        pd.testing.assert_series_equal(multicol_vif_expected, multicol_vif_result, atol=1e-4)

        multicol_corr_expected = pd.Series({
            'pclass': -0.730578,
            'age': -0.369226,
            'sibsp': 0.426955,
            'parch': 0.426955,
            'fare': -0.730578,
        }, name="highest_correlation").sort_index()
        multicol_corr_result = (interactions_linear_block.
                                block_config.tables[2].table["highest_correlation"].sort_index())
        pd.testing.assert_series_equal(multicol_corr_expected, multicol_corr_result, atol=1e-4)

        assert len(interactions_linear_block.block_config.visualizations) == 7

        # interactions non-linear block
        interactions_nonlinear_block = next(
            b for b in report.blocks if b.block_config.title == "Non-linear relations"
        )
        highest_nonlin_deps_expected = pd.DataFrame([
            ["alive", 0.988806],
            ["adult_male",0.580525],
            ["who", 0.593623],
            ["sex", 0.532007],
            ["age", 0.251045]
            ],
            columns = ["X", "coef"]
        )
        highest_nonlin_deps_result = interactions_nonlinear_block.block_config.tables[0].table[
            ["X", "coef"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(highest_nonlin_deps_expected, highest_nonlin_deps_result, atol=0.02)

        assert len(interactions_nonlinear_block.block_config.visualizations) == 2
    finally:
        report.close_figures()


# -------------------------------
# Tests for get_interactions_blocks
# -------------------------------

def test_get_interactions_blocks_linear_block_always_created(df_mixed):
    blocks = get_interactions_blocks(df_mixed)
    try:
        assert len(blocks) >= 1
        assert blocks[0].block_config.title == "Linear relations"
    finally:
        for block in blocks:
            block.close_figures()

def test_get_interactions_blocks_nonlinear_block_conditionally_added(df_mixed):
    blocks = get_interactions_blocks(df_mixed)
    try:
        titles = [b.block_config.title for b in blocks]

        assert "Linear relations" in titles
        # Non-linear may be included, and may be not included. check consistency.
        assert len(titles) in (1, 2)
    finally:
        for block in blocks:
            block.close_figures()

def test_get_interactions_blocks_feature_assignment_priority(df_mixed):

    blocks = get_interactions_blocks(
        df_mixed, numerical_names=["x1"], categorical_names=["c1"],
        target_numerical_name="y")
    try:
        linear_block = blocks[0]

        # x2 should not be included in linear-relations tables
        for table in linear_block.block_config.tables:
            assert "x2" not in table.table.to_string()
    finally:
        for block in blocks:
            block.close_figures()

def test_get_interactions_blocks_mapping_input_supported():
    data = {
        "x1": [1, 2, 3],
        "x2": [4, 5, 6],
        "y": [0, 1, 0],
    }

    blocks = get_interactions_blocks(data)
    try:
        assert len(blocks) >= 1
        assert blocks[0].block_config.title == "Linear relations"
    finally:
        for block in blocks:
            block.close_figures()

def test_get_interactions_blocks_no_numeric_features():
    df = pd.DataFrame({"c1": ["a", "b", "c"]})

    blocks = get_interactions_blocks(df)
    try:
        # block should not be included
        assert all(b.block_config.title != "Linear relations" for b in blocks)
    finally:
        for block in blocks:
            block.close_figures()

# -------------------------------
# Tests for get_interactions_report
# -------------------------------

def test_get_interactions_report_smoke(mocker):
    df = pd.DataFrame({"x": [1, 2, 3]})

    fake_blocks = []

    mocked_blocks = mocker.patch(
        "explorica.reports.presets.interactions.get_interactions_blocks",
        return_value=fake_blocks,
    )

    report = get_interactions_report(
        df,
        categorical_threshold=42,
        round_digits=7,
        nan_policy="raise",
    )
    try:
        # Report is constructed correctly
        assert isinstance(report, Report)
        assert report.title == "Interaction analysis"
        assert len(report.blocks) == len(fake_blocks)

        # Parameters are forwarded
        mocked_blocks.assert_called_once_with(
            df,
            numerical_names=None,
            categorical_names=None,
            target_name=None,
            categorical_threshold=42,
            round_digits=7,
            nan_policy="raise",
        )
    finally:
        report.close_figures()

# -------------------------------
# Tests for get_data_quality_blocks
# -------------------------------

def test_get_data_quality_blocks_smoke(mocker):
    data = {"a": [1, 2, 3]}
    round_digits = 3

    card_block = Block({"title": "cardinality"})
    dist_block = Block({"title": "distributions"})
    out_block = Block({"title": "outliers"})

    card_mock = mocker.patch(
        "explorica.reports.presets.data_quality.get_cardinality_block",
        return_value=card_block,
    )
    dist_mock = mocker.patch(
        "explorica.reports.presets.data_quality.get_distributions_block",
        return_value=dist_block,
    )
    out_mock = mocker.patch(
        "explorica.reports.presets.data_quality.get_outliers_block",
        return_value=out_block,
    )

    blocks = get_data_quality_blocks(data, round_digits=round_digits)
    try:
        # Result
        assert blocks[0].block_config.title == "outliers"
        assert blocks[1].block_config.title == "distributions"
        assert blocks[2].block_config.title == "cardinality"

        # Calls
        card_mock.assert_called_once_with(data, round_digits=round_digits, nan_policy="drop_with_split")
        dist_mock.assert_called_once_with(data, round_digits=round_digits, nan_policy="drop_with_split")
        out_mock.assert_called_once_with(data, nan_policy="drop_with_split")
    finally:
        for block in blocks:
            block.close_figures()


# -------------------------------
# Tests for get_data_quality_report
# -------------------------------

def test_get_data_quality_report_smoke(mocker):
    df = pd.DataFrame({"a": [1, 2, 3]})

    fake_blocks = [
        Block({"title": "block0"}),
        Block({"title": "block1"}),
        Block({"title": "block2"}),
    ]

    blocks_mock = mocker.patch(
        "explorica.reports.presets.data_quality.get_data_quality_blocks",
        return_value=fake_blocks,
    )
    report = get_data_quality_report(df, 3)
    try:
        # orchestration
        blocks_mock.assert_called_once_with(df, 3, nan_policy="drop_with_split")

        # report object
        assert isinstance(report, Report)
        assert report.blocks[0].block_config.title == "block0"
        assert report.blocks[1].block_config.title == "block1" 
        assert report.blocks[2].block_config.title == "block2" 
        assert report.title == "Data quality"
    finally:
        report.close_figures()

# -------------------------------
# Tests for get_data_overview_blocks
# -------------------------------

def test_get_data_overview_blocks_basic():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"]
    })

    blocks = get_data_overview_blocks(df, round_digits=3)
    try:
        # Check block type and count
        assert isinstance(blocks, list)
        assert len(blocks) == 3
        assert all(isinstance(b, Block) for b in blocks)

        # Check block headers
        titles = [b.block_config.title for b in blocks]
        assert "Basic statistics for the dataset" in titles
        assert "Dataset shape" in titles
        assert "Data quality quick summary" in titles
    finally:
        for block in blocks:
            block.close_figures()

# -------------------------------
# Tests for get_data_overview_report
# -------------------------------

def test_get_data_overview_report_with_mocked_blocks(mocker):
    # make fake data
    dummy_blocks = [
        Block(BlockConfig(title="block-1")),
        Block(BlockConfig(title="block-2"))
    ]

    # mock get_data_overview_blocks
    mocker.patch(
        "explorica.reports.presets.data_overview.get_data_overview_blocks",
        return_value=dummy_blocks
    )

    report = get_data_overview_report(data={"a": [1, 2, 3]})
    try:
        # check type and attrs
        assert isinstance(report, Report)
        assert report.blocks[0].block_config.title == dummy_blocks[0].block_config.title
        assert report.blocks[1].block_config.title == dummy_blocks[1].block_config.title
        assert report.title == "Data overview"
        assert "Short overview of the dataset" in report.description
    finally:
        report.close_figures()