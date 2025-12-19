import pdb
from pathlib import Path
from io import BytesIO

from pypdf import PdfReader
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

from explorica.reports import BlockConfig, Block, Report, render_pdf


# -------------------------------
# Helper functions
# -------------------------------

def make_sample_dataframe(n=500, seed=42):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "age": rng.integers(18, 65, size=n),
        "income": rng.normal(80_000, 20_000, size=n),
        "tenure_months": rng.integers(1, 120, size=n),
        "country": rng.choice(["DE", "FR", "NL", "PL"], size=n, p=[0.4, 0.25, 0.2, 0.15]),
    })

    # realistic issues
    df.loc[rng.random(n) < 0.1, "income"] = np.nan
    df.loc[rng.random(n) < 0.05, "age"] = np.nan

    return df

def make_data_quality_block(df: pd.DataFrame) -> BlockConfig:
    missing_rate = df.isna().mean()

    fig, ax = plt.subplots(figsize=(6, 3))
    missing_rate.sort_values(ascending=False).plot.bar(ax=ax)
    ax.set_title("Missing Values Rate")
    ax.set_ylabel("Share")

    return BlockConfig(
        title="Data Quality Overview",
        description="High-level overview of dataset completeness.",
        metrics=[
            {"name": "Rows", "value": len(df)},
            {"name": "Columns", "value": df.shape[1]},
            {"name": "Avg missing rate", "value": round(missing_rate.mean(), 3)},
        ],
        visualizations=[fig],
    )

def make_distribution_block(df: pd.DataFrame) -> BlockConfig:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    df["age"].dropna().plot.hist(bins=20, ax=axes[0])
    axes[0].set_title("Age distribution")

    df["income"].dropna().plot.hist(bins=20, ax=axes[1])
    axes[1].set_title("Income distribution")

    return BlockConfig(
        title="Feature Distributions",
        description="Key numerical feature distributions.",
        metrics=[
            {"name": "Avg age", "value": round(df["age"].mean(), 1)},
            {"name": "Median income", "value": round(df["income"].median(), 0)},
        ],
        visualizations=[fig],
    )

def make_variance_block(df: pd.DataFrame) -> BlockConfig:
    fig, ax = plt.subplots(figsize=(6, 3))
    df[["age", "income", "tenure_months"]].boxplot(ax=ax)
    ax.set_title("Feature Spread")

    return BlockConfig(
        title="Variance Analysis",
        description="Spread and stability of key metrics.",
        metrics=[
            {"name": "Age std", "value": round(df["age"].std(), 2)},
            {"name": "Income std", "value": round(df["income"].std(), 2)},
        ],
        visualizations=[fig],
    )

def close_report_figures(report: Report):
    for block in report:
        for vis in block.block_config.visualizations:
            if isinstance(vis.figure, matplotlib.figure.Figure):
                plt.close(vis.figure)

# -------------------------------
# Tests for render_pdf
# -------------------------------


def test_render_pdf_example_based(tmp_path):
    df = make_sample_dataframe()

    blocks = [
        Block(make_data_quality_block(df)),
        Block(make_distribution_block(df)),
        Block(make_variance_block(df)),
    ]

    report = Report(
        blocks=blocks,
        title="Customer Dataset EDA",
        description="Exploratory data analysis for internal review."
    )

    pdf_bytes = render_pdf(report, path=Path(tmp_path), report_name="eda_report", mpl_fig_scale=50)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 10_000
    assert (tmp_path / "eda_report.pdf").exists()

    # Parse bytes build

    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) > 0  # pages exists

    text_content = "\n".join([reader.pages[i].extract_text() for i in range(len(reader.pages))])
    
    # assert page width/height ratio (~1/sqrt(2) for A4)

    pagesizes = [page.mediabox for page in reader.pages]
    for psize in pagesizes:
        assert np.isclose(float(psize.width) / float(psize.height), 0.7070706878738762, atol=0.001)


    # assert headers
    headers = ["Customer Dataset EDA",
               "Data Quality Overview",
               "Feature Distributions",
               "Variance Analysis"]
    for header in headers:
        assert header in text_content
    
    # assert descriptions
    descs = ["Exploratory data analysis for internal review",
             "High-level overview of dataset completeness.",
             "Key numerical feature distributions.",
             "Spread and stability of key metrics."]
    for desc in descs:
        assert desc in text_content

    # assert metrics
    metrics = ["Rows: 500",
               "Columns: 4",
               "Avg missing rate: 0.034",
               "Avg age: 41.0",
               "Median income: 78661.0",
               "Age std: 13.48",
               "Income std: 20857.46",]
    for metric in metrics:
        assert metric  in text_content

    # assert images count
    images = []

    for page in reader.pages:
        xobjects = page["/Resources"].get("/XObject", {})
        for obj in xobjects.values():
            if obj.get("/Subtype") == "/Image":
                images.append(obj)
    assert len(images) == 3

    close_report_figures(report)