import pdb
from pathlib import Path
from io import BytesIO

from pypdf import PdfReader
import pytest
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

from explorica.types import TableResult
from explorica.reports import BlockConfig, Block, Report, render_pdf, render_html


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

def make_data_overview_block(df: pd.DataFrame) -> BlockConfig:
    # Prepare summary tables
    central_tendency = df.describe().loc[["mean", "50%", "std"]].rename(index={"50%": "median"})
    central_tendency_table = TableResult(
        table=central_tendency,
        title="Central Tendency",
        description="Mean, median, and standard deviation per numerical feature."
    )

    value_ranges = df.describe().loc[["min", "max"]]
    value_ranges_table = TableResult(
        table=value_ranges,
        title="Feature Ranges",
        description="Minimum and maximum values per numerical feature."
    )

    block_cfg = BlockConfig(
        title="Data Overview",
        description="Summary tables of central tendency and ranges.",
        tables=[central_tendency_table, value_ranges_table],
        metrics=[
            {"name": "Numeric features", "value": len(df.select_dtypes("number").columns)},
            {"name": "Categorical features", "value": len(df.select_dtypes("object").columns)}
        ],
        visualizations=[]
    )
    return block_cfg


def make_data_quality_block(df: pd.DataFrame) -> BlockConfig:
    missing_rate = df.isna().mean()

    fig, ax = plt.subplots(figsize=(6, 3))
    missing_rate.sort_values(ascending=False).plot.bar(ax=ax)
    ax.set_title("Missing Values Rate")
    ax.set_ylabel("Share")

    return BlockConfig(
        title="Data Quality",
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


# -------------------------------
# Tests for render_html
# -------------------------------

def test_render_html_example_based(tmp_path):
    df = make_sample_dataframe()

    blocks = [
        Block(make_data_overview_block(df)),
        Block(make_data_quality_block(df)),
        Block(make_distribution_block(df)),
        Block(make_variance_block(df)),
    ]
    report = Report(
        blocks=blocks,
        title="Customer Dataset EDA",
        description="Exploratory data analysis for internal review."
    )
    try:
        html_page = render_html(report, path=str(tmp_path), report_name="eda_report")

        assert isinstance(html_page, str)
        assert (tmp_path/"eda_report.html").exists()

        # parse html
        soup = BeautifulSoup(html_page, "html.parser")

        # check report container
        report_div = soup.find("div", class_="eda_report")
        assert report_div is not None

        # check header
        h1 = report_div.find("h1")
        assert h1.text == "Customer Dataset EDA"

        # check description
        p_desc = report_div.find("p")
        assert "Exploratory data analysis" in p_desc.text

        # check blocks
        blocks_divs = report_div.find_all("div", class_=lambda c: c and "eda_report_block" in c)
        assert len(blocks_divs) == 4

        # check block titles and descriptions
        expected_titles = ["Data Overview", "Data Quality", "Feature Distributions", "Variance Analysis"]
        expected_descs = [
            "Summary tables of central tendency and ranges.",
            "High-level overview of dataset completeness.",
            "Key numerical feature distributions.",
            "Spread and stability of key metrics."
        ]
        for bdiv, title, desc in zip(blocks_divs, expected_titles, expected_descs):
            h2 = bdiv.find("h2")
            assert h2.text == title
            p = bdiv.find("p")
            assert desc in p.text

        # check metrics
        metrics_texts = ["Rows: 500", "Columns: 4", "Avg missing rate: 0.034",
                        "Avg age: 41.0", "Median income: 78661.0",
                        "Age std: 13.48", "Income std: 20857.46"]
        full_text = soup.get_text()
        for metric in metrics_texts:
            assert metric in full_text

        # check inline styles / font-family
        style_tag = soup.find("style")
        assert style_tag is not None
        assert "font-family" in style_tag.text
        assert "Arial" in style_tag.text  # default font_family

        # check images/iframes
        images = report_div.find_all(["img", "iframe"])
        assert len(images) >= 3

       # check tables in Data Overview block
        tables = blocks_divs[0].find_all("table", class_="explorica-dataframe")
        assert len(tables) == 2  # 2 tables: Central Tendency и Feature Ranges

        # table 1: Central Tendency
        
        # check columns
        table1 = tables[0]
        columns = [col.get_text(strip=True) for col in table1.find('thead').find_all('tr')[-1].find_all("th")]
        expected_columns = ["age", "income", "tenure_months"]
        for c in columns:
            assert c in expected_columns or c == ""
        
        # check indices
        rows1 = table1.find_all("tr")
        expected_indices = ["", "mean", "median", "std"]

        for ind, exp_ind in zip([row.find("th").get_text(strip=True) for row in rows1], expected_indices):
            assert ind == exp_ind

        # check rows
        expected_rows1 = [["41.012658", "78676.672694", "59.214000"],
                         ["41.000000", "78660.936542", "58.500000"],
                         ["13.482110", "20857.456077", "34.597662"]]
        for row, expected_row in zip(rows1[1:], expected_rows1):
            cells = [td.text for td in row.find_all("td")]
            for cell, exp in zip(cells, expected_row):
                assert cell == exp

        # table 2: Feature Ranges

        # check columns
        table2 = tables[1]
        columns = [col.get_text(strip=True) for col in table2.find('thead').find_all('tr')[-1].find_all("th")]
        expected_columns = ["age", "income", "tenure_months"]
        for c in columns:
            assert c in expected_columns or c == ""

        # check indices
        rows2 = table2.find_all("tr")
        expected_indices = ["", "min", "max"]

        for ind, exp_ind in zip([row.find("th").get_text(strip=True) for row in rows2], expected_indices):
            assert ind == exp_ind
        
        # check rows
        expected_rows2 = [["18.0", "20709.423243", "1.0"],
                          ["64.0", "138101.343385", "119.0"]]
        for row, expected_row in zip(rows2[1:], expected_rows2):
            cells = [td.text for td in row.find_all("td")]
            for cell, exp in zip(cells, expected_row):
                assert cell == exp
    finally:
        report.close_figures()

# -------------------------------
# Tests for render_pdf
# -------------------------------


def test_render_pdf_example_based(tmp_path):
    df = make_sample_dataframe()

    blocks = [
        Block(make_data_overview_block(df)),
        Block(make_data_quality_block(df)),
        Block(make_distribution_block(df)),
        Block(make_variance_block(df)),
    ]

    report = Report(
        blocks=blocks,
        title="Customer Dataset EDA",
        description="Exploratory data analysis for internal review."
    )
    try:
        pdf_bytes = render_pdf(report, path=str(tmp_path), report_name="eda_report")

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
                "Data Quality",
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

        # assert first block tables (Data Overview)
        expected_table_texts = [
            # Central Tendency
            "mean\n41.0126582278481\n78676.67269442741\n59.214",
            "median\n41.0\n78660.93654203598\n58.5",
            "std\n13.482109675083448\n20857.456076924787\n34.59766177649585",
            # Feature Ranges
            "min\n18.0\n20709.423243166973\n1.0",
            "max\n64.0\n138101.34338480813\n119.0"
        ]
        for txt in expected_table_texts:
            assert txt in text_content
    finally:
        report.close_figures()
