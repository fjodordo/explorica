"""
Core (low-level) module for Explorica reports.

This module contains the foundational classes for constructing and rendering
report blocks (`Block`), managing block configuration (`BlockConfig`), and
a placeholder for aggregated reports (`Report`).

Classes
-------
BlockConfig
    Dataclass for storing configuration of a report block, including title,
    description, metrics, and visualizations.
Block
    Main class for a report block. Initializes with a `BlockConfig` and provides
    rendering methods for HTML and PDF.
Report
    High-level container for aggregating multiple report blocks into a single
    structured report, with utilities for composition and rendering.

Examples
--------
>>> from explorica.reports import Block, BlockConfig, Report
>>> import matplotlib.pyplot as plt
>>> import plotly.graph_objects as go

# Create a BlockConfig with some metrics and visualizations
>>> config = BlockConfig(
...     title="Sample Block",
...     description="Example block for demonstration",
...     metrics=[{"name": "accuracy", "value": 0.95}],
...     visualizations=[]
... )
>>> block = Block(config)

# Add some visualizations
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3], [4, 5, 6])
>>> block.add_visualization(fig)

>>> figly = go.Figure(data=go.Bar(y=[2, 3, 1]))
>>> block.add_visualization(figly)

# Render individual block
>>> html_output = block.render_html(path=None)
>>> pdf_bytes = block.render_pdf(path=None)

# Create a report with multiple blocks
>>> report = Report(blocks=[block], title="My Report", description="Demo report")
>>> html_report = report.render_html(path=None)
>>> pdf_report = report.render_pdf(path=None)

# Close figures
>>> plt.close(fig)
>>> report.close_figures()
"""

from .block import Block, BlockConfig
from .report import Report

__all__ = [
    "Block",
    "BlockConfig",
    "Report",
]
