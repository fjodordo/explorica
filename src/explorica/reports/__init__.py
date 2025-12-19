"""
Facade for Explorica reporting functionality.

This package provides a unified interface for constructing and rendering
analytical reports in Explorica. It exposes high-level building blocks
(`Block`, `BlockConfig`) as well as rendering entrypoints for exporting
reports and report blocks to different formats.

The reports module is designed around the concept of *blocks* — independent,
self-contained report units that combine textual content, metrics, and
visualizations. Blocks can be rendered individually or later aggregated
into full reports.

Currently supported output formats include:
- HTML (with interactive Plotly visualizations)
- PDF (static layout; Plotly figures are replaced with placeholders)

Classes
-------
Block(block_config)
    Core report building unit representing a single report block.
    A block contains a title, description, metrics, and visualizations,
    and provides methods for rendering itself into supported formats.
BlockConfig
    Configuration dataclass defining the content of a `Block`, including
    textual metadata, metrics, and visualizations.
Report
    High-level container for aggregating multiple report blocks into a single
    structured report, with utilities for composition and rendering.

Functions
---------
render_html(data, path, report_name)
    Render a `Block` or `Report` object into an HTML representation.
render_pdf(data, path, report_name)
    Render a `Block` or `Report` object into a PDF byte stream.
render_block_html(block)
    Render a single `Block` into an HTML fragment.
render_block_pdf(block, doc_template_kws)
    Render a single `Block` into a PDF byte stream.
normalize_visualization(figure)
    Normalize Matplotlib and Plotly figures into a unified
    `VisualizationResult` representation.

Notes
-----
- Visualization objects provided to `BlockConfig.visualizations` are
  automatically normalized into `VisualizationResult` instances during
  `Block` initialization.
- Interactive Plotly figures are supported only in HTML output.
  For PDF rendering, Plotly figures are replaced with standardized
  placeholders.
- The `Report` abstraction is currently a placeholder and will be extended
  to support aggregation and layout of multiple blocks.

Examples
--------
# Minimal example of creating and rendering a block:
>>> from explorica.reports import Block, BlockConfig, render_html
>>> import matplotlib.pyplot as plt
>>>
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3], [4, 5, 6])
>>>
>>> config = BlockConfig(
...     title="Example Block",
...     description="Simple report block example",
...     metrics=[{"name": "Accuracy", "value": 0.92}],
...     visualizations=[fig],
... )
>>>
>>> block = Block(config)
>>> html = render_html(block)
>>> html[:50]
'<h2>Example Block</h2>'
"""

from .core import Block, BlockConfig, Report
from .utils import normalize_visualization
from .renderers import render_block_html, render_block_pdf, render_html, render_pdf

__all__ = [
    "Block",
    "BlockConfig",
    "Report",
    "normalize_visualization",
    "render_block_html",
    "render_block_pdf",
    "render_html",
    "render_pdf",
]
