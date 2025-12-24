"""
Rendering utilities for Explorica reports.

This module provides HTML and PDF rendering functionality for Explorica
report objects. It defines high-level rendering entrypoints for end users,
as well as lower-level helpers responsible for rendering individual report
blocks.

The rendering system is built around two core concepts:

- **Block** — a single, self-contained report unit consisting of metadata,
  metrics, and visualizations.
- **Report** — a higher-level abstraction intended to aggregate multiple
  blocks into a complete report (currently a placeholder).

Two rendering formats are supported:

- **HTML** — supports both static (Matplotlib) and interactive (Plotly)
  visualizations.
- **PDF** — supports static Matplotlib figures; Plotly figures are replaced
  with standardized placeholders due to the lack of interactivity in PDF.

Functions
---------
render_html(data, path, font, **kwargs)
    Render a Block or Report into an HTML representation.
render_pdf(data, path, font, doc_template_kws, **kwargs)
    Render a Block or Report into a PDF byte stream.
render_block_html(block, add_css_style, font, **kwargs)
    Render a single Block into an HTML fragment.
render_block_pdf(block, mpl_fig_scale, plotly_fig_scale, reportlab_styles, block_name)
    Render a single Block into a PDF byte stream.

See Also
--------
explorica.reports.core.Block
    Core block abstraction used as the primary rendering unit.
explorica.reports.utils.normalize_visualization
    Utility for standardizing visualization objects across backends.

Notes
-----
- `render_html` applies CSS styles automatically:
    - for a single Block, the `<style>` is scoped to that block;
    - for a Report, the `<style>` wraps the entire report container.
- Rendering entrypoints (`render_html`, `render_pdf`) are responsible for
  dispatching based on object type (Block vs Report) and optionally saving
  results to disk.
- Both renderers respect visualization scaling (`mpl_fig_scale` and `plotly_fig_scale`).
- Block-level renderers (`render_block_html`, `render_block_pdf`) are lower-
  level utilities and are not intended to be used as primary entrypoints,
  though they remain part of the public API.
- It is assumed that ``Block.block_config.visualizations`` contains
  visualizations normalized into ``VisualizationResult`` objects during
  ``Block`` initialization. If this invariant is violated (e.g. by manual
  mutation or mocking), rendering behavior is undefined.
- PDF rendering is inherently static; interactive visualizations are not
  preserved and are replaced with placeholders where necessary.


Examples
--------
# Render a single block to HTML:
>>> from explorica.reports.renderers import render_html
>>> html = render_html(block)
>>> html[:50]
'<h2>My Block Title</h2>'

# Render a report with multiple blocks to HTML:
>>> from explorica.reports.renderers import render_html
>>> report = Report(
...   [block1, block2], title="My Report", description="Report description")
>>> html_report = render_html(
...   report, font=["Arial", "DejaVu Sans"], report_name="my_report")
>>> html_report[:50]
'<div class=\'my_report\'><h1>My Report</h1>'

# Render a block and save it as a PDF:
>>> from explorica.reports import render_pdf
>>> pdf_bytes = render_pdf(block, path="./reports", report_name="example")

# Render a report to PDF:
>>> report_bytes = render_pdf(report, path="./reports")
"""

from .html import render_block_html, render_html
from .pdf import render_block_pdf, render_pdf

__all__ = ["render_block_html", "render_html", "render_block_pdf", "render_pdf"]
