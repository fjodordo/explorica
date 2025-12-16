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
render_html(data, path, report_name)
    Render a Block or Report into an HTML representation.
render_pdf(data, path, report_name)
    Render a Block or Report into a PDF byte stream.
render_block_html(block)
    Render a single Block into an HTML fragment.
render_block_pdf(block, doc_template_kws)
    Render a single Block into a PDF byte streams.

See Also
--------
explorica.reports.core.Block
    Core block abstraction used as the primary rendering unit.
explorica.reports.utils.normalize_visualization
    Utility for standardizing visualization objects across backends.

Notes
-----
- Rendering entrypoints (`render_html`, `render_pdf`) are responsible for
  dispatching based on object type (Block vs Report) and optionally saving
  results to disk.
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
>>> from explorica.reports import render_html
>>> html = render_html(block)
>>> html[:50]
'<h2>My Block Title</h2>'

# Render a block and save it as a PDF:
>>> from explorica.reports import render_pdf
>>> pdf_bytes = render_pdf(block, path="./reports", report_name="example")
"""

from pathlib import Path
from io import BytesIO
import base64

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image
import plotly.graph_objects
import plotly.io as pio
import matplotlib.figure

from explorica.visualizations._utils import get_empty_plot


def render_pdf(
    data, path: str | Path | None = None, report_name: str = "report"
) -> bytes:
    """
    Render a Block or Report into PDF format.

    This function renders a single `Block` or a `Report` (sequence of blocks)
    into a PDF format. It serves as the main entry point for PDF export
    functionality in the reports module. Currently, only rendering of individual
    `Block` objects is fully supported. Rendering of `Report` objects is planned
    for future implementation.

    Parameters
    ----------
    data : Block or Report
        The data object to render. Currently, only single Blocks are
        fully supported. Rendering a Report is not implemented yet.
    path : str or Path, optional
        If provided, the generated PDF bytes will also be saved to this
        location. Can be either a directory (in which case the default
        report name will be used) or a full path ending with '.pdf'.
    report_name : str, default="report"
        The base name used when saving the PDF if `path` is a directory.
        Also serves as a placeholder name in logs or error messages.

    Returns
    -------
    bytes
        PDF content as bytes.

    Raises
    ------
    TypeError
        If `data` is not a Block or Report.
    NotImplementedError
        If `data` is a Report.

    Notes
    -----
    - This function provides a unified interface for PDF generation from both
      individual blocks and eventually from full report sequences.
    - Users can optionally save the generated PDF by providing the `path`
      parameter.
    - It is assumed that `Block().block_config.visualizations` contains a list
      of figures wrapped as `VisualizationResult` instances. During `Block`
      initialization, all figures are automatically normalized into
      `VisualizationResult` objects, so downstream rendering or processing
      functions can safely rely on a uniform interface.

    Examples
    --------
    >>> from explorica.reports.renderers import render_pdf
    >>> from explorica.reports.core import Block, BlockConfig
    >>> import matplotlib.pyplot as plt

    # Block render usage
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> block_cfg = BlockConfig(
    ...     title="Example Block",
    ...     description="A simple block with one plot",
    ...     metrics=[{"name": "sum", "value": 15}],
    ...     visualizations=[fig]
    ... )
    >>> block = Block(block_cfg)

    >>> # Render PDF as bytes
    >>> pdf_bytes = render_pdf(block)

    >>> # Render and save to a directory
    >>> render_pdf(block, path="./reports", report_name="my_block")
    """
    if data.typename == "Block":
        pdf_bytes = render_block_pdf(data)
        if path is not None:
            _save_pdf(pdf_bytes, path, report_name=report_name)
        return pdf_bytes
    if data.typename == "Report":
        raise NotImplementedError("PDF rendering for Report is not implemented yet")

    raise TypeError("'data' must be Block or Report")


def render_block_pdf(block, doc_template_kws: dict = None) -> bytes:
    """
    Render a single Block object into a PDF byte stream.

    The PDF includes the block's title, description, metrics, and visualizations.
    Plotly figures are replaced with a standardized placeholder, since
    interactive visualizations cannot be rendered directly in PDF.

    Parameters
    ----------
    block : Block
        The block to render.
    doc_template_kws : dict, optional
        Additional keyword arguments to pass to `SimpleDocTemplate` for
        customizing PDF layout (e.g., margins, page size).

    Returns
    -------
    bytes
        PDF content as a byte string.

    Notes
    -----
    - Matplotlib figures are saved directly into the PDF.
    - Plotly figures are replaced by a placeholder using `get_empty_plot`.
    - The PDF layout uses A4 pages with fixed margins and spacing between
      elements.
    - This is a lower-level utility function. For end users, the higher-level
      `render_pdf` function is the recommended entry point.
    - It is assumed that `block.block_config.visualizations` contains a list
      of figures wrapped as `VisualizationResult` instances. During `Block`
      initialization, all figures are automatically normalized into
      `VisualizationResult` objects, so downstream rendering or processing
      functions can safely rely on a uniform interface.

    Examples
    --------
    >>> from explorica.reports.renderers import render_block_pdf
    >>> from explorica.reports.core import Block, BlockConfig
    >>> import matplotlib.pyplot as plt

    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> block_cfg = BlockConfig(
    ...     title="Example Block",
    ...     description="A simple block with one plot",
    ...     metrics=[{"name": "sum", "value": 15}],
    ...     visualizations=[fig]
    ... )
    >>> block = Block(block_cfg)

    >>> pdf_bytes = render_block_pdf(block)
    >>> type(pdf_bytes)
    <class 'bytes'>
    """
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        **(doc_template_kws or {}),
    )

    styles = getSampleStyleSheet()
    story = []

    # title
    story.append(Paragraph(block.block_config.title, styles["Heading2"]))
    story.append(Spacer(1, 12))

    # description
    if block.block_config.description:
        story.append(Paragraph(block.block_config.description, styles["BodyText"]))
        story.append(Spacer(1, 12))

    # metrics
    for metric in block.block_config.metrics or []:
        name = metric.get("name", "")
        value = metric.get("value", "")
        desc = metric.get("description", "")

        text = f"<b>{name}</b>: {value}"
        if desc:
            text += f"<br/><i>{desc}</i>"

        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 8))

    placeholder_plotly = get_empty_plot(
        message=(
            "Unsupported figure type for PDF."
            "Plotly figures are only supported in HTML render."
        ),
        engine="matplotlib",
    )[0]
    for vis_result in block.block_config.visualizations:
        img_buffer = BytesIO()
        if isinstance(vis_result.figure, matplotlib.figure.Figure):
            vis_result.figure.savefig(img_buffer, format="png", bbox_inches="tight")
        elif isinstance(vis_result.figure, plotly.graph_objects.Figure):
            placeholder_plotly.savefig(img_buffer, format="png", bbox_inches="tight")
        else:
            continue
        img_buffer.seek(0)

        story.append(Image(img_buffer, width=400, height=300))
        story.append(Spacer(1, 12))

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def _save_pdf(pdf_bytes: bytes, path: str | Path, report_name: str = "report"):
    """
    Save PDF bytes to the specified path.

    This is a low-level helper function for writing PDF content to disk.

    Parameters
    ----------
    pdf_bytes : bytes
        PDF content to save.
    path : str or Path
        Directory or full path where the PDF will be saved. If a directory is
        provided, the PDF will be saved as 'report.pdf' by default.
    report_name : str, default='report'
        Name to use for the PDF file if a directory is provided.

    Notes
    -----
    - This function does not generate PDF content; it only writes existing
      PDF bytes to disk.
    - Intended as a low-level helper for internal use. Public-facing functions
      like `render_pdf` handle PDF creation and call this utility.

    Examples
    --------
    >>> from io import BytesIO
    >>> pdf_bytes = b"%PDF-1.4\\n%..."  # Minimal PDF bytes example
    >>> _save_pdf(pdf_bytes, "output_dir")  # saves as 'output_dir/report.pdf'
    >>> _save_pdf(pdf_bytes, "output_file.pdf")  # saves as given path
    """
    path = Path(path)

    if path.suffix == "":
        path /= f"{report_name}.pdf"
    elif path.suffix != ".pdf":
        raise ValueError("'path' must be a directory or have .pdf extension")

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        f.write(pdf_bytes)


def render_html(data, path: str = None, report_name: str = "report"):
    """
    Render a Block or Report object into HTML format.

    This function generates an HTML representation of a single `Block` or
    a `Report`. Currently, only single `Block` instances are fully supported.
    For `Report` objects, HTML rendering is not implemented yet.

    Parameters
    ----------
    data : Block or Report
        The object to render. Should be an instance of `Block` for now.
    path : str, optional
        If provided, the HTML string will also be saved to the specified location.
        Can be either a directory (the file will be saved as "{report_name}.html")
        or a full file path ending with ".html".
    report_name : str, default='report'
        The base name to use when saving the HTML file if `path` is a directory.

    Returns
    -------
    str
        HTML content as a string.

    Raises
    ------
    TypeError
        If `data` is not an instance of `Block` or `Report`.
    NotImplementedError
        If `data` is a `Report` object.

    Notes
    -----
    - Plotly figures are fully supported in HTML output; interactive elements
      are preserved.
    - It is assumed that `Block().block_config.visualizations` contains a list
      of figures wrapped as `VisualizationResult` instances. During `Block`
      initialization, all figures are automatically normalized into
      `VisualizationResult` objects, so downstream rendering or processing
      functions can safely rely on a uniform interface.

    Examples
    --------
    >>> from explorica import Block, BlockConfig
    >>> from explorica.reports import render_html
    >>> import matplotlib.pyplot as plt

    # Minimal Block with Matplotlib visualization
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> block_config = BlockConfig(
    ...     title="Sample Block",
    ...     description="A minimal example block",
    ...     metrics=[{"name": "Metric 1", "value": 42}],
    ...     visualizations=[fig]
    ... )
    >>> block = Block(block_config)
    >>> html_output = render_html(block)
    >>> print(html_output[:100])  # Preview first 100 characters

    # Optionally save to disk
    >>> render_html(block, path="./reports", report_name="my_block")
    """
    if data.typename == "Block":
        report = render_block_html(data)
        if path is not None:
            _save_html(report, path, report_name=report_name)
        return report
    if data.typename == "Report":
        raise NotImplementedError("HTML rendering for Report is not implemented yet")
    raise TypeError("'data' must be Block or Report")


def render_block_html(block) -> str:
    """
    Render a single Block object into an HTML string.

    This function generates an HTML representation of a `Block` including
    its title, description, metrics, and visualizations. Both Matplotlib and
    Plotly figures are supported. Matplotlib figures are embedded as
    base64-encoded PNG images, while Plotly figures preserve interactivity.

    Parameters
    ----------
    block : Block
        The block to render.
    Returns
    -------
    str
        HTML content as a string.

    Notes
    -----
    - Plotly figures are rendered as interactive HTML components.
    - Matplotlib figures are rasterized as PNG images and embedded inline
      using base64 encoding.
    - It is assumed that `block.block_config.visualizations` contains a list
      of figures wrapped as `VisualizationResult` instances. During `Block`
      initialization, all figures are automatically normalized into
      `VisualizationResult` objects, so downstream rendering or processing
      functions can safely rely on a uniform interface.

    Examples
    --------
    >>> from explorica.core import Block, BlockConfig
    >>> from explorica.reports.renderers import render_block_html
    >>> import matplotlib.pyplot as plt
    >>> import plotly.graph_objects as go

    # Matplotlib example
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> block_config = BlockConfig(
    ...     title="Matplotlib Block",
    ...     description="Block with a Matplotlib figure",
    ...     metrics=[{"name": "Metric 1", "value": 42}],
    ...     visualizations=[fig]
    ... )
    >>> block = Block(block_config)
    >>> html_output = render_block_html(block)
    >>> print(html_output[:100])  # Preview first 100 characters

    # Plotly example
    >>> fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    >>> block_config = BlockConfig(
    ...     title="Plotly Block",
    ...     description="Block with a Plotly figure",
    ...     metrics=[],
    ...     visualizations=[fig]
    ... )
    >>> block = Block(block_config)
    >>> html_output = render_block_html(block)
    >>> print(html_output[:100])
    """
    html_parts = [f"<h2>{block.block_config.title}</h2>"]
    html_parts.append(f"<p>{block.block_config.description}</p>")
    # metrics
    if block.block_config.metrics:
        html_parts.append("<ul>")
        for metric in block.block_config.metrics:
            name = metric.get("name", "")
            value = metric.get("value", "")
            desc = metric.get("description", "")
            html_parts.append(f"<li><b>{name}</b>: {value} <i>{desc}</i></li>")
        html_parts.append("</ul>")

    for fig in block.block_config.visualizations:
        if isinstance(fig, plotly.graph_objects.Figure):
            html_parts.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))
        if isinstance(fig, matplotlib.figure.Figure):
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            html_parts.append(f'<img src="data:image/png;base64,{img_b64}"/>')
            buf.close()
    html_str = "\n".join(html_parts)
    return html_str


def _save_html(report, path, report_name="report"):
    """
    Save an HTML report to disk.

    This is a low-level helper function responsible for writing an HTML
    string to a file. It is primarily used internally by HTML renderers
    and is not intended to be a public entry point.

    Parameters
    ----------
    report : str
        HTML content to save.
    path : str or Path
        Target directory or full path where the HTML file will be written.
        - If a directory is provided, the file will be saved as
          ``f"{report_name}.html"``.
        - If a file path is provided, it must end with ``.html``.
    report_name : str, default='report'
        Base name to use for the output file when `path` is a directory.
        This parameter is typically used for naming reports consistently
        in logs, error messages.
    """
    path = Path(path)
    if path.suffix == "":
        path /= f"{report_name}.html"
    elif path.suffix != ".html":
        raise ValueError(
            "'path' must contain a directory (a folder without ext) or have .html ext"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
