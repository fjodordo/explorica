"""
PDF renderers for Explorica reports.

This module provides functionality for converting `Block` and `Report`
objects into PDF format using ReportLab as the underlying rendering engine.
It exposes both high-level APIs for end-to-end rendering and lower-level
utilities for working with individual blocks.

Functions
---------
render_pdf(data, path, font, doc_template_kws, **kwargs)
    Render a Block or Report into a PDF byte stream, optionally saving
    to disk and supporting customization of fonts and page layout.
render_block_pdf(block, mpl_fig_scale, plotly_fig_scale, reportlab_styles, block_name)
    Render a single Block into a sequence of ReportLab Flowables, including
    title, description, metrics, and visualizations.

See Also
--------
explorica.reports.core.Block
    Core block abstraction used as the primary rendering unit.
explorica.reports.utils.normalize_visualization
    Utility for standardizing visualization objects across backends.

Notes
-----
- It is assumed that ``Block.block_config.visualizations`` contains
  visualizations normalized into ``VisualizationResult`` objects during
  ``Block`` initialization. If this invariant is violated (e.g. by manual
  mutation or mocking), rendering behavior is undefined.
- High-level function `render_pdf` handles both single blocks and full reports.
- Visualizations that exceed the available page frame are automatically scaled
  to fit while preserving aspect ratio.
- Matplotlib figures are rendered directly; Plotly figures are replaced with
  placeholders to maintain layout consistency.
- `render_block_pdf` returns Flowables only; it does not build or save a PDF file.
- End users should typically use `render_pdf` unless they need fine-grained
  control over block-level Flowables.

Examples
--------
>>> from explorica.reports.renderers import render_pdf, render_block_pdf
>>> from explorica.reports.core import Block, BlockConfig
>>> import matplotlib.pyplot as plt

# Block-level rendering
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3], [4, 5, 6])
>>> block_cfg = BlockConfig(
...     title="Example Block",
...     description="A simple block with one plot",
...     metrics=[{"name": "sum", "value": 15}],
...     visualizations=[fig]
... )
>>> block = Block(block_cfg)
>>> flowables = render_block_pdf(block)
>>> isinstance(flowables, list)
True

# End-to-end PDF rendering
>>> pdf_bytes = render_pdf(block)
>>> len(pdf_bytes) > 0
True

# Rendering a full report
>>> blocks = [block, block, block]
>>> report = Report(blocks, title="Example Report", description="Report description")
>>> report_bytes = render_pdf(report, path="./reports")
"""

from pathlib import Path
from contextlib import nullcontext
from io import BytesIO
import logging

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Flowable,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors


from explorica._utils import (
    temp_log_level,
    read_config,
    enable_io_logs,
    validate_path,
    convert_filepath,
)
from explorica.visualizations._utils import get_empty_plot
from ...types import VisualizationResult, TableResult
from ..utils import normalize_visualization

logger = logging.getLogger(__name__)

ERR_MSG_UNSUPPORTED_STRING_FLAG_F = read_config("messages")["errors"][
    "unsupported_method_f"
]


TTF_SANS = (
    Path(__file__).absolute().parent.parent.parent / "assets/fonts/DejaVuSans.ttf"
)
TTF_SERIF = (
    Path(__file__).absolute().parent.parent.parent / "assets/fonts/DejaVuSerif.ttf"
)

pdfmetrics.registerFont(TTFont("DejaVuSans", TTF_SANS))
pdfmetrics.registerFont(TTFont("DejaVuSerif", TTF_SERIF))


def render_pdf(
    data,
    path: str = None,
    font: str = "DejaVuSans",
    doc_template_kws: dict = None,
    **kwargs,
) -> bytes:
    """
    Render a Block or Report into PDF format.

    This function renders a single `Block` or a `Report` (sequence of blocks)
    into a PDF format. It serves as the main entry point for PDF export
    functionality in the reports module.

    Parameters
    ----------
    data : Block or Report
        The object to render. Can be a single `Block` or a `Report`
        (sequence of blocks).
    path : str, optional
        Directory path or full file path where the PDF report should be saved.
        - If a directory is provided, the report is saved as
        ``f"{report_name}.pdf"`` inside that directory.
        - If a full file path ending with ``.pdf`` is provided, the report
        is saved to that exact location.
        If None, the rendered PDF is returned as bytes without saving.
    font : str, default="DejaVuSans"
        Font to use for rendering text in the PDF. Supports the following:
        - `"DejaVuSans"` (default)
        - `"DejaVuSerif"`
        - path to a custom TTF font file, which will be registered as
          `UserProvidedFont`.
    doc_template_kws : dict, optional
        Additional keyword arguments to pass to `SimpleDocTemplate` for
        customizing PDF layout (e.g., margins, page size).

    Other parameters
    ----------------
    report_name : str, default="report"
        The base name used when saving the PDF if `path` is a directory.
        Also serves as a placeholder name in logs or error messages.
    verbose : bool, default=False
        Enables info-level logging during the function execution.
    debug : bool, default=False
        Enables debug-level logging during the function execution.
        Takes precedence over the 'verbose' parameter.
    mpl_fig_scale : float, default=80.0
        Scaling factor applied to Matplotlib figure dimensions when converting
        to PDF. Accounts for figure size being specified in inches.
    plotly_fig_scale : float, default=1.0
        Scaling factor applied to Plotly figure placeholders in the PDF.

    Returns
    -------
    bytes
        PDF content as bytes.

    Raises
    ------
    TypeError
        If `data` is not a Block or Report.
    ValueError
        If `font` is not a recognized alias and the specified file path does not exist.

    Notes
    -----
    - This function provides a unified interface for PDF generation from both
      individual blocks and eventually from full report sequences.
    - The function always returns the generated PDF content as bytes,
      even if `path` is provided for saving.
    - Users can optionally save the generated PDF by providing the `path`
      parameter.
    - It is assumed that `Block().block_config.visualizations` contains a list
      of figures wrapped as `VisualizationResult` instances. During `Block`
      initialization, all figures are automatically normalized into
      `VisualizationResult` objects, so downstream rendering or processing
      functions can safely rely on a uniform interface.
    - Visualizations are rendered into the PDF within the available page frame.
      If a visualization exceeds the frame width (taking page size and margins
      into account), it will be automatically scaled down to fit the frame while
      preserving its aspect ratio.
    - This behavior applies to all figure-based visualizations (e.g. Matplotlib
      or Plotly figures) and ensures that content never overflows page boundaries.
      As a result, the final rendered size of a visualization in the PDF may differ
      from its original size.

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

    # Report render usage
    >>> blocks = [Block(cfg1), Block(cfg2), Block(cfg3)]
    >>> report = Report(blocks, title = "Example Report",
    ...                 description = "Description to example report")
    >>> report_bytes = render_pdf(report)

    >>> # Save report to ./reports/report.pdf
    >>> render_pdf(report, path = "./reports")

    # Usage with 'doc_template_kws'
    >>> from reportlab.lib.pagesizes import A3
    >>> report_bytes = render_pdf(report,
    ...     doc_template_kws = {
    ...         "pagesize": A3,
    ...         "rightMargin": 35,
    ...         "leftMargin": 35,
    ...         "topMargin": 40,
    ...         "bottomMargin": 0,
    ...     }
    """
    params = {
        "plotly_fig_scale": kwargs.get("plotly_fig_scale", 1),
        "mpl_fig_scale": kwargs.get("mpl_fig_scale", 80),
        "report_name": kwargs.get("report_name", "report"),
        "verbose": kwargs.get("verbose", False),
        "debug": kwargs.get("debug", False),
    }
    if not (hasattr(data, "typename") and data.typename in {"Block", "Report"}):
        raise TypeError(f"Expected Block or Report, got {type(data).__name__}")
    # Optionally enable temp log context
    if params["debug"]:
        context = temp_log_level(logger, logging.DEBUG)
    elif params["verbose"]:
        context = temp_log_level(logger, logging.INFO)
    else:
        context = nullcontext()
    with context:
        logger.info("Rendering '%s' in pdf", params["report_name"])
        font, reportlab_styles = _preprocess_font(font)
        # Render pipeline for a single Block
        if data.typename == "Block":
            block_base_name = f"{params['report_name']}_block"
            story = render_block_pdf(
                data,
                mpl_fig_scale=params["mpl_fig_scale"],
                plotly_fig_scale=params["plotly_fig_scale"],
                reportlab_styles=reportlab_styles,
                block_name=block_base_name,
            )
        # Render pipeline for Report with multiple Blocks
        else:
            story = []
            # Make a header
            if data.title:
                story.append(Paragraph(data.title, reportlab_styles["Heading1"]))
                story.append(Spacer(1, 16))
            if data.description:
                story.append(Paragraph(data.description, reportlab_styles["BodyText"]))
                story.append(Spacer(1, 24))
            # Make a report body
            for i, block in enumerate(data.blocks):
                block_base_name = f"{params['report_name']}_block{i}"
                story.extend(
                    render_block_pdf(
                        block,
                        mpl_fig_scale=params["mpl_fig_scale"],
                        plotly_fig_scale=params["plotly_fig_scale"],
                        reportlab_styles=reportlab_styles,
                        block_name=block_base_name,
                    )
                )
                logger.debug("Block '%s' was successfully rendered", block_base_name)
            if len(story) == 0:
                story.append(
                    _get_placeholder_pdf(
                        "pdf",
                        reportlab_styles=reportlab_styles,
                    )
                )
                logger.warning(
                    "Report '%s' has no blocks; inserting placeholder",
                    params["report_name"],
                )
        pdf_bytes = _get_build_pdf(story, doc_template_kws=doc_template_kws)
        if path is not None:
            _save_pdf(pdf_bytes, path, report_name=params["report_name"])
        logger.info("'%s' was successfully rendered", params["report_name"])
        return pdf_bytes


def _get_build_pdf(story: list, doc_template_kws: dict = None) -> bytes:
    """
    Build a PDF document from a prepared ReportLab story.

    This helper function constructs a PDF using ReportLab's
    ``SimpleDocTemplate`` and returns the rendered document as raw bytes.
    It is intended for internal use by the PDF rendering orchestrator.

    Parameters
    ----------
    story : list
        A list of ReportLab flowables (e.g., ``Paragraph``, ``Image``,
        ``Spacer``) representing the fully prepared document content.
    doc_template_kws : dict, optional
        Additional keyword arguments passed directly to
        ``reportlab.platypus.SimpleDocTemplate``. These can be used to
        override layout parameters such as page size or margins.

    Returns
    -------
    bytes
        The rendered PDF document as a byte sequence.

    Notes
    -----
    - A temporary in-memory buffer (``BytesIO``) is used to build the PDF.
    - Default layout parameters (A4 page size and standard margins) are
      applied unless explicitly overridden via ``doc_template_kws``.
    - This function does not perform any file I/O; saving the resulting
      bytes is handled by higher-level orchestration logic.
    """
    buffer = BytesIO()
    try:
        kwargs = {
            "filename": buffer,
            "pagesize": A4,
            "rightMargin": 40,
            "leftMargin": 40,
            "topMargin": 40,
            "bottomMargin": 40,
            **(doc_template_kws or {}),
        }
        doc = SimpleDocTemplate(**kwargs)
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        return pdf_bytes
    finally:
        buffer.close()


def _preprocess_font(font: str):
    """
    Validate and register a font for ReportLab PDF rendering.

    Parameters
    ----------
    font : str
        Font alias or path to a TTF font file. Supported aliases:
        - "DejaVuSans"
        - "DejaVuSerif"
        If a path is provided, the font is registered as "UserProvidedFont".

    Returns
    -------
    tuple
        - font_name : str — the font name to use in ReportLab styles
        - reportlab_styles : dict — a dictionary of ReportLab paragraph styles
          updated to use the specified font

    Raises
    ------
    ValueError
        If `font` is not a recognized alias and the specified file path does not exist.

    Notes
    -----
    - Updates standard ReportLab styles: Normal, BodyText, Heading1–3
    """
    reportlab_styles = getSampleStyleSheet()
    if font not in {"DejaVuSans", "DejaVuSerif"}:
        if not Path(font).exists():
            raise ValueError(
                f"There is no font in the specified directory: {font}."
                f"Please, provide an existing path to font or alias"
                "('DejaVuSans', 'DejaVuSerif')"
            )
        pdfmetrics.registerFont(TTFont("UserProvidedFont", Path(font)))
        logger.info("Using user-provided font from path: %s", font)
        font = "UserProvidedFont"
    reportlab_styles["Normal"].fontName = font
    reportlab_styles["BodyText"].fontName = font
    reportlab_styles["Heading1"].fontName = font
    reportlab_styles["Heading2"].fontName = font
    reportlab_styles["Heading3"].fontName = font
    reportlab_styles["Heading4"].fontName = font
    reportlab_styles["Italic"].fontName = font
    return font, reportlab_styles


def render_block_pdf(
    block,
    mpl_fig_scale: float = 80.0,
    plotly_fig_scale: float = 1.0,
    reportlab_styles=None,
    block_name: str = "block",
) -> list[Flowable]:
    """
    Render a single Block object into ReportLab Flowables for PDF generation.

    This function converts a `Block` into a sequence of ReportLab `Flowable`
    objects, representing the block's content including title, description,
    metrics, and visualizations.

    Parameters
    ----------
    block : Block
        The block to render.
    mpl_fig_scale : float, default=80.0
        Scaling factor applied to Matplotlib figure dimensions when converting
        to PDF. Accounts for figure size being specified in inches.
    plotly_fig_scale : float, default=1.0
        Scaling factor applied to Plotly figure placeholders in the PDF.
    reportlab_styles : dict or None, optional
        Predefined ReportLab styles to use for rendering. If None, default
        styles from `getSampleStyleSheet()` are used.
    block_name : str, default='block'
        Name of the block used in logging and error messages. It is not used
        elsewhere in the rendering.

    Returns
    -------
    list[reportlab.platypus.Flowable]
        Sequence of Platypus flowable elements representing the rendered block.

    Notes
    -----
    - Matplotlib figures are saved directly into the PDF.
    - Plotly figures are replaced by a placeholder using `get_empty_plot`.
    - This is a lower-level utility function. For end users, the higher-level
      `render_pdf` function is the recommended entry point.
    - It is assumed that `block.block_config.visualizations` contains a list
      of figures wrapped as `VisualizationResult` instances. During `Block`
      initialization, all figures are automatically normalized into
      `VisualizationResult` objects, so downstream rendering or processing
      functions can safely rely on a uniform interface.
    - This function does not build or save a PDF file.
    - It returns a list of ReportLab `Flowable` objects intended to be consumed
      by higher-level rendering utilities such as `render_pdf`

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
    >>> flowables = render_block_pdf(block)
    >>> type(flowables)
    <class 'list'>
    """
    if reportlab_styles is None:
        reportlab_styles = getSampleStyleSheet()
    story = []

    # title
    if block.block_config.title is not None:
        story.append(Paragraph(block.block_config.title, reportlab_styles["Heading2"]))
        story.append(Spacer(1, 12))

    # description
    if block.block_config.description:
        story.append(
            Paragraph(block.block_config.description, reportlab_styles["BodyText"])
        )
        story.append(Spacer(1, 12))

    # metrics
    log_counter = 0
    for metric in block.block_config.metrics or []:
        name = metric.get("name", "")
        value = metric.get("value", "")
        desc = metric.get("description", "")

        text = f"<b>{name}</b>: {value}"
        if desc:
            text += f"<br/><i>{desc}</i>"
        story.append(Paragraph(text, reportlab_styles["BodyText"]))
        story.append(Spacer(1, 8))
        log_counter += 1

    if log_counter > 0:
        logger.debug("Added %d metrics to '%s'.", log_counter, block_name)

    # tables
    story.extend(
        _render_block_pdf_build_tables(
            block.block_config.tables,
            reportlab_styles=reportlab_styles,
            block_name=block_name,
        )
    )

    # visualizations
    story.extend(
        _render_block_pdf_build_visualizations(
            block.block_config.visualizations,
            mpl_fig_scale,
            plotly_fig_scale,
            block_name=block_name,
        )
    )
    return story


def _render_block_pdf_build_tables(
    tables: list[TableResult],
    reportlab_styles=None,
    block_name: str = "block",
) -> list[Flowable]:
    """
    Convert TableResult objects into ReportLab Flowables.

    Parameters
    ----------
    tables : list[TableResult]
        List of tables to render.
    reportlab_styles : dict or None, optional
        Predefined ReportLab styles to use for rendering. If None, default
        styles from `getSampleStyleSheet()` are used.
    block_name : str, default='block'
        Block identifier used for logging.

    Returns
    -------
    list[Flowable]
        List of Platypus Flowables representing tables.

    Examples
    --------
    >>> import pandas as pd
    >>> from reportlab.platypus import SimpleDocTemplate
    >>> df = pd.DataFrame(
    ...     {"age": [41.0, 42.0], "income": [78000.0, 82000.0]},
    ...     index=["mean", "median"],
    ... )
    >>> tr = TableResult(
    ...     table=df,
    ...     title="Central Tendency",
    ...     description="Summary statistics for numeric features.",
    ...     render_extra={"show_index": True, "show_columns": True},
    ... )
    >>> story = _render_block_pdf_build_tables([tr])
    >>> doc = SimpleDocTemplate("example.pdf")
    >>> doc.build(story)
    """
    story: list[Flowable] = []

    if reportlab_styles is None:
        reportlab_styles = getSampleStyleSheet()
    log_counter = 0

    for tr in tables:
        # --- title ---
        if tr.title:
            story.append(Paragraph(tr.title, reportlab_styles["Heading4"]))
            story.append(Spacer(1, 6))

        # --- description ---
        if tr.description:
            story.append(Paragraph(tr.description, reportlab_styles["Italic"]))
            story.append(Spacer(1, 6))

        df = tr.table
        show_index = (tr.render_extra or {}).get("show_index", True)
        show_columns = (tr.render_extra or {}).get("show_columns", True)

        table_data = []

        # --- header row ---
        if show_columns:
            indeces = []
            if show_index:
                indeces.append("")  # top-left empty cell
            indeces.extend(map(str, df.columns))
            table_data.append(indeces)

        # --- body ---
        for idx, row in df.iterrows():
            indeces = []
            if show_index:
                indeces.append(str(idx))
            indeces.extend(map(str, row.values))
            table_data.append(indeces)

        rl_table = Table(table_data, repeatRows=(1 if show_columns else 0))
        styles = [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", ((1 if show_index else 0), 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
        if show_columns:
            styles.append(("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke))
            styles.append(
                ("TOPPADDING", (0, 0), (-1, 0), 6),
            )
            styles.append(("BOTTOMPADDING", (0, 0), (-1, 0), 6))
        rl_table.setStyle(TableStyle(styles))

        story.append(rl_table)
        story.append(Spacer(1, 12))
        log_counter += 1

    if log_counter:
        logger.debug("Added %d tables to '%s'", log_counter, block_name)

    return story


def _render_block_pdf_build_visualizations(
    visualizations: list[VisualizationResult],
    mpl_fig_scale: float,
    plotly_fig_scale: float,
    block_name: str = "block",
) -> list[Flowable]:
    """
    Convert a list of VisualizationResult objects into ReportLab Flowables.

    This is an internal helper function used by `render_block_pdf` to process
    visualizations attached to a Block. It handles Matplotlib figures directly
    and replaces Plotly figures with a placeholder.

    Parameters
    ----------
    visualizations : list[VisualizationResult]
        List of visualization results to render.
    mpl_fig_scale : float
        Scaling factor applied to Matplotlib figures.
    plotly_fig_scale : float
        Scaling factor applied to Plotly figure placeholders.
    block_name : str, default='block'
        Name of the block for logging purposes. Used in debug logs
        to identify which block the visualizations belong to.

    Returns
    -------
    list[Flowable]
        A list of ReportLab Flowable objects representing the rendered figures.

    Notes
    -----
    - Matplotlib figures are added as images to the Flowables.
    - Plotly figures are substituted with a Matplotlib placeholder since
      PDF rendering does not support Plotly directly.
    - If a Plotly visualization is dimensionless (i.e. `width` or `height`
      is not defined), the placeholder size is derived from the placeholder
      figure's intrinsic size in inches and multiplied by `mpl_fig_scale`.
    - The function logs debug information for each figure added.

    Raises
    ------
    ValueError
        If a VisualizationResult contains an unsupported engine type.

    Examples
    --------
    >>> vis_results = [VisualizationResult(
    ...     engine="matplotlib", figure=fig, width=6, height=4)]
    >>> flowables = _render_block_pdf_build_visualizations(
    ...     vis_results, 100, 1.0, block_name="TestBlock")
    """
    story = []
    log_counter = 0
    placeholder_plotly = normalize_visualization(
        get_empty_plot(
            message=(
                "Unsupported figure type for PDF."
                "Plotly figures are only supported in HTML render."
            ),
            engine="matplotlib",
        )[0]
    )

    for vis_result in visualizations:
        img_buffer = BytesIO()
        if vis_result.engine == "matplotlib":
            vis_result.figure.savefig(img_buffer, format="png", bbox_inches="tight")
            w, h = vis_result.width * mpl_fig_scale, vis_result.height * mpl_fig_scale

        elif vis_result.engine == "plotly":
            placeholder_plotly.figure.savefig(
                img_buffer, format="png", bbox_inches="tight"
            )
            if vis_result.width is not None and vis_result.height is not None:
                w, h = (
                    vis_result.width * plotly_fig_scale,
                    vis_result.height * plotly_fig_scale,
                )
            else:
                w, h = (
                    placeholder_plotly.width * mpl_fig_scale,
                    placeholder_plotly.height * mpl_fig_scale,
                )
        else:
            raise ValueError(
                f"Unsupported engine '{vis_result.engine}'"
                "in provided Block's visualizations."
            )
        img_buffer.seek(0)
        story.append(Image(img_buffer, width=w, height=h, kind="proportional"))
        logger.debug(
            "%s figure added with size %.2f x %.2f",
            vis_result.engine.capitalize(),
            w,
            h,
        )
        story.append(Spacer(1, 12))
        log_counter += 1
    if log_counter > 0:
        logger.debug("Added %d visualizations to '%s'", log_counter, block_name)
        logger.debug(log_counter)
    return story


@enable_io_logs(logger)
def _save_pdf(
    pdf_bytes: bytes,
    path: str | Path,
    overwrite: bool = True,
    report_name: str = "report",
):
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
    overwrite : bool, default=True
        Controls whether existing files can be overwritten.
        - If True (default), existing files at the target path will be
          silently overwritten.
        - If False, a `FileExistsError` is raised when attempting to save to a path
          that already exists, preventing accidental overwriting.

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
    path = convert_filepath(path, f"{report_name}.pdf")
    need_overwrite_check = not overwrite

    validate_path(path, overwrite_check=need_overwrite_check)

    if path.suffix != ".pdf":
        raise ValueError("'path' must be a directory or have .pdf extension")

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        f.write(pdf_bytes)


def _get_placeholder_pdf(
    message: str = ("There are no render blocks in this report."), reportlab_styles=None
):
    if reportlab_styles is None:
        reportlab_styles = getSampleStyleSheet()
    return Paragraph(message, reportlab_styles["Heading2"])
