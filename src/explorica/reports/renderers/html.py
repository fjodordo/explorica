"""
HTML renderers for Explorica reports and blocks.

This module provides utilities to render `Block` and `Report` objects into
HTML format. It defines the public rendering entry points used to generate
standalone HTML documents or reusable HTML fragments, including optional
CSS styling and figure scaling.

The module supports mixed visualization backends (e.g. Matplotlib and
Plotly) and handles their normalization and embedding into HTML in a
backend-aware manner.


Functions
---------
render_html(data, path, font, **kwargs)
    Render a `Block` or `Report` object into an HTML document. Acts as the
    main public entry point for HTML rendering and optionally saves the
    result to disk.
render_block_html(block, add_css_style, font, **kwargs)
    Render a single `Block` into an HTML fragment. This function is primarily
    used internally by `render_html`, but may also be useful when embedding
    individual blocks into external HTML layouts.

Notes
-----
- When rendering a single `Block`, CSS styles are injected locally and apply
  only to that block.
- When rendering a `Report`, CSS styles are injected once at the report level
  and shared across all contained blocks.
- Internal helper functions (e.g. CSS generation, placeholder rendering)
  are not part of the public API and may change without notice.
- Font handling relies on CSS `font-family` rules and assumes that the
  specified fonts are available in the target rendering environment.


Examples
--------
# Render a single block to HTML:
>>> from explorica.reports import Block, BlockConfig
>>> from explorica.reports.renderers.html import render_html
>>> import matplotlib.pyplot as plt
>>>
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3], [4, 5, 6])
>>>
>>> block = Block(BlockConfig(
...     title="Example Block",
...     visualizations=[fig]
... ))
>>>
>>> html = render_html(block)

# Render a report with multiple blocks:
>>> from explorica.reports import Report
>>>
>>> report = Report(
...     blocks=[block],
...     title="Example Report",
...     description="Demonstration of HTML rendering"
... )
>>>
>>> html = render_html(report)
"""

from contextlib import nullcontext
from typing import Sequence
from pathlib import Path
from io import BytesIO
import base64
import logging
from copy import deepcopy

import plotly.io as pio

from explorica._utils import (
    read_config,
    enable_io_logs,
    validate_path,
    convert_filepath,
    temp_log_level,
)

logger = logging.getLogger(__name__)

ERR_MSG_UNSUPPORTED_STRING_FLAG_F = read_config("messages")["errors"][
    "unsupported_method_f"
]


def render_html(
    data,
    path: str = None,
    font: str | list[str] = ("Arial", "DejaVu Serif", "DejaVu Sans", "sans-serif"),
    **kwargs,
):
    """
    Render a Block or Report object into HTML format.

    This function generates an HTML representation of either a single `Block`
    or a `Report` containing multiple blocks.

    - When rendering a `Block`, the output HTML is fully self-contained and
    includes a CSS `<style>` block scoped to that block only.
    - When rendering a `Report`, all contained blocks are rendered sequentially,
    wrapped in a single report-level container, and a single shared CSS
    `<style>` block is applied to the entire report.

    If a `Report` contains no blocks, a placeholder message is inserted into
    the report body.

    Parameters
    ----------
    data : Block or Report
        The object to render. Can be either a single `Block` instance or a
        `Report` containing multiple blocks.
    path : str, optional
        If provided, the rendered HTML will also be saved to this location.
        Can be either:
        - a directory path (the file will be saved as "{report_name}.html"), or
        - a full file path ending with ".html".
    font : str or Sequence[str], optional
        Font family name(s) to be used in the CSS ``font-family`` property.

        This argument represents system font family names as understood by
        the browser (e.g. "Arial", "DejaVu Sans", "sans-serif").

        It can be provided either as a single font family name or as a sequence
        of names defining a fallback order. When a sequence is given, it is
        normalized into a single CSS ``font-family`` declaration.

        By default, a common cross-platform fallback sequence is used:
        ``("Arial", "DejaVu Serif", "DejaVu Sans", "sans-serif")``.

        Font availability and final selection are handled entirely by the browser.

    Other parameters
    ----------------
    overwrite : bool, default=True
        Controls whether an existing file at the target path can be overwritten.
        If False and the target file already exists, a `FileExistsError` is raised.
    max_width : int, default=800
        Maximum width of the outer container in pixels. Applied via the
        `max-width` CSS property.
    mpl_fig_scale : float, default=80.0
        Scaling factor for Matplotlib figures. The figure size in inches
        (`figure.width` and `figure.height`) is multiplied by this factor
        to determine pixel dimensions for the embedded PNG.
    plotly_fig_scale : float, default=1.0
        Scaling factor for Plotly figures. The figure size in pixels
        (`figure.layout.width` and `figure.layout.height`) is multiplied
        by this factor **only if both width and height are explicitly defined**.
        Figures with undefined dimensions are included without scaling.
    verbose : bool, default=False
        Enables info-level logging during the function execution.
    debug : bool, default=False
        Enables debug-level logging during the function execution.
        Takes precedence over the 'verbose' parameter.
    vis_container_class : str, default='visualizations'
        CSS class name for the div wrapping the visualizations. This
        allows external CSS or higher-level wrappers to style all
        visualizations consistently.
    tables_container_class : str, default='explorica-tables'
        CSS class name applied to the <div> wrapping all tables in a block
        or report. This allows users to apply consistent styling to tables,
        for example scrollable containers, max-height constraints, or
        custom CSS rules for all tables. Passed down to internal block
        rendering functions.
    report_name : str, default='report'
        Base name used:
        - as the CSS class for the report container, and
        - as the output file name when `path` is a directory.
        - as an identifier in log messages emitted during rendering.

    Returns
    -------
    str
        HTML content as a string.

    Raises
    ------
    TypeError
        If `data` is not an instance of `Block` or `Report`.

    Notes
    -----
    - Plotly figures are fully supported in HTML output; interactive elements
      are preserved.
    - It is assumed that `Block().block_config.visualizations` contains a list
      of figures wrapped as `VisualizationResult` instances. During `Block`
      initialization, all figures are automatically normalized into
      `VisualizationResult` objects, so downstream rendering or processing
      functions can safely rely on a uniform interface.
    - CSS injection behavior depends on the input type:
        - For `Block`, CSS is injected and scoped to that block only.
        - For `Report`, a single CSS block is injected and shared by all blocks
          within the report container.

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
    # Single block
    >>> html_output = render_html(block)
    >>> print(html_output[:100])

    # Report with multiple blocks
    >>> report = Report(blocks=[block], title="My Report", description="Example report")
    >>> html_output = render_html(report)
    >>> print(html_output[:100])

    # Optionally save to disk
    >>> render_html(block, path="./reports", report_name="my_block")
    """
    if not (hasattr(data, "typename") and data.typename in {"Block", "Report"}):
        raise TypeError(f"Expected Block or Report, got {type(data).__name__}")
    params = {
        "max_width": kwargs.get("max_width", 800),
        "report_name": kwargs.get("report_name", "report"),
        "vis_container_class": kwargs.get(
            "vis_container_class", "explorica-visualizations"
        ),
        "tables_container_class": kwargs.get(
            "tables_container_class", "explorica-tables"
        ),
        "mpl_fig_scale": kwargs.get("mpl_fig_scale", 100),
        "plotly_fig_scale": kwargs.get("plotly_fig_scale", 1),
        "debug": kwargs.get("debug", False),
        "verbose": kwargs.get("verbose", False),
    }
    font_family = _normalize_font_family(font)

    # Optionally enable temp log context
    if params["debug"]:
        context = temp_log_level(logger, level=logging.DEBUG)
    elif params["verbose"]:
        context = temp_log_level(logger, level=logging.INFO)
    else:
        context = nullcontext()
    with context:
        logger.info("Rendering '%s' in html", params["report_name"])
        # Render pipeline for a single Block
        if data.typename == "Block":
            block_base_name = f"explorica-{params['report_name']}_block"
            report = render_block_html(
                data,
                add_css_style=True,
                font=font_family,
                block_name=block_base_name,
                max_width=params["max_width"],
                vis_container_class=params["vis_container_class"],
                tables_container_class=params["tables_container_class"],
                mpl_fig_scale=params["mpl_fig_scale"],
                plotly_fig_scale=params["plotly_fig_scale"],
            )
        # Render pipeline for Report with multiple Blocks
        else:
            rendered_blocks = []
            # Make a report body
            for i, block in enumerate(data.blocks):
                block_base_name = f"explorica-{params['report_name']}_block-num{i}"
                rendered_blocks.append(
                    render_block_html(
                        block,
                        add_css_style=False,
                        font=font_family,
                        block_name=block_base_name,
                        max_width=params["max_width"],
                        vis_container_class=params["vis_container_class"],
                        tables_container_class=params["tables_container_class"],
                        mpl_fig_scale=params["mpl_fig_scale"],
                        plotly_fig_scale=params["plotly_fig_scale"],
                    )
                )
                logger.debug("Block '%s' was successfully rendered", block_base_name)
            if not rendered_blocks:
                rendered_blocks = [_get_placeholder_html()]
                logger.warning(
                    "Report '%s' has no blocks; inserting placeholder",
                    params["report_name"],
                )
            # Make a header
            header_html = ""
            if data.title:
                header_html += f"<h1>{data.title}</h1>\n"
            if data.description:
                header_html += f"<p>{data.description}</p>\n"
            html_parts = [header_html] + rendered_blocks
            html_parts.insert(0, f"<div class='{params['report_name']}'>")
            html_parts.append("</div>")
            html_parts.insert(
                0,
                _get_css_style(
                    report_class_name=params["report_name"],
                    vis_class_name=params["vis_container_class"],
                    tables_class_name=params["tables_container_class"],
                    max_width=params["max_width"],
                    font_family=font_family,
                ),
            )
            report = "\n".join(html_parts)
        if path is not None:
            _save_html(
                report,
                path,
                overwrite=kwargs.get("overwrite", True),
                report_name=params["report_name"],
            )
        logger.info("'%s' was successfully rendered", params["report_name"])
        return report


def render_block_html(
    block,
    add_css_style: bool = False,
    font: str | Sequence[str] = ("Arial", "DejaVu Serif", "DejaVu Sans", "sans-serif"),
    **kwargs,
) -> str:
    """
    Render a single Block object into an HTML string.

    This function generates a full HTML representation of a `Block`,
    including its title, description, metrics, and visualizations.
    Matplotlib figures are embedded as base64-encoded PNG images,
    while Plotly figures are rendered as interactive HTML components.
    Optional CSS can be applied to style the block and its visualizations.

    Parameters
    ----------
    block : Block
        The Block object to render. Its `block_config.visualizations`
        should contain a list of `VisualizationResult` instances.
    add_css_style : bool, optional
        If True, embed default CSS styling for the block and its visualizations.
        Defaults to False.
    font : str or Sequence[str], optional
        Font family or sequence of font families to be applied to textual
        elements (title, description, metrics) in the block. If a sequence
        is provided, the first available font installed on the system will
        be used. For reliable rendering, it is recommended that the specified
        fonts are pre-installed on the system where the HTML will be viewed.
        The final CSS `font-family` property is constructed from this argument.

    Other parameters
    ----------------
    max_width : int, optional, default=800
        Maximum width of the outer block in pixels. This value is applied
        via the `max-width` CSS property of the block container **only if
        `add_css_style=True`**. When disabled, this parameter has no effect
        on layout. Useful to constrain content width when rendering multiple
        blocks on the same page.
    block_name : str, optional, default='block'
        CSS class name for the outer wrapper div of the block. Used both
        for styling and for logging purposes to identify the block in logs
        or debug output.
    vis_container_class : str, optional, default='explorica-visualizations'
        CSS class name for the div wrapping the visualizations. This
        allows external CSS or higher-level wrappers to style all
        visualizations consistently.
    tables_container_class : str, optional, default='explorica-tables'
        CSS class name for the div wrapping all tables in the block. This
        allows external CSS to style all tables consistently. It is passed
        to the internal `_render_block_html_build_tables` function.
    mpl_fig_scale : float, optional, default=80.0
        Scaling factor for Matplotlib figures. The figure size in inches
        (`figure.width` and `figure.height`) is multiplied by this factor
        to determine pixel dimensions for the embedded PNG.
    plotly_fig_scale : float, optional, default=1.0
        Scaling factor for Plotly figures. The figure size in pixels
        (`figure.layout.width` and `figure.layout.height`) is multiplied
        by this factor **only if both width and height are explicitly defined**.
        Figures with undefined dimensions are included without scaling.

    Returns
    -------
    str
        HTML content as a string representing the rendered block.

    Notes
    -----
    - Plotly figures retain interactivity; scaling is applied only if width and
      height are explicitly set.
    - Matplotlib figures are rasterized as PNG images and embedded inline.
    - The visualization div uses the `vis_container_class` class, allowing external
      styling when multiple blocks are rendered together.
    - Images and iframes are styled with `object-fit: contain` to maintain
      aspect ratio while scaling.

    Examples
    --------
    >>> from explorica.core import Block, BlockConfig
    >>> import matplotlib.pyplot as plt
    >>> import plotly.graph_objects as go
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> plotly_fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    >>> block_config = BlockConfig(
    ...     title="Example Block",
    ...     description="Block with visualizations",
    ...     metrics=[{"name": "Metric A", "value": 42}],
    ...     visualizations=[fig, plotly_fig]
    ... )
    >>> block = Block(block_config)
    >>> html_output = render_block_html(
    ...     block, add_css_style=True, font=["Arial", "sans-serif"])
    >>> print(html_output[:200])  # Preview first 200 characters
    """
    params = {
        "max_width": 800,
        "block_name": "explorica-block",
        "vis_container_class": "explorica-visualizations",
        "tables_container_class": "explorica-tables",
        "mpl_fig_scale": 80.0,
        "plotly_fig_scale": 1.0,
        **kwargs,
    }
    html_parts = []
    if block.block_config.title is not None:
        html_parts.append(f"<h2>{block.block_config.title}</h2>")
    if block.block_config.description is not None:
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

    # tables
    html_parts.extend(
        _render_block_html_build_tables(
            block, container_class=params["tables_container_class"]
        )
    )

    # visualizations
    html_parts.extend(
        _render_block_html_build_visualizations(
            block,
            mpl_fig_scale=params["mpl_fig_scale"],
            plotly_fig_scale=params["plotly_fig_scale"],
            name_css=params["vis_container_class"],
        )
    )

    # wrap and design CSS class
    html_parts.insert(0, f"<div class='{params['block_name']}'>")
    html_parts.append("</div>")

    if add_css_style:
        font_family = _normalize_font_family(font)
        html_parts.insert(
            0,
            _get_css_style(
                report_class_name=params["block_name"],
                vis_class_name=params["vis_container_class"],
                tables_class_name=params["tables_container_class"],
                max_width=params["max_width"],
                font_family=font_family,
            ),
        )

    html_str = "\n".join(html_parts)
    return html_str


def _render_block_html_build_tables(
    block, container_class: str = "explorica-tables"
) -> list[str]:
    """
    Generate HTML fragments for all tables in a report block.

    This function iterates over `TableResult` objects in `block.block_config.tables`
    and converts each table into an HTML string, optionally including its title and
    description. The resulting HTML fragments can then be inserted into the block's
    overall HTML structure.

    Parameters
    ----------
    block : Block
        The report block containing tables to render.
    container_class : str, default='explorica-tables'
        CSS class to apply to the outer <div> wrapping all tables

    Returns
    -------
    list[str]
        A list of HTML string fragments representing the tables, titles,
        and descriptions. Can be concatenated to form the full block content.

    Notes
    -----
    - Each table is rendered using Pandas' `DataFrame.to_html()` method.
    - `TableResult.render_extra` can contain optional rendering settings:
        - `show_index` : bool, default True — whether to display the index column.
        - `show_columns` : bool, default True — whether to display column headers.
    - Titles and descriptions of tables are included as <h4> and <i> elements.

    Examples
    --------
    >>> html_fragments = _render_block_html_build_tables(
    ...     block, container_class="my-tables")
    >>> for frag in html_fragments:
    ...     print(frag)  # or join fragments into a full HTML string
    """
    tables = block.block_config.tables
    if not tables:
        return []

    html_parts = []
    html_parts.append(f"<div class='{container_class}'>")
    for table_result in tables:
        if table_result.title:
            html_parts.append(f"<h4>{table_result.title}</h4>")

        if table_result.description:
            html_parts.append(
                f'<i class="explorica-table-description">'
                f"{table_result.description}"
                f"</i>"
            )
        i, h = (
            table_result.render_extra.get("show_index", True),
            table_result.render_extra.get("show_columns", True),
        )
        html_parts.append(
            table_result.table.to_html(
                classes="explorica-dataframe",
                index=i,
                header=h,
                border=0,
            )
        )
    html_parts.append("</div>")
    return html_parts


def _render_block_html_build_visualizations(
    block,
    mpl_fig_scale: float,
    plotly_fig_scale: float,
    name_css: str = "visualizations",
) -> list[str]:
    """
    Generate HTML snippets for all visualizations in a single Block.

    This internal helper function converts the visualizations contained
    in a `Block` into HTML elements, suitable for embedding in a report.
    Both Matplotlib and Plotly figures are supported. Matplotlib figures
    are rasterized and embedded as base64-encoded PNG images, while Plotly
    figures are rendered as interactive HTML iframes.

    Parameters
    ----------
    block : Block
        The Block instance containing visualizations to render. It is assumed
        that all figures have been normalized into `VisualizationResult` objects.
    mpl_fig_scale : float
        Scaling factor for Matplotlib figures. Figure width and height
        will be multiplied by this value to determine the rendered size.
    plotly_fig_scale : float
        Scaling factor for Plotly figures. Only applied if the figure
        has defined width and height. Figures without explicit dimensions
        will be rendered with default Plotly sizing.
    name_css : str
        CSS class name to wrap all visualization HTML elements. This allows
        external CSS to style or position visualizations consistently.

    Returns
    -------
    list[str]
        A list of HTML strings representing each visualization, wrapped in
        a single `<div>` with the given CSS class. If the block has no
        visualizations, an empty list is returned.

    Notes
    -----
    - Matplotlib figures are saved to PNG in memory and embedded inline using
      base64 encoding.
    - Plotly figures are deep-copied when scaling is applied to avoid modifying
      the original figure.
    - Scaling is applied multiplicatively to the original width/height of
      the figures. For Plotly figures without explicit width/height, the
      scale is ignored.
    - The resulting HTML is intended to be used within a larger report
      container. The CSS class provided in `name_css` is responsible for
      positioning and styling the visualization elements.

    Examples
    --------
    >>> from explorica.core import Block, BlockConfig
    >>> import matplotlib.pyplot as plt
    >>> import plotly.graph_objects as go
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1,2,3], [4,5,6])
    >>> block_config = BlockConfig(
    ...     title="Test Block",
    ...     description="Block with figures",
    ...     visualizations=[fig]
    ... )
    >>> block = Block(block_config)
    >>> html_snippets = _render_block_html_build_visualizations(
    ...     block, mpl_fig_scale=80.0, plotly_fig_scale=1.0, name_css="visualizations")
    >>> full_html = "\n".join(html_snippets)
    >>> print(full_html[:100])  # Preview first 100 characters of the full HTML snippet
    """
    visualization_parts = []
    for vis_result in block.block_config.visualizations:
        if vis_result.engine == "plotly":
            if vis_result.width is not None and vis_result.height is not None:
                w, h = (
                    int(vis_result.width * plotly_fig_scale),
                    int(vis_result.height * plotly_fig_scale),
                )
                fig = deepcopy(vis_result.figure)
                fig.update_layout(width=w, height=h, autosize=False)
            else:
                fig = vis_result.figure
            visualization_parts.append(
                pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
            )
        if vis_result.engine == "matplotlib":
            buf = BytesIO()
            vis_result.figure.savefig(buf, format="png", bbox_inches="tight")
            w, h = (
                int(vis_result.width * mpl_fig_scale),
                int(vis_result.height * mpl_fig_scale),
            )
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            visualization_parts.append(
                f'<img src="data:image/png;base64,{img_b64}" '
                f'width="{w}" height="{h}"/>'
            )
            buf.close()
    if len(visualization_parts) > 0:
        html = [f'<div class="{name_css}">'] + visualization_parts + ["</div>"]
        return html
    return []


def _get_css_style(
    report_class_name: str,
    vis_class_name: str,
    tables_class_name: str,
    max_width: int,
    font_family: str,
) -> str:
    """
    Generate an embedded CSS <style> block for HTML reports.

    This internal helper constructs a preformatted CSS string wrapped in a
    `<style>` tag. The generated styles are scoped to the provided CSS class
    names and are intended to be injected directly into the rendered HTML
    output.

    The styles define:
    - layout and width constraints for the report container,
    - typography for textual elements (headers, paragraphs, lists),
    - responsive behavior for images and embedded visualizations.

    Parameters
    ----------
    report_class_name : str
        CSS class name of the outer report or block container. All report-level
        styles are scoped under this class.
    vis_class_name : str
        CSS class name used to wrap visualization elements (images, iframes).
        Allows consistent styling of all visualizations within the report.
    tables_class_name : str
        CSS class name used to wrap table elements (e.g., tables generated
        from TableResult). This class allows tables to have consistent
        styling, such as scrollable container, max-height, or other table-specific
        CSS rules.
    max_width : int
        Maximum width (in pixels) applied to the report container via the
        `max-width` CSS property.
    font_family : str
        Value for the CSS `font-family` property. This should be a valid
        CSS font-family string (e.g. a comma-separated list of font families).

    Returns
    -------
    str
        A string containing a complete `<style>` block with scoped CSS rules.

    Notes
    -----
    - This function is considered private and internal to the rendering
      pipeline. Its signature and behavior are not part of the public API and
      may change without notice.
    - The returned CSS is embedded directly into the HTML output rather than
      written to an external stylesheet, ensuring self-contained reports.
    """
    return f"""<style>
                   .{report_class_name} {{
                      border: 1px solid #ccc;
                      padding: 10px;
                      margin: 20px auto;
                      max-width: {max_width}px;
                  }}
                  .{report_class_name} h2 {{
                      margin-top: 0;
                      font-family: {font_family};
                  }}
                  .{report_class_name} p {{
                      font-size: 14px;
                      font-family: {font_family};
                  }}
                  .{report_class_name} ul {{
                      font-family: {font_family};
                  }}
                  .{report_class_name} img {{
                      display: block;
                      margin: 10px auto;
                      max-width: 100%;
                      height: auto;
                   }}
                   .{vis_class_name} {{
                       text-align: center;
                       margin: 16px auto;
                   }}
                   .{vis_class_name} img,
                   .{vis_class_name} iframe {{
                       display: block;
                       margin: 0 auto;
                       max-width: 100%;
                       max-height: 600px;
                       object-fit: contain;
                   }}
                   .{tables_class_name} {{
                        max-height: 400px;
                        overflow: auto;
                        }}
                        </style>"""


@enable_io_logs(logger)
def _save_html(
    report: str,
    path: str | Path,
    overwrite: bool = True,
    report_name: str = "report",
):
    """
    Save HTML content to the specified path.

    This is a low-level helper function for writing an HTML report
    string to disk. It does not generate HTML content; it only writes
    the provided string. Public-facing HTML renderers call this
    function internally.

    Parameters
    ----------
    report : str
        HTML content to save.
    path : str or Path
        Directory or full file path where the HTML will be saved.
        - If a directory is provided, the file will be saved as
          ``f"{report_name}.html"``.
        - If a file path is provided, it must end with ``.html``.
    report_name : str, default='report'
        Base name used when `path` is a directory. This is also used
        in logging and error messages for consistent naming.
    overwrite : bool, default=True
        Controls whether existing files can be overwritten:
        - True (default): existing files will be silently overwritten.
        - False: raises a `FileExistsError` if the target file exists.

    Raises
    ------
    ValueError
        If `path` is a file path not ending with ``.html``.
    FileExistsError
        If `overwrite=False` and the target file already exists.

    Notes
    -----
    - This function ensures that the parent directory exists.
    - Intended as an internal helper; use higher-level HTML renderers
      to generate and save full reports.
    - Logging is enabled via the `@enable_io_logs` decorator.

    Examples
    --------
    >>> html_content = "<html><body><h1>Report</h1></body></html>"
    >>> _save_html(html_content, "output_dir")
    >>> _save_html(html_content, "output_file.html")
    """
    path = convert_filepath(path, f"{report_name}.html")
    need_overwrite_check = not overwrite
    validate_path(path, overwrite_check=need_overwrite_check)

    if path.suffix != ".html":
        raise ValueError("'path' must be a directory or have .html extension")

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)


def _get_placeholder_html(
    message: str = ("There are no render blocks in this report."),
) -> str:
    """
    Generate a placeholder HTML snippet for empty reports.

    This internal helper returns a minimal HTML paragraph element containing
    a user-facing message. It is used when a `Report` instance has no blocks
    to render, ensuring that the resulting HTML output is never empty.

    The returned markup is intentionally simple and relies on inherited CSS
    styles from the surrounding report container (e.g. font-family, font size).

    Parameters
    ----------
    message : str, optional
        Text to display inside the placeholder paragraph.

    Returns
    -------
    str
        HTML string containing a single `<p>` element.

    Notes
    -----
    - This function is internal and not part of the public API.
    - Styling is not applied directly and is expected to be inherited from
      the report-level CSS rules.
    """
    return f"<p>{message}</p>"


def _normalize_font_family(font_family) -> str:
    """
    Normalize a font family input into a CSS-compatible string.

    This internal helper converts a font specification into a string suitable
    for use in CSS `font-family` properties.

    Parameters
    ----------
    font_family : str or Sequence[str]
        - If a string is provided, it is returned unchanged.
        - If a sequence of strings is provided, the entries are joined with
          commas to form a CSS-compatible font-family list, e.g.,
          ``["Arial", "sans-serif"]`` -> ``"Arial, sans-serif"``.

    Returns
    -------
    str
        CSS-compatible font-family string.

    Raises
    ------
    TypeError
        If `font_family` is not a string or a sequence of strings.

    Examples
    --------
    >>> _normalize_font_family("Arial")
    'Arial'

    >>> _normalize_font_family(["Arial", "DejaVu Sans", "sans-serif"])
    'Arial, DejaVu Sans, sans-serif'
    """
    if isinstance(font_family, str):
        return font_family
    if isinstance(font_family, Sequence):
        return ", ".join(font_family)
    raise TypeError("font_family must be str or sequence of str")
