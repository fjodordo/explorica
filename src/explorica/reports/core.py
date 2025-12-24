"""
Core module for Explorica reports.

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
>>> from explorica.reports import Block, BlockConfig

>>> config = BlockConfig(
...     title="Sample Block",
...     description="Example block for demonstration",
...     metrics=[{"name": "accuracy", "value": 0.95}],
...     visualizations=[]
... )
>>> block = Block(config)
>>> html = block.render_block("html")
>>> pdf_bytes = block.render_block("pdf")
"""

from copy import deepcopy
from typing import Any, Sequence
from dataclasses import dataclass, is_dataclass, field

import plotly.graph_objects
import matplotlib.figure

from explorica.types import VisualizationResult
from .utils import normalize_visualization
from .renderers import render_pdf, render_html


@dataclass
class BlockConfig:
    """
    Configuration container for a single report block in Explorica.

    This dataclass stores all information necessary to render a block in
    a report. It includes metadata, metrics, and visualizations.

    Attributes
    ----------
    title : str
        The title of the block, displayed prominently in the report.
    description : str
        A textual description of the block, providing context or
        explanation for the included metrics and visualizations.
    metrics : list of dict
        A list of metric dictionaries to display in the block. Each dictionary
        can include keys such as:
        - ``name`` : str, required — name of the metric
        - ``value`` : Any, required — value of the metric
        - ``description`` : str, optional — additional context
    visualizations : list of Matplotlib or Plotly figures, or VisualizationResult
        A list of visualizations associated with the block. Each element can be:
        - ``matplotlib.figure.Figure`` — a Matplotlib figure
        - ``plotly.graph_objects.Figure`` — a Plotly figure
        - ``VisualizationResult`` — a pre-normalized visualization object
        Figures provided as Matplotlib or Plotly objects are automatically
        normalized into ``VisualizationResult`` when a `Block` instance
        is created. Users may provide raw figures or already normalized
        visualizations.

    Notes
    -----
    - This class is primarily used internally by the `Block` class.
    - The normalization of figures into `VisualizationResult` occurs
      automatically during `Block` initialization; this ensures that
      all visualizations are consistently formatted for rendering.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> from plotly import graph_objects as go
    >>> from explorica.reports import BlockConfig
    >>> fig_mpl, ax = plt.subplots()
    >>> fig_mpl.plot([1, 2, 3], [4, 5, 6])
    >>> fig_plotly = go.Figure(data=go.Bar(y=[1, 2, 3]))
    >>> config = BlockConfig(
    ...     title="Example Block",
    ...     description="This block shows example metrics and figures.",
    ...     metrics=[{"name": "Mean", "value": 5.0}],
    ...     visualizations=[fig_mpl, fig_plotly]
    ... )
    """

    title: str = ""
    description: str = ""
    metrics: list[dict[str, Any]] = field(default_factory=list)
    visualizations: list[
        plotly.graph_objects.Figure | matplotlib.figure.Figure | VisualizationResult
    ] = field(default_factory=list)


class Block:
    """
    A container for a report block in Explorica.

    This class wraps a `BlockConfig` dataclass and provides utilities
    for rendering the block into HTML or PDF. During initialization,
    all figures in `block_config.visualizations` are normalized into
    `VisualizationResult` objects for uniform downstream processing.

    Parameters
    ----------
    block_config : dict or BlockConfig
        The configuration of the block. If a dictionary is provided,
        it will be converted into a `BlockConfig` instance.

    Attributes
    ----------
    block_config : BlockConfig
        The configuration of the block including title, description,
        metrics, and visualizations (normalized to `VisualizationResult`).
    typename : str
        The name of the class, always 'Block'. Useful for type-checking
        without direct imports.

    Methods
    -------
    render_block(format, path=None, report_name="report_block")
        Render the block into the specified format ('html' or 'pdf').

    Examples
    --------
    # Simple usage
    >>> from explorica.reports import Block, BlockConfig
    >>> import matplotlib.pyplot as plt
    >>> import plotly.graph_objects as go

    >>> # Get matplotlib figure
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])

    >>> # Get plotly figure
    >>> figly = go.Figure(data=go.Bar(y=[2, 3, 1]))

    >>> # Block config
    >>> block_cfg = BlockConfig(
    ...     title="Example Block",
    ...     description="A minimal example of Block usage.",
    ...     metrics=[{"name": "Mean", "value": 5.0}],
    ...     visualizations=[fig, figly]
    ... )

    >>> # Block init
    >>> block = Block(block_cfg)

    >>> # Block render tp HTML
    >>> html_output = block.render_block(format="html")
    >>> print(type(html_output))
    <class 'str'>

    >>> # Block render to PDF (oprtional)
    >>> # pdf_bytes = block.render_block(format="pdf")
    >>> # print(type(pdf_bytes))
    <class 'bytes'>
    """

    def __init__(self, block_config):
        """
        Initialize a Block instance and normalize its visualizations.
        """
        if isinstance(block_config, dict):
            self.block_config = BlockConfig(**block_config)
        elif is_dataclass(block_config):
            self.block_config = block_config
        else:
            raise ValueError("'block_config' must be a dict or dataclass")

        # Normalize visualizations
        self.block_config.visualizations = [
            normalize_visualization(vis) for vis in self.block_config.visualizations
        ]

    @property
    def typename(self):
        """
        Return the class name.

        Returns
        -------
        str
            The name of the class ('Block').
        """
        return self.__class__.__name__

    def render_block(
        self, output_format: str, path: str = None, report_name="report_block"
    ):
        """
        Render the block into HTML or PDF.

        Parameters
        ----------
        output_format : str
            The format to render. Must be 'html' or 'pdf'.
        path : str, optional
            Path to save the rendered output. If None, the output is returned.
        report_name : str, default='report_block'
            Name to use for the saved file (without extension).

        Returns
        -------
        str or bytes
            HTML string if `output_format='html'` or PDF bytes if `output_format='pdf'`.

        Raises
        ------
        ValueError
            If `output_format` is not 'html' or 'pdf'.
        """
        if output_format.lower() == "html":
            return render_html(self, path, report_name=report_name)
        if output_format.lower() == "pdf":
            return render_pdf(self, path, report_name=report_name)
        raise ValueError("'output_format' must be 'html' or 'pdf'.")


class Report:
    """
    Aggregates multiple report blocks into a single structured report.

    The Report class serves as a container for `Block` instances, providing
    utilities to compose, manage, and render aggregated reports in different
    formats (HTML, PDF). It supports both in-place and functional operations
    for adding, inserting, or removing blocks, as well as iteration and
    length queries like a standard Python sequence.

    Parameters
    ----------
    blocks : list[Block], optional
        Initial list[Block] instances to include in the report. If None,
        an empty report is created.
    title : str, optional
        Title of the report.
    description : str, optional
        Short description of the report.

    Attributes
    ----------
    blocks : list[Block]
        The list of `Block` instances in this report.
    title : str or None
        The title of the report.
    description : str or None
        Description of the report.

    Methods
    -------
    render_html(path=None, report_name="report")
        Render the report to HTML format.
    render_pdf(path=None, doc_template_kws=None, report_name="report")
        Render the report to PDF format.
    insert_block(block, index=-1)
        Insert a block at the specified index in the report.
    remove_block(index)
        Remove a block at the specified index.
    typename
        Property returning the class name as a string.
    __iadd__(other)
        Add a Block or list of Blocks to the report in-place using `+=`.
    __add__(other)
        Return a new Report with `other` block(s) appended.
    __len__()
        Return the number of blocks in the report.
    __iter__()
        Iterate over the blocks in the report.

    Examples
    --------
    # Create a report with a single block:
    >>> report = Report(
    ...     blocks=[block1], title="My Report",
    ...     description="Example report")
    >>> len(report)
    1

    # Add another block in-place:
    >>> report += block2
    >>> len(report)
    2

    # Create a new report by combining reports/blocks:
    >>> report2 = report + [block3, block4]
    >>> len(report2)
    4
    >>> len(report)  # original report unchanged
    2

    # Iterate through blocks:
    >>> for blk in report:
    ...     print(blk.typename)

    # Render report (HTML/PDF):
    >>> report.render_report(output_format="html", path="output/")
    >>> report.render_report(output_format="pdf")
    """

    def __init__(
        self,
        blocks: list[Block] = None,
        title: str = None,
        description: str = None,
    ):
        """
        Initialize a Report instance and normalize its visualizations.

        Notes
        -----
        - All blocks are deep-copied to avoid side-effects from external references.
        - Visualizations inside each block are automatically normalized.
        """
        self.blocks = []
        blocks_copy = deepcopy(blocks or [])
        # Ensures, that all visualizations in blocks are VisualizationResult objs
        for block in blocks_copy:
            block.block_config.visualizations = [
                normalize_visualization(vis)
                for vis in block.block_config.visualizations
            ]
        self.blocks.extend(blocks_copy)
        self.title = title
        self.description = description

    def __iadd__(self, other: Block | list[Block]):
        """
        Add a Block or a list of Blocks to the report in-place using `+=`.

        This method modifies the current Report instance by appending one or
        more blocks directly to its `blocks` list. The original Report object
        is updated, unlike `__add__` which returns a new Report.

        Parameters
        ----------
        other : Block or list[Block]
            A single `Block` or a list of `Block` instances to add to the report.

        Returns
        -------
        self : Report
            The updated Report instance, allowing method chaining.

        Raises
        ------
        TypeError
            If `other` is neither a `Block` nor a list of `Block`.

        Examples
        --------
        # Add a single block:
        >>> report = Report(blocks = [block1])
        >>> report += block2
        >>> report.blocks
        [<explorica.reports.core.Block,
         <explorica.reports.core.Block]

        # Add multiple blocks:
        >>> report = Report(blocks = [block1])
        >>> report += [block2, block3]
        >>> report.blocks
        [<explorica.reports.core.Block,
         <explorica.reports.core.Block,
         <explorica.reports.core.Block]
        """
        if isinstance(other, Block):
            self.blocks.append(other)
        elif isinstance(other, list) and all(isinstance(b, Block) for b in other):
            self.blocks.extend(other)
        else:
            raise TypeError("Can only add a Block or a list of Blocks")
        return self

    def __add__(self, other: Block | list[Block]):
        """
        Return a new Report with `other` block(s) appended.

        The original Report is not modified. a deep copy of the Report is created
        and the new block(s) are added to it.

        Parameters
        ----------
        other : Block or list[Block]
            The block or list of blocks to add to the report.

        Returns
        -------
        Report
            A new Report instance containing the original blocks plus `other`.

        Raises
        ------
        TypeError
            If `other` is not a Block or list of Block.

        Notes
        -----
        - The original report remains unchanged.
        - Blocks are appended to the end of the new report's block list.

        Examples
        --------
        >>> report1 = Report([block1])
        >>> report2 = report1 + block2
        >>> len(report2)
        2
        >>> len(report1)  # original report unchanged
        1

        >>> report3 = report1 + [block2, block3]
        >>> len(report3)
        3
        """
        new_report = deepcopy(self)
        if isinstance(other, Block):
            new_report.blocks.append(other)
        elif isinstance(other, list) and all(isinstance(b, Block) for b in other):
            new_report.blocks.extend(other)
        else:
            raise TypeError("Can only add a Block or a list of Blocks")
        return new_report

    def __len__(self):
        """
        Return the number of blocks in the report.

        This allows using Python's built-in `len()` function on a Report instance.

        Returns
        -------
        int
            The number of `Block` instances contained in the report.

        Examples
        --------
        >>> report = Report(blocks=[block1, block2])
        >>> len(report)
        2
        """
        return len(self.blocks)

    def __iter__(self):
        """
        Return an iterator over the blocks in the report.

        This allows iterating directly over a Report instance using a for-loop.

        Returns
        -------
        iterator of Block
            An iterator over the `Block` instances contained in the report.

        Examples
        --------
        >>> report = Report(blocks=[block1, block2])
        >>> for block in report:
        ...     print(block.typename)
        Block
        Block
        """
        return iter(self.blocks)

    @property
    def typename(self):
        """
        Return the class name.

        Returns
        -------
        str
            The name of the class ('Report').

        Examples
        --------
        >>> report = Report()
        >>> report.typename
        'Report'
        """
        return self.__class__.__name__

    def insert_block(self, block, index=-1):
        """
        Insert a block at the specified position in the report.

        If `index` is not provided or is -1, the block is appended to the end
        of the report (equivalent to `append`).

        Parameters
        ----------
        block : Block
            The block to insert into the report.
        index : int, optional
            Position at which to insert the block. Supports negative indexing
            (default is -1, which appends at the end).

        Examples
        --------
        # Append a block to the end of the report:
        >>> report = Report(blocks=[block1])
        >>> report.insert_block(block2)
        >>> len(report.blocks)
        2

        # Insert a block at the beginning:
        >>> report.insert_block(block3, index=0)
        >>> report.blocks[0] is block3
        True
        """
        self.blocks.insert(index, block)

    def remove_block(self, index: int):
        """
        Remove a block from the report at the specified index.

        This method modifies the current Report instance by removing the block
        at the given position. Supports negative indexing (like Python lists).

        Parameters
        ----------
        index : int
            Position of the block to remove. Negative values count from the end
            (e.g., -1 removes the last block).

        Raises
        ------
        IndexError
            If the index is out of range.

        Examples
        --------
        # Remove the first block:
        >>> report = Report(blocks=[block1, block2])
        >>> report.remove_block(0)
        >>> len(report.blocks)
        1

        # Remove the last block using negative indexing:
        >>> report.remove_block(-1)
        >>> len(report.blocks)
        0
        """
        try:
            self.blocks.pop(index)
        except IndexError as e:
            raise IndexError(f"No block at index {index}") from e

    def render_html(
        self,
        path: str = None,
        font: str | Sequence[str] = (
            "Arial",
            "DejaVu Serif",
            "DejaVu Sans",
            "sans-serif",
        ),
        **kwargs,
    ) -> str:
        """
        Render the report to HTML format.

        This method is a thin convenience wrapper around
        :func:`explorica.reports.renderers.html.render_html` and exposes the same
        rendering functionality directly on a ``Report`` instance.

        All parameters are forwarded to the underlying renderer without
        modification.

        Parameters
        ----------
        path : str, optional
            If provided, the HTML content will be saved to this location.
            Can be a directory (in which case the file will be saved as
            ``"{report_name}.html"``) or a full file path ending with ``.html``.
        font : str or Sequence[str], optional
            Font family or sequence of font families to be applied to textual
            elements (title, description, metrics) in the report.
            If a sequence is provided, the first available font installed on
            the system will be used. Defaults to
            ``("Arial", "DejaVu Serif", "DejaVu Sans", "sans-serif")``.
        **kwargs : dict
            Additional parameters passed through to the underlying
            :func:`render_html` function. This includes, for example, `report_name`,
            `max_width`, `mpl_fig_scale`, `plotly_fig_scale`, `verbose`, `debug`, etc.

        Returns
        -------
        str
            HTML content as a string.

        Notes
        -----
        - CSS is automatically applied for the report container and its blocks.
        - Font-family is determined from the `font` argument and propagated to all
        textual elements.
        - Visualizations (Matplotlib and Plotly) are rendered according to the
        scaling factors provided via `kwargs`.
        - This method does not modify the underlying Report instance.

        Examples
        --------
        # Render HTML without saving
        >>> report = Report(blocks=[block1, block2], title="My Report")
        >>> html_content = report.render_html()

        # Render and save to a directory (file name derived from `report_name`)
        >>> report.render_html(path="./output", report_name="customer_report")

        # Render and save to a full file path
        >>> report.render_html(path="./output/my_report.html")
        """
        params = {
            "path": path,
            "font": font,
            **kwargs,
        }
        return render_html(self, **params)

    def render_pdf(
        self,
        path: str = None,
        font: str = "DejaVuSans",
        doc_template_kws: dict = None,
        **kwargs,
    ) -> bytes:
        """
        Render the report to PDF format.

        This method is a thin convenience wrapper around
        :func:`explorica.reports.renderers.pdf.render_pdf` and exposes the same
        rendering functionality directly on a ``Report`` instance.

        All parameters are forwarded to the underlying renderer without
        modification.

        Parameters
        ----------
        path : str or None, optional
            Directory path or full file path where the PDF report should be saved.
            - If a directory is provided, the report is saved as
            ``f"{report_name}.pdf"`` inside that directory.
            - If a full file path ending with ``.pdf`` is provided, the report
            is saved to that exact location.
            If None, the rendered PDF is returned as bytes without saving.
        doc_template_kws : dict, optional
            Keyword arguments passed directly to
            ``reportlab.platypus.SimpleDocTemplate`` to control PDF layout
            (e.g., margins, page size). Provides full access to all currently supported
            customization options.
        report_name : str, default="report"
            Base name used for the output file when `path` is a directory.

        Returns
        -------
        bytes
            Rendered PDF content as bytes.

        Notes
        -----
        This method exposes the full capabilities of the PDF renderer.
        Users can rely on this method to access all functionality currently
        implemented in ``explorica.reports.renderers.render_pdf``.

        Examples
        --------
        >>> pdf_bytes = report.render_pdf()
        >>> report.render_pdf(path="output/")
        >>> report.render_pdf(
        ...     path="output/my_report.pdf",
        ...     doc_template_kws={"pagesize": A3}
        ... )
        """
        params = {
            "path": path,
            "font": font,
            "doc_template_kws": doc_template_kws,
            **kwargs,
        }
        return render_pdf(self, **params)
