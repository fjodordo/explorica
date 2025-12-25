"""
Report module of Explorica Reports

This module contains the `Report` class, which aggregates multiple
`Block` instances into a single structured report and provides
utilities for rendering, managing, and iterating over blocks.

Classes
-------
Report(blocks=None, title=None, description=None)
    Aggregates multiple report blocks into a structured report,
    supports rendering to HTML/PDF, block management, iteration,
    and safe figure handling.

Notes
-----
- Matplotlib figures contained in blocks are not automatically closed.
  To free memory after processing or rendering, call `Report.close_figures()`.
- Blocks passed to `Report` are deep-copied to ensure safety and avoid
  unintended side-effects on external references.
- For consistent behavior, ensure that all blocks contain properly
  normalized visualizations (`VisualizationResult` instances).

Examples
--------
>>> from explorica.reports import Block, BlockConfig
>>> from explorica.reports.core import Report
>>> import matplotlib.pyplot as plt
>>> import plotly.graph_objects as go

# Create some figures
>>> fig1, ax1 = plt.subplots()
>>> ax1.plot([1, 2, 3], [4, 5, 6])
>>> fig2, ax2 = plt.subplots()
>>> ax2.plot([10, 20, 30], [3, 6, 9])
>>> figly = go.Figure(data=go.Bar(y=[2, 3, 1]))

# Initialize blocks with metrics and visualizations
>>> block1_cfg = BlockConfig(
...     title="Block 1",
...     description="First block",
...     metrics=[{"name": "Mean", "value": 5.0}],
...     visualizations=[fig1, figly]
... )
>>> block2_cfg = BlockConfig(
...     title="Block 2",
...     description="Second block",
...     metrics=[{"name": "Std", "value": 1.2}],
...     visualizations=[fig2]
... )
>>> block1 = Block(block1_cfg)
>>> block2 = Block(block2_cfg)

# Create a report with initial block
>>> report = Report(blocks=[block1], title="My Report", description="Example report")
>>> len(report)
1

# Add another block in-place
>>> report += block2
>>> len(report)
2

# Insert a new metric into block1
>>> block1.add_metric("Max", 10.0)

# Iterate through blocks
>>> for blk in report:
...     print(blk.typename)
Block
Block

# Render report (HTML/PDF)
>>> report.render_html(path=None)
>>> report.render_pdf(path=None)

# Close Matplotlib figures safely
>>> report.close_figures()
>>> plt.close(fig)
>>> plt.get_fignums()  # all report figures are closed
[]
"""

from copy import deepcopy
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.figure

from ..utils import normalize_visualization
from ..renderers import render_pdf, render_html
from .block import Block


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
    close_figures()
        Close all active Matplotlib figures stored in this report's blocks.
    insert_block(block, index)
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

    Notes
    -----
    - All blocks passed to the constructor are deep-copied, ensuring that
    modifications to blocks within the report do not affect external
    references.
    - Visualizations in each block are automatically normalized into
    `VisualizationResult` objects.
    - Matplotlib figures are not automatically closed. To free memory,
    use `Report.close_figures()` or ensure that `Block` provides a method
    to close its visualizations.

    Examples
    --------
    >>> from explorica.reports import Block, BlockConfig
    >>> from explorica.reports.core import Report
    >>> import matplotlib.pyplot as plt
    >>> import plotly.graph_objects as go

    # Create some figures
    >>> fig1, ax1 = plt.subplots()
    >>> ax1.plot([1, 2, 3], [4, 5, 6])

    >>> fig2, ax2 = plt.subplots()
    >>> ax2.plot([10, 20, 30], [3, 6, 9])

    >>> figly = go.Figure(data=go.Bar(y=[2, 3, 1]))

    # Initialize blocks with metrics and visualizations
    >>> block1_cfg = BlockConfig(
    ...     title="Block 1",
    ...     description="First block",
    ...     metrics=[{"name": "Mean", "value": 5.0}],
    ...     visualizations=[fig1, figly]
    ... )
    >>> block2_cfg = BlockConfig(
    ...     title="Block 2",
    ...     description="Second block",
    ...     metrics=[{"name": "Std", "value": 1.2}],
    ...     visualizations=[fig2]
    ... )

    >>> block1 = Block(block1_cfg)
    >>> block2 = Block(block2_cfg)

    # Create a report with initial block
    >>> report = Report(
    ...     blocks=[block1], title="My Report", description="Example report")
    >>> len(report)
    1

    # Add another block in-place
    >>> report += block2
    >>> len(report)
    2

    # Insert a new metric into block1
    >>> block1.add_metric("Max", 10.0)

    # Iterate through blocks
    >>> for blk in report:
    ...     print(blk.typename)
    Block
    Block

    # Render report (HTML/PDF)
    >>> report.render_html(path=None)
    >>> report.render_pdf(path=None)

    # Close Matplotlib figures safely
    >>> report.close_figures()
    >>> plt.close(fig)
    >>> plt.get_fignums()  # all report figures are closed
    []
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
        - Matplotlib figures are not automatically closed. To free memory,
          use `Report.close_figures()` or ensure that `Block` provides a method
          to close its visualizations.
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

    def insert_block(self, block: Block, index: int):
        """
        Insert a block at the specified position in the report.

        If `index` is not provided or is -1, the block is appended to the end
        of the report (equivalent to `append`).

        Parameters
        ----------
        block : Block
            The block to insert into the report.
        index : int, optional
            Position at which to insert the block. Supports negative indexing.

        Examples
        --------
        # Insert a block at the beginning:
        >>> report = Report(blocks=[block1])
        >>> report.insert_block(block2, index=0)
        >>> report.blocks[0] is block2
        True

        # Insert a block at the end (using len(report.blocks) as index):
        >>> report.insert_block(block3, index=len(report.blocks))
        >>> report.blocks[-1] is block3
        True
        """
        self.blocks.insert(index, block)

    def remove_block(self, index: int) -> Block:
        """
        Remove a block from the report at the specified index.

        This method modifies the current Report instance by removing the block
        at the given position. Supports negative indexing (like Python lists).

        Parameters
        ----------
        index : int
            Position of the block to remove. Negative values count from the end
            (e.g., -1 removes the last block).

        Returns
        -------
        Block
            The removed block instance.

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
            return self.blocks.pop(index)
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

        See also
        --------
        explorica.reports.renderers.html.render_html
            Entrypoint to render a `Block` or `Report`
            object into an HTML representation.

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

        See also
        --------
        explorica.reports.renderers.pdf.render_pdf
            Entrypoint to render a `Block` or `Report`
            object into an PDF representation.

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
            "font": font,
            "path": path,
            "doc_template_kws": doc_template_kws,
            **kwargs,
        }
        return render_pdf(self, **params)

    def close_figures(self):
        """
        Close all active Matplotlib figures stored in this report's blocks.

        This method iterates over all visualizations in all `Block` instances
        contained within the report and closes any Matplotlib figures to free
        up memory and resources.

        Notes
        -----
        - Only Matplotlib figures are affected; Plotly figures or other visualization
        objects are ignored.
        - It is recommended to call this method once all processing or rendering
        with the `Report` instance is finished, to avoid memory leaks from
        lingering figure objects.

        Examples
        --------
        >>> report = Report(blocks=[block1, block2])
        >>> # ... generate visualizations and render report ...
        >>> report.close_figures()  # safely close all Matplotlib figures
        """
        for block in self.blocks:
            for vis in block.block_config.visualizations:
                if isinstance(vis.figure, matplotlib.figure.Figure):
                    plt.close(vis.figure)
