"""
Block module of Explorica Reports

This module provides the `Block` class, a core building block
for constructing analytical reports in Explorica. It allows
managing metrics and visualizations within a report block
and rendering them to HTML or PDF formats.

Classes
-------
Block(block_config=None)
    A container for a report block in Explorica.
BlockConfig
    Dataclass for storing configuration of a report block, including title,
    description, metrics, and visualizations.

Notes
-----
- `Block` wraps a `BlockConfig` instance and normalizes all visualizations
  into `VisualizationResult` objects for consistent rendering.
- HTML and PDF rendering methods are thin wrappers around the renderers,
  forwarding all parameters for flexibility.

Examples
--------
# Simple usage
>>> from explorica.reports import Block, BlockConfig
>>> import matplotlib.pyplot as plt
>>> import plotly.graph_objects as go

>>> # Create a matplotlib figure
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3], [4, 5, 6])

>>> # Create a plotly figure
>>> figly = go.Figure(data=go.Bar(y=[2, 3, 1]))

>>> # Initialize BlockConfig
>>> block_cfg = BlockConfig(
...     title="Example Block",
...     description="A minimal example of Block usage.",
...     metrics=[{"name": "Mean", "value": 5.0}],
...     visualizations=[fig, figly]
... )

>>> # Initialize Block
>>> block = Block(block_cfg)

# Usage of management methods
>>> # Add a new metric
>>> block.add_metric("Std", 1.2, description="Standard deviation")

>>> # Insert a metric at index 0
>>> block.insert_metric(0, "Min", 2.0)

>>> # Add a new visualization
>>> fig2, ax2 = plt.subplots()
>>> ax2.plot([10, 20, 30], [3, 6, 9])
>>> block.add_visualization(fig2)

>>> # Insert a visualization at index 1
>>> fig3 = go.Figure(data=go.Scatter(y=[1, 4, 9]))
>>> block.insert_visualization(fig3, 1)

# Render to HTML
>>> html_output = block.render_html(path=None)
>>> print(type(html_output))
<class 'str'>

# Render to PDF
>>> pdf_bytes = block.render_pdf(path=None)
>>> print(type(pdf_bytes))
<class 'bytes'>
"""

from typing import Any, Sequence, Hashable, Mapping
from dataclasses import dataclass, field

import plotly.graph_objects
import matplotlib.figure

from ...types import VisualizationResult, TableResult
from ..utils import normalize_visualization, normalize_table
from ..renderers import render_pdf, render_html


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
    tables : list of array-like, mapping, or TableResult
        A list of tabular results associated with the block. Each element can be:
        - a 1D or 2D sequence (e.g., list of lists, tuple of tuples)
        - a mapping (e.g., dict-like structure)
        - ``TableResult`` - a pre-normalized tabular container

        Array-like or mapping inputs are automatically normalized into
        ``TableResult`` when a `Block` instance is created. Users may provide
        raw tabular data or already normalized table objects.

        If a ``TableResult`` is provided, its ``title`` and ``description``
        attributes are used during rendering to label and describe the table
        in the report output.

    Notes
    -----
    - This class is primarily used internally by the `Block` class.
    - Normalization of figures into ``VisualizationResult`` and tabular data
      into ``TableResult`` occurs automatically during `Block` initialization.
      This ensures that all visual and tabular content is consistently
      formatted for rendering.
    - ``BlockConfig`` itself does not perform validation or normalization;
      this responsibility belongs to the `Block` class.

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
    tables: list[
        Sequence[Sequence[Any]] | Mapping[str, Sequence[Any]] | TableResult
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
    block_config : dict or BlockConfig, optional
        Configuration for the block.
        - If `dict` -> converted to `BlockConfig`.
        - If `BlockConfig` -> used as-is.
        - If `None` -> a new empty `BlockConfig` is created.

    Attributes
    ----------
    block_config : BlockConfig
        The configuration of the block including title, description,
        metrics, and visualizations (normalized to `VisualizationResult`).

    Methods
    -------
    add_visualization(visualization)
        Add a visualization to the block.
    insert_visualization(visualization, index)
        Insert a visualization into the block at a given position.
    remove_visualization(index)
        Remove a visualization from the block by index.
    clear_visualizations()
        Remove all visualizations from the block.
    add_metric(name, value, description = None)
        Add a scalar metric to the block.
    insert_metric(index, name, value, description = None)
        Insert a scalar metric into the block at a given position.
    remove_metric(index)
        Remove a metric from the block at a given index.
    add_table(table, title = None, description = None)
        Add a table to the block.
    insert_table(index, table, title = None, description = None)
        Insert a table into the block at a given position.
    remove_table(index)
        Remove and return a table from the block by index.
    render_pdf(path = None, font = "DejaVuSans", doc_template_kws = None, **kwargs)
        Render the block to PDF format.
    render_html(
        path=None,
        font=("Arial", "DejaVu Serif", "DejaVu Sans", "sans-serif"),
        **kwargs
    )
        Render the block to HTML format.
    typename : str
        The name of the class, always 'Block'. Useful for type-checking
        without direct imports.

    Notes
    -----
    - During initialization, all figures in `block_config.visualizations`
    are normalized into `VisualizationResult` objects via `normalize_visualization`.
    - All tables in `block_config.tables` are normalized into `TableResult`
    objects via `normalize_table`.
    - This ensures consistent handling for rendering in HTML or PDF and
    standardized table representation.
    - `block_config.visualizations` and `block_config.tables` can be empty or None;
    in that case, they are initialized as empty lists.

    Examples
    --------
    # Simple usage
    >>> from explorica.reports import Block, BlockConfig
    >>> import matplotlib.pyplot as plt
    >>> import plotly.graph_objects as go

    >>> # Create a matplotlib figure
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])

    >>> # Create a plotly figure
    >>> figly = go.Figure(data=go.Bar(y=[2, 3, 1]))

    >>> # Initialize BlockConfig
    >>> block_cfg = BlockConfig(
    ...     title="Example Block",
    ...     description="A minimal example of Block usage.",
    ...     metrics=[{"name": "Mean", "value": 5.0}],
    ...     visualizations=[fig, figly]
    ... )

    >>> # Initialize Block
    >>> block = Block(block_cfg)

    # Usage of management methods
    >>> # Add a new metric
    >>> block.add_metric("Std", 1.2, description="Standard deviation")

    >>> # Insert a metric at index 0
    >>> block.insert_metric(0, "Min", 2.0)

    >>> # Add a new visualization
    >>> fig2, ax2 = plt.subplots()
    >>> ax2.plot([10, 20, 30], [3, 6, 9])
    >>> block.add_visualization(fig2)

    >>> # Insert a visualization at index 1
    >>> fig3 = go.Figure(data=go.Scatter(y=[1, 4, 9]))
    >>> block.insert_visualization(fig3, 1)

    # Render to HTML
    >>> html_output = block.render_html(path=None)
    >>> print(type(html_output))
    <class 'str'>

    # Render to PDF
    >>> pdf_bytes = block.render_pdf(path=None)
    >>> print(type(pdf_bytes))
    <class 'bytes'>
    """

    def __init__(self, block_config=None):
        """
        Initialize a Block instance and normalize its visualizations.
        """
        if block_config is None:
            self.block_config = BlockConfig()
        elif isinstance(block_config, dict):
            self.block_config = BlockConfig(**block_config)
        elif isinstance(block_config, BlockConfig):
            self.block_config = block_config
        else:
            raise ValueError("'block_config' must be a dict or dataclass")

        # Normalize visualizations
        self.block_config.visualizations = [
            normalize_visualization(vis)
            for vis in self.block_config.visualizations or []
        ]

        # Normalize tables
        self.block_config.tables = [
            normalize_table(table) for table in self.block_config.tables or []
        ]

    @property
    def typename(self):
        """
        Return the class name.

        Returns
        -------
        str
            The name of the class ('Block').

        Examples
        --------
        >>> block = Block()
        >>> block.typename
        'Block'
        """
        return self.__class__.__name__

    def add_visualization(
        self,
        visualization: (
            matplotlib.figure.Figure | plotly.graph_objects.Figure | VisualizationResult
        ),
    ):
        """
        Add a visualization to the block.

        The visualization is normalized into a `VisualizationResult` object
        before being appended to the block's list of visualizations.

        Parameters
        ----------
        visualization : matplotlib.figure.Figure, plotly.graph_objects.Figure,
                        or VisualizationResult
            The figure or visualization object to add to the block.

        Notes
        -----
        - `normalize_visualization` is applied automatically to ensure
        a uniform internal representation.
        - This method does not render the visualization; it only stores it
        for future rendering via `render_html` or `render_pdf`.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from explorica.reports import Block
        >>> fig, ax = plt.subplots()
        >>> block = Block()
        >>> block.add_visualization(fig)
        """
        self.block_config.visualizations.append(normalize_visualization(visualization))

    def insert_visualization(
        self,
        visualization: (
            matplotlib.figure.Figure | plotly.graph_objects.Figure | VisualizationResult
        ),
        index: int,
    ):
        """
        Insert a visualization into the block at a given position.

        The visualization is normalized into a `VisualizationResult` object
        before being inserted into the block's list of visualizations.

        Parameters
        ----------
        visualization : matplotlib.figure.Figure, plotly.graph_objects.Figure,
                        or VisualizationResult
            The figure or visualization object to insert.
        index : int
            Position at which the visualization is inserted.
            Uses standard Python list insertion semantics.

        Notes
        -----
        - `normalize_visualization` is applied automatically to ensure
          a uniform internal representation.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from explorica.reports import Block
        >>> fig, ax = plt.subplots()
        >>> block = Block()
        >>> block.insert_visualization(fig, index=0)
        """
        self.block_config.visualizations.insert(
            index, normalize_visualization(visualization)
        )

    def remove_visualization(self, index: int) -> VisualizationResult:
        """
        Remove a visualization from the block by index.

        Parameters
        ----------
        index : int
            Index of the visualization to remove. Follows standard Python
            list indexing semantics.

        Returns
        -------
        VisualizationResult
            The removed visualization.

        Raises
        ------
        IndexError
            If there is no visualization at the specified index.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from explorica.reports import Block, BlockConfig
        >>> fig, ax = plt.subplots()
        >>> block = Block(BlockConfig(visualizations=[fig]))
        >>> block.remove_visualization(index=0)
        >>> block.block_config.visualizations
        []
        """
        try:
            return self.block_config.visualizations.pop(index)
        except IndexError as e:
            raise IndexError(f"No visualization at index {index}") from e

    def clear_visualizations(self):
        """
        Remove all visualizations from the block.

        This method clears the list of visualizations associated with the block.

        Returns
        -------
        None

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from explorica.reports import Block, BlockConfig
        >>> fig, ax = plt.subplots()
        >>> block = Block(BlockConfig(visualizations=[fig]))
        >>> block.clear_visualizations()
        >>> block.block_config.visualizations
        []
        """
        self.block_config.visualizations = []

    def _validate_metric(self, metric: dict):
        """
        Validate a metric dictionary before adding it to the block.

        This method enforces the internal metric contract used by
        :meth:`add_metric` & :meth:`insert_metric`. It checks that metric components
        are compatible with downstream rendering and storage requirements.

        Parameters
        ----------
        metric : dict
            Metric specification with the following keys:

            - ``name`` : Hashable
            Metric name. Must be hashable and renderable as text.
            - ``value`` : Number
            Scalar numeric value. Must be hashable and renderable as text.
            - ``description`` : Hashable or None
            Optional metric description. Must be hashable and renderable
            as text if provided.

        Raises
        ------
        TypeError
            If ``name``, ``description`` or ``value`` are not hashable.
        NotImplementedError
            If ``value`` is a numeric sequence (sequence metrics are not
            supported yet).

        Notes
        -----
        This method is intended for internal use only and should not be
        called directly by users.
        """
        # Name validation
        if not isinstance(metric["name"], Hashable):
            raise TypeError(
                "Expected `name` to be hashable or None,"
                f"got {type(metric['name']).__name__}"
            )
        # Description validation
        if metric["description"] is not None and (
            not isinstance(metric["description"], Hashable)
        ):
            raise TypeError(
                "Expected `description` to be hashable or None,"
                f"got {type(metric['description']).__name__}"
            )
        # Value validation
        if not isinstance(metric["value"], Hashable):
            raise TypeError(
                "Expected `value` to be hashable or None,"
                f"got {type(metric['value']).__name__}"
            )

    def add_metric(
        self,
        name: Hashable,
        value: Hashable | Sequence[Hashable],
        description: Hashable = None,
    ):
        """
        Add a scalar metric to the block.

        Metrics represent key numeric indicators that are rendered alongside
        visualizations in reports.

        Parameters
        ----------
        name : Hashable
            Metric name. Must be hashable and renderable as text
            (e.g. ``str``, ``int``, ``Enum``).
        value : Number
            Metric value. Can be any hashable scalar or sequence of hashable
            scalars. Non-hashable types are not supported.
        description : Hashable, optional
            Optional metric description. Must be hashable and renderable
            as text if provided.

        Raises
        ------
        TypeError
            If ``name``, ``description`` or ``value`` are not hashable.

        Notes
        -----
        - Metrics are stored internally as dictionaries with keys
        ``name``, ``value`` and ``description``.

        Examples
        --------
        >>> block.add_metric("mean", 5.2)
        >>> block.add_metric("std", 1.3, description="Standard deviation")
        """
        metric = {"name": name, "value": value, "description": description}
        self._validate_metric(metric)
        self.block_config.metrics.append(metric)

    def insert_metric(
        self,
        index: int,
        name: Hashable,
        value: Hashable | Sequence[Hashable],
        description: Hashable = None,
    ):
        """
        Insert a scalar metric into the block at a given position.

        This method behaves like :meth:`add_metric`, but allows explicit
        control over the insertion index. Metric validation rules are
        identical to those of :meth:`add_metric`.

        Parameters
        ----------
        index : int
            Position at which the metric should be inserted.
            Follows standard Python ``list.insert`` semantics.
        name : Hashable
            Metric name. Must be hashable and renderable as text
            (e.g. ``str``, ``int``, ``Enum``).
        value : Hashable or Sequence[Hashable]
            Metric value. Can be any hashable scalar or sequence of hashable
            scalars.
        description : Hashable, optional
            Optional metric description. Must be hashable and renderable
            as text if provided.

        Raises
        ------
        TypeError
            If ``name``, ``description`` or ``value`` are not hashable.

        Notes
        -----
        - This method follows standard Python ``list.insert`` behavior for
        index handling.
        - Negative indices are supported and follow standard Python semantics.
        - Metric validation is performed via the internal
        :meth:`_validate_metric` helper.

        Examples
        --------
        >>> block.insert_metric(0, "mean", 5.2)
        >>> block.insert_metric(1, "std", 1.3, description="Standard deviation")
        """
        metric = {"name": name, "value": value, "description": description}
        self._validate_metric(metric)
        self.block_config.metrics.insert(index, metric)

    def remove_metric(self, index: int) -> dict:
        """
        Remove a metric from the block at a given index.

        Parameters
        ----------
        index : int
            Position of the metric to remove. Follows standard Python
            list indexing semantics, including support for negative indices.

        Returns
        -------
        dict
            The removed metric, containing keys ``name``, ``value`` and
            ``description``.

        Raises
        ------
        IndexError
            If no metric exists at the given index.

        Notes
        -----
        - This is a direct wrapper around Python's list ``pop`` method
        for the internal ``metrics`` list.
        - The returned metric is the exact dictionary stored internally.

        Examples
        --------
        >>> block.add_metric("mean", 5.0)
        >>> removed = block.remove_metric(0)
        >>> print(removed)
        {'name': 'mean', 'value': 5.0, 'description': None}
        """
        try:
            return self.block_config.metrics.pop(index)
        except IndexError as e:
            raise IndexError(f"No metric at index {index}") from e

    def add_table(
        self,
        table: Sequence[Any] | Mapping[str, Sequence],
        title: str = None,
        description: str = None,
    ):
        """
        Add a table to the block.

        The provided table data is normalized into a :class:`TableResult`
        instance during insertion. Supported inputs include 1D/2D sequences
        and mapping-like objects (e.g. dictionaries), which are internally
        converted to a pandas DataFrame.

        Parameters
        ----------
        table : Sequence[Any] or Mapping[str, Sequence]
            Tabular data to add. Can be a 1D or 2D sequence, or a mapping of sequences.
        title : str, optional
            Optional table title. If `table` is already a `TableResult`,
            this value will overwrite its existing `title`. Used during rendering.
        description : str, optional
            Optional table description. If `table` is already a `TableResult`,
            this value will overwrite its existing `description`.
            Used during rendering.

        Notes
        -----
        - The input data is normalized and wrapped
        into a :class:`TableResult`.
        - MultiIndex or multi-column DataFrames are not supported and may
        raise an error during normalization.

        Examples
        --------
        >>> block.add_table(
        ...     {"mean": [1.2], "std": [0.3]},
        ...     title="Summary statistics"
        ... )
        """
        tr = normalize_table(table)
        if title is not None:
            tr.title = title
        if description is not None:
            tr.description = description
        self.block_config.tables.append(tr)

    def insert_table(
        self,
        index: int,
        table: Sequence[Any] | Mapping[str, Sequence],
        title: str = None,
        description: str = None,
    ):
        """
        Insert a table into the block at a given position.

        This method behaves like :meth:`add_table`, but allows explicit
        control over the insertion index. Index handling follows standard
        Python ``list.insert`` semantics.

        Parameters
        ----------
        index : int
            Position at which the table should be inserted.
            Supports negative indices.
        table : Sequence[Any] or Mapping[str, Sequence]
            Tabular data to insert.
            Can be a 1D or 2D sequence, or a mapping of sequences.
        title : str, optional
            Optional table title. If `table` is already a `TableResult`,
            this value will overwrite its existing `title`. Used during rendering.
        description : str, optional
            Optional table description. If `table` is already a `TableResult`,
            this value will overwrite its existing `description`.
            Used during rendering.

        Notes
        -----
        - The table is normalized into a :class:`TableResult` before insertion.
        - Negative indices follow standard Python list behavior.

        Examples
        --------
        >>> block.insert_table(
        ...     0,
        ...     [[1, 2], [3, 4]],
        ...     title="Raw values"
        ... )
        """
        tr = normalize_table(table)
        if title is not None:
            tr.title = title
        if description is not None:
            tr.description = description
        self.block_config.tables.insert(index, tr)

    def remove_table(self, index: int) -> TableResult:
        """
        Remove and return a table from the block by index.

        Parameters
        ----------
        index : int
            Position of the table to remove. Supports negative indexing.

        Returns
        -------
        TableResult
            The removed table.

        Raises
        ------
        IndexError
            If no table exists at the given index.

        Examples
        --------
        >>> removed = block.remove_table(0)
        >>> removed.title
        'Summary statistics'
        """
        try:
            return self.block_config.tables.pop(index)
        except IndexError as e:
            raise IndexError(f"No metric at index {index}") from e

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
        Render the block to HTML format.

        This method is a thin convenience wrapper around
        :func:`explorica.reports.renderers.html.render_html` that allows rendering
        a single ``Block`` instance directly via its public API.

        All parameters are forwarded to the underlying renderer without
        modification.

        Parameters
        ----------
        path : str, optional
            Directory path or full file path where the HTML output should be saved.
            If a directory is provided, the file name is derived automatically.
            If None, the rendered HTML is returned as a string without saving.
        font : str or Sequence[str], optional
            Font family (or sequence of font families) used to construct the CSS
            ``font-family`` property for textual elements.
            If a sequence is provided, it is joined into a comma-separated CSS
            font-family declaration.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`explorica.reports.renderers.html.render_html`
            (e.g. ``max_width``, ``verbose``, ``debug``, ``mpl_fig_scale``).

        Returns
        -------
        str
            Rendered HTML content as a string.

        See Also
        --------
        explorica.reports.renderers.html.render_html
            Entrypoint to render a `Block` or `Report` object into an HTML
            representation.

        Notes
        -----
        - This method does not introduce any new rendering logic.
        It exists purely for ergonomic reasons, allowing HTML rendering
        directly from a ``Block`` instance.
        - CSS styles are applied at the block level when rendering a single block.

        Examples
        --------
        >>> html = block.render_html()
        >>> block.render_html(path="./reports")
        >>> block.render_html(path="./reports/my_block.html")
        """
        params = {"path": path, "font": font, **kwargs}
        return render_html(self, **params)

    def render_pdf(
        self,
        path: str = None,
        font: str = "DejaVuSans",
        doc_template_kws: dict = None,
        **kwargs,
    ) -> bytes:
        """
        Render the block to PDF format.

        This method is a thin convenience wrapper around
        :func:`explorica.reports.renderers.pdf.render_pdf`, enabling PDF rendering
        directly from a ``Block`` instance.

        All parameters are forwarded to the underlying renderer without
        modification.

        Parameters
        ----------
        path : str, optional
            Directory path or full file path where the PDF output should be saved.
            If None, the rendered PDF is returned as bytes without saving.
        font : str, default="DejaVuSans"
            Font used for rendering text in the PDF. Can be a supported font alias
            or a path to a custom TTF font file.
        doc_template_kws : dict, optional
            Additional keyword arguments passed to ReportLab's
            ``SimpleDocTemplate`` to customize page layout (e.g. margins, page size).
        **kwargs
            Additional keyword arguments forwarded to
            :func:`explorica.reports.renderers.pdf.render_pdf`
            (e.g. ``report_name``, ``verbose``, ``debug``,
            ``mpl_fig_scale``, ``plotly_fig_scale``).

        Returns
        -------
        bytes
            Rendered PDF content as bytes.

        See also
        --------
        explorica.reports.renderers.pdf.render_pdf
            Entrypoint to render a `Block` or `Report` object into an PDF
            representation.

        Notes
        -----
        - This method provides a convenient, object-oriented interface for PDF
        rendering but does not alter rendering behavior.
        - Plotly visualizations are rendered as placeholders due to the static
        nature of the PDF format.

        Examples
        --------
        >>> pdf_bytes = block.render_pdf()
        >>> block.render_pdf(path="./reports")
        >>> block.render_pdf(path="./reports/my_block.pdf")
        """
        params = {
            "path": path,
            "font": font,
            "doc_template_kws": doc_template_kws,
            **kwargs,
        }
        return render_pdf(self, **params)
