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
    Placeholder class for future aggregation of multiple report blocks and
    report construction utilities.

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

from typing import Any
from dataclasses import dataclass, is_dataclass

import plotly.graph_objects
import matplotlib.figure

from explorica.types import VisualizationResult
from .utils import normalize_visualization
from .renderers import render_pdf, render_html


# pylint: disable=too-few-public-methods
class Report:
    """
    Placeholder class for Explorica reports aggregation.

    This class is intended to serve as a future container for multiple
    report blocks (`Block` instances), providing utilities to compose,
    manage, and render aggregated reports. Currently, it is an
    architectural stub without implemented functionality.
    """


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

    title: str
    description: str
    metrics: list[dict[str, Any]]
    visualizations: list[
        plotly.graph_objects.Figure | matplotlib.figure.Figure | VisualizationResult
    ]


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
