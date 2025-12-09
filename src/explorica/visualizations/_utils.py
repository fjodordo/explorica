"""
Internal utility module for Explorica's visualizations subpackage.

This module provides necessary infrastructure for visualization methods (such as
distplot and scatterplot), handling concerns like plot saving, stylistic
consistency, and graceful error handling for edge cases (e.g., empty data).
These utilities ensure that public methods remain clean, focused, and maintain a
consistent user experience.

Functions
-------
save_plot(fig, directory, overwrite, plot_name, verbose)
    Saves a Matplotlib Figure object to disk with file handling and logging.
get_empty_plot(message, figsize)
    Generates a placeholder Figure/Axes for instances where data is unavailable.
temp_plot_theme(palette, style, cmap)
    Context manager to temporarily set multiple Seaborn and Matplotlib styles.
temp_plot_cmap(cmap)
    Context manager to temporarily set the default Matplotlib color map.

Notes
-----
This module is strictly for internal use within the Explorica library. It is not
intended to be imported or used directly by end-users. It relies heavily on
standard Python context managers (`contextlib`) and Matplotlib/Seaborn utilities
to manage global state safely.

Examples
--------
>>> fig, ax = get_empty_plot(message="Data validation failed")
>>> # fig and ax are now Matplotlib objects that can be displayed or saved.
"""

import logging
import contextlib
from typing import Sequence
from contextlib import contextmanager, ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.graph_objs import Figure as PxFigure

from explorica._utils import temp_log_level, read_config

logger = logging.getLogger(__name__)

DEFAULT_MPL_PLOT_PARAMS = {
    "title": "",
    "xlabel": "",
    "ylabel": "",
    "style": None,
    "figsize": (10, 6),
    "directory": None,
    "nan_policy": "drop",
    "verbose": False,
    "plot_kws": {},
}

WRN_MSG_CATEGORIES_EXCEEDS_PALETTE_F = read_config("messages")["warns"][
    "DataVisualizer"
]["categories_exceeds_palette_f"]
WRN_MSG_EMPTY_DATA = read_config("messages")["warns"]["DataVisualizer"]["empty_data_f"]
ERR_MSG_UNSUPPORTED_METHOD = read_config("messages")["errors"]["unsupported_method_f"]
ERR_MSG_UNSUPPORTED_METHOD_F = read_config("messages")["errors"]["unsupported_method_f"]
ERR_MSG_ARRAYS_LENS_MISMATCH = read_config("messages")["errors"][
    "arrays_lens_mismatch_f"
]
ERR_MSG_ARRAYS_LENS_MISMATCH_F = read_config("messages")["errors"][
    "arrays_lens_mismatch_f"
]


def validate_file_format(ext: str, engine: str):
    """
    Validate that a given file extension is supported by the specified engine.

    This is an internal utility used by `save_plot` to ensure that only
    supported formats are written to disk.

    Parameters
    ----------
    ext : str
        File extension to validate (without leading dot, e.g., 'png', 'html').
    engine : str
        Plotting engine, either 'matplotlib' or 'plotly'.

    Raises
    ------
    ValueError
        If the extension is not supported for the given engine.

    Notes
    -----
    - Matplotlib supports multiple static image formats.
    - Plotly supports only 'html'.
    """
    supported_formats = {
        "matplotlib": {
            "png",
            "jpg",
            "jpeg",
            "svg",
            "pdf",
            "eps",
            "pgf",
            "ps",
            "raw",
            "rgba",
            "svgz",
            "tif",
            "tiff",
            "webp",
        },
        "plotly": {"html"},
    }
    if ext not in supported_formats[engine]:
        supported = ", ".join(sorted(supported_formats[engine]))
        raise ValueError(
            f"Unsupported file format '{ext}' for engine '{engine}'. "
            f"Supported formats are: {supported}."
        )


def resolve_plot_path(directory: str, plot_name: str, engine: str):
    """
    Resolve the absolute path and file extension for a plot file.

    If the input path does not include an extension, a default filename
    is generated based on the engine ('plot_name.html' for Plotly,
    'plot_name.png' for Matplotlib).

    Parameters
    ----------
    directory : str
        Target directory or path where the plot will be saved.
    plot_name : str
        Base name for the plot file if the directory does not include an extension.
    engine : str
        Plotting engine, either 'matplotlib' or 'plotly'.

    Returns
    -------
    tuple[Path, str]
        - Absolute path to the file including filename.
        - File extension (without leading dot).
    """
    directory = Path(directory).absolute()
    ext = directory.suffix.lower()
    # If the extension is not specified, assume the input is a directory path
    if ext == "":
        directory = (
            directory / f"{plot_name}.html"
            if engine == "plotly"
            else directory / f"{plot_name}.png"
        )
        ext = directory.suffix.lower()
    return directory, ext[1:]


def _save_plot_validate_plot_name(plot_name: str):
    """
    Internal utility to validate that a plot name is non-empty.

    Raises a ValueError if the plot name is an empty string.

    Parameters
    ----------
    plot_name : str
        Name of the plot file.

    Raises
    ------
    ValueError
        If `plot_name` is empty.
    """
    if plot_name == "":
        err_msg = (
            "The 'plot_name' cannot be empty. "
            "Please provide a non-empty name or use the default value 'plot'."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)


def save_plot(
    fig: plt.Figure | PxFigure,
    directory: str = ".",
    overwrite: bool = True,
    plot_name: str = "plot",
    **kwargs,
):
    """
    Save a matplotlib or Plotly figure to disk with flexible format handling,
    logging, and directory management.

    This function validates the figure object, resolves output paths, checks
    for overwrite policies, and saves the figure in the specified format.
    It supports both static Matplotlib figures and interactive Plotly figures.

    Parameters
    ----------
    fig : matplotlib.pyplot.Figure or plotly.graph_objs._figure.Figure
        The figure object to save. Must be a Matplotlib Figure for engine="matplotlib"
        or a Plotly Figure for engine="plotly".
    directory : str, default="."
        Target file path or directory for saving the figure.
        - If the path includes an extension (e.g., ".png" or ".html"), the figure is
          saved with that filename.
        - If no extension is provided, the figure is saved as
          '{plot_name}.png' (for Matplotlib)
        or '{plot_name}.html' (for Plotly) within the directory.
    overwrite : bool, default=True
        If True, the target file will be overwritten if it exists.
        If False, a FileExistsError is raised.
    plot_name : str, default="plot"
        Name of the plot used for logging and for generating filenames when
        directory is a path without extension. Cannot be empty.
    verbose : bool, default=False
        If True, enables informational logging.
    engine : str, default="matplotlib"
        Specifies which plotting engine to use for saving:
        - "matplotlib": saves Matplotlib figures in one of the supported static
          formats.
        - "plotly": saves Plotly figures as interactive HTML files.
        Only one engine can be used at a time. The fig type must match the chosen
        engine.

    Raises
    ------
    TypeError
        If fig is not a matplotlib Figure object.
        If 'directory' is not a string object.
    ValueError
        If 'plot_name' is empty.
        If 'directory' path is empty, or if any segment of the path
        (folder name) consists only of whitespace.
        If provided file format is unsupported.
    PermissionError
        If unable to write to the specified directory.
    FileExistsError
        If the output file already exists and 'overwrite' is set to False.

    Notes
    -----
    Supported file formats:
        Matplotlib (engine="matplotlib"): "png", "eps", "jpg", "jpeg", "pdf", "pgf",
        "ps",  "raw", "rgba", "svg", "svgz", "tif", "tiff", "webp".
        Plotly (engine="plotly"): "html" only. Plotly figures are
        saved as interactive HTML pages
        that can be opened in a browser, preserving hover info, zooming, and panning.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3])
    >>> save_plot(fig, "./plot.png", "my_plot", verbose=True)
    """

    # ----------------------------------------------------------------------
    # 1. SETUP AND API SANITY CHECKS (FAIL-FAST)
    # ----------------------------------------------------------------------

    log_context = (
        temp_log_level(logger, logging.INFO)
        if kwargs.get("verbose", False)
        else contextlib.nullcontext()
    )
    # 1.1. Check 'fig' type
    if not isinstance(fig, (plt.Figure, PxFigure)):
        logger.error(
            "Failed to save '%s' to %s: "
            "Expected matplotlib or plotly Figure object, got %s.",
            plot_name,
            directory,
            type(fig).__name__,
        )
        raise TypeError(
            f"Expected matplotlib or plotly Figure object, got {type(fig).__name__}. "
            f"Please create a figure using plt.subplots() or plt.figure() first."
        )

    # 1.2. Check 'directory' type
    if not isinstance(directory, str):
        logger.error(
            "Invalid type for argument 'directory'. Expected str, but received %s.",
            type(directory).__name__,
        )
        raise TypeError(
            f"Invalid type for argument '{directory}'. Expected str or pathlib.Path, "
            f"but received {type(directory).__name__}."
        )

    # 1.3. Check 'directory' for non empty path
    if not directory:
        err_msg = "Directory path must not be empty."
        logger.error(err_msg)
        raise ValueError(err_msg)

    # 1.4. Check "directory" for no parts whitespace-only
    for part in Path(directory).parts:
        if not part.strip():
            err_msg = (
                "Invalid path structure detected in argument 'directory'. "
                "Path segment cannot be empty or only whitespace. "
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    # 1.5. Check 'plot_name' for non empty value
    _save_plot_validate_plot_name(plot_name)

    # ----------------------------------------------------------------------
    # 2. PATH RESOLUTION, FORMAT DETECTION, AND I/O OPERATIONS
    # ----------------------------------------------------------------------
    try:
        # 2.1. Determine output path:
        # If the extension is not specified, assume the input is a directory path
        directory, file_format = resolve_plot_path(
            directory, plot_name, kwargs.get("engine", "matplotlib")
        )

        # 2.2. Check format support
        validate_file_format(file_format, kwargs.get("engine", "matplotlib"))
        path = directory.parent
        # 2.3. Create parent directories if they do not exist
        if not path.exists():
            path.mkdir(parents=True)
            logger.warning("Directory '%s' was created automatically.", path)
        # 2.4. Check for overwrite policy (Fail-Safe)
        if directory.exists() and not overwrite:
            logger.error(
                "Attempted to save plot to existing path "
                "without 'overwrite=True'. Path: %s",
                directory,
            )
            raise FileExistsError(
                (
                    f"Attempted to save plot to existing path "
                    f"without 'overwrite=True'. Path: {directory}"
                )
            )
        # 2.5. Execute final save call
        if kwargs.get("engine", "matplotlib") == "matplotlib":
            fig.savefig(directory)
        elif kwargs.get("engine", "matplotlib") == "plotly":
            fig.write_html(directory)
        else:
            raise ValueError(
                (
                    f"Invalid 'engine' parameter: {kwargs.get('engine')}. "
                    f"Supported engines are 'matplotlib' and 'plotly'."
                )
            )
        with log_context:
            logger.info("'%s' saved to %s", plot_name, directory)
    # ----------------------------------------------------------------------
    # 3. ERROR HANDLING (REACTIVE SYSTEM PROCESSES)
    # ----------------------------------------------------------------------

    # 3.1. Catching permissions related errors (common I/O error)
    except PermissionError as e:
        logger.error("Permission denied saving '%s' to %s: %s", plot_name, directory, e)
        raise
    # 3.2 Catching all other system I/O errors (invalid characters, disk full, etc.)
    except Exception as e:
        logger.error("Failed to save '%s' to %s: %s", plot_name, directory, e)
        raise


def get_empty_plot(
    message: str = "No data available for visualization",
    figsize: Sequence[float] = (10, 6),
    engine: str = "matplotlib",
):
    """
    Generate a placeholder empty plot for Matplotlib or Plotly.

    This utility creates a standardized empty visualization canvas with a
    centered informational message. It is typically used when input data is
    empty, invalid, or removed during preprocessing, ensuring that the calling
    function still returns a valid figure object.

    Parameters
    ----------
    message : str, default="No data available for visualization"
        The informational text displayed at the center of the plot.
    figsize : Sequence[float], default=(10, 6)
        The size of the resulting figure.
        - For ``engine='matplotlib'``: interpreted as inches and passed
          directly to ``matplotlib.pyplot.subplots``.
        - For ``engine='plotly'``: interpreted as pixel dimensions for
          ``figure.update_layout(width=..., height=...)``.
    engine : {'matplotlib', 'plotly'}, default='matplotlib'
        Visualization backend to use for generating the empty plot.

    Returns
    -------
    tuple
        If ``engine='matplotlib'``:
            (matplotlib.figure.Figure, matplotlib.axes.Axes)
            The Matplotlib figure and axes objects.
    plotly.graph_objects.Figure
        If ``engine='plotly'``:
            A Plotly figure object with a centered annotation.

    Notes
    -----
    - The Plotly version uses a minimal ``simple_white`` template and hides axes.
    - This function is backend-agnostic and serves as a shared mechanism for
      handling empty-data cases across visualization utilities.

    Examples
    --------
    >>> fig, ax = get_empty_plot(message="No data", engine="matplotlib")
    >>> fig

    >>> fig = get_empty_plot(message="Nothing to display", engine="plotly")
    >>> fig
    """
    if engine == "matplotlib":
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
            style="italic",
        )
        output = fig, ax
    elif engine == "plotly":
        fig = PxFigure()
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 16, "color": "gray"},
        )
        fig.update_layout(
            template="simple_white",
            xaxis={"visible": False},
            yaxis={"visible": False},
            width=figsize[0],
            height=figsize[1],
        )
        output = fig
    else:
        output = None
    return output


@contextmanager
def temp_plot_theme(palette: str = None, style: str = None, cmap: str = None):
    """
    Temporarily set Matplotlib and Seaborn plotting styles and color maps.

    This context manager allows users to temporarily change the Seaborn style,
    Seaborn color palette, and/or Matplotlib color map (for heatmap-like plots)
    within a specific block of code. Changes are automatically reverted upon
    exiting the context.

    Parameters
    ----------
    palette : str or list of colors, optional
        Name of a valid Seaborn color palette (e.g., "viridis", "husl") or a
        list of colors. Sets the color cycle for categorical plots.
    style : str, optional
        Name of a valid Seaborn style context (e.g., "darkgrid", "whitegrid",
        "ticks"). Affects axis appearance, background, and gridlines.
    cmap : str or matplotlib.colors.Colormap, optional
        Name of a Matplotlib color map (e.g., "RdBu", "plasma"). Sets the
        default color map used for scalar-to-color mapping (e.g., in heatmaps).

    Examples
    --------
    Temporarily apply a dark theme and custom palette:

    >>> import pandas as pd
    >>> import seaborn as sns
    >>> data = pd.Series([1, 2, 3, 4])
    >>>
    >>> with temp_plot_theme(style="darkgrid", palette="Set2"):
    ...     # All plots generated inside this block will use "darkgrid" and "Set2"
    ...     sns.histplot(data)
    ...
    >>> # Plots generated after the block reverts to the previous global settings.
    """
    contexts = []
    if style is not None:
        contexts.append(sns.axes_style(style))
    if palette is not None:
        contexts.append(sns.color_palette(palette))
    if cmap is not None:
        contexts.append(temp_plot_cmap(cmap))

    if not contexts:
        contexts.append(contextlib.nullcontext())

    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)
        yield


@contextmanager
def temp_plot_cmap(cmap: str):
    """
    Temporarily set the default Matplotlib color map (cmap).

    This context manager modifies the global Matplotlib parameter
    'image.cmap' to the specified color map name. This affects functions
    that map scalar data to colors, such as those used for heatmaps
    or contour plots. The original color map setting is safely restored
    upon exiting the context.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        The name of the color map to use (e.g., "viridis", "RdBu") or a
        `Colormap` object. This value is assigned to
        `plt.rcParams['image.cmap']`.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> data = np.random.rand(10, 10)
    >>>
    >>> with temp_plot_cmap("plasma"):
    ...     # This plot will use the 'plasma' color map
    ...     plt.imshow(data)
    ...
    >>> # Plots generated after this block will revert to the original cmap
    """
    original_cmap = plt.rcParams["image.cmap"]
    plt.rcParams["image.cmap"] = cmap
    try:
        yield
    finally:
        plt.rcParams["image.cmap"] = original_cmap


def resolve_plotly_palette(palette: str | Sequence[str], categorical=True):
    """
    Resolve a Plotly color palette for categorical or sequential data.

    This utility function returns a list of colors suitable for use in Plotly
    visualizations. It supports built-in Plotly palettes as well as custom
    user-provided sequences of colors. The function automatically selects
    an appropriate default palette if none is provided.

    Parameters
    ----------
    palette : str or Sequence[str]
        The name of a built-in Plotly palette (e.g., "Plotly" for categorical,
        "Viridis" for sequential) or a custom sequence of color strings (hex codes
        or named colors). If `None`, a default palette is used.
    categorical : bool, default=True
        Whether to treat the palette as categorical (`True`) or sequential (`False`).
        Determines the set of default palettes and the namespace to search for named
        palettes.

    Returns
    -------
    List[str]
        A list of color strings suitable for Plotly visualizations.

    Raises
    ------
    ValueError
        If a string is provided that does not correspond to a known Plotly palette
        in the appropriate namespace.

    Notes
    -----
    - Categorical palettes are found in `plotly.express.colors.qualitative`.
    - Sequential palettes are found in `plotly.express.colors.sequential`.
    - This function is intended to standardize palette selection for consistent
    plotting behavior across Explorica visualizations.

    Examples
    --------
    >>> import plotly.express as px
    >>> resolve_plotly_palette("Plotly")
    ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', ...]
    >>> resolve_plotly_palette(["#FF0000", "#00FF00", "#0000FF"])
    ['#FF0000', '#00FF00', '#0000FF']
    >>> resolve_plotly_palette(None, categorical=False)
    ['#440154', '#482878', '#3E4989', ...]  # Viridis sequential palette
    """
    if palette is None:
        return (
            px.colors.qualitative.Plotly
            if categorical
            else px.colors.sequential.Viridis
        )
    if isinstance(palette, str):
        if categorical:
            try:
                colors = getattr(px.colors.qualitative, palette)
            except AttributeError as exc:
                raise ValueError(
                    f"Unknown categorical palette {palette} for Plotly"
                ) from exc
        else:
            try:
                colors = getattr(px.colors.sequential, palette)
            except AttributeError as exc:
                raise ValueError(
                    f"Unknown sequential palette {palette} for Plotly"
                ) from exc
    else:
        colors = palette
    return colors
