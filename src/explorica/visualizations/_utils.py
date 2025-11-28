"""
Internal utility module for Explorica's visualizations subpackage.

This module provides necessary infrastructure for visualization methods (such as
distplot and scatterplot), handling concerns like plot saving, stylistic
consistency, and graceful error handling for edge cases (e.g., empty data).
These utilities ensure that public methods remain clean, focused, and maintain a
consistent user experience.

Methods
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

from explorica._utils import temp_log_level

logger = logging.getLogger(__name__)

def save_plot(
    fig: plt.Figure,
    directory: str = ".",
    overwrite: bool = True,
    plot_name: str = "plot",
    verbose: bool = False,
    ):
    """
    Save a matplotlib figure to disk with optional logging and format handling.
    
    This function handles file path validation, format detection, and error handling
    for saving matplotlib figures. It supports the following formats:
    "png", "eps", "jpg", "jpeg", "pdf", "pgf", "ps",  "raw", "rgba", "svg",
    "svgz", "tif", "tiff", "webp".

    Parameters
    ----------
    fig : matplotlib.pyplot.Figure
        Figure object to save.
    directory : str, default="."
        File path where the figure will be saved. If the path has no file extension,
        it is treated as a directory path and the plot is saved as '{plot_name}.png' 
        within it. If the path includes an extension, the plot is saved with that
        exact filename.
    overwrite : bool, default=True
        If True, the target file will be overwritten if it exists.
        If False, a FileExistsError is raised.
    plot_name : str, default="plot"
        Name of the plot used for logging and for generating filenames when 
        directory is a path without extension. Cannot be empty.
    verbose : bool, default=False
        If True, enables informational logging.

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

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3])
    >>> save_plot(fig, "./plot.png", "my_plot", verbose=True)
    """

    # ----------------------------------------------------------------------
    # 1. SETUP AND API SANITY CHECKS (FAIL-FAST)
    # ----------------------------------------------------------------------

    log_context = (temp_log_level(logger, logging.INFO) if
                    verbose else contextlib.nullcontext())
    # 1.1. Check 'fig' type
    if not isinstance(fig, plt.Figure):
        logger.error("Failed to save '%s' to %s: "
                     "Expected matplotlib Figure object, got %s.",
                     plot_name, directory, type(fig).__name__)
        raise TypeError(
            f"Expected matplotlib Figure object, got {type(fig).__name__}. "
            f"Please create a figure using plt.subplots() or plt.figure() first."
        )

    # 1.2. Check 'directory' type
    if not isinstance(directory, str):
        logger.error("Invalid type for argument 'directory'. Expected str, "
            "but received %s.", type(directory).__name__)
        raise TypeError(f"Invalid type for argument '{directory}'. Expected str or pathlib.Path, "
            f"but received {type(directory).__name__}.")

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
    if plot_name == "":
        err_msg = ("The 'plot_name' cannot be empty. "
        "Please provide a non-empty name or use the default value 'plot'.")
        logger.error(err_msg)
        raise ValueError(err_msg)

    # ----------------------------------------------------------------------
    # 2. PATH RESOLUTION, FORMAT DETECTION, AND I/O OPERATIONS
    # ----------------------------------------------------------------------
    try:
        directory = Path(directory).absolute()

        file_format = directory.suffix.lower()

        # 2.1. Determine output path:
        # If the extension is not specified, assume the input is a directory path
        if file_format == "":
            # No extension - treat as directory and add filename
            directory = directory / f"{plot_name}.png"
            file_format = ".png"
        file_format = file_format[1:] # Remove the dot

        # 2.2. Check format support
        supported_formats = {
            "png", "eps", "jpg",
            "jpeg", "pdf", "pgf",
            "ps",  "raw", "rgba",
            "svg", "svgz", "tif",
            "tiff", "webp"
        }
        if file_format not in supported_formats:
            raise ValueError(
                f"Unsupported format: '{file_format}'. "
                f"Supported formats: {sorted(supported_formats)}"
            )
        path = directory.parent
        # 2.3. Create parent directories if they do not exist
        if not path.exists():
            path.mkdir(parents=True)
            logger.warning("Directory '%s' was created automatically.", path)
        # 2.4. Check for overwrite policy (Fail-Safe)
        if directory.exists() and not overwrite:
            logger.error("Attempted to save plot to existing path "
                          "without 'overwrite=True'. Path: %s", directory)
            raise FileExistsError((f"Attempted to save plot to existing path "
                                   f"without 'overwrite=True'. Path: {directory}"))
        # 2.5. Execute final save call
        fig.savefig(directory)
        with log_context:
            logger.info("'%s' saved to %s", plot_name, directory)
    # ----------------------------------------------------------------------
    # 3. ERROR HANDLING (REACTIVE SYSTEM PROCESSES)
    # ----------------------------------------------------------------------

    # 3.1. Catching permissions related errors (common I/O error)
    except PermissionError as e:
        logger.error("Permission denied saving '%s' to %s: %s",
                     plot_name, directory, e)
        raise
    # 3.2 Catching all other system I/O errors (invalid characters, disk full, etc.)
    except Exception as e:
        logger.error("Failed to save '%s' to %s: %s", plot_name, directory, e)
        raise

def get_empty_plot(message: str = "No data available for visualization",
                   figsize: Sequence[float] = (10, 6)):
    """
    Generate a placeholder Matplotlib Figure and Axes object.

    This utility function is used to create a standard, empty plot canvas
    with a centered informational message. This is typically used to handle
    edge cases where input data is empty, invalid, or requires special
    handling that prevents generating a standard visualization, ensuring
    the calling function still returns valid Matplotlib objects.

    Parameters
    ----------
    message : str, default="No data available for visualization"
        The informational text displayed at the center of the plot.
    figsize : Sequence[float], default=(10, 6)
        The size of the resulting figure, passed directly to
        `matplotlib.pyplot.subplots`.
    
    Returns
    -------
    tuple (matplotlib.figure.Figure, matplotlib.axes.Axes)
        Figure and axes objects for further customization.

    Examples
    --------
    >>> fig, ax = get_empty_plot(message="Data validation failed")
    >>> # fig and ax are now Matplotlib objects that can be displayed or saved.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message,
        ha="center", va="center", transform=ax.transAxes,
        fontsize=12, color="gray", style="italic")
    return fig, ax

@contextmanager
def temp_plot_theme(palette: str = None,
                    style: str = None,
                    cmap: str = None):
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
    original_cmap = plt.rcParams['image.cmap']
    plt.rcParams['image.cmap'] = cmap
    try:
        yield
    finally:
        plt.rcParams['image.cmap'] = original_cmap
