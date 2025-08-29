from typing import Union

import matplotlib.pyplot as plt


def handle_plot_output_matplotlib(
    plot: plt.figure,
    show_plot: bool = True,
    return_plot: bool = False,
    directory: str = None,
) -> Union[plt.Figure | None]:
    """
    Handle display, saving, and optional returning of a matplotlib figure.

    This utility function provides a unified interface to either show a plot,
    save it to disk, and/or return the figure object for further manipulation.

    Parameters
    ----------
    plot : matplotlib.figure.Figure
        The figure object to be displayed or saved.
    show_plot : bool, default=True
        If True, the plot will be displayed using `plt.show()`.
    return_plot : bool, default=False
        If True, the figure object is returned to the caller.
    dir : str, optional
        File path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure, optional
        The figure object, if `return_plot` is True. Otherwise, returns None.

    Notes
    -----
    - If neither `show_plot` nor `dir` is provided, the function will only return
      the figure if `return_plot` is True.
    - The figure can be further modified by the caller after being returned.
    """
    if directory is not None:
        plot.savefig(directory)
    if show_plot:
        plt.show()
    if return_plot:
        return plot
