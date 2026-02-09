"""
Facade for creating visualizations.

This subpackage provides a high-level interface for generating both
statistical and categorical plots. The main class, DataVisualizer,
acts as a unified entry point for various specialized visualization
tools.

Classes available at the top level include:
- DataVisualizer: high-level facade for statistical and categorical plots.

Notes
-----
- In Jupyter/IPython: figures are automatically displayed when returned.
- In scripts: call `plt.show()` to display figures after plotting.

Examples:
---------
# In Jupyter - automatic display
distplot(data)

# In scripts - explicit display
distplot(data)
plt.show()

# For customization - get figure/axes
fig, ax = distplot(data, return_plot=True)
ax.set_title("Custom Title")
plt.show()
"""

from .plots import piechart, barchart, mapbox
from .statistical_plots import distplot, boxplot, heatmap, hexbin
from .scatterplot import scatterplot

__all__ = [
    "piechart",
    "barchart",
    "mapbox",
    "distplot",
    "boxplot",
    "scatterplot",
    "heatmap",
    "hexbin",
]
