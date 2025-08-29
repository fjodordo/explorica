"""
Facade for creating visualizations.

This subpackage provides a high-level interface for generating both
statistical and categorical plots. The main class, DataVisualizer,
acts as a unified entry point for various specialized visualization
tools.

Classes available at the top level include:
- DataVisualizer: high-level facade for statistical and categorical plots
"""

from .visualizer import DataVisualizer

__all__ = ["DataVisualizer"]
