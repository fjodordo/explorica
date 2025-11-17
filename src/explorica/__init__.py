"""
Explorica — a Python framework for simplified exploratory data analysis (EDA).

Features include:
- Data cleaning and categorization
- Automated feature engineering
- Outlier detection and handling
- Data visualization
- Feature interaction analysis
"""
import logging

from .visualizer import DataVisualizer

__version__ = "0.1.0"

__all__ = ["DataVisualizer"]

logger = logging.getLogger("explorica")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.WARNING)
