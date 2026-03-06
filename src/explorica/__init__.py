"""
Explorica - a Python framework for simplified exploratory data analysis (EDA).

Features include:
- Data cleaning and categorization
- Automated feature engineering
- Outlier detection and handling
- Data visualization
- Feature interaction analysis
- EDA reports automation
"""

import logging

from . import data_quality
from . import interactions
from . import reports
from . import visualizations
from . import types

__all__ = [
    "data_quality",
    "interactions",
    "reports",
    "visualizations",
    "types",
]

__version__ = "1.0.0"
__author__ = "LaplaceDevil"
__email__ = "LaplaceDevil@proton.me"


def info():
    """Print basic info about Explorica package."""
    print(f"Explorica v{__version__} by {__author__}")


logger = logging.getLogger("explorica")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.WARNING)
