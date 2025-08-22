"""
Explorica — a Python framework for simplified exploratory data analysis (EDA).

Features include:
- Data cleaning and categorization
- Automated feature engineering
- Outlier detection and handling
- Data visualization
- Feature interaction analysis
"""

from .visualizer import DataVisualizer
from .data_preprocessor import DataPreprocessor
from .feature_engineer import FeatureEngineer
from .outlier_handler import OutlierHandler

__version__ = "0.1.0"

__all__ = ["DataPreprocessor",
           "FeatureEngineer",
           "OutlierHandler",
           "DataVisualizer"]
