"""
Outlier detection and handling utilities.

This subpackage provides classes for detecting, handling, and
analyzing outliers in numerical datasets. Key utilities exported
at the top level:

- DetectionMethods: outlier detection via IQR, Z-score, etc.
- HandlingMethods: remove or replace outliers in sequences or DataFrames.
- DistributionMetrics: compute skewness, kurtosis, and describe distribution shapes.
"""

from .detection import DetectionMethods
from .handling import HandlingMethods
from .stats import DistributionMetrics

__all__ = [
    "HandlingMethods",
    "DetectionMethods",
    "DistributionMetrics",
]
