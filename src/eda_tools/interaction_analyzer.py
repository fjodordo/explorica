"""
interaction_analyzer.py

Module for analyzing interactions between variables in a dataset,
including correlation matrices for numeric features and association
measures for categorical features (such as Cramér's V).
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

class InteractionAnalyzer:
    """
    A utility class for analyzing statistical interactions between variables.

    Provides tools for computing correlation matrices using various methods,
    including Pearson, Spearman, and Cramér's V. Designed to support exploratory
    data analysis (EDA) by revealing linear or categorical associations between features.
    """

    def __init__(self):
        pass

    def corr_matrix(self, dataset: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
        """
        Compute a correlation matrix using the specified method.

        Parameters
        ----------
        dataset : pd.DataFrame
            DataFrame with numeric or categorical variables.
        method : str, optional
            Correlation method to use:
            - 'pearson': standard correlation for continuous features
            - 'spearman': rank-based correlation (non-parametric)
            - 'cramer_v': for categorical features (based on chi-squared statistic)

        Returns
        -------
        pd.DataFrame
            A symmetric matrix of pairwise correlation values.

        Raises
        ------
        ValueError
            If the specified method is not supported.
        """
        supported_methods = {"pearson", "spearman", "cramer_v"}
        if method not in supported_methods:
            raise ValueError(f"Unsupported method '{method}'. Choose from: {supported_methods}")

        if method in {"pearson", "spearman"}:
            return dataset.corr(method=method)

        if method == "cramer_v":
            cols = dataset.columns
            n = len(cols)
            matrix = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)

            for i, col1 in enumerate(cols):
                for j in range(i):
                    col2 = cols[j]
                    v = self.cramer_v(dataset[col1], dataset[col2])
                    matrix.iloc[i, j] = v
                    matrix.iloc[j, i] = v  # ensure symmetry
        return matrix

    def high_corr_pairs(self, dataset: pd.DataFrame, method: str = 'pearson',
                        threshold: float = 0.7) -> pd.DataFrame | None:
        """
        Find pairs of features with high correlation/association.

        Parameters
        ----------
        dataset : pd.DataFrame
            DataFrame with numeric or categorical features.
        method : str, optional
            Correlation method: 'pearson', 'spearman', or 'cramer_v'.
        threshold : float, optional
            Minimum absolute value of correlation to consider.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns ['1st', '2nd', 'coef'] for each pair
            exceeding the given threshold. Returns None if no such pairs found.
        """
        correlation_matrix = self.corr_matrix(dataset, method=method)
        corr_pairs = []

        cols = correlation_matrix.columns
        for i, col1 in enumerate(cols):
            for j in range(i + 1, len(cols)):  # only the upper triangle
                coef = correlation_matrix.iloc[i, j]
                if abs(coef) >= threshold:
                    corr_pairs.append([col1, cols[j], coef])

        if not corr_pairs:
            return None

        return pd.DataFrame(corr_pairs, columns=["1st", "2nd", "coef"])\
             .sort_values(by="coef", key=abs, ascending=False)

    def cramer_v(
        self,
        x: pd.Series,
        y: pd.Series,
        bias_correction: bool = True,
        yates_correction: bool = False
    ) -> float:
        """
        Calculates Cramér's V statistic for measuring the association between two categorical
        variables.

        Parameters
        ----------
        x : pd.Series
            First categorical variable.
        y : pd.Series
            Second categorical variable.
        bias_correction : bool, optional, default=True
            Whether to apply bias correction (recommended for small samples).
        yates_correction : bool, optional, default=False
            Whether to apply Yates' correction for continuity (only applies to 2x2 tables; usually
            set to False when using Cramér's V).

        Returns
        -------
        float
            Cramér's V value, ranging from 0 (no association) to 1 (perfect association).
            Returns 0 if the statistic is undefined (e.g., due to zero denominator).
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix, correction=yates_correction)[0]
        n = confusion_matrix.to_numpy().sum()
        r, k = confusion_matrix.shape
        min_dim = min(r - 1, k - 1)

        if min_dim == 0:
            return 0.0

        if bias_correction:
            correction = ((r - 1) * (k - 1)) / n
            return np.sqrt((chi2 / n - correction) / min_dim)
        else:
            return np.sqrt(chi2 / (n * min(k - 1, r - 1)))
