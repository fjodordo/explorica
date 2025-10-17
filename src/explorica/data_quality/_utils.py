from typing import Optional

import numpy as np
import pandas as pd


class Replacers:
    """
    Utility class for replacing values in a pandas Series.

    Provides methods to replace outliers or specific indices with:
    - measures of central tendency (mean, median, mode)
    - random values sampled from the Series
    - explicit scalar values
    """

    @staticmethod
    def replace_mct(
        feature: pd.Series,
        to_replace: pd.Index,
        include_to_replace: bool = False,
        measure: Optional[str] = "mean",
    ) -> pd.Series:
        """
        Replace values at specified indices using a measure of central tendency.

        Parameters
        ----------
        feature : pd.Series
            The Series in which values will be replaced.
        to_replace : pd.Index
            Indices of values to replace.
        include_to_replace : bool, default False
            Whether to include the `to_replace` values in the
            calculation of the measure.
        measure : {'mean', 'median', 'mode'}, default 'mean'
            Measure to compute for replacement.

        Returns
        -------
        pd.Series
            Series with values at `to_replace` replaced by the calculated measure.
            Rounds the replacement values if the Series dtype is integer.
        """
        supported_measures = {
            "mean": lambda series: series.mean(),
            "median": lambda series: series.median(),
            "mode": lambda series: series.mode().iloc[0],
        }
        replaced = feature.copy()
        if not include_to_replace:
            fill_value = supported_measures[measure](replaced.drop(to_replace))
        else:
            fill_value = supported_measures[measure](feature)
        replaced = Replacers.replace(replaced, to_replace, fill_value)
        return replaced

    @staticmethod
    def replace_random(
        feature: pd.Series, to_replace: pd.Index, seed: float = None
    ) -> pd.Series:
        """
        Replace values at specified indices with a random value sampled
        from the Series excluding the `to_replace` indices.

        Parameters
        ----------
        feature : pd.Series
            The Series in which values will be replaced.
        to_replace : pd.Index
            Indices of values to replace.
        seed : int or float, optional, default None
            Seed for the random number generator.
            If None, the random selection will be non-deterministic.

        Returns
        -------
        pd.Series
            Series with values at `to_replace` replaced by a random value.
            Rounds the replacement values if the Series dtype is integer.
        """
        replaced = feature.copy()
        fill_value = replaced.drop(to_replace).sample(1, random_state=seed).iloc[0]
        replaced = Replacers.replace(replaced, to_replace, fill_value)
        return replaced

    @staticmethod
    def replace(feature: pd.Series, to_replace: pd.Index, value) -> pd.Series:
        """
        Replace values at specified indices with a provided value.

        Parameters
        ----------
        feature : pd.Series
            The Series in which values will be replaced.
        to_replace : pd.Index
            Indices of values to replace.
        value : scalar
            Value to assign to the specified indices.

        Returns
        -------
        pd.Series
            Series with values at `to_replace` replaced by `value`.
            Rounds the replacement values if the Series dtype is integer.
        """
        replaced = feature.copy()
        if pd.api.types.is_integer_dtype(feature):
            replaced[to_replace] = np.round(value)
        else:
            replaced[to_replace] = value
        return replaced
