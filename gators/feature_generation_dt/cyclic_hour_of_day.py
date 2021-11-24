# Licence Apache-2.0
from math import pi
from typing import List, TypeVar

import numpy as np

import feature_gen_dt

from ..util import util
from ._base_datetime_feature import _BaseDatetimeFeature

PREFACTOR = 2 * pi / 23.0


DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class CyclicHourOfDay(_BaseDatetimeFeature):
    """Create new columns based on the cyclic mapping of the hour of the day.

    Parameters
    ----------
    columns : List[str]
        List of columns.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation_dt import CyclicHourOfDay
    >>> X = pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicHourOfDay(columns=['A'])
    >>> obj.fit_transform(X)
                        A  B  A__hour_of_day_cos  A__hour_of_day_sin
    0 2020-01-01 23:00:00  0            1.000000       -2.449294e-16
    1 2020-12-15 18:00:00  1            0.203456       -9.790841e-01
    2                 NaT  0                 NaN                 NaN

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_dt import CyclicHourOfDay
    >>> X = ks.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicHourOfDay(columns=['A'])
    >>> obj.fit_transform(X)
                        A  B  A__hour_of_day_cos  A__hour_of_day_sin
    0 2020-01-01 23:00:00  0            1.000000       -2.449294e-16
    1 2020-12-15 18:00:00  1            0.203456       -9.790841e-01
    2                 NaT  0                 NaN                 NaN

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation_dt import CyclicHourOfDay
    >>> X = pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicHourOfDay(columns=['A'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, 1.0,
            -2.4492935982947064e-16],
           [Timestamp('2020-12-15 18:00:00'), 1, 0.20345601305263328,
            -0.979084087682323],
           [NaT, 0, nan, nan]], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_dt import CyclicHourOfDay
    >>> X = ks.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicHourOfDay(columns=['A'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, 1.0,
            -2.4492935982947064e-16],
           [Timestamp('2020-12-15 18:00:00'), 1, 0.20345601305263328,
            -0.979084087682323],
           [NaT, 0, nan, nan]], dtype=object)


    """

    def __init__(self, columns: List[str]):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        column_names = self.get_cyclic_column_names(columns, "hour_of_day")
        column_mapping = {
            name: col for name, col in zip(column_names, columns + columns)
        }
        _BaseDatetimeFeature.__init__(self, columns, column_names, column_mapping)

    def transform(self, X: DataFrame) -> DataFrame:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.

        Returns
        -------
        DataFrame
            Transformed dataframe.
        """
        self.check_dataframe(X)
        return self.compute_cyclic_hour_of_day(X)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the NumPy array `X`.

        Parameters
        ----------
        X  : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Transformed array.
        """
        self.check_array(X)
        return feature_gen_dt.cyclic_hour_of_day(X, self.idx_columns, PREFACTOR)

    def compute_cyclic_hour_of_day(
        self,
        X: DataFrame,
    ) -> DataFrame:
        """Compute the cyclic hours of the day features.

        Parameters
        ----------
        X_datetime : DataFrame
            Dataframe of datetime columns.

        Returns
        -------
        DataFrame
            Dataframe of cyclic hours of the day features.
        """

        for i, c in enumerate(self.columns):
            hour = X[c].dt.hour
            X[self.column_names[2 * i]] = np.cos(PREFACTOR * hour)
            X[self.column_names[2 * i + 1]] = np.sin(PREFACTOR * hour)
        return X
