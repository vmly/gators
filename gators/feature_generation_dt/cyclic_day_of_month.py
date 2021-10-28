# Licence Apache-2.0
from math import pi
from typing import List, TypeVar

import numpy as np

import feature_gen_dt

from ..util import util
from ._base_datetime_feature import _BaseDatetimeFeature

TWO_PI = 2 * pi


DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class CyclicDayOfMonth(_BaseDatetimeFeature):
    """Create new columns based on the cyclic mapping of the day of the month.

    Parameters
    ----------
    columns: List[str]
        List of columns.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation_dt import CyclicDayOfMonth
    >>> X = pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicDayOfMonth(columns=['A'])
    >>> obj.fit_transform(X)
                        A  B  A__day_of_month_cos  A__day_of_month_sin
    0 2020-01-01 23:00:00  0             1.000000             0.000000
    1 2020-12-15 18:00:00  1            -0.978148             0.207912
    2                 NaT  0                  NaN                  NaN

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_dt import CyclicDayOfMonth
    >>> X = ks.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicDayOfMonth(columns=['A'])
    >>> obj.fit_transform(X)
                        A  B  A__day_of_month_cos  A__day_of_month_sin
    0 2020-01-01 23:00:00  0             1.000000             0.000000
    1 2020-12-15 18:00:00  1            -0.978148             0.207912
    2                 NaT  0                  NaN                  NaN

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation_dt import CyclicDayOfMonth
    >>> X = pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicDayOfMonth(columns=['A'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, 1.0, 0.0],
           [Timestamp('2020-12-15 18:00:00'), 1, -0.9781476007338057,
            0.2079116908177593],
           [NaT, 0, nan, nan]], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_dt import CyclicDayOfMonth
    >>> X = ks.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicDayOfMonth(columns=['A'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, 1.0, 0.0],
           [Timestamp('2020-12-15 18:00:00'), 1, -0.9781476007338057,
            0.2079116908177593],
           [NaT, 0, nan, nan]], dtype=object)


    """

    def __init__(self, columns: List[str]):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        column_names = self.get_cyclic_column_names(columns, "day_of_month")
        column_mapping = {
            name: col for name, col in zip(column_names, columns + columns)
        }
        _BaseDatetimeFeature.__init__(self, columns, column_names, column_mapping)

    def transform(self, X: DataFrame) -> DataFrame:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.

        Returns
        -------
        DataFrame
            Transformed dataframe.
        """
        self.check_dataframe(X)
        return self.compute_cyclic_day_of_month(X)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the NumPy array `X`.

        Parameters
        ----------
        X : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Transformed array.
        """

        self.check_array(X)
        return feature_gen_dt.cyclic_day_of_month(X, self.idx_columns)

    def compute_cyclic_day_of_month(self, X: DataFrame) -> DataFrame:
        """Compute the cyclic day of the month features.

        Parameters
        ----------
        X : DataFrame
            Dataframe of datetime columns.

        Returns
        -------
        DataFrame
            Dataframe of cyclic day of the month features.
        """

        def f_cos(x):
            day_of_month = x.dt.day - 1
            n_days_in_month = x.dt.daysinmonth - 1
            prefactors = 2 * np.pi / n_days_in_month
            return np.cos(prefactors * day_of_month)

        def f_sin(x):
            day_of_month = x.dt.day - 1
            n_days_in_month = x.dt.daysinmonth - 1
            prefactors = 2 * np.pi / n_days_in_month
            return np.sin(prefactors * day_of_month)

        X_cos = util.get_function(X).apply(X[self.columns], f_cos)
        X_sin = util.get_function(X).apply(X[self.columns], f_sin)
        X_cos.columns = self.column_names[::2]
        X_sin.columns = self.column_names[1::2]
        X_new = X_cos.join(X_sin)
        return X.join(X_new[self.column_names])
