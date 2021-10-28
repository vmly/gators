# Licence Apache-2.0
from math import pi
from typing import List, TypeVar

import numpy as np

import feature_gen_dt

from ..util import util
from ._base_datetime_feature import _BaseDatetimeFeature

PREFACTOR = 2 * pi / 6.0


DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class CyclicDayOfWeek(_BaseDatetimeFeature):
    """Create new columns based on the cyclic mapping of the day of the week.

    Parameters
    ----------
    columns : List[str]
        List of columns.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation_dt import CyclicDayOfWeek
    >>> X = pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicDayOfWeek(columns=['A'])
    >>> obj.fit_transform(X)
                        A  B  A__day_of_week_cos  A__day_of_week_sin
    0 2020-01-01 23:00:00  0                -0.5            0.866025
    1 2020-12-15 18:00:00  1                 0.5            0.866025
    2                 NaT  0                 NaN                 NaN

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_dt import CyclicDayOfWeek
    >>> X = ks.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicDayOfWeek(columns=['A'])
    >>> obj.fit_transform(X)
                        A  B  A__day_of_week_cos  A__day_of_week_sin
    0 2020-01-01 23:00:00  0                -0.5            0.866025
    1 2020-12-15 18:00:00  1                 0.5            0.866025
    2                 NaT  0                 NaN                 NaN

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation_dt import CyclicDayOfWeek
    >>> X = pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicDayOfWeek(columns=['A'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, -0.4999999999999998,
            0.8660254037844388],
           [Timestamp('2020-12-15 18:00:00'), 1, 0.5000000000000001,
            0.8660254037844386],
           [NaT, 0, nan, nan]], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_dt import CyclicDayOfWeek
    >>> X = ks.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicDayOfWeek(columns=['A'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, -0.4999999999999998,
            0.8660254037844388],
           [Timestamp('2020-12-15 18:00:00'), 1, 0.5000000000000001,
            0.8660254037844386],
           [NaT, 0, nan, nan]], dtype=object)
    """

    def __init__(self, columns: List[str]):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        column_names = self.get_cyclic_column_names(columns, "day_of_week")
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
        return self.compute_cyclic_day_of_week(X)

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
        return feature_gen_dt.cyclic_day_of_week(X, self.idx_columns, PREFACTOR)

    def compute_cyclic_day_of_week(self, X: DataFrame) -> DataFrame:
        """Compute the cyclic day of the week features.

        Parameters
        ----------
        X : DataFrame
            Dataframe of datetime columns.

        Returns
        -------
        DataFrame
            Dataframe of cyclic day of the week features.
        """

        def f_cos(x):
            return np.cos(PREFACTOR * x.dt.dayofweek)

        def f_sin(x):
            return np.sin(PREFACTOR * x.dt.dayofweek)

        X_cos = util.get_function(X).apply(X[self.columns], f_cos)
        X_sin = util.get_function(X).apply(X[self.columns], f_sin)
        X_cos.columns = self.column_names[::2]
        X_sin.columns = self.column_names[1::2]
        X_new = X_cos.join(X_sin)
        return X.join(X_new[self.column_names])
