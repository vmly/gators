# Licence Apache-2.0

from typing import List, TypeVar

import numpy as np

import feature_gen_dt

from ..util import util
from ._base_datetime_feature import _BaseDatetimeFeature

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class OrdinalMonthOfYear(_BaseDatetimeFeature):
    """Create new columns based on the month of the year.

    Parameters
    ----------
    columns : List[str]
        List of columns.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation_dt import OrdinalMonthOfYear
    >>> X = pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = OrdinalMonthOfYear(columns=['A'])
    >>> obj.fit_transform(X)
                        A  B A__month_of_year
    0 2020-01-01 23:00:00  0              1.0
    1 2020-12-15 18:00:00  1             12.0
    2                 NaT  0              nan

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_dt import OrdinalMonthOfYear
    >>> X = ks.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = OrdinalMonthOfYear(columns=['A'])
    >>> obj.fit_transform(X)
                        A  B A__month_of_year
    0 2020-01-01 23:00:00  0              1.0
    1 2020-12-15 18:00:00  1             12.0
    2                 NaT  0              nan

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation_dt import OrdinalMonthOfYear
    >>> X = pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = OrdinalMonthOfYear(columns=['A'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, '1.0'],
           [Timestamp('2020-12-15 18:00:00'), 1, '12.0'],
           [NaT, 0, 'nan']], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_dt import OrdinalMonthOfYear
    >>> X = ks.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = OrdinalMonthOfYear(columns=['A'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, '1.0'],
           [Timestamp('2020-12-15 18:00:00'), 1, '12.0'],
           [NaT, 0, 'nan']], dtype=object)


    """

    def __init__(self, columns: List[str]):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        column_names = [f"{c}__month_of_year" for c in columns]
        column_mapping = dict(zip(column_names, columns))
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

        for name, col in zip(self.column_names, self.columns):
            X[name] = X[col].dt.month.astype(np.float64).astype(str)

        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array X.

        Parameters
        ----------
        X : np.ndarray
            Input array.

        Returns
        -------
            np.ndarray: Dataset with the Tree features.
        """
        self.check_array(X)
        return feature_gen_dt.ordinal_month_of_year(X, self.idx_columns)
