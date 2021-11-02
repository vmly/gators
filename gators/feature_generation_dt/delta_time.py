# Licence Apache-2.0

from typing import List, TypeVar

import numpy as np

import feature_gen_dt

from ..transformers import Transformer
from ..util import util

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


# class ComputerPandas(ComputerFactory):
#     def compute(self, X, column_names, columns_a, columns_b, deltatime_dtype):
#         for name, c_a, c_b in zip(column_names, columns_a, columns_b):
#             X[name] = (X[c_a] - X[c_b])
#         return X


# class ComputerKoalas(ComputerFactory):
#     def compute(self, X, column_names, columns_a, columns_b, deltatime_dtype):
#         for name, c_a, c_b in zip(column_names, columns_a, columns_b):
#             X = X.assign(dummy=(X[c_a].astype(float) - X[c_b].astype(float))).rename(
#                 columns={"dummy": name}
#             )


class DeltaTime(Transformer):
    """Create new columns based on the time difference in sec. between two columns.

    Parameters
    ----------
    columns : List[str]
        List of columns.

    Examples
    ---------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation_dt import DeltaTime
    >>> X = pd.DataFrame({
    ... 'A': ['2020-01-01T23', '2020-01-15T18', pd.NaT],
    ... 'B': [0, 1, 0],
    ... 'C': ['2020-01-02T05', '2020-01-15T23', pd.NaT]})
    >>> obj = DeltaTime(columns_a=['C'], columns_b=['A'])
    >>> obj.fit_transform(X)
                        A  B                   C  C__A__Deltatime[s]
    0 2020-01-01 23:00:00  0 2020-01-02 05:00:00             21600.0
    1 2020-01-15 18:00:00  1 2020-01-15 23:00:00             18000.0
    2                 NaT  0                 NaT                 NaN

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_dt import DeltaTime
    >>> X = ks.DataFrame({
    ... 'A': ['2020-01-01T23', '2020-01-15T18', pd.NaT],
    ... 'B': [0, 1, 0],
    ... 'C': ['2020-01-02T05', '2020-01-15T23', pd.NaT]})
    >>> obj = DeltaTime(columns_a=['C'], columns_b=['A'])
    >>> obj.fit_transform(X)
                        A  B                   C  C__A__Deltatime[s]
    0 2020-01-01 23:00:00  0 2020-01-02 05:00:00             21600.0
    1 2020-01-15 18:00:00  1 2020-01-15 23:00:00             18000.0
    2                 NaT  0                 NaT                 NaN

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation_dt import DeltaTime
    >>> X = pd.DataFrame({
    ... 'A': ['2020-01-01T23', '2020-01-15T18', pd.NaT],
    ... 'B': [0, 1, 0],
    ... 'C': ['2020-01-02T05', '2020-01-15T23', pd.NaT]})
    >>> obj = DeltaTime(columns_a=['C'], columns_b=['A'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0,
            Timestamp('2020-01-02 05:00:00'), 21600.0],
           [Timestamp('2020-01-15 18:00:00'), 1,
            Timestamp('2020-01-15 23:00:00'), 18000.0],
           [NaT, 0, NaT, nan]], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_dt import DeltaTime
    >>> X = ks.DataFrame({
    ... 'A': ['2020-01-01T23', '2020-01-15T18', pd.NaT],
    ... 'B': [0, 1, 0],
    ... 'C': ['2020-01-02T05', '2020-01-15T23', pd.NaT]})
    >>> obj = DeltaTime(columns_a=['C'], columns_b=['A'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0,
            Timestamp('2020-01-02 05:00:00'), 21600.0],
           [Timestamp('2020-01-15 18:00:00'), 1,
            Timestamp('2020-01-15 23:00:00'), 18000.0],
           [NaT, 0, NaT, nan]], dtype=object)
    """

    def __init__(self, columns_a: List[str], columns_b: List[str]):
        Transformer.__init__(self)
        if not isinstance(columns_a, list):
            raise TypeError("`columns_a` should be a list.")
        if not columns_a:
            raise ValueError("`columns_a` should not be empty.")
        if not isinstance(columns_b, list):
            raise TypeError("`columns_b` should be a list.")
        if not columns_b:
            raise ValueError("`columns_b` should not be empty.")
        if len(columns_b) != len(columns_a):
            raise ValueError("`columns_a` and `columns_b` should have the same length.")
        self.unit = "s"
        self.columns_a = columns_a
        self.columns_b = columns_b
        self.deltatime_dtype = f"timedelta64[{self.unit}]"
        self.column_names = [
            f"{c_a}__{c_b}__Deltatime[{self.unit}]"
            for c_a, c_b in zip(columns_a, columns_b)
        ]
        self.column_mapping = {
            name: [c_a, c_b]
            for name, c_a, c_b in zip(self.column_names, columns_a, columns_b)
        }

    def fit(self, X: DataFrame, y: Series = None) -> "DeltaTime":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe.
        y : Series, default to None.
            Target values.

        Returns
        -------
        DeltaTime
            Instance of itself.
        """
        self.check_dataframe(X)
        columns = list(set(self.columns_a + self.columns_b))
        X_datetime_dtype = X.dtypes
        for column in columns:
            if not np.issubdtype(X_datetime_dtype[column], np.datetime64):
                raise TypeError(
                    """
                    Datetime columns should be of subtype np.datetime64.
                    Use `ConvertColumnDatatype` to convert the dtype.
                """
                )
        self.idx_columns_a = util.get_idx_columns(
            columns=X.columns,
            selected_columns=self.columns_a,
        )
        self.idx_columns_b = util.get_idx_columns(
            columns=X.columns,
            selected_columns=self.columns_b,
        )
        return self

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
        X = util.get_function(X).delta_time(
            X, self.column_names, self.columns_a, self.columns_b, self.deltatime_dtype
        )
        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array X.

        Parameters
        ----------
        X : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray:
            Array with the datetime features added.
        """
        self.check_array(X)
        return feature_gen_dt.deltatime(X, self.idx_columns_a, self.idx_columns_b)
