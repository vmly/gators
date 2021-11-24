# License: Apache-2.0
from typing import TypeVar

import numpy as np

from scaler import minmax_scaler

from ..transformers.transformer import Transformer
from ..util import util

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class MinMaxScaler(Transformer):
    """Scale each column to the [0, 1] range.

    Parameters
    ----------
    dtype : type, default to np.float64.
        Numerical datatype of the output data.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.scalers import MinMaxScaler
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = MinMaxScaler()
    >>> obj.fit_transform(X)
         A     B
    0  0.0  0.00
    1  0.5  0.75
    2  1.0  1.00

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.scalers import MinMaxScaler
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = MinMaxScaler()
    >>> obj.fit_transform(X)
         A     B
    0  0.0  0.00
    1  0.5  0.75
    2  1.0  1.00

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.scalers import MinMaxScaler
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = MinMaxScaler()
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[0.  , 0.  ],
           [0.5 , 0.75],
           [1.  , 1.  ]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.scalers import MinMaxScaler
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = MinMaxScaler()
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[0.  , 0.  ],
           [0.5 , 0.75],
           [1.  , 1.  ]])

    """

    def __init__(self, dtype: type = np.float64):
        self.dtype = dtype
        self.X_min: DataFrame = None
        self.X_max: DataFrame = None
        self.X_min_np = np.array([])
        self.X_max_np = np.array([])

    def fit(self, X: DataFrame, y: Series = None) -> "MinMaxScaler":
        """Fit the transformer on the pandas/koalas dataframe X.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default to None.
            Labels.

        Returns
        -------
            'MinMaxScaler': Instance of itself.
        """
        self.check_dataframe(X)
        self.check_dataframe_is_numerics(X)
        self.columns = list(X.columns)
        self.X_min = util.get_function(X).to_pandas(X.min()).astype(self.dtype)
        self.X_max = util.get_function(X).to_pandas(X.max()).astype(self.dtype)
        self.X_min_np = util.get_function(self.X_min).to_numpy(self.X_min)
        self.X_max_np = util.get_function(self.X_max).to_numpy(self.X_max)
        self.X_min = self.X_min.to_dict()
        self.X_max = self.X_max.to_dict()
        return self

    def transform(self, X):
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
        self.check_dataframe_is_numerics(X)

        for col in self.columns:
            X[col] = (X[col] - self.X_min[col]) / (self.X_max[col] - self.X_min[col])
        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array X.

        Parameters
        ----------
        X (np.ndarray): Input ndarray.

        Returns
        -------
            np.ndarray: Imputed ndarray.
        """
        self.check_array(X)
        return minmax_scaler(X.astype(self.dtype), self.X_min_np, self.X_max_np)
