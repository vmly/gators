# License: Apache-2.0
from typing import TypeVar

import databricks.koalas as ks
import numpy as np

from scaler import standard_scaler

from ..transformers.transformer import Transformer
from ..util import util

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class StandardScaler(Transformer):
    """Scale each column by setting the mean to 0 and the standard deviation to 1.



    Parameters
    ----------
    dtype : type, default to np.float64.
        Numerical datatype of the output data.

    Examples
    --------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.scalers import StandardScaler
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = StandardScaler()
    >>> obj.fit_transform(X)
         A         B
    0 -1.0 -1.120897
    1  0.0  0.320256
    2  1.0  0.800641

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.scalers import StandardScaler
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = StandardScaler()
    >>> obj.fit_transform(X)
         A         B
    0 -1.0 -1.120897
    1  0.0  0.320256
    2  1.0  0.800641

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.scalers import StandardScaler
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = StandardScaler()
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[-1.        , -1.12089708],
           [ 0.        ,  0.32025631],
           [ 1.        ,  0.80064077]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.scalers import StandardScaler
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = StandardScaler()
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[-1.        , -1.12089708],
           [ 0.        ,  0.32025631],
           [ 1.        ,  0.80064077]])

    """

    def __init__(self, dtype: type = np.float64):
        self.dtype = dtype
        self.X_mean: DataFrame = None
        self.X_std: DataFrame = None
        self.X_mean_np = np.array([])
        self.X_std_np = np.array([])

    def fit(self, X: DataFrame, y: Series = None) -> "StandardScaler":
        """Fit the transformer on the pandas/koalas dataframe X.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default to None.
            Target values.

        Returns
        -------
            'StandardScaler': Instance of itself.
        """
        self.check_dataframe(X)
        self.check_dataframe_is_numerics(X)
        self.X_mean = util.get_function(X).to_pandas(X.mean()).astype(self.dtype)
        self.X_std = util.get_function(X).to_pandas(X.std()).astype(self.dtype)
        self.X_mean_np = util.get_function(self.X_mean).to_numpy(self.X_mean)
        self.X_std_np = util.get_function(self.X_std).to_numpy(self.X_std)
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

        def f(x, X_mean, X_std):
            c = x.name
            return (x - X_mean[c]) / X_std[c]

        return util.get_function(X).apply(
            X.astype(self.dtype), f, args=(self.X_mean, self.X_std)
        )

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the numpy ndarray X.

        Parameters
        ----------
        X (np.ndarray): Input ndarray.

        Returns
        -------
            np.ndarray: Imputed ndarray.
        """
        self.check_array(X)
        self.check_array_is_numerics(X)
        return standard_scaler(X.astype(self.dtype), self.X_mean_np, self.X_std_np)
