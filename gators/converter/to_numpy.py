from typing import Tuple, TypeVar

import numpy as np

from ..transformers.transformer_xy import TransformerXY
from ..util import util

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class ToNumpy(TransformerXY):
    """Convert dataframe and series to NumPy arrays.

    Examples
    ---------
    * transform with pandas

    >>> import pandas as pd
    >>> from gators.converter import ToNumpy
    >>> X = pd.DataFrame({
    ...     'q': [0.0, 3.0, 6.0],
    ...     'w': [1.0, 4.0, 7.0],
    ...     'e': [2.0, 5.0, 8.0]})
    >>> y = pd.Series([0, 0, 1], name='TARGET')
    >>> obj = ToNumpy()
    >>> X, y = obj.transform(X, y)
    >>> X
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])
    >>> y
    array([0, 0, 1])

    * transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.converter.to_numpy import ToNumpy
    >>> X = ks.DataFrame({
    ...     'q': [0.0, 3.0, 6.0],
    ...     'w': [1.0, 4.0, 7.0],
    ...     'e': [2.0, 5.0, 8.0]})
    >>> y = ks.Series([0, 0, 1], name='TARGET')
    >>> obj = ToNumpy()
    >>> X, y = obj.transform(X, y)
    >>> X
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])
    >>> y
    array([0, 0, 1])

    """

    def __init__(self):
        TransformerXY.__init__(self)

    def transform(
        self,
        X: DataFrame,
        y: Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Dataframe.
        y : [pd.Series, ks.Series]:
            Target values.

        Returns
        -------
        X : np.ndarray
            Array.
        y : np.ndarray
            Target values.
        """
        self.check_dataframe(X)
        self.check_y(X, y)
        return util.get_function(X).to_numpy(X), util.get_function(X).to_numpy(y)
