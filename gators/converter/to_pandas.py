from typing import Tuple, TypeVar

import numpy as np

from ..transformers.transformer_xy import TransformerXY
from ..util import util

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class ToPandas(TransformerXY):
    """Convert dataframe and series to a pandas dataframe and series.

    Examples
    ---------
    * transform with pandas

    >>> import databricks.koalas as ks
    >>> from gators.converter import ToPandas
    >>> X = ks.DataFrame({
    ... 'q': {0: 0.0, 1: 3.0, 2: 6.0},
    ... 'w': {0: 1.0, 1: 4.0, 2: 7.0},
    ... 'e': {0: 2.0, 1: 5.0, 2: 8.0}})
    >>> y = ks.Series([0, 0, 1], name='TARGET')
    >>> obj = ToPandas()
    >>> X, y = obj.transform(X, y)
    >>> X
         q    w    e
    0  0.0  1.0  2.0
    1  3.0  4.0  5.0
    2  6.0  7.0  8.0
    >>> y
    0    0
    1    0
    2    1
    Name: TARGET, dtype: int64
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
            Input dataframe.
        y : [pd.Series, ks.Series]:
            Target values.

        Returns
        -------
        X : pd.DataFrame
            Dataframe.
        y : np.ndarray
            Target values.
        """
        self.check_dataframe(X)
        self.check_y(X, y)
        return util.get_function(X).to_pandas(X), util.get_function(X).to_pandas(y)
