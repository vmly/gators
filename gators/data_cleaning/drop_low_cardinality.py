# License: Apache-2.0
from typing import List, TypeVar

from ..util import util
from ._base_data_cleaning import _BaseDataCleaning

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class DropLowCardinality(_BaseDataCleaning):
    """Drop the catgorical columns having a low cardinality.

    Parameters
    ----------
    min_categories : int
        Min categories allowed.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.data_cleaning import DropLowCardinality
    >>> X = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'd']})
    >>> obj = DropLowCardinality(min_categories=2)
    >>> obj.fit_transform(X)
       A
    0  a
    1  b
    2  c

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.data_cleaning import DropLowCardinality
    >>> X = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'd']})
    >>> obj = DropLowCardinality(min_categories=2)
    >>> obj.fit_transform(X)
       A
    0  a
    1  b
    2  c

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.data_cleaning import DropLowCardinality
    >>> X = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'd']})
    >>> obj = DropLowCardinality(min_categories=2)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['a'],
           ['b'],
           ['c']], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.data_cleaning import DropLowCardinality
    >>> X = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'd']})
    >>> obj = DropLowCardinality(min_categories=2)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['a'],
           ['b'],
           ['c']], dtype=object)

    """

    def __init__(self, min_categories: int):
        if not isinstance(min_categories, int):
            raise TypeError("`min_categories` should be an int.")
        _BaseDataCleaning.__init__(self)
        self.min_categories: int = min_categories

    def fit(self, X: DataFrame, y=None) -> "DropLowCardinality":
        """Fit the transformer on the dataframe X.

        Get the list of column names to remove and the array of
           indices to be kept.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : None
           None
        Returns
        -------
        DropLowCardinality: Instance of itself.
        """
        self.check_dataframe(X)
        self.columns = self.get_columns_to_drop(X=X, min_categories=self.min_categories)
        self.columns_to_keep = util.exclude_columns(
            columns=X.columns,
            excluded_columns=self.columns,
        )
        self.idx_columns_to_keep = self.get_idx_columns_to_keep(
            columns=X.columns, columns_to_drop=self.columns
        )
        return self

    @staticmethod
    def get_columns_to_drop(X: DataFrame, min_categories: float) -> List[str]:
        """Get the list of column names to remove.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset.
        min_categories : int
            Min categories allowed.

        Returns
        -------
        List[str]
            List of column names to drop.
        """
        object_columns = util.get_datatype_columns(X, "object")
        if not object_columns:
            return []

        def f(x):
            return x.nunique()

        X_nunique = util.get_function(X).apply_to_pandas(X, f)
        mask_columns = X_nunique < min_categories
        columns_to_drop = mask_columns[mask_columns].index
        return list(columns_to_drop)
