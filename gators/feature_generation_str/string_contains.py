# License: Apache-2.0
from typing import List, TypeVar

import databricks.koalas as ks
import numpy as np
import pandas as pd

from feature_gen_str import contains

from ._base_string_feature import _BaseStringFeature
from ..util import util

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class StringContains(_BaseStringFeature):
    """Create new binary columns.

    The value is 1 if the element contains the given substring and 0 otherwise.

    Parameters
    ----------
    columns : List[str]
        List of columns.
    contains_vec : List[int]
        List of substrings.
    column_names : List[int], default to None.
        List new column names.

    Examples
    ---------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation_str import StringContains
    >>> X = pd.DataFrame({'A': ['qwe', 'qwd', 'zwe'], 'B': [1, 2, 3]})
    >>> obj = StringContains(columns=['A', 'A'], contains_vec=['qw', 'we'])
    >>> obj.fit_transform(X)
         A  B  A__contains_qw  A__contains_we
    0  qwe  1             1.0             1.0
    1  qwd  2             1.0             0.0
    2  zwe  3             0.0             1.0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_str import StringLength
    >>> X = ks.DataFrame({'A': ['qwe', 'asd', 'zxc'], 'B': [1, 2, 3]})
    >>> obj = StringContains(columns=['A', 'A'], contains_vec=['qw', 'we'])
    >>> obj.fit_transform(X)
         A  B  A__contains_qw  A__contains_we
    0  qwe  1             1.0             1.0
    1  asd  2             0.0             0.0
    2  zxc  3             0.0             0.0

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation_str import StringLength
    >>> X = ks.DataFrame({'A': ['qwe', 'asd', 'zxc'], 'B': [1, 2, 3]})
    >>> obj = StringContains(columns=['A', 'A'], contains_vec=['qw', 'we'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['qwe', 1, 1.0, 1.0],
           ['asd', 2, 0.0, 0.0],
           ['zxc', 3, 0.0, 0.0]], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_str import StringLength
    >>> X = ks.DataFrame({'A': ['qwe', 'asd', 'zxc'], 'B': [1, 2, 3]})
    >>> obj = StringContains(columns=['A', 'A'], contains_vec=['qw', 'we'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['qwe', 1, 1.0, 1.0],
           ['asd', 2, 0.0, 0.0],
           ['zxc', 3, 0.0, 0.0]], dtype=object)

    """

    def __init__(
        self,
        columns: List[str],
        contains_vec: List[str],
        column_names: List[str] = None,
    ):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")
        if not isinstance(contains_vec, list):
            raise TypeError("`contains_vec` should be a list.")
        if len(columns) != len(contains_vec):
            raise ValueError("Length of `columns` and `contains_vec` should match.")
        if not column_names:
            column_names = [
                f"{col}__contains_{val}" for col, val in zip(columns, contains_vec)
            ]
        _BaseStringFeature.__init__(self, columns, column_names)
        self.contains_vec = np.array(contains_vec, str).astype(object)

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

        def f(x, contain_dict):
            return x.str.contains(contain_dict[x.name], regex=False).astype(np.float64)

        X_new = X[self.columns]
        X_new.columns = self.column_names
        contain_dict = dict(zip(self.column_names, self.contains_vec))
        meta = pd.DataFrame(columns=self.column_names, dtype=np.float64)
        X_new = util.get_apply(X).apply(X_new, f, (contain_dict,), meta)
        return X.join(X_new)

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
        return contains(X, self.idx_columns, self.contains_vec)
