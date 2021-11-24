# License: Apache-2.0
from typing import List, TypeVar

import numpy as np

from feature_gen_str import lower_case

from ..util import util
from ._base_string_feature import _BaseStringFeature

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class LowerCase(_BaseStringFeature):
    """Convert the selected columns to lower case.

    Parameters
    ----------
    columns : List[str]
        List of columns.

    Examples
    ---------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation_str import LowerCase
    >>> X = pd.DataFrame({'A': ['abC', 'Ab', ''], 'B': ['ABc', 'aB', None]})
    >>> obj = LowerCase(columns=['A', 'B'])
    >>> obj.fit_transform(X)
         A     B
    0  abc   abc
    1   ab    ab
    2       None

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_str import LowerCase
    >>> X = ks.DataFrame({'A': ['abC', 'Ab', ''], 'B': ['ABc', 'aB', None]})
    >>> obj = LowerCase(columns=['A', 'B'])
    >>> obj.fit_transform(X)
         A     B
    0  abc   abc
    1   ab    ab
    2       None

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation_str import LowerCase
    >>> X = pd.DataFrame({'A': ['abC', 'Ab', ''], 'B': ['ABc', 'aB', None]})
    >>> obj = LowerCase(columns=['A', 'B'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['abc', 'abc'],
           ['ab', 'ab'],
           ['', None]], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_str import LowerCase
    >>> X = ks.DataFrame({'A': ['abC', 'Ab', ''], 'B': ['ABc', 'aB', None]})
    >>> obj = LowerCase(columns=['A', 'B'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['abc', 'abc'],
           ['ab', 'ab'],
           ['', None]], dtype=object)


    """

    def __init__(self, columns: List[str]):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        self.columns = columns

    def fit(self, X: DataFrame, y: Series = None) -> "LowerCase":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default to None.
            Target values.

        Returns
        -------
        LowerCase
            Instance of itself.
        """
        self.check_dataframe(X)
        self.idx_columns = util.get_idx_columns(
            columns=X.columns, selected_columns=self.columns
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

        for col in self.columns:
            X[col] = X[col].astype(str).str.lower().replace({"none": None})
        return X

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
        return lower_case(X, self.idx_columns)
