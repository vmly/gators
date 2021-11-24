# License: Apache-2.0
import warnings
from typing import TypeVar

import numpy as np

from encoder import onehot_encoder

from ..util import util
from ._base_encoder import _BaseEncoder
from .ordinal_encoder import OrdinalEncoder

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class OneHotEncoder(_BaseEncoder):
    """Encode the categorical columns as one-hot numeric columns.

    Parameters
    ----------
    dtype : type, default to np.float64.
        Numerical datatype of the output data.

    Examples
    ---------

    Imports and initialization:

    >>> from gators.encoders import OneHotEncoder
    >>> obj = OneHotEncoder()

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes,

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']}), npartitions=1)

    * `koalas` dataframes,

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
       A__b  A__a  B__d  B__c
    0   0.0   1.0   0.0   1.0
    1   0.0   1.0   1.0   0.0
    2   1.0   0.0   1.0   0.0

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([[0., 1., 0., 1.],
           [0., 1., 1., 0.],
           [1., 0., 1., 0.]])
    """

    def __init__(self, dtype: type = np.float64):
        _BaseEncoder.__init__(self, dtype=dtype)
        self.ordinal_encoder = OrdinalEncoder(dtype=dtype, add_other_columns=False)
        self.idx_numerical_columns = np.array([])
        self.onehot_columns = []
        self.numerical_columns = []
        self.column_mapping = {}

    def fit(self, X: DataFrame, y: Series = None) -> "OneHotEncoder":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default to None.
            Target values.

        Returns
        -------
        OneHotEncoder: Instance of itself.
        """
        self.check_dataframe(X)
        # self.check_nans(X, self.columns)
        self.columns = util.get_datatype_columns(X, object)
        columns = list(X.columns)
        if not self.columns:
            warnings.warn(
                f"""`X` does not contain object columns:
                `{self.__class__.__name__}` is not needed"""
            )
            return self
        self.onehot_columns = list(
            util.get_function(X).get_dummies(
                X[self.columns], self.columns, prefix_sep="__"
            )
        )
        self.onehot_columns = sorted(self.onehot_columns)
        object_columns = [
            "__".join(col.split("__")[:-1]) for col in self.onehot_columns
        ]
        self.idx_columns = util.get_idx_columns(X, object_columns)
        self.idx_columns_to_keep = [
            i
            for i, col in enumerate(columns + self.onehot_columns)
            if col not in self.columns
        ]

        self.cats = np.array(
            [col.split("__")[-1] for col in self.onehot_columns]
        ).astype(object)
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
        if not self.columns:
            return X
        for onehot_col in self.onehot_columns:
            dummy = onehot_col.split("__")
            col, cat = "__".join(dummy[:-1]), dummy[-1]
            X[onehot_col] = X[col] == cat
        return X.drop(self.columns, axis=1).astype(self.dtype)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the input array.

        Parameters
        ----------
        X  : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray: Encoded array.
        """
        self.check_array(X)
        if self.idx_columns.size == 0:
            return X
        return onehot_encoder(X, self.idx_columns, self.cats)[
            :, self.idx_columns_to_keep
        ].astype(self.dtype)
