# License: Apache-2.0
from abc import ABC
from typing import List, TypeVar

import numpy as np

from binning import discretizer, discretizer_inplace

from ..transformers.transformer import Transformer
from ..util import util
from .bin_factory import get_bin

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class _BaseDiscretizer(Transformer):
    """Base discretizer transformer class.

    Parameters
    ----------
    n_bins : int
        Number of bins to use.
    inplace : bool
        If False, return the dataframe with the new discretized columns
        with the names '`column_name`__bin'). Otherwise,
        return the dataframe with the existing binned columns.

    """

    def __init__(self, n_bins: int, inplace: bool):
        if (not isinstance(n_bins, int)) or (n_bins <= 0):
            raise TypeError("`n_bins` should be a positive int.")
        if not isinstance(inplace, bool):
            raise TypeError("`inplace` should be a bool.")
        Transformer.__init__(self)
        self.n_bins = n_bins
        self.inplace = inplace
        self.columns = []
        self.output_columns = []
        self.idx_columns = np.array([])
        self.bins = {}
        self.bins_np = np.array([])

    def fit(self, X: DataFrame, y=None) -> "_BaseDiscretizer":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default to None.
            Labels.

        Returns
        -------
        'Discretizer'
            Instance of itself.
        """
        self.check_dataframe(X)
        self.columns = util.get_numerical_columns(X)
        self.output_columns = [f"{c}__bin" for c in self.columns]
        self.idx_columns = util.get_idx_columns(X.columns, self.columns)
        if self.idx_columns.size == 0:
            return self

        self.bins, self.bins_np = self.compute_bins(X[self.columns], self.n_bins)
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.

        Returns
        -------
        DataFrame
            Transformed dataframe.
        """
        bin = get_bin(X)
        self.check_dataframe(X)
        if self.idx_columns.size == 0:
            return X
        if self.inplace:
            return bin.bin_inplace(X, self.bins, self.columns, self.output_columns)
        return bin.bin(X, self.bins, self.columns, self.output_columns)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the NumPy array.

        Parameters
        ----------
        X : np.ndarray
            NumPy array.

        Returns
        -------
        np.ndarray
            Transformed NumPy array.
        """
        self.check_array(X)
        if self.idx_columns.size == 0:
            return X
        if self.inplace:
            if X.dtype == object:
                return discretizer_inplace(X, self.bins_np, self.idx_columns)
            return discretizer_inplace(X.astype(object), self.bins_np, self.idx_columns)
        if X.dtype == object:
            return discretizer(X, self.bins_np, self.idx_columns)
        return discretizer(X.astype(object), self.bins_np, self.idx_columns)
