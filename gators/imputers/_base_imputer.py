# License: Apache-2.0
from abc import ABC, abstractmethod
from typing import Dict, List, TypeVar, Union

import numpy as np
import pandas as pd

from ..transformers.transformer import Transformer

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class ComputerFactory(ABC):
    @abstractmethod
    def compute_statistics_mean():
        pass

    @abstractmethod
    def compute_statistics_median():
        pass

    @abstractmethod
    def compute_statistics_most_frequent():
        pass

    @abstractmethod
    def transform():
        pass


class ComputerPandas(ComputerFactory):
    def compute_statistics_mean(self, X):
        return X.mean().to_dict()

    def compute_statistics_median(self, X):
        return X.median().to_dict()

    def compute_statistics_most_frequent(self, X):
        columns = list(X.columns)
        values = [X[c].value_counts().index.to_numpy()[0] for c in columns]
        return dict(zip(columns, values))

    def transform(self, X, statistics):
        return X.fillna(statistics)


class ComputerKoalas(ComputerFactory):
    def compute_statistics_mean(self, X):
        return X.mean().to_dict()

    def compute_statistics_median(self, X):
        return X.median().to_dict()

    def compute_statistics_most_frequent(self, X):
        columns = list(X.columns)
        values = [X[c].value_counts().index.to_numpy()[0] for c in columns]
        return dict(zip(columns, values))

    def transform(self, X, statistics):
        for col, val in statistics.items():
            X[col] = X[col].fillna(val)
        return X


class ComputerDask(ComputerFactory):
    def compute_statistics_mean(self, X):
        return X.mean().compute().to_dict()

    def compute_statistics_median(self, X):
        return X.median().compute().to_dict()

    def transform(self, X, statistics):
        return X.fillna(statistics)

    def compute_statistics_most_frequent(self, X):
        columns = list(X.columns)
        values = [X[c].value_counts().compute().index[0] for c in columns]
        return dict(zip(columns, values))


def get_computer(X):
    factories = {
        "<class 'pandas.core.frame.DataFrame'>": ComputerPandas(),
        "<class 'databricks.koalas.frame.DataFrame'>": ComputerKoalas(),
        "<class 'dask.dataframe.core.DataFrame'>": ComputerDask(),
    }
    return factories[str(type(X))]


class _BaseImputer(Transformer):
    """Base imputer transformer class.

    Parameters
    ----------
    strategy : str
        Imputation strategy. The possible values are:

        * constant
        * most_frequent (only for the FloatImputer class)
        * mean (only for the FloatImputer class)
        * median (only for the FloatImputer class)

    value (Union[float, str, None]): Imputation value, default to None.
        used for `strategy=constant`.
    columns: List[str], default to None.
        List of columns.
    """

    def __init__(
        self, strategy: str, value: Union[float, str, None], columns: List[str]
    ):
        if not isinstance(strategy, str):
            raise TypeError("`strategy` should be a string.")
        if strategy == "constant" and value is None:
            raise ValueError('if `strategy` is "constant", `value` should not be None.')
        if strategy not in ["constant", "mean", "median", "most_frequent"]:
            raise ValueError("Imputation `strategy` not implemented.")
        if not isinstance(columns, list) and columns is not None:
            raise TypeError("`columns` should be a list or None.")

        Transformer.__init__(self)
        self.strategy = strategy
        self.value = value
        self.columns = columns
        self.statistics: Dict = {}
        self.statistics_values: np.ndarray = None
        self.idx_columns: np.ndarray = None
        self.X_dtypes: Series = None

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
        if isinstance(X, pd.DataFrame):
            return X.fillna(self.statistics)
        for col, val in self.statistics.items():
            X[col] = X[col].fillna(val)
        return X

    def compute_statistics(
        self, X: DataFrame, value: Union[float, int, str, None]
    ) -> Dict[str, Union[float, int, str]]:
        """Compute the imputation values.

        Parameters
        ----------
        X : DataFrame
            Dataframe used to compute the imputation values.
        value : Union[float, int, str, None]
            Value used for imputation.

        Returns
        -------
        Dict[str, TypeVar[float, int, str]]
            Imputation value mapping.
        """
        if self.strategy == "mean":
            statistics = self.computer.compute_statistics_mean(X[self.columns])
        elif self.strategy == "median":
            statistics = self.computer.compute_statistics_median(X[self.columns])
        elif self.strategy == "most_frequent":
            statistics = self.computer.compute_statistics_most_frequent(X[self.columns])
        else:  # strategy == 'constant'
            values = len(self.columns) * [value]
            statistics = dict(zip(self.columns, values))
        if pd.Series(statistics).isnull().sum():
            raise ValueError(
                """Some columns contains only NaN values and the
                imputation values cannot be calculated.
                Remove these columns
                before performing the imputation
                (e.g. with `gators.data_cleaning.drop_high_nan_ratio()`)."""
            )
        return statistics
