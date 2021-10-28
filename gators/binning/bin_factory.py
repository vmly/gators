# License: Apache-2.0
from abc import ABC
from typing import List, TypeVar

import pandas as pd
import numpy as np
from ..util import util

EPSILON = 1e-10

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class BinFactory(ABC):
    def bin(self) -> DataFrame:
        """Retun a dataframe with the new binned columns."""
        
    def bin_inplace(self) -> DataFrame:
        """Retun a dataframe with binned columns."""

class BinPandas(BinFactory):
    def bin(self, X: DataFrame , bins: pd.DataFrame, columns: List[str], output_columns: List[str]):
        def f(x, bins):
            name = x.name
            return (
                pd.cut(
                    x,
                    bins=bins[name],
                    labels=np.arange(len(bins[name]) - 1),
                    duplicates="drop",
                )
                .fillna(0)
                .astype(float)
                .astype(str)
            )

        return X.join(
            X[columns]
            .apply(f, args=(bins,))
            .astype(object)
            .rename(columns=dict(zip(columns, output_columns)))
        )

    def bin_inplace(self, X: DataFrame , bins: pd.DataFrame, columns: List[str], output_columns: List[str]):
        def f(x, bins, columns):
            name = x.name
            if name not in columns:
                return x
            return (
                pd.cut(
                    x,
                    bins=bins[name],
                    labels=np.arange(len(bins[name]) - 1),
                    duplicates="drop",
                )
                .fillna(0)
                .astype(float)
                .astype(str)
            )

        return util.get_function(X).apply(X, f, args=(bins, columns))


class BinKoalas(BinFactory):
    def bin(self, X: DataFrame , bins: pd.DataFrame, columns: List[str], output_columns: List[str]):
        from pyspark.ml.feature import Bucketizer

        bins_np = [np.unique(b) + EPSILON for b in bins.values()]
        X = (
            Bucketizer(
                splitsArray=bins_np, inputCols=columns, outputCols=output_columns
            )
            .transform(X.to_spark())
            .to_koalas()
        )
        X[output_columns] = X[output_columns].astype(str)
        return X

    def bin_inplace(self, X: DataFrame , bins: pd.DataFrame, columns: List[str], output_columns: List[str]):
        from pyspark.ml.feature import Bucketizer

        bins_np = [np.unique(b) + EPSILON for b in bins.values()]
        cols = X.columns  # used to ensure column order
        X = (
            Bucketizer(
                splitsArray=bins_np, inputCols=columns, outputCols=output_columns
            )
            .transform(X.to_spark())
            .to_koalas()
            .drop(columns, axis=1)
            .rename(columns=dict(zip(output_columns, columns)))
        )
        X[columns] = X[columns].astype(str)
        return X[cols]


class BinDask(BinFactory):
    def bin(self, X: DataFrame , bins: pd.DataFrame, columns: List[str], output_columns: List[str]):
        def f(x, bins):
            name = x.name
            return (
                pd.cut(
                    x,
                    bins=bins[name],
                    labels=np.arange(len(bins[name]) - 1),
                    duplicates="drop",
                )
                .fillna(0)
                .astype(float)
                .astype(str)
            )

        return X.join(
            util.get_function(X)
            .apply(X[columns], f, args=(bins,))
            .rename(columns=dict(zip(columns, output_columns)))
        )

    def bin_inplace(self, X: DataFrame , bins: pd.DataFrame, columns: List[str], output_columns: List[str]):
        def f(x, bins, columns):
            name = x.name
            if name not in columns:
                return x
            return (
                pd.cut(
                    x,
                    bins=bins[name],
                    labels=np.arange(len(bins[name]) - 1),
                    duplicates="drop",
                )
                .fillna(0)
                .astype(float)
                .astype(str)
            )

        return util.get_function(X).apply(X, f, args=(bins, columns))


def get_bin(X):
    factories = {
        "<class 'pandas.core.frame.DataFrame'>": BinPandas(),
        "<class 'databricks.koalas.frame.DataFrame'>": BinKoalas(),
        "<class 'dask.dataframe.core.DataFrame'>": BinDask(),
    }
    return factories[str(type(X))]
