# License: Apache-2.0
from abc import ABC
from typing import List, TypeVar

import numpy as np
import pandas as pd

from ..util import util

EPSILON = 1e-10

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class BinFactory(ABC):
    def bin(self) -> DataFrame:
        """Retun a dataframe with the new binned columns."""

    def bin_inplace(self) -> DataFrame:
        """Retun a dataframe with binned columns."""


class BinPandas(BinFactory):
    def bin(
        self,
        X: DataFrame,
        bins: pd.DataFrame,
        columns: List[str],
        output_columns: List[str],
    ):
        for name, col in zip(output_columns, columns):
            labels = [f"_{x}" for x in np.arange(len(bins[col]) - 1)]
            X[name] = (
                pd.cut(
                    X[col],
                    bins=bins[col],
                    labels=labels,
                    duplicates="drop",
                )
                .fillna("_0")
                .astype(object)
            )
        return X

    def bin_inplace(
        self,
        X: DataFrame,
        bins: pd.DataFrame,
        columns: List[str],
        output_columns: List[str],
    ):
        for name, col in zip(output_columns, columns):
            labels = [f"_{x}" for x in np.arange(len(bins[col]) - 1)]
            X[col] = (
                pd.cut(
                    X[col],
                    bins=bins[col],
                    labels=labels,
                    duplicates="drop",
                )
                .fillna("_0")
                .astype(object)
            )
        return X
        # def f(x, bins, columns):
        #     name = x.name
        #     if name not in columns:
        #         return x
        #     return (
        #         pd.cut(
        #             x,
        #             bins=bins[name],
        #             labels=[f'_{x}'for x in np.arange(len(bins[name]) - 1)],
        #             duplicates="drop",
        #         )
        #         .fillna('_0').astype(object)
        #     )

        # return util.get_function(X).apply(X, f, args=(bins, columns))


class BinKoalas(BinFactory):
    def bin(
        self,
        X: DataFrame,
        bins: pd.DataFrame,
        columns: List[str],
        output_columns: List[str],
    ):
        from pyspark.ml.feature import Bucketizer

        bins_np = [np.unique(b) + EPSILON for b in bins.values()]
        X = (
            Bucketizer(
                splitsArray=bins_np, inputCols=columns, outputCols=output_columns
            )
            .transform(X.to_spark())
            .to_koalas()
        )
        X[output_columns] = "_" + X[output_columns].astype(int).astype(str)
        return X

    def bin_inplace(
        self,
        X: DataFrame,
        bins: pd.DataFrame,
        columns: List[str],
        output_columns: List[str],
    ):
        from pyspark.ml.feature import Bucketizer

        bins_np = [np.unique(b) + EPSILON for b in bins.values()]
        ordered_columns = X.columns
        X = (
            Bucketizer(
                splitsArray=bins_np, inputCols=columns, outputCols=output_columns
            )
            .transform(X.to_spark())
            .to_koalas()
            .drop(columns, axis=1)
            .rename(columns=dict(zip(output_columns, columns)))
        )
        X[columns] = "_" + X[columns].astype(int).astype(str)
        return X[ordered_columns]


class BinDask(BinFactory):
    # def bin(
    #     self,
    #     X: DataFrame,
    #     bins: pd.DataFrame,
    #     columns: List[str],
    #     output_columns: List[str],
    # ):
    #     import dask.dataframe as dd
    #     for name, col in zip(output_columns, columns):
    #         labels = [f'_{x}'for x in np.arange(len(bins[col]) - 1)]
    #         X[name] = dd.cut(
    #                 X[col],
    #                 bins=bins[col],
    #                 labels=labels,
    #                 duplicates="drop",
    #             ).fillna('_0').astype(object)
    #     return X

    # def bin_inplace(
    #     self,
    #     X: DataFrame,
    #     bins: pd.DataFrame,
    #     columns: List[str],
    #     output_columns: List[str],
    # ):
    #     import dask.dataframe as dd
    #     for name, col in zip(output_columns, columns):
    #         labels = [f'_{x}'for x in np.arange(len(bins[col]) - 1)]
    #         X[col] = dd.cut(
    #                 X[col],
    #                 bins=bins[col],
    #                 labels=labels,
    #                 duplicates="drop",
    #             ).fillna('_0').astype(object)
    #     return X

    def bin(
        self,
        X: DataFrame,
        bins: pd.DataFrame,
        columns: List[str],
        output_columns: List[str],
    ):
        def f(x, bins):
            name = x.name
            return (
                pd.cut(
                    x,
                    bins=bins[name],
                    labels=[f"_{x}" for x in np.arange(len(bins[name]) - 1)],
                    duplicates="drop",
                )
                .fillna("_0")
                .astype(object)
            )

        return X.join(
            util.get_function(X)
            .apply(X[columns], f, args=(bins,))
            .rename(columns=dict(zip(columns, output_columns)))
        )

    def bin_inplace(
        self,
        X: DataFrame,
        bins: pd.DataFrame,
        columns: List[str],
        output_columns: List[str],
    ):
        def f(x, bins, columns):
            name = x.name
            if name not in columns:
                return x
            return (
                pd.cut(
                    x,
                    bins=bins[name],
                    labels=[f"_{x}" for x in np.arange(len(bins[name]) - 1)],
                    duplicates="drop",
                )
                .fillna("_0")
                .astype(object)
            )

        return util.get_function(X).apply(X, f, args=(bins, columns))


def get_bin(X):
    factories = {
        "<class 'pandas.core.frame.DataFrame'>": BinPandas(),
        "<class 'databricks.koalas.frame.DataFrame'>": BinKoalas(),
        "<class 'dask.dataframe.core.DataFrame'>": BinDask(),
    }
    return factories[str(type(X))]
