# License: Apache-2.0
from abc import ABC, abstractmethod
from typing import List, TypeVar

import numpy as np
import pandas as pd

from ..util import util

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.Series, ks.Series, dd.Series]")

NUMERICS_DTYPES = [np.int16, np.int32, np.int64, np.float32, np.float64]
PRINT_NUMERICS_DTYPES = ", ".join([dtype.__name__ for dtype in NUMERICS_DTYPES])


class Transformer(ABC):
    """Abstract **gators** transformer class.

    Examples
    ---------

    * A Gators transformer.

    >>> import pandas as pd
    >>> import databricks.koalas as ks
    >>> import numpy as np
    >>> from gators.transformers import Transformer
    >>> class GetFirstColumn(Transformer):
    ...     def fit(self, X, y=None):
    ...         return self
    ...     def transform(self, X: pd.DataFrame):
    ...         return X[[X.columns[0]]]
    ...     def transform_numpy(self, X: np.ndarray):
    ...         return X[:, 0].reshape(-1, 1)

    * fit & transform with `pandas`

    >>> GetFirstColumn().fit_transform(
    ... pd.DataFrame({'A':[1, 2], 'B':[3, 4]}))
       A
    0  1
    1  2

    * fit with `pandas` & transform with `NumPy`

    >>> obj = GetFirstColumn().fit(
    ... pd.DataFrame({'A':[1, 2], 'B':[3, 4]}))
    >>> obj.transform_numpy(np.array([[5, 6], [7, 8]]))
    array([[5],
           [7]])

    * fit & transform with `koalas`

    >>> GetFirstColumn().fit_transform(
    ... ks.DataFrame({'A':[1, 2], 'B':[3, 4]}))
       A
    0  1
    1  2

    * fit with `koalas` & transform with `NumPy`

    >>> obj = GetFirstColumn().fit(
    ... ks.DataFrame({'A':[1, 2], 'B':[3, 4]}))
    >>> obj.transform_numpy(np.array([[5, 6], [7, 8]]))
    array([[5],
           [7]])
    """

    @abstractmethod
    def fit(self, X: DataFrame, y: Series = None) -> "Transformer":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default to None.
            Target values.

        Returns
        -------
            Transformer: Instance of itself.
        """

    @abstractmethod
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

    @abstractmethod
    def transform_numpy(self, X: Series, y: Series = None):
        """Transform the array X.

        Parameters
        ----------
        X : np.ndarray
            Array
        """

    def fit_transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Fit and Transform the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default to None.
            Input target.

        Returns
        -------
        DataFrame
            Transformed dataframe.
        """
        _ = self.fit(X, y)
        return self.transform(X)

    @staticmethod
    def check_dataframe(X: DataFrame):
        """Validate dataframe.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        """
        util.get_function(X)
        for c in X.columns:
            if not isinstance(c, str):
                raise TypeError("Column names of `X` should be of type str.")

    @staticmethod
    def check_y(X: DataFrame, y: Series):
        """Validate target.

        Parameters
        ----------
        X : DataFrame
            Dataframe
        y : Series
            Labels
        """
        util.get_function(X).raise_y_dtype_error(y)
        if not isinstance(y.name, str):
            raise TypeError("Name of `y` should be a str.")
        shape = util.get_function(X).shape
        if shape(X)[0] != shape(y)[0]:
            raise ValueError("Length of `X` and `y` should match.")

    @staticmethod
    def check_array(X: np.ndarray):
        """Validate array.

        Parameters
        ----------
            X (np.ndarray): Array.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("`X` should be a NumPy array.")

    @staticmethod
    def check_dataframe_is_numerics(X: DataFrame):
        """Check if dataframe is only numerics.

        Parameters
        ----------
        X : DataFrame
            Dataframe
        """
        X_dtypes = X.dtypes
        for x_dtype in X_dtypes:
            if x_dtype not in NUMERICS_DTYPES:
                raise ValueError(f"`X` should be of type {PRINT_NUMERICS_DTYPES}.")

    def check_datatype(self, dtype, accepted_dtypes):
        """Check if dataframe is only numerics.

        Parameters
        ----------
        X : DataFrame
            Dataframe
        """
        print_dtypes = ", ".join([dtype.__name__ for dtype in accepted_dtypes])
        if dtype not in accepted_dtypes:
            raise ValueError(
                f"""`X` should be of type {print_dtypes}.
                        Use gators.converter.ConvertColumnDatatype before
                        calling the transformer {self.__class__.__name__}."""
            )

    @staticmethod
    def check_binary_target(X: DataFrame, y: Series):
        """Raise an error if the target datatype is not binary.

        Parameters
        ----------
        y : Series
            Target values.
        """
        if util.get_function(X).nunique(y) != 2 or "int" not in str(y.dtype):
            raise ValueError("`y` should be binary.")

    @staticmethod
    def check_multiclass_target(y: Series):
        """Raise an error if the target datatype is not discrete.

        Parameters
        ----------
        y : Series
            Target values.
        """
        if "int" not in str(y.dtype):
            raise ValueError("`y` should be discrete.")

    @staticmethod
    def check_regression_target(y: Series):
        """Raise an error if the target datatype is not discrete.

        Parameters
        ----------
        y : Series
            Target values.
        """
        if "float" not in str(y.dtype):
            raise ValueError("`y` should be float.")

    @staticmethod
    def check_dataframe_contains_numerics(X: DataFrame):
        """Check if dataframe is only numerics.

        Parameters
        ----------
        X : DataFrame
            Dataframe
        """
        X_dtypes = X.dtypes
        for x_dtype in X_dtypes:
            if x_dtype in NUMERICS_DTYPES:
                return
        raise ValueError(
            f"""`X` should contains one of the following float dtypes:
            {PRINT_NUMERICS_DTYPES}.
            Use gators.converter.ConvertColumnDatatype before
            calling this transformer."""
        )

    def check_dataframe_with_objects(self, X: DataFrame):
        """Check if dataframe contains object columns.

        Parameters
        ----------
        X : DataFrame
            Dataframe
        """
        X_dtypes = X.dtypes
        contains_object = object in X_dtypes
        if not contains_object:
            raise ValueError(
                f"""`X` should contains object columns to use the transformer
                {self.__class__.__name__}."""
            )

    def check_array_is_numerics(self, X: np.ndarray):
        """Check if array is only numerics.

        Parameters
        ----------
        X : np.ndarray
            Array
        """
        if X.dtype not in NUMERICS_DTYPES:
            raise ValueError(
                f"""`X` should be of type {PRINT_NUMERICS_DTYPES}
                to use the transformer {self.__class__.__name__}.
                Use gators.converter.ConvertColumnDatatype before calling it.
                """
            )

    def check_nans(self, X: DataFrame, columns: List[str]):
        """Raise an error if X contains NaN values.

        Parameters
        ----------
        X : DataFrame
            Dataframe.
        columns : List[str]
            List of columns.
        """
        # if util.get_function(X).to_pandas(X[columns].isnull().sum()).sum() != 0:
        if util.get_function(X).to_numpy(X[columns].isnull().sum()).any():
            raise ValueError(
                f"""The object columns should not contain NaN values
                to use the transformer {self.__class__.__name__}.
                Use `gators.imputers.ObjectImputer` before calling it."""
            )
