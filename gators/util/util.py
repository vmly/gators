# License: Apache-2.0
from abc import ABC, abstractmethod
from typing import List, TypeVar

import numpy as np
import pandas as pd

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
KSDataFrame = TypeVar("ks.DataFrame")


class FunctionFactory(ABC):
    @abstractmethod
    def apply():
        pass

    @abstractmethod
    def apply_to_pandas():
        pass
    
    @abstractmethod
    def cut():
        pass
    

class FunctionPandas(FunctionFactory):
    def apply(self, X, f, args=None, meta=None):
        return X.apply(f, args=args)

    def apply_to_pandas(self, X, f, args=None, meta=None):
        return X.apply(f, args=args)

    def cut(self, X, **kwargs):
        return pd.cut(X, **kwargs)
    
    def concat(self, objs, axis=0):
        return pd.concat(objs, axis=axis)
class FunctionKoalas(FunctionFactory):
    def apply(self, X, f, args=None, meta=None):
        return X.apply(f, args=args)

    def apply_to_pandas(self, X, f, args=None, meta=None):
        return X.apply(f, args=args)

    def cut(self, X, **kwargs):
        import databricks.koalas as ks
        return ks.cut(X, **kwargs)

    def concat(self, objs, axis=0):
        import databricks.koalas as ks
        return ks.concat(objs, axis=axis)
    
class FunctionDask(FunctionFactory):

    def to_pandas(self, X):
        return X.compute()
    
    def apply(self, X, f, args=None, meta=None):
        if meta is None:
            return X.map_partitions(lambda x: x.apply(f, args=args))
        return X.map_partitions(lambda x: x.apply(f, args=args), meta=meta)

    def apply_to_pandas(self, X, f, args=None, meta=None):
        if meta is None:
            return X.map_partitions(lambda x: x.apply(f, args=args))
        return X.map_partitions(lambda x: x.apply(f, args=args), meta=meta)

    def cut(self, X, **kwargs):
        import dask.dataframe as dd
        return dd.cut(X, **kwargs)

    def concat(self, objs, axis=0):
        import dask.dataframe as dd
        return dd.concat(objs, axis=axis)
    
def get_function(X):
    factories = {
        "<class 'pandas.core.frame.DataFrame'>": FunctionPandas(),
        "<class 'databricks.koalas.frame.DataFrame'>": FunctionKoalas(),
        "<class 'dask.dataframe.core.DataFrame'>": FunctionDask(),
    }
    return factories[str(type(X))]


class ComputerFactory(ABC):
    @abstractmethod
    def to_pandas(self, X):
        pass

    @abstractmethod
    def mask_object():
        pass

    @abstractmethod
    def raise_y_dtype_error():
        pass


class ComputerPandas(ComputerFactory):

    def to_pandas(self, X):
        return X

    def mask_object(self, X_dtypes):
        return X_dtypes == object

    def raise_y_dtype_error(self, X):
        if not isinstance(y, pd.Series):
            raise TypeError("`y` should be a pandas series.")


class ComputerKoalas(ComputerFactory):

    def mask_object(self, X_dtypes):
        return (X_dtypes.astype(str).str.startswith("<U")) | (X_dtypes == object)

    def raise_y_dtype_error(self, X):
        import databricks.koalas as ks

        if not isinstance(y, ks.Series):
            raise TypeError("`y` should be a koalas series.")


class ComputerDask(ComputerFactory):
    def concat(self, objs, axis):
        import dask.dataframe as dd

        return dd.concat(objs, axis=axis)

    def to_pandas(self, X):
        return X.compute()
    
    def mask_object(self, X_dtypes):
        return X_dtypes == object

    def raise_y_dtype_error(self, y):
        import dask.dataframe as dd

        if not isinstance(y, dd.Series):
            raise TypeError("`y` should be a dask series.")


def get_computer(X):
    factories = {
        "<class 'pandas.core.frame.DataFrame'>": ComputerPandas(),
        "<class 'databricks.koalas.frame.DataFrame'>": ComputerKoalas(),
        "<class 'dask.dataframe.core.DataFrame'>": ComputerDask(),
        "<class 'pandas.core.frame.Series'>": ComputerPandas(),
        "<class 'databricks.koalas.frame.Series'>": ComputerKoalas(),
        "<class 'dask.dataframe.core.Series'>": ComputerDask(),
    }
    str_type = str(type(X))
    if str(type(X)) not in factories:
        raise TypeError("""`X` should be a pandas, koalas, or dask dataframe.""")
    return factories[str(type(X))]


def get_bounds(X_dtype: type) -> List:
    """Return upper and lower of the input numerical NumPy datatype.

    Parameters
    ----------
    X_dtype : type, default to np.float64
        Numerical NumPy datatype.

    Returns
    -------
    List
        Lower ad upper bounds.
    """
    if "float" in str(X_dtype):
        info = np.finfo(X_dtype)
        return info.min, info.max
    elif "int" in str(X_dtype):
        info = np.iinfo(X_dtype)
        return info.min, info.max


def concat(objs: List[DataFrame], axis: int = 0) -> DataFrame:
    """Concatenate the `objs` along an axis.

    Parameters
    ----------
    objs : List[DataFrame]
        List of dataframes to concatenate.
    axis : int, default to 0.
        The axis to concatenate along.

    Returns
    -------
    DataFrame
        Concatenated dataframe.
    """
    return get_computer(objs[0]).concat(objs, axis=axis)


def get_datatype_columns(X: DataFrame, datatype: type) -> List[str]:
    """Return the columns of the specified datatype.

    Parameters
    ----------
    X : DataFrame
            Input dataframe.
    datatype : type
        Datatype.

    Returns
    -------
    List[str]
        List of columns
    """
    computer = get_computer(X)
    X_dtypes = computer.dtypes(X)
    if datatype != object:
        mask = X_dtypes == datatype
    else:
        mask = computer.mask_object(X_dtypes)
    datatype_columns = [c for c, m in zip(X_dtypes.index, mask) if m]
    return datatype_columns


def exclude_columns(columns: List[str], excluded_columns: List[str]) -> List[str]:
    """Return the columns in `columns` not in `selected_columns`.

    Parameters
    ----------
    columns : List[str]
        List of columns.
    excluded_columns : List[str]
        List of columns.

    Returns
    -------
    List[str]
        List of columns.
    """
    return [c for c in columns if c not in excluded_columns]


def get_idx_columns(columns: List[str], selected_columns: List[str]) -> np.ndarray:
    """Return the indices of the columns in `columns`
      and `selected_columns`.

    Parameters
    ----------
    columns : List[str]
        List of columns.
    selected_columns : List[str]
        List of columns.

    Returns
    -------
    np.ndarray
        Array of indices.
    """
    selected_idx_columns = []
    for selected_column in selected_columns:
        for i, column in enumerate(columns):
            if column == selected_column:
                selected_idx_columns.append(i)
    return np.array(selected_idx_columns)


def exclude_idx_columns(columns: List[str], excluded_columns: List[str]) -> np.ndarray:
    """Return the indices of the columns in `columns`
        and not in `excluded_columns`.

    Parameters
    ----------
    columns : List[str]
        List of columns.
    excluded_columns : List[str]
        List of columns.

    Returns
    -------
    np.ndarray
        Array of indices.
    """

    selected_idx_columns = [
        i for i, c in enumerate(columns) if c not in excluded_columns
    ]
    return np.array(selected_idx_columns)


def get_numerical_columns(X: DataFrame) -> List[str]:
    """Return the float columns.

    Parameters
    ----------
    X : DataFrame
        Input dataframe.

    Returns
    -------
    List[str]
        List of columns.
    """
    X_dtypes = get_computer(X).dtypes(X)
    mask = (
        (X_dtypes == np.float64)
        | (X_dtypes == np.int64)
        | (X_dtypes == np.float32)
        | (X_dtypes == np.int32)
        | (X_dtypes == np.float16)
        | (X_dtypes == np.int16)
    )
    numerical_columns = [c for c, m in zip(X_dtypes.index, mask) if m]
    return numerical_columns


def generate_spark_dataframe(X: KSDataFrame, y=None):
    """
    Generates a Spark dataframe and transforms the features
    to one column, ready for training in a SparkML model

    Parameters
    ----------
    X : ks.DataFrame
        Feature set.
    y : ks.Series, default to None.
        Target column. Defaults to None.

    Returns
    -------
    pyspark.DataFrame
        Contains the features transformed into one column.
    """
    from pyspark.ml.feature import VectorAssembler

    columns = list(X.columns)
    if y is None:
        spark_df = X.to_spark()
    else:
        spark_df = X.join(y).to_spark()
    vector_assembler = VectorAssembler(inputCols=columns, outputCol="features")
    transformed_spark_df = vector_assembler.transform(spark_df)
    return transformed_spark_df


def flatten_list(list_to_flatten: List):
    """Flatten list.

    Parameters
    ----------
    list_to_flatten : List
        List to flatten

    Returns
    -------
    List
        Flatten list
    """
    list_flatten = []
    for i, l in enumerate(list_to_flatten):
        if not isinstance(l, list):
            list_flatten.append(l)
        else:
            list_flatten += l
    return list_flatten
