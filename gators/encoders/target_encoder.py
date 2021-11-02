# License: Apache-2.
import warnings
from typing import Dict, List, TypeVar

import numpy as np
import pandas as pd

from ..util import util
from ._base_encoder import _BaseEncoder

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class TargetEncoder(_BaseEncoder):
    """Encode the categorical variable using the target encoding technique.

    Parameters
    ----------
    dtype : type, default to np.float64.
        Numerical datatype of the output data.

    Examples
    --------

    Imports and initialization:

    >>> from gators.encoders import TargetEncoder
    >>> obj = TargetEncoder()

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes,

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']}), npartitions=1)
    >>> y = dd.from_pandas(pd.Series([1, 1, 0], name='TARGET'), npartitions=1)

    * `koalas` dataframes,

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> y = ks.Series([1, 1, 0], name='TARGET')

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> y = pd.Series([1, 1, 0], name='TARGET')

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X, y)
         A    B
    0  1.0  1.0
    1  1.0  0.5
    2  0.0  0.5

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([[1. , 1. ],
           [1. , 0.5],
           [0. , 0.5]])
    """

    def __init__(self, dtype: type = np.float64):
        _BaseEncoder.__init__(self, dtype=dtype)

    def fit(self, X: DataFrame, y: Series) -> "TargetEncoder":
        """Fit the encoder.

        Parameters
        ----------
        X : DataFrame:
            Input dataframe.
        y : Series, default to None.
            Labels.

        Returns
        -------
        TargetEncoder:
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_y(X, y)
        # # self.check_binary_target(X, y)
        # self.check_nans(X, self.columns)
        self.columns = util.get_datatype_columns(X, object)
        if not self.columns:
            warnings.warn(
                f"""`X` does not contain object columns:
                `{self.__class__.__name__}` is not needed"""
            )
            return self
        self.mapping = self.generate_mapping(X[self.columns], y)
        self.num_categories_vec = np.array([len(m) for m in self.mapping.values()])
        columns, self.values_vec, self.encoded_values_vec = self.decompose_mapping(
            mapping=self.mapping
        )
        self.idx_columns = util.get_idx_columns(
            columns=X.columns, selected_columns=columns
        )
        return self

    def generate_mapping(self, X: DataFrame, y: Series) -> Dict[str, Dict[str, float]]:
        """Generate the mapping to perform the encoding.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series:
             Labels.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping.
        """
        mapping_list = []
        y_name = y.name
        columns = X.columns
        X = util.get_function(X).join(X, y.to_frame())
        for name in columns:
            dummy = util.get_function(X).to_pandas(
                X[[name, y_name]].groupby(name).mean()
            )[y_name]
            dummy.name = name
            mapping_list.append(dummy)
        mapping = pd.concat(mapping_list, axis=1).to_dict()
        X = X.drop(y_name, axis=1)
        return self.clean_mapping(mapping)

    @staticmethod
    def clean_mapping(
        mapping: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, List[float]]]:
        mapping = {
            col: {k: v for k, v in mapping[col].items() if v == v}
            for col in mapping.keys()
        }
        for m in mapping.values():
            if "OTHERS" not in m:
                m["OTHERS"] = 0.0
            if "MISSING" not in m:
                m["MISSING"] = 0.0
        return mapping
