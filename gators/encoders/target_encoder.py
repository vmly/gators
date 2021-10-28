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

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.encoders import TargetEncoder
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> y = pd.Series([1, 1, 0], name='TARGET')
    >>> obj = TargetEncoder()
    >>> obj.fit_transform(X, y)
         A    B
    0  1.0  1.0
    1  1.0  0.5
    2  0.0  0.5

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.encoders import TargetEncoder
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> y = ks.Series([1, 1, 0], name='TARGET')
    >>> obj = TargetEncoder()
    >>> obj.fit_transform(X, y)
         A    B
    0  1.0  1.0
    1  1.0  0.5
    2  0.0  0.5

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.encoders import TargetEncoder
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> y = pd.Series([1, 1, 0], name='TARGET')
    >>> obj = TargetEncoder()
    >>> _ = obj.fit(X, y)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1. , 1. ],
           [1. , 0.5],
           [0. , 0.5]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.encoders import TargetEncoder
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> y = ks.Series([1, 1, 0], name='TARGET')
    >>> obj = TargetEncoder()
    >>> _ = obj.fit(X, y)
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
        self.check_binary_target(X, y)
        self.check_nans(X, self.columns)
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
        function = util.get_function(X)
        columns = X.columns
        X = X.join(y.to_frame())
        for name in columns:
            dummy = function.to_pandas(X[[name, y_name]].groupby(name).mean()[y_name])
            dummy.name = name
            mapping_list.append(dummy)
        X = X.drop(y_name, axis=1)
        mapping = pd.concat(mapping_list, axis=1).to_dict()
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
