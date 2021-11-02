# License: Apache-2.0
import warnings
from typing import Dict, List, TypeVar

import numpy as np

from ..util import util
from ._base_encoder import _BaseEncoder

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class OrdinalEncoder(_BaseEncoder):
    """Encode the categorical columns as integer columns.

    Parameters
    ----------
    dtype : type, default to np.float64.
        Numerical datatype of the output data.
    add_other_columns: bool, default to True.
        If True, add the columns 'OTHERS' and 'MISSING'
        to the mapping even if the categories are not
        present in the data.

    Examples
    ---------

    Imports and initialization:

    >>> from gators.encoders import OrdinalEncoder
    >>> obj = OrdinalEncoder()

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
         A    B
    0  1.0  1.0
    1  1.0  0.0
    2  0.0  0.0

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([[1., 1.],
           [1., 0.],
           [0., 0.]])
    """

    def __init__(self, dtype: type = np.float64, add_other_columns: bool = True):
        _BaseEncoder.__init__(self, dtype=dtype)
        if not isinstance(add_other_columns, bool):
            raise TypeError("`add_other_columns` shouldbe a bool.")
        self.add_other_columns = add_other_columns

    def fit(self, X: DataFrame, y: Series = None) -> "OrdinalEncoder":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default to None.
            Labels.

        Returns
        -------
        OrdinalEncoder: Instance of itself.
        """
        self.check_dataframe(X)
        self.columns = util.get_datatype_columns(X, object)
        # self.check_nans(X, self.columns)
        if not self.columns:
            warnings.warn(
                f"""`X` does not contain object columns:
                `{self.__class__.__name__}` is not needed"""
            )
            return self
        self.mapping = self.generate_mapping(X, self.columns, self.add_other_columns)
        self.num_categories_vec = np.array([len(m) for m in self.mapping.values()])
        columns, self.values_vec, self.encoded_values_vec = self.decompose_mapping(
            mapping=self.mapping,
        )
        self.idx_columns = util.get_idx_columns(
            columns=X.columns, selected_columns=columns
        )
        return self

    def generate_mapping(
        self, X: DataFrame, columns: List[str], add_other_columns: bool
    ) -> Dict[str, Dict[str, float]]:
        """Generate the mapping to perform the encoding.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        self.columns : List[str]
            List of  columns.
        add_other_columns: bool
            If True, add the columns 'OTHERS' and 'MISSING'
            to the mapping even if the categories are not
            present in the data.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping.
        """
        mapping = {}
        for c in columns:
            categories = (
                util.get_function(X)
                .to_pandas(X[c].value_counts(dropna=False))
                .to_dict()
            )
            n_categories = len(categories)
            category_names = list(categories.keys())
            category_names = sorted(category_names)
            category_mapping = dict(
                zip(category_names, np.arange(n_categories - 1, -1, -1).astype(str))
            )
            if add_other_columns and "MISSING" not in category_mapping:
                category_mapping["MISSING"] = str(len(category_mapping))
            if add_other_columns and "OTHERS" not in category_mapping:
                category_mapping["OTHERS"] = str(len(category_mapping))
            mapping[c] = category_mapping
        return mapping
