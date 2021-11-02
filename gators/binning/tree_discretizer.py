# License: Apache-2.0
from typing import List, Tuple, TypeVar

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..util import util
from ._base_discretizer import _BaseDiscretizer


DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class TreeDiscretizer(_BaseDiscretizer):
    """Discretize the columns using decision tree based splits.

    The discretization can be done inplace or by adding the discretized
    columns to the existing data.

    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree.
    inplace : bool, default to False
        If False, return the dataframe with the new discretized columns
        with the names '`column_name`__bin'). Otherwise, return
        the dataframe with the existing binned columns.

    Examples
    ---------
    * fit & transform with `pandas`

        - inplace discretization

            >>> import pandas as pd
            >>> from gators.binning import TreeDiscretizer
            >>> X = pd.DataFrame({
            ... 'A': [1.07, -2.59, -1.54, 1.72],
            ... 'B': [-1.19, -0.22, -0.28, 1.28],
            ... 'C': [-1.15, 1.92, 1.09, -0.95]})
            >>> y = pd.Series([0, 1, 0, 1], name="TARGET")
            >>> obj = TreeDiscretizer(max_depth=2, inplace=True)
            >>> obj.fit_transform(X, y)
                 A    B    C
            0  1.0  0.0  0.0
            1  0.0  1.0  2.0
            2  1.0  0.0  2.0
            3  2.0  1.0  1.0

        - add discretization

            >>> import pandas as pd
            >>> from gators.binning import TreeDiscretizer
            >>> X = pd.DataFrame({
            ... 'A': [1.07, -2.59, -1.54, 1.72],
            ... 'B': [-1.19, -0.22, -0.28, 1.28],
            ... 'C': [-1.15, 1.92, 1.09, -0.95]})
            >>> y = pd.Series([0, 1, 0, 1], name="TARGET")
            >>> obj = TreeDiscretizer(max_depth=2, inplace=False)
            >>> obj.fit_transform(X, y)
                  A     B     C A__bin B__bin C__bin
            0  1.07 -1.19 -1.15    1.0    0.0    0.0
            1 -2.59 -0.22  1.92    0.0    1.0    2.0
            2 -1.54 -0.28  1.09    1.0    0.0    2.0
            3  1.72  1.28 -0.95    2.0    1.0    1.0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.binning import TreeDiscretizer
    >>> X = ks.DataFrame({
    ... 'A': [1.07, -2.59, -1.54, 1.72],
    ... 'B': [-1.19, -0.22, -0.28, 1.28],
    ... 'C': [-1.15, 1.92, 1.09, -0.95]})
    >>> y = ks.Series([0, 1, 0, 1], name="TARGET")
    >>> obj = TreeDiscretizer(max_depth=2, inplace=False)
    >>> obj.fit_transform(X, y)
          A     B     C A__bin B__bin C__bin
    0  1.07 -1.19 -1.15    1.0    0.0    0.0
    1 -2.59 -0.22  1.92    0.0    1.0    2.0
    2 -1.54 -0.28  1.09    1.0    0.0    2.0
    3  1.72  1.28 -0.95    2.0    1.0    1.0

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.binning import TreeDiscretizer
    >>> X = pd.DataFrame({
    ... 'A': [1.07, -2.59, -1.54, 1.72],
    ... 'B': [-1.19, -0.22, -0.28, 1.28],
    ... 'C': [-1.15, 1.92, 1.09, -0.95]})
    >>> y = pd.Series([0, 1, 0, 1], name="TARGET")
    >>> obj = TreeDiscretizer(max_depth=2)
    >>> _ = obj.fit(X, y)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1.07, -1.19, -1.15, '1.0', '0.0', '0.0'],
           [-2.59, -0.22, 1.92, '0.0', '1.0', '2.0'],
           [-1.54, -0.28, 1.09, '1.0', '0.0', '2.0'],
           [1.72, 1.28, -0.95, '2.0', '1.0', '1.0']], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.binning import TreeDiscretizer
    >>> X = ks.DataFrame({
    ... 'A': [1.07, -2.59, -1.54, 1.72],
    ... 'B': [-1.19, -0.22, -0.28, 1.28],
    ... 'C': [-1.15, 1.92, 1.09, -0.95]})
    >>> y = ks.Series([0, 1, 0, 1], name="TARGET")
    >>> obj = TreeDiscretizer(max_depth=2)
    >>> _ = obj.fit(X, y)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1.07, -1.19, -1.15, '1.0', '0.0', '0.0'],
           [-2.59, -0.22, 1.92, '0.0', '1.0', '2.0'],
           [-1.54, -0.28, 1.09, '1.0', '0.0', '2.0'],
           [1.72, 1.28, -0.95, '2.0', '1.0', '1.0']], dtype=object)

    See Also
    --------
    gators.binning.CustomDiscretizer
        Discretize using the variable quantiles.
    gators.binning.Discretizer
        Discretize using equal splits.
    gators.binning.CustomDiscretizer
        Discretize using the variable quantiles.

    """

    def __init__(self, max_depth, inplace=False):
        if (not isinstance(max_depth, int)) or (max_depth <= 0):
            raise TypeError("`max_depth` should be a positive int.")
        _BaseDiscretizer.__init__(self, n_bins=1, inplace=inplace)
        self.max_depth = max_depth

    def fit(self, X: DataFrame, y) -> "TreeDiscretizer":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default to None.
            Labels.

        Returns
        -------
        "TreeDiscretizer"
            Instance of itself.
        """
        self.check_dataframe(X)
        self.columns = util.get_numerical_columns(X)
        self.output_columns = [f"{c}__bin" for c in self.columns]
        self.idx_columns = util.get_idx_columns(X.columns, self.columns)
        n_cols = self.idx_columns.size
        if n_cols == 0:
            return self
        self.bins = {}
        dt = self.get_tree(y.dtype, self.max_depth)
        for c in self.columns:
            dt.fit(
                util.get_function(X).to_numpy(X[[c]]).astype(np.float32),
                util.get_function(X).to_numpy(y).astype(np.int32),
            )
            self.bins[c] = np.unique(
                [-np.inf]
                + sorted(
                    [
                        float(node.split("<=")[1])
                        for node in tree.export_text(dt, decimals=6).split("|   ")
                        if "<=" in node
                    ]
                )
                + [np.inf]
            )
        max_bins = max([len(v) for v in self.bins.values()])
        self.bins_np = np.inf * np.ones((max_bins, n_cols))
        for i, b in enumerate(self.bins.values()):
            self.bins_np[: len(b), i] = b
        return self

    @staticmethod
    def get_tree(y_dtype, max_depth):
        if "int" in str(y_dtype):
            return DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        return DecisionTreeRegressor(max_depth=max_depth, random_state=0)
