# License: Apache-2.0
from typing import TypeVar

import databricks.koalas as ks
import pandas as pd

from ..converter import ToPandas
from ..util import util
from ._base_feature_selection import _BaseFeatureSelection

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class SelectFromModel(_BaseFeatureSelection):
    """Select From Model Transformer.

    Select the top *k* features based on the feature importance
    of the given machine learning model.

    Parameters
    ----------
    model : model
        Machine learning model.
    k : int
        Number of features to keep.

    Examples
    ---------
    * fit & transform with `pandas`


    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier as RFC
    >>> from gators.feature_selection import SelectFromModel
    >>> X = pd.DataFrame(
    ... {'A': [22.0, 38.0, 26.0, 35.0, 35.0, 28.11, 54.0, 2.0, 27.0, 14.0],
    ... 'B': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
    ... 'C': [3.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0, 3.0, 2.0]})
    >>> y = pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name='TARGET')
    >>> model = RFC(n_estimators=1, max_depth=2, random_state=0)
    >>> obj = SelectFromModel(model=model, k=1)
    >>> obj.fit_transform(X, y)
           A
    0  22.00
    1  38.00
    2  26.00
    3  35.00
    4  35.00
    5  28.11
    6  54.00
    7   2.00
    8  27.00
    9  14.00

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from pyspark.ml.classification import RandomForestClassifier as RFCSpark
    >>> from gators.feature_selection import SelectFromModel
    >>> X = ks.DataFrame(
    ... {'A': [22.0, 38.0, 26.0, 35.0, 35.0, 28.11, 54.0, 2.0, 27.0, 14.0],
    ... 'B': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
    ... 'C': [3.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0, 3.0, 2.0]})
    >>> y = ks.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name='TARGET')
    >>> model = RFCSpark(numTrees=1, maxDepth=2, labelCol=y.name, seed=0)
    >>> obj = SelectFromModel(model=model, k=1)
    >>> obj.fit_transform(X, y)
           A
    0  22.00
    1  38.00
    2  26.00
    3  35.00
    4  35.00
    5  28.11
    6  54.00
    7   2.00
    8  27.00
    9  14.00

    See Also
    --------
    gators.feature_selection.SelectFromModels
        Similar method using multiple models.

    """

    def __init__(self, model, k: int):
        if not isinstance(k, int):
            raise TypeError("`k` should be an int.")
        if not hasattr(model, "fit"):
            raise TypeError("`model` should have the attribute `fit`.")
        _BaseFeatureSelection.__init__(self)
        self.model = model
        self.k = k

    def fit(self, X: DataFrame, y: Series = None) -> "SelectFromModel":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default to None.
            Labels.

        Returns
        -------
            SelectFromModel: Instance of itself.
        """
        self.check_dataframe(X)
        self.check_y(X, y)
        columns = list(X.columns)
        self.feature_importances_ = util.get_function(X).feature_importances_(
            self.model, X, y
        )
        mask = self.feature_importances_ != 0
        self.feature_importances_ = self.feature_importances_[mask]
        self.feature_importances_.sort_values(ascending=False, inplace=True)
        self.selected_columns = list(self.feature_importances_.index[: self.k])
        self.columns_to_drop = [c for c in columns if c not in self.selected_columns]
        self.idx_selected_columns = util.get_idx_columns(
            X.columns, self.selected_columns
        )
        return self
