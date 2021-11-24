# License: Apache-2.0
from typing import List, TypeVar

import numpy as np
import pandas as pd

from ..scalers.minmax_scaler import MinMaxScaler
from ..util import util
from ._base_feature_selection import _BaseFeatureSelection

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")


class SelectFromModels(_BaseFeatureSelection):
    """Select From Models By Vote Transformer.

    Select the top *k* features based on the feature importance
    of the given machine learning models.

    Parameters
    ----------
    models : List[model]
        List of machine learning models.
    k : int
        Number of features to keep.

    Examples
    ---------
    * fit & transform with `koalas`

    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier as RFC
    >>> from gators.feature_selection import SelectFromModels
    >>> X = pd.DataFrame({
    ... 'A': [22.0, 38.0, 26.0, 35.0, 35.0, 28.11, 54.0, 2.0, 27.0, 14.0],
    ... 'B': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
    ... 'C': [3.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0, 3.0, 2.0]})
    >>> y = pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name='TARGET')
    >>> models = [RFC(n_estimators=1, max_depth=1, random_state=0),
    ... RFC(n_estimators=1, max_depth=2, random_state=1)]
    >>> obj = SelectFromModels(models=models, k=1)
    >>> obj.fit_transform(X, y)
           B
    0   7.25
    1  71.28
    2   7.92
    3  53.10
    4   8.05
    5   8.46
    6  51.86
    7  21.08
    8  11.13
    9  30.07

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from pyspark.ml.classification import RandomForestClassifier as RFCSpark
    >>> from gators.feature_selection import SelectFromModels
    >>> X = ks.DataFrame({
    ... 'A': [22.0, 38.0, 26.0, 35.0, 35.0, 28.11, 54.0, 2.0, 27.0, 14.0],
    ... 'B': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
    ... 'C': [3.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0, 3.0, 2.0]})
    >>> y = ks.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name='TARGET')
    >>> models = [RFCSpark(numTrees=1, maxDepth=1, labelCol=y.name, seed=0),
    ... RFCSpark(numTrees=1, maxDepth=2, labelCol=y.name, seed=1)]
    >>> obj = SelectFromModels(models=models, k=1)
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
    gators.feature_selection.SelectFromMode
        Similar method using one model.

    """

    def __init__(self, models: List[object], k: int):
        if not isinstance(models, list):
            raise TypeError("`models` should be a list.")
        if not isinstance(k, int):
            raise TypeError("`k` should be an int.")
        for model in models:
            if not hasattr(model, "fit"):
                raise TypeError(
                    "All the elements of `models` should have the attribute `fit`."
                )
        _BaseFeatureSelection.__init__(self)
        self.models = models
        self.k = k

    def fit(self, X: DataFrame, y: Series = None) -> "SelectFromModels":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default to None.
            Labels.

        Returns
        -------
        SelectFromModels: Instance of itself.
        """
        self.check_dataframe(X)
        self.check_y(X, y)
        self.feature_importances_ = self.get_feature_importances_frame(X, self.models)
        for col, model in zip(self.feature_importances_.columns, self.models):
            self.feature_importances_[col] = util.get_function(X).feature_importances_(
                model, X, y
            )
        self.feature_importances_ = self.clean_feature_importances_frame(
            self.feature_importances_
        )
        self.selected_columns = list(
            self.feature_importances_["count"].iloc[: self.k].index
        )
        self.columns_to_drop = [
            c for c in self.feature_importances_.index if c not in self.selected_columns
        ]
        self.idx_selected_columns = util.get_idx_columns(
            X.columns, self.selected_columns
        )
        return self

    @staticmethod
    def get_feature_importances_frame(X, models):
        index = np.array(list(X.columns))
        columns = []
        for i, model in enumerate(models):
            col = str(model).split("(")[0]
            columns.append(col + "_" + str(i))
        return pd.DataFrame(columns=columns, index=index, dtype=np.float64)

    @staticmethod
    def clean_feature_importances_frame(feature_importances):
        feature_importances = MinMaxScaler().fit_transform(feature_importances)
        feature_importances_sum = feature_importances.sum(1)
        feature_importances_count = (feature_importances != 0).sum(1)
        feature_importances["sum"] = feature_importances_sum
        feature_importances["count"] = feature_importances_count
        feature_importances.sort_values(
            by=["count", "sum"], ascending=False, inplace=True
        )
        return feature_importances
