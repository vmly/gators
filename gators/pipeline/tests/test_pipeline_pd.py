# License: Apache-2.0import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from gators.feature_selection.select_from_model import SelectFromModel
from gators.pipeline.pipeline import Pipeline
from gators.transformers.transformer import Transformer

data = load_iris()


class MultiplyTransformer(Transformer):
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        return self.multiplier * X

    def transform_numpy(self, X):
        return self.multiplier * X


class NameTransformer(Transformer):
    def fit(self, X, y=None):
        self.column_names = [f"{c}_new" for c in X.columns]
        self.column_mapping = dict(zip(self.column_names, [[c] for c in X.columns]))
        return self

    def transform(self, X):
        return X.rename(columns=dict(zip(X.columns, self.column_names)))

    def transform_numpy(self, X):
        return X


@pytest.fixture
def pipeline_example():
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    steps = [
        MultiplyTransformer(4.0),
        MultiplyTransformer(0.5),
        NameTransformer(),
    ]
    pipe = Pipeline(steps)
    return pipe, X


@pytest.fixture
def pipeline_with_feature_selection_example():
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    y = pd.Series(data["target"], name="TARGET")

    model = RandomForestClassifier(n_estimators=6, max_depth=4, random_state=0)
    steps = [
        MultiplyTransformer(4.0),
        MultiplyTransformer(0.5),
        NameTransformer(),
        SelectFromModel(model=model, k=3),
    ]
    pipe = Pipeline(steps).fit(X, y)
    return pipe, X


@pytest.fixture
def pipeline_with_model_example():
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    y = pd.Series(data["target"], name="TARGET")
    model = RandomForestClassifier(n_estimators=6, max_depth=4, random_state=0)
    steps = [
        MultiplyTransformer(4.0),
        MultiplyTransformer(0.5),
        NameTransformer(),
        model,
    ]
    pipe = Pipeline(steps).fit(X, y)
    return pipe, X


def test_pipeline_fit_and_transform(pipeline_example):
    pipe, X = pipeline_example
    _ = pipe.fit(X)
    X_new = pipe.transform(X)
    assert X_new.shape == (150, 4)
    assert list(X_new.columns) == [
        "sepal length (cm)_new",
        "sepal width (cm)_new",
        "petal length (cm)_new",
        "petal width (cm)_new",
    ]


def test_fit_transform_pipeline(pipeline_example):
    pipe, X = pipeline_example
    X_new = pipe.fit_transform(X)
    assert X_new.shape == (150, 4)
    assert list(X_new.columns) == [
        "sepal length (cm)_new",
        "sepal width (cm)_new",
        "petal length (cm)_new",
        "petal width (cm)_new",
    ]


def test_pipeline_predict(pipeline_with_model_example):
    pipe, X = pipeline_with_model_example
    y_pred = pipe.predict(X)
    assert y_pred.shape == (150,)


def test_pipeline_predict_proba(pipeline_with_model_example):
    pipe, X = pipeline_with_model_example
    y_pred = pipe.predict_proba(X)
    assert y_pred.shape == (150, 3)


def test_pipeline_numpy(pipeline_example):
    pipe, X = pipeline_example
    _ = pipe.fit(X)
    X_numpy_new = pipe.transform_numpy(X.to_numpy())
    assert X_numpy_new.shape == (150, 4)


def test_pipeline_predict_numpy(pipeline_with_model_example):
    pipe, X = pipeline_with_model_example
    y_pred = pipe.predict_numpy(X.to_numpy())
    assert y_pred.shape == (150,)


def test_pipeline_predict_proba_numpy(pipeline_with_model_example):
    pipe, X = pipeline_with_model_example
    y_pred = pipe.predict_proba_numpy(X.to_numpy())
    assert y_pred.shape == (150, 3)


def test_init():
    with pytest.raises(TypeError):
        _ = Pipeline(0)
    with pytest.raises(TypeError):
        _ = Pipeline([])


def test_pipeline_transform_input_data(pipeline_example):
    pipe, X = pipeline_example
    _ = pipe.fit(X)
    with pytest.raises(TypeError):
        _ = pipe.transform(X.to_numpy())
    with pytest.raises(TypeError):
        _ = pipe.transform(X, X)
    with pytest.raises(TypeError):
        _ = pipe.transform_numpy(X)


def test_get_feature_importances(pipeline_with_feature_selection_example):
    pipe, _ = pipeline_with_feature_selection_example
    feature_importances_expected = pd.Series(
        {"petal width (cm)_new": 0.6427904, "petal length (cm)_new": 0.18456636}
    )
    feature_importances = pipe.get_feature_importances(k=2)
    assert_series_equal(feature_importances, feature_importances_expected)


def test_get_features(pipeline_with_feature_selection_example):
    pipe, _ = pipeline_with_feature_selection_example
    assert [
        "petal width (cm)_new",
        "petal length (cm)_new",
        "sepal length (cm)_new",
    ] == pipe.get_features()


def test_get_feature_importances_no_feature_selection(pipeline_example):
    pipe, _ = pipeline_example
    with pytest.raises(AttributeError):
        pipe.get_feature_importances(k=2)


def test_get_features_no_feature_selection(pipeline_example):
    pipe, _ = pipeline_example
    with pytest.raises(AttributeError):
        pipe.get_features()


def test_get_production_columns(pipeline_with_feature_selection_example):
    pipe, _ = pipeline_with_feature_selection_example
    assert sorted(pipe.get_production_columns()) == sorted(
        ["sepal length (cm)", "petal length (cm)", "petal width (cm)"]
    )


def test_get_production_columns_no_feature_selection(pipeline_example):
    pipe, _ = pipeline_example
    with pytest.raises(AttributeError):
        pipe.get_production_columns()
