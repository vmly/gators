# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning import TreeDiscretizer


@pytest.fixture
def data():
    max_depth = 2
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [1.07, -2.59, -1.54, 1.72],
                "B": [-1.19, -0.22, -0.28, 1.28],
                "C": [-1.15, 1.92, 1.09, -0.95],
                "D": ["a", "b", "c", "d"],
            }
        ),
        npartitions=1,
    )
    y = dd.from_pandas(pd.Series([0, 1, 0, 1], name="TARGET"), npartitions=1)
    X_expected = pd.DataFrame(
        {
            "A": [1.07, -2.59, -1.54, 1.72],
            "B": [-1.19, -0.22, -0.28, 1.28],
            "C": [-1.15, 1.92, 1.09, -0.95],
            "D": ["a", "b", "c", "d"],
            "A__bin": ["_1", "_0", "_1", "_2"],
            "B__bin": ["_0", "_1", "_0", "_1"],
            "C__bin": ["_0", "_2", "_2", "_1"],
        }
    )
    obj = TreeDiscretizer(max_depth).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_regression():
    max_depth = 2
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [-0.1, 1.45, 0.98, -0.98],
                "B": [-0.15, 0.14, 0.4, 1.87],
                "C": [0.95, 0.41, 1.76, 2.24],
            }
        ),
        npartitions=1,
    )
    y = dd.from_pandas(
        pd.Series([39.9596835, 36.65644911, 137.24445075, 300.15325913], name="TARGET"),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": ["_1", "_2", "_1", "_0"],
            "B": ["_0", "_0", "_1", "_2"],
            "C": ["_0", "_0", "_1", "_2"],
        }
    )
    obj = TreeDiscretizer(max_depth, inplace=True).fit(X, y)
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


def test_regression_dd(data_regression):
    obj, X, X_expected = data_regression
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_regression_dd_np(data_regression):
    obj, X, X_expected = data_regression
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))
