# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.encoders import OrdinalEncoder


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["Q", "Q", "W"],
                "B": ["Q", "W", "W"],
                "C": ["W", "Q", "W"],
                "D": [1, 2, 3],
            }
        ),
        npartitions=2,
    )
    X_expected = pd.DataFrame(
        [[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 3.0]],
        columns=list("ABCD"),
    )
    obj = OrdinalEncoder().fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_int16():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["Q", "Q", "W"],
                "B": ["Q", "W", "W"],
                "C": ["W", "Q", "W"],
                "D": [1, 2, 3],
            }
        ),
        npartitions=2,
    )
    X_expected = pd.DataFrame(
        [[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 3.0]],
        columns=list("ABCD"),
    ).astype(np.int16)
    obj = OrdinalEncoder(dtype=np.int16).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_cat():
    X = dd.from_pandas(
        pd.DataFrame(
            np.zeros((3, 3)),
            columns=list("ABC"),
        ),
        npartitions=2,
    )
    obj = OrdinalEncoder().fit(X)
    return obj, X, X.compute().copy()


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy = X.compute().to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_int16_dd(data_int16):
    obj, X, X_expected = data_int16
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_int16_dd_np(data_int16):
    obj, X, X_expected = data_int16
    X_numpy = X.compute().to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_no_cat_dd(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_no_cat_dd_np(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_numpy = X.compute().to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)
