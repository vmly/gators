# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.is_null import IsNull


@pytest.fixture
def data_num():
    X = pd.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
    )
    X = X.mask(X < 3, np.nan)
    X_expected = pd.DataFrame(
        [
            [np.nan, np.nan, np.nan, 1.0, 1.0, 1.0],
            [3.0, 4.0, 5.0, 0.0, 0.0, 0.0],
            [6.0, 7.0, 8.0, 0.0, 0.0, 0.0],
        ],
        columns=["A", "B", "C", "A__is_null", "B__is_null", "C__is_null"],
    )
    obj = IsNull(columns=list("ABC")).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_float32_num():
    X = pd.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
    ).astype(np.float32)
    X = X.mask(X < 3, np.nan)

    X_expected = pd.DataFrame(
        [
            [np.nan, np.nan, np.nan, 1.0, 1.0, 1.0],
            [3.0, 4.0, 5.0, 0.0, 0.0, 0.0],
            [6.0, 7.0, 8.0, 0.0, 0.0, 0.0],
        ],
        columns=["A", "B", "C", "A__is_null", "B__is_null", "C__is_null"],
    ).astype(np.float32)
    obj = IsNull(columns=list("ABC"), dtype=np.float32).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_names():
    X = pd.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
    )
    X = X.mask(X < 3, np.nan)
    X_expected = pd.DataFrame(
        [
            [np.nan, np.nan, np.nan, 1.0, 1.0, 1.0],
            [3.0, 4.0, 5.0, 0.0, 0.0, 0.0],
            [6.0, 7.0, 8.0, 0.0, 0.0, 0.0],
        ],
        columns=["A", "B", "C", "AIsNull", "BIsNull", "CIsNull"],
    )
    obj = IsNull(
        columns=list("ABC"), column_names=["AIsNull", "BIsNull", "CIsNull"]
    ).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_obj():
    X = pd.DataFrame(
        {
            "A": [None, "a", "b"],
            "B": [None, "c", "d"],
            "C": [None, "e", "f"],
            "D": [0, 1, np.nan],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [None, "a", "b"],
            "B": [None, "c", "d"],
            "C": [None, "e", "f"],
            "D": [0, 1, np.nan],
            "A__is_null": [1.0, 0.0, 0.0],
            "B__is_null": [1.0, 0.0, 0.0],
            "C__is_null": [1.0, 0.0, 0.0],
            "D__is_null": [0.0, 0.0, 1.0],
        }
    )
    obj = IsNull(columns=list("ABCD")).fit(X)
    return obj, X, X_expected


def test_pd(data_num):
    obj, X, X_expected = data_num
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data_num):
    obj, X, X_expected = data_num
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_float32_pd(data_float32_num):
    obj, X, X_expected = data_float32_num
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_float32_pd_np(data_float32_num):
    obj, X, X_expected = data_float32_num
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_names_pd(data_names):
    obj, X, X_expected = data_names
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_names_pd_np(data_names):
    obj, X, X_expected = data_names
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_obj(data_obj):
    obj, X, X_expected = data_obj
    X_new = obj.transform(X)
    assert_frame_equal(
        X_new.iloc[:, 4:].astype(float), X_expected.iloc[:, 4:].astype(float)
    )


def test_obj_np(data_obj):
    obj, X, X_expected = data_obj
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(
        X_new.iloc[:, 4:].astype(float), X_expected.iloc[:, 4:].astype(float)
    )


def test_init():
    with pytest.raises(TypeError):
        _ = IsNull(columns=0)
    with pytest.raises(ValueError):
        _ = IsNull(columns=[], column_names=["AIsNull"])
    with pytest.raises(TypeError):
        _ = IsNull(columns=list("ABC"), column_names=0)
    with pytest.raises(ValueError):
        _ = IsNull(columns=list("ABC"), column_names=["a", "b"])
