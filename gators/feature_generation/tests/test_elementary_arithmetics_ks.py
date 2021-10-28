# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.elementary_arithmethics import ElementaryArithmetics


@pytest.fixture
def data_add_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
    )

    X_expected = pd.DataFrame(
        np.array(
            [
                [0.0, 1.0, 2.0, -2.0, -4.0],
                [3.0, 4.0, 5.0, -5.0, -7.0],
                [6.0, 7.0, 8.0, -8.0, -10.0],
            ]
        ),
        columns=["A", "B", "C", "A__-__B", "A__-__C"],
    )
    obj = ElementaryArithmetics(
        columns_a=list("AA"), columns_b=list("BC"), coef=-2.0, operator="+"
    ).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_float32_add_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
    ).astype(np.float32)

    X_expected = pd.DataFrame(
        np.array(
            [
                [0.0, 1.0, 2.0, -2.0, -4.0],
                [3.0, 4.0, 5.0, -5.0, -7.0],
                [6.0, 7.0, 8.0, -8.0, -10.0],
            ]
        ),
        columns=["A", "B", "C", "A__-__B", "A__-__C"],
    ).astype(np.float32)
    obj = ElementaryArithmetics(
        columns_a=list("AA"),
        columns_b=list("BC"),
        coef=-2.0,
        operator="+",
        dtype=np.float32,
    ).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_name_add_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
    )

    X_expected = pd.DataFrame(
        np.array(
            [
                [0.0, 1.0, 2.0, -2.0, -4.0],
                [3.0, 4.0, 5.0, -5.0, -7.0],
                [6.0, 7.0, 8.0, -8.0, -10.0],
            ]
        ),
        columns=["A", "B", "C", "A+B", "A+C"],
    )
    obj = ElementaryArithmetics(
        columns_a=list("AA"),
        columns_b=list("BC"),
        coef=-2.0,
        operator="+",
        column_names=["A+B", "A+C"],
    ).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_mult_ks():
    X = ks.DataFrame(np.arange(9).reshape(3, 3), columns=list("ABC"), dtype=np.float64)
    X_expected = pd.DataFrame(
        np.array(
            [
                [0.0, 1.0, 2.0, 0.0, 0.0],
                [3.0, 4.0, 5.0, 12.0, 15.0],
                [6.0, 7.0, 8.0, 42.0, 48.0],
            ]
        ),
        columns=["A", "B", "C", "A__*__B", "A__*__C"],
    )
    obj = ElementaryArithmetics(
        columns_a=list("AA"), columns_b=list("BC"), operator="*"
    ).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_div_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
    )

    X_expected = pd.DataFrame(
        np.array(
            [
                [0.0, 1.0, 2.0, 0.0, 0],
                [3.0, 4.0, 5.0, 0.75, 0.59999988],
                [6.0, 7.0, 8.0, 0.85714286, 0.7499999],
            ]
        ),
        columns=["A", "B", "C", "A__/__B", "A__/__C"],
    )
    obj = ElementaryArithmetics(
        columns_a=list("AA"), columns_b=list("BC"), operator="/"
    ).fit(X)
    return obj, X, X_expected


@pytest.mark.koalas
def test_add_ks(data_add_ks):
    obj, X, X_expected = data_add_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_add_ks_np(data_add_ks):
    obj, X, X_expected = data_add_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_float32_add_ks_ks(data_float32_add_ks):
    obj, X, X_expected = data_float32_add_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_float32_add_ks_np_ks(data_float32_add_ks):
    obj, X, X_expected = data_float32_add_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_mult_ks(data_mult_ks):
    obj, X, X_expected = data_mult_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_mult_ks_np(data_mult_ks):
    obj, X, X_expected = data_mult_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_div_ks(data_div_ks):
    obj, X, X_expected = data_div_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_div_ks_np(data_div_ks):
    obj, X, X_expected = data_div_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_name_add_ks_ks(data_name_add_ks):
    obj, X, X_expected = data_name_add_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_name_add_ks_np_ks(data_name_add_ks):
    obj, X, X_expected = data_name_add_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)
