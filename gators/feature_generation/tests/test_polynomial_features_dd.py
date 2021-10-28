import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.polynomial_features import PolynomialFeatures


@pytest.fixture
def data_inter():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 3.0, 6.0],
                "B": [1.0, 4.0, 7.0],
                "C": [2.0, 5.0, 8.0],
            }
        ),
        npartitions=1,
    )
    obj = PolynomialFeatures(interaction_only=True, columns=["A", "B", "C"]).fit(X)
    X_expected = pd.DataFrame(
        np.array(
            [
                [0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
                [3.0, 4.0, 5.0, 12.0, 15.0, 20.0],
                [6.0, 7.0, 8.0, 42.0, 48.0, 56.0],
            ]
        ),
        columns=["A", "B", "C", "A__x__B", "A__x__C", "B__x__C"],
    )
    return obj, X, X_expected


@pytest.fixture
def data_int16_inter():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 3.0, 6.0],
                "B": [1.0, 4.0, 7.0],
                "C": [2.0, 5.0, 8.0],
            },
            dtype=np.int16,
        ),
        npartitions=1,
    )
    obj = PolynomialFeatures(interaction_only=True, columns=["A", "B", "C"]).fit(X)
    X_expected = pd.DataFrame(
        np.array(
            [
                [0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
                [3.0, 4.0, 5.0, 12.0, 15.0, 20.0],
                [6.0, 7.0, 8.0, 42.0, 48.0, 56.0],
            ]
        ),
        columns=["A", "B", "C", "A__x__B", "A__x__C", "B__x__C"],
    ).astype(np.int16)
    return obj, X, X_expected


@pytest.fixture
def data_all():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 3.0, 6.0],
                "B": [1.0, 4.0, 7.0],
                "C": [2.0, 5.0, 8.0],
            },
            dtype=np.float32,
        ),
        npartitions=1,
    )

    obj = PolynomialFeatures(interaction_only=False, columns=["A", "B", "C"]).fit(X)
    X_expected = pd.DataFrame(
        np.array(
            [
                [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 4.0],
                [3.0, 4.0, 5.0, 9.0, 12.0, 15.0, 16.0, 20.0, 25.0],
                [6.0, 7.0, 8.0, 36.0, 42.0, 48.0, 49.0, 56.0, 64.0],
            ]
        ),
        columns=[
            "A",
            "B",
            "C",
            "A__x__A",
            "A__x__B",
            "A__x__C",
            "B__x__B",
            "B__x__C",
            "C__x__C",
        ],
    ).astype(np.float32)
    return obj, X, X_expected


@pytest.fixture
def data_degree():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 3.0, 6.0],
                "B": [1.0, 4.0, 7.0],
                "C": [2.0, 5.0, 8.0],
            }
        ),
        npartitions=1,
    )
    obj = PolynomialFeatures(
        interaction_only=False, degree=3, columns=["A", "B", "C"]
    ).fit(X)
    X_expected = pd.DataFrame(
        np.array(
            [
                [
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    4.0,
                    8.0,
                ],
                [
                    3.0,
                    4.0,
                    5.0,
                    9.0,
                    12.0,
                    15.0,
                    16.0,
                    20.0,
                    25.0,
                    27.0,
                    36.0,
                    45.0,
                    48.0,
                    60.0,
                    75.0,
                    64.0,
                    80.0,
                    100.0,
                    125.0,
                ],
                [
                    6.0,
                    7.0,
                    8.0,
                    36.0,
                    42.0,
                    48.0,
                    49.0,
                    56.0,
                    64.0,
                    216.0,
                    252.0,
                    288.0,
                    294.0,
                    336.0,
                    384.0,
                    343.0,
                    392.0,
                    448.0,
                    512.0,
                ],
            ]
        ),
        columns=[
            "A",
            "B",
            "C",
            "A__x__A",
            "A__x__B",
            "A__x__C",
            "B__x__B",
            "B__x__C",
            "C__x__C",
            "A__x__A__x__A",
            "A__x__A__x__B",
            "A__x__A__x__C",
            "A__x__B__x__B",
            "A__x__B__x__C",
            "A__x__C__x__C",
            "B__x__B__x__B",
            "B__x__B__x__C",
            "B__x__C__x__C",
            "C__x__C__x__C",
        ],
    )
    return obj, X, X_expected


@pytest.fixture
def data_inter_degree():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 3.0, 6.0],
                "B": [1.0, 4.0, 7.0],
                "C": [2.0, 5.0, 8.0],
            }
        ),
        npartitions=1,
    )
    obj = PolynomialFeatures(
        interaction_only=True, degree=3, columns=["A", "B", "C"]
    ).fit(X)
    X_expected = pd.DataFrame(
        np.array(
            [
                [0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 0.0],
                [3.0, 4.0, 5.0, 12.0, 15.0, 20.0, 60.0],
                [6.0, 7.0, 8.0, 42.0, 48.0, 56.0, 336.0],
            ]
        ),
        columns=["A", "B", "C", "A__x__B", "A__x__C", "B__x__C", "A__x__B__x__C"],
    )
    return obj, X, X_expected


@pytest.fixture
def data_subset():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 4.0, 8.0],
                "B": [1.0, 5.0, 9.0],
                "C": [2.0, 6.0, 10.0],
                "D": [3.0, 7.0, 11.0],
            }
        ),
        npartitions=1,
    )
    obj = PolynomialFeatures(
        columns=["A", "B", "C"], interaction_only=True, degree=2
    ).fit(X)
    X_expected = pd.DataFrame(
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 2.0],
                [4.0, 5.0, 6.0, 7.0, 20.0, 24.0, 30.0],
                [8.0, 9.0, 10.0, 11.0, 72.0, 80.0, 90.0],
            ]
        ),
        columns=["A", "B", "C", "D", "A__x__B", "A__x__C", "B__x__C"],
    )
    return obj, X, X_expected


def test_inter_dd(data_inter):
    obj, X, X_expected = data_inter
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_inter_dd_np(data_inter):
    obj, X, X_expected = data_inter
    X_new = obj.transform_numpy(X.compute().to_numpy())
    assert np.allclose(X_new, X_expected)


def test_int16_inter_dd(data_int16_inter):
    obj, X, X_expected = data_int16_inter
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_int16_inter_dd_np(data_int16_inter):
    obj, X, X_expected = data_int16_inter
    X_new = obj.transform_numpy(X.compute().to_numpy())
    assert np.allclose(X_new, X_expected)


def test_all_dd(data_all):
    obj, X, X_expected = data_all
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_all_dd_np(data_all):
    obj, X, X_expected = data_all
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_degree_dd(data_degree):
    obj, X, X_expected = data_degree
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_degree_dd_np(data_degree):
    obj, X, X_expected = data_degree
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_inter_degree_dd(data_inter_degree):
    obj, X, X_expected = data_inter_degree
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_inter_degree_dd_np(data_inter_degree):
    obj, X, X_expected = data_inter_degree
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_subset_dd(data_subset):
    obj, X, X_expected = data_subset
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_subset_dd_np(data_subset):
    obj, X, X_expected = data_subset
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)
