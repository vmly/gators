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
                "A": [0., 3., 6.],
                "B": [1., 4., 7.],
                "C": [2., 5., 8.],
            }
        ),
        npartitions=1,
    )
    obj = PolynomialFeatures(interaction_only=True, columns=["A", "B", "C"]).fit(X)
    X_expected = pd.DataFrame(
        np.array(
            [
                [0., 1., 2., 0., 0., 2.0],
                [3., 4., 5., 12., 15., 20.0],
                [6., 7., 8., 42., 48., 56.0],
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
                "A": [0., 3., 6.],
                "B": [1., 4., 7.],
                "C": [2., 5., 8.],
            ],
            dtype=np.int16,
        ),
        npartitions=1,
    )
    obj = PolynomialFeatures(interaction_only=True, columns=["A", "B", "C"]).fit(X)
    X_expected = pd.DataFrame(
        np.array(
            [
                [0., 1., 2., 0., 0., 2.0],
                [3., 4., 5., 12., 15., 20.0],
                [6., 7., 8., 42., 48., 56.0],
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
                "A": [0., 3., 6.],
                "B": [1., 4., 7.],
                "C": [2., 5., 8.],
            ],
            dtype=np.float32,
        ),
        npartitions=1,
    )

    obj = PolynomialFeatures(interaction_only=False, columns=["A", "B", "C"]).fit(X)
    X_expected = pd.DataFrame(
        np.array(
            [
                [0., 1., 2., 0., 0., 0., 1., 2., 4.0],
                [3., 4., 5., 9., 12., 15., 16., 20., 25.0],
                [6., 7., 8., 36., 42., 48., 49., 56., 64.0],
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
                "A": [0., 3., 6.],
                "B": [1., 4., 7.],
                "C": [2., 5., 8.],
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
                [0., 1., 2., 0., 0., 0., 1., 2., 4., 0., 0., 0., 0., 0., 0., 1., 2., 4., 8.],
                [3., 4., 5., 9., 12., 15., 16., 20., 25., 27., 36., 45., 48., 60., 75., 64., 80., 100., 125.],
                [6., 7., 8., 36., 42., 48., 49., 56., 64., 216., 252., 288., 294., 336., 384., 343., 392., 448., 512.],
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
                "A": [0., 3., 6.],
                "B": [1., 4., 7.],
                "C": [2., 5., 8.],
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
                [0., 1., 2., 0., 0., 2., 0.],
                [3., 4., 5., 12., 15., 20., 60.],
                [6., 7., 8., 42., 48., 56., 336.],
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
                "A": [0., 4., 8.0],
                "B": [1., 5., 9.0],
                "C": [2., 6., 10.0],
                "D": [3., 7., 11.0],
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
                [0., 1., 2., 3., 0., 0., 2.0],
                [4., 5., 6., 7., 20., 24., 30.0],
                [8., 9., 10., 11., 72., 80., 90.0],
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
