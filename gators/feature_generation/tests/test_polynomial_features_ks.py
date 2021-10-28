import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.polynomial_features import PolynomialFeatures

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_inter_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
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
def data_int16_inter_ks():
    X = ks.DataFrame(np.arange(9).reshape(3, 3), columns=list("ABC"), dtype=np.int16)
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
def data_all_ks():
    X = ks.DataFrame(np.arange(9).reshape(3, 3), columns=list("ABC"), dtype=np.float32)
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
def data_degree_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
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
def data_inter_degree_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
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
def data_subset_ks():
    X = ks.DataFrame(
        np.arange(12).reshape(3, 4), columns=list("ABCD"), dtype=np.float64
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


@pytest.mark.koalas
def test_inter_ks(data_inter_ks):
    obj, X, X_expected = data_inter_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_inter_ks_np(data_inter_ks):
    obj, X, X_expected = data_inter_ks
    X_new = obj.transform_numpy(X.to_numpy())
    assert np.allclose(X_new, X_expected)


@pytest.mark.koalas
def test_int16_inter_ks(data_int16_inter_ks):
    obj, X, X_expected = data_int16_inter_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_int16_inter_ks_np(data_int16_inter_ks):
    obj, X, X_expected = data_int16_inter_ks
    X_new = obj.transform_numpy(X.to_numpy())
    assert np.allclose(X_new, X_expected)


@pytest.mark.koalas
def test_all_ks(data_all_ks):
    obj, X, X_expected = data_all_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_all_ks_np(data_all_ks):
    obj, X, X_expected = data_all_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_degree_ks(data_degree_ks):
    obj, X, X_expected = data_degree_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_degree_ks_np(data_degree_ks):
    obj, X, X_expected = data_degree_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_inter_degree_ks(data_inter_degree_ks):
    obj, X, X_expected = data_inter_degree_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_inter_degree_ks_np(data_inter_degree_ks):
    obj, X, X_expected = data_inter_degree_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_subset_ks(data_subset_ks):
    obj, X, X_expected = data_subset_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_subset_ks_np(data_subset_ks):
    obj, X, X_expected = data_subset_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)
