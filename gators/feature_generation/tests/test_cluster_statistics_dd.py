# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.cluster_statistics import ClusterStatistics


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(np.arange(9, dtype=float).reshape(3, 3), columns=list("ABC")),
        npartitions=1,
    )
    clusters_dict = {
        "cluster_name_a": list("AB"),
        "cluster_name_b": list("AC"),
        "cluster_name_c": list("BC"),
    }
    obj = ClusterStatistics(clusters_dict=clusters_dict).fit(X)
    X_expected = pd.DataFrame(
        [
            [
                0.0,
                1.0,
                2.0,
                0.5,
                0.7071067811865476,
                1.0,
                1.4142135623730951,
                1.5,
                0.7071067811865476,
            ],
            [
                3.0,
                4.0,
                5.0,
                3.5,
                0.7071067811865476,
                4.0,
                1.4142135623730951,
                4.5,
                0.7071067811865476,
            ],
            [
                6.0,
                7.0,
                8.0,
                6.5,
                0.7071067811865476,
                7.0,
                1.4142135623730951,
                7.5,
                0.7071067811865476,
            ],
        ],
        columns=[
            "A",
            "B",
            "C",
            "cluster_name_a__mean",
            "cluster_name_a__std",
            "cluster_name_b__mean",
            "cluster_name_b__std",
            "cluster_name_c__mean",
            "cluster_name_c__std",
        ],
    )
    return obj, X, X_expected


@pytest.fixture
def data_float32():
    X = dd.from_pandas(
        pd.DataFrame(np.arange(9, dtype=np.int16).reshape(3, 3), columns=list("ABC")),
        npartitions=1,
    )

    clusters_dict = {
        "cluster_name_a": list("AB"),
        "cluster_name_b": list("AC"),
        "cluster_name_c": list("BC"),
    }
    obj = ClusterStatistics(clusters_dict=clusters_dict, dtype=np.float32).fit(X)
    X_expected = pd.DataFrame(
        [
            [
                0.0,
                1.0,
                2.0,
                0.5,
                0.7071067811865476,
                1.0,
                1.4142135623730951,
                1.5,
                0.7071067811865476,
            ],
            [
                3.0,
                4.0,
                5.0,
                3.5,
                0.7071067811865476,
                4.0,
                1.4142135623730951,
                4.5,
                0.7071067811865476,
            ],
            [
                6.0,
                7.0,
                8.0,
                6.5,
                0.7071067811865476,
                7.0,
                1.4142135623730951,
                7.5,
                0.7071067811865476,
            ],
        ],
        columns=[
            "A",
            "B",
            "C",
            "cluster_name_a__mean",
            "cluster_name_a__std",
            "cluster_name_b__mean",
            "cluster_name_b__std",
            "cluster_name_c__mean",
            "cluster_name_c__std",
        ],
    ).astype(np.float32)
    return obj, X, X_expected


@pytest.fixture
def data_names():
    X = dd.from_pandas(
        pd.DataFrame(np.arange(9, dtype=float).reshape(3, 3), columns=list("ABC")),
        npartitions=1,
    )

    clusters_dict = {
        "cluster_name_a": list("AB"),
        "cluster_name_b": list("AC"),
        "cluster_name_c": list("BC"),
    }
    obj = ClusterStatistics(
        clusters_dict=clusters_dict,
        column_names=["a_mean", "a_std", "bb_mean", "bb_std", "ccc_mean", "ccc_std"],
    ).fit(X)
    X_expected = pd.DataFrame(
        [
            [
                0.0,
                1.0,
                2.0,
                0.5,
                0.7071067811865476,
                1.0,
                1.4142135623730951,
                1.5,
                0.7071067811865476,
            ],
            [
                3.0,
                4.0,
                5.0,
                3.5,
                0.7071067811865476,
                4.0,
                1.4142135623730951,
                4.5,
                0.7071067811865476,
            ],
            [
                6.0,
                7.0,
                8.0,
                6.5,
                0.7071067811865476,
                7.0,
                1.4142135623730951,
                7.5,
                0.7071067811865476,
            ],
        ],
        columns=[
            "A",
            "B",
            "C",
            "a_mean",
            "a_std",
            "bb_mean",
            "bb_std",
            "ccc_mean",
            "ccc_std",
        ],
    )
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_float32_dd(data_float32):
    obj, X, X_expected = data_float32
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_float32_dd_np(data_float32):
    obj, X, X_expected = data_float32
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_names_dd(data_names):
    obj, X, X_expected = data_names
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_names_np(data_names):
    obj, X, X_expected = data_names
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values.astype(np.float64))
    assert_frame_equal(X_new, X_expected)
