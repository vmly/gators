# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning import QuantileDiscretizer
from gators.encoders import RegressionEncoder, WOEEncoder


@pytest.fixture
def data():
    n_bins = 3
    X = pd.DataFrame(
        {
            "A": ["Q", "Q", "Q", "W", "W", "W"],
            "B": ["Q", "Q", "W", "W", "W", "W"],
            "C": ["Q", "Q", "Q", "Q", "W", "W"],
            "D": [1, 2, 3, 4, 5, 6],
        }
    )
    y = pd.Series([0.11, -0.1, 5.55, 233.9, 4.66, 255.1], name="TARGET")
    obj = RegressionEncoder(
        WOEEncoder(), discretizer=QuantileDiscretizer(n_bins=n_bins, inplace=True)
    ).fit(X, y)
    X_expected = pd.DataFrame(
        {
            "D": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "A__TARGET_1_WOEEncoder": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "B__TARGET_1_WOEEncoder": [
                0.0,
                0.0,
                0.6931471805599453,
                0.6931471805599453,
                0.6931471805599453,
                0.6931471805599453,
            ],
            "C__TARGET_1_WOEEncoder": [
                -0.40546510810816444,
                -0.40546510810816444,
                -0.40546510810816444,
                -0.40546510810816444,
                0.6931471805599453,
                0.6931471805599453,
            ],
            "A__TARGET_2_WOEEncoder": [
                0.0,
                0.0,
                0.0,
                1.3862943611198906,
                1.3862943611198906,
                1.3862943611198906,
            ],
            "B__TARGET_2_WOEEncoder": [
                0.0,
                0.0,
                0.6931471805599453,
                0.6931471805599453,
                0.6931471805599453,
                0.6931471805599453,
            ],
            "C__TARGET_2_WOEEncoder": [
                -0.40546510810816444,
                -0.40546510810816444,
                -0.40546510810816444,
                -0.40546510810816444,
                0.6931471805599453,
                0.6931471805599453,
            ],
        }
    )
    return obj, X, X_expected


@pytest.fixture
def data_float32():
    n_bins = 3
    X = pd.DataFrame(
        {
            "A": ["Q", "Q", "Q", "W", "W", "W"],
            "B": ["Q", "Q", "W", "W", "W", "W"],
            "C": ["Q", "Q", "Q", "Q", "W", "W"],
            "D": [1, 2, 3, 4, 5, 6],
        }
    )
    y = pd.Series([0.11, -0.1, 5.55, 233.9, 4.66, 255.1], name="TARGET")
    obj = RegressionEncoder(
        WOEEncoder(),
        discretizer=QuantileDiscretizer(n_bins=n_bins, inplace=True),
        dtype=np.float32,
    ).fit(X, y)
    X_expected = pd.DataFrame(
        {
            "D": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "A__TARGET_1_WOEEncoder": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "B__TARGET_1_WOEEncoder": [
                0.0,
                0.0,
                0.6931471805599453,
                0.6931471805599453,
                0.6931471805599453,
                0.6931471805599453,
            ],
            "C__TARGET_1_WOEEncoder": [
                -0.40546510810816444,
                -0.40546510810816444,
                -0.40546510810816444,
                -0.40546510810816444,
                0.6931471805599453,
                0.6931471805599453,
            ],
            "A__TARGET_2_WOEEncoder": [
                0.0,
                0.0,
                0.0,
                1.3862943611198906,
                1.3862943611198906,
                1.3862943611198906,
            ],
            "B__TARGET_2_WOEEncoder": [
                0.0,
                0.0,
                0.6931471805599453,
                0.6931471805599453,
                0.6931471805599453,
                0.6931471805599453,
            ],
            "C__TARGET_2_WOEEncoder": [
                -0.40546510810816444,
                -0.40546510810816444,
                -0.40546510810816444,
                -0.40546510810816444,
                0.6931471805599453,
                0.6931471805599453,
            ],
        }
    ).astype(np.float32)
    return obj, X, X_expected


@pytest.fixture
def data_no_cat():
    n_bins = 3
    X = pd.DataFrame(
        np.zeros((3, 6)),
        columns=list("qweasd"),
    )
    y = pd.Series([1.0, 2.0, 0.0], name="TARGET")
    obj = RegressionEncoder(
        WOEEncoder(), discretizer=QuantileDiscretizer(n_bins=n_bins, inplace=True)
    ).fit(X, y)
    return obj, X, X.copy()


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_float32_pd(data_float32):
    obj, X, X_expected = data_float32
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_float32_pd_np(data_float32):
    obj, X, X_expected = data_float32
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_without_cat_pd(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_without_cat_pd_np(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_init():
    discretizer = QuantileDiscretizer(n_bins=2)
    with pytest.raises(TypeError):
        _ = RegressionEncoder(encoder="q", discretizer=discretizer)
    with pytest.raises(TypeError):
        _ = RegressionEncoder(encoder=WOEEncoder(), discretizer="q")
