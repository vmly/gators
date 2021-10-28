# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.encoders import MultiClassEncoder, WOEEncoder

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": ["Q", "Q", "Q", "W", "W", "W"],
            "B": ["Q", "Q", "W", "W", "W", "W"],
            "C": ["Q", "Q", "Q", "Q", "W", "W"],
            "D": [1, 2, 3, 4, 5, 6],
        })
    y = ks.Series([0, 0, 1, 2, 1, 2], name="TARGET")
    obj = MultiClassEncoder(WOEEncoder()).fit(X, y)
    X_expected = pd.DataFrame(
        {
            "D": [ 1.0,2.0,3.0,4.0, 5.0, 6.0],
            "A__TARGET_1_WOEEncoder": [ 0.0,0.0,0.0,0.0, 0.0, 0.0],
            "B__TARGET_1_WOEEncoder": {
                0: 0.0,
               0.0,
               0.6931471805599453,
               0.6931471805599453,
                0.6931471805599453,
                0.6931471805599453,
            ],
            "C__TARGET_1_WOEEncoder": {
                0: -0.40546510810816444,
               -0.40546510810816444,
               -0.40546510810816444,
               -0.40546510810816444,
                0.6931471805599453,
                0.6931471805599453,
            ],
            "A__TARGET_2_WOEEncoder": {
                0: 0.0,
               0.0,
               0.0,
               1.3862943611198906,
                1.3862943611198906,
                1.3862943611198906,
            ],
            "B__TARGET_2_WOEEncoder": {
                0: 0.0,
               0.0,
               0.6931471805599453,
               0.6931471805599453,
                0.6931471805599453,
                0.6931471805599453,
            ],
            "C__TARGET_2_WOEEncoder": {
                0: -0.40546510810816444,
               -0.40546510810816444,
               -0.40546510810816444,
               -0.40546510810816444,
                0.6931471805599453,
                0.6931471805599453,
            ],
        })
    return obj, X, X_expected


@pytest.fixture
def data_float32_ks():
    X = ks.DataFrame(
        {
            "A": ["Q", "Q", "Q", "W", "W", "W"],
            "B": ["Q", "Q", "W", "W", "W", "W"],
            "C": ["Q", "Q", "Q", "Q", "W", "W"],
            "D": [1, 2, 3, 4, 5, 6],
        })
    y = ks.Series([0, 0, 1, 2, 1, 2], name="TARGET")
    obj = MultiClassEncoder(WOEEncoder(), dtype=np.float32).fit(X, y)
    X_expected = pd.DataFrame(
        {
            "D": [ 1.0,2.0,3.0,4.0, 5.0, 6.0],
            "A__TARGET_1_WOEEncoder": [ 0.0,0.0,0.0,0.0, 0.0, 0.0],
            "B__TARGET_1_WOEEncoder": {
                0: 0.0,
               0.0,
               0.6931471805599453,
               0.6931471805599453,
                0.6931471805599453,
                0.6931471805599453,
            ],
            "C__TARGET_1_WOEEncoder": {
                0: -0.40546510810816444,
               -0.40546510810816444,
               -0.40546510810816444,
               -0.40546510810816444,
                0.6931471805599453,
                0.6931471805599453,
            ],
            "A__TARGET_2_WOEEncoder": {
                0: 0.0,
               0.0,
               0.0,
               1.3862943611198906,
                1.3862943611198906,
                1.3862943611198906,
            ],
            "B__TARGET_2_WOEEncoder": {
                0: 0.0,
               0.0,
               0.6931471805599453,
               0.6931471805599453,
                0.6931471805599453,
                0.6931471805599453,
            ],
            "C__TARGET_2_WOEEncoder": {
                0: -0.40546510810816444,
               -0.40546510810816444,
               -0.40546510810816444,
               -0.40546510810816444,
                0.6931471805599453,
                0.6931471805599453,
            ],
        }).astype(np.float32)
    return obj, X, X_expected


@pytest.fixture
def data_no_cat_ks():
    X = ks.DataFrame(
        np.zeros((3, 6)),
        columns=list("qweasd"),
    )
    y = ks.Series([1, 2, 0], name="TARGET")
    obj = MultiClassEncoder(WOEEncoder()).fit(X, y)
    return obj, X, X.to_pandas().copy()


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_float32_ks(data_float32_ks):
    obj, X, X_expected = data_float32_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_float32_ks_np(data_float32_ks):
    obj, X, X_expected = data_float32_ks
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_without_cat_ks(data_no_cat_ks):
    obj, X, X_expected = data_no_cat_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_without_cat_ks_np(data_no_cat_ks):
    obj, X, X_expected = data_no_cat_ks
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)
