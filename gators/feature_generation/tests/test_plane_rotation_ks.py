# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.plane_rotation import PlaneRotation

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        [[200.0, 140.0, 100.0], [210.0, 160.0, 125.0]], columns=list("XYZ")
    )
    X_expected = pd.DataFrame(
        {
            "X": [200.0, 210.0],
            "Y": [140.0, 160.0],
            "Z": [100.0, 125.0],
            "XY_x_45deg": [42.42640687119287, 35.35533905932739],
            "XY_y_45deg": [240.41630560342614, 261.62950903902254],
            "XY_x_60deg": [-21.243556529821376, -33.56406460551014],
            "XY_y_60deg": [243.20508075688775, 261.8653347947321],
            "XZ_x_45deg": [70.71067811865477, 60.104076400856556],
            "XZ_y_45deg": [212.13203435596424, 236.8807716974934],
            "XZ_x_60deg": [13.397459621556166, -3.253175473054796],
            "XZ_y_60deg": [223.20508075688775, 244.36533479473212],
        }
    )
    obj = PlaneRotation(columns=[["X", "Y"], ["X", "Z"]], theta_vec=[45, 60]).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_float32_ks():
    X = ks.DataFrame([[200, 140, 100], [210, 160, 125]], columns=list("XYZ")).astype(
        np.float32
    )
    X_expected = pd.DataFrame(
        {
            "X": [200.0, 210.0],
            "Y": [140.0, 160.0],
            "Z": [100.0, 125.0],
            "XY_x_45deg": [42.42640687119287, 35.35533905932739],
            "XY_y_45deg": [240.41630560342614, 261.62950903902254],
            "XY_x_60deg": [-21.243556529821376, -33.56406460551014],
            "XY_y_60deg": [243.20508075688775, 261.8653347947321],
            "XZ_x_45deg": [70.71067811865477, 60.104076400856556],
            "XZ_y_45deg": [212.13203435596424, 236.8807716974934],
            "XZ_x_60deg": [13.397459621556166, -3.253175473054796],
            "XZ_y_60deg": [223.20508075688775, 244.36533479473212],
        }
    ).astype(np.float32)
    obj = PlaneRotation(
        columns=[["X", "Y"], ["X", "Z"]], theta_vec=[45, 60], dtype=np.float32
    ).fit(X)
    return obj, X, X_expected


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected)


@pytest.mark.koalas
def test_float32_ks(data_float32_ks):
    obj, X, X_expected = data_float32_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_float32_ks_np(data_float32_ks):
    obj, X, X_expected = data_float32_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected)
