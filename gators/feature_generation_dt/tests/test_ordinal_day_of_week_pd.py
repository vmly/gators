# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from gators.feature_generation_dt import OrdinalDayOfWeek


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": ["2020-05-04 00:00:00", np.nan],
            "B": ["2020-05-06 06:00:00", np.nan],
            "C": ["2020-05-08 23:00:00", pd.NaT],
            "D": ["2020-05-09 06:00:00", None],
            "E": ["2020-05-10 06:00:00", None],
            "X": ["x", "x"],
        }
    )
    columns = ["A", "B", "C", "D", "E"]
    X["A"] = X["A"].astype("datetime64[ns]")
    X["B"] = X["B"].astype("datetime64[ms]")
    X["C"] = X["C"].astype("datetime64[s]")
    X["D"] = X["D"].astype("datetime64[m]")
    X["E"] = X["E"].astype("datetime64[h]")

    X_expected = pd.DataFrame(
        {
            "A__day_of_week": ["0.0", "nan"],
            "B__day_of_week": ["2.0", "nan"],
            "C__day_of_week": ["4.0", "nan"],
            "D__day_of_week": ["5.0", "nan"],
            "E__day_of_week": ["6.0", "nan"],
        }
    )
    X_expected = pd.concat([X.copy(), X_expected], axis=1)
    obj = OrdinalDayOfWeek(columns=columns).fit(X)
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_series_equal(X_new.dtypes, X_expected.dtypes)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_series_equal(X_new.dtypes, X_expected.dtypes)
    assert_frame_equal(X_new, X_expected)


def test_init():
    with pytest.raises(TypeError):
        _ = OrdinalDayOfWeek(columns=0)
    with pytest.raises(ValueError):
        _ = OrdinalDayOfWeek(columns=[])
