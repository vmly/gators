# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from gators.feature_generation_dt import OrdinalDayOfWeek

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
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
    X[columns] = X[columns].astype("datetime64[ns]")

    X_expected = pd.DataFrame(
        {
            "A__day_of_week": ["0.0", "nan"],
            "B__day_of_week": ["2.0", "nan"],
            "C__day_of_week": ["4.0", "nan"],
            "D__day_of_week": ["5.0", "nan"],
            "E__day_of_week": ["6.0", "nan"],
        }
    )
    X_expected = pd.concat([X.to_pandas().copy(), X_expected], axis=1)
    obj = OrdinalDayOfWeek(columns=columns).fit(X)
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
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)
