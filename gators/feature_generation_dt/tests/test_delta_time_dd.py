# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt.delta_time import DeltaTime


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["2020-05-04 00:00:00", pd.NaT],
                "B": ["2020-05-04 06:00:00", pd.NaT],
                "C": ["2020-05-04 12:00:00", pd.NaT],
                "D": ["2020-05-04 18:00:00", pd.NaT],
                "E": ["2020-05-04 23:00:00", pd.NaT],
                "X": ["x", "x"],
            }
        ),
        npartitions=1,
    )
    X["A"] = X["A"].astype("datetime64[ns]")
    X["B"] = X["B"].astype("datetime64[ms]")
    X["C"] = X["C"].astype("datetime64[s]")
    X["D"] = X["D"].astype("datetime64[m]")
    X["E"] = X["E"].astype("datetime64[h]")

    X_expected = pd.DataFrame(
        {
            "B__A__Deltatime[s]": [21600.0, np.nan],
            "C__A__Deltatime[s]": [43200.0, np.nan],
            "D__A__Deltatime[s]": [64800.0, np.nan],
            "E__A__Deltatime[s]": [82800.0, np.nan],
        }
    )
    X_expected = pd.concat([X.compute().copy(), X_expected], axis=1)
    obj = DeltaTime(columns_a=["B", "C", "D", "E"], columns_b=["A", "A", "A", "A"]).fit(
        X
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
