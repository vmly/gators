# License: Apache-2.0
import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt import OrdinalMinuteOfHour


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["2020-05-04-T00:00:00", pd.NaT],
                "B": ["2020-05-06-T00:10:00", pd.NaT],
                "C": ["2020-05-08-T00:20:00", pd.NaT],
                "D": ["2020-05-09-T00:40:00", pd.NaT],
                "E": ["2020-05-09-T00:59:00", pd.NaT],
                "X": ["x", "x"],
            }
        ),
        npartitions=1,
    )
    columns = ["A", "B", "C", "D", "E"]
    X["A"] = X["A"].astype("datetime64[ns]")
    X["B"] = X["B"].astype("datetime64[ms]")
    X["C"] = X["C"].astype("datetime64[s]")
    X["D"] = X["D"].astype("datetime64[m]")
    X["E"] = X["E"].astype("datetime64[m]")

    X_expected = pd.DataFrame(
        {
            "A__minute_of_hour": ["0.0", "nan"],
            "B__minute_of_hour": ["10.0", "nan"],
            "C__minute_of_hour": ["20.0", "nan"],
            "D__minute_of_hour": ["40.0", "nan"],
            "E__minute_of_hour": ["59.0", "nan"],
        }
    )
    X_expected = pd.concat([X.compute().copy(), X_expected], axis=1)
    obj = OrdinalMinuteOfHour(columns=columns).fit(X)
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
