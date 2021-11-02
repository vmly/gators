# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt.delta_time import DeltaTime

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": ["2020-05-04 00:00:00", pd.NaT],
            "B": ["2020-05-04 06:00:00", pd.NaT],
            "C": ["2020-05-04 12:00:00", pd.NaT],
            "D": ["2020-05-04 18:00:00", pd.NaT],
            "E": ["2020-05-04 23:00:00", pd.NaT],
            "X": ["x", "x"],
        }
    )
    columns = ["A", "B", "C", "D", "E"]
    X[columns] = X[columns].astype("datetime64[ns]")
    X_expected = pd.DataFrame(
        {
            "B__A__Deltatime[s]": [21600.0, np.nan],
            "C__A__Deltatime[s]": [43200.0, np.nan],
            "D__A__Deltatime[s]": [64800.0, np.nan],
            "E__A__Deltatime[s]": [82800.0, np.nan],
        }
    )
    X_expected = pd.concat([X.to_pandas().copy(), X_expected], axis=1)
    obj = DeltaTime(columns_a=["B", "C", "D", "E"], columns_b=["A", "A", "A", "A"]).fit(
        X
    )
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
