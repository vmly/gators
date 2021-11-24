# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_str import UpperCase


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 0.0, 0.0],
                "B": [0.0, 0.0, 0.0],
                "C": [0.0, 0.0, 0.0],
                "D": ["q", "qq", "QQq"],
                "E": ["w", "WW", "WWw"],
                "F": ["nan", None, ""],
            }
        ),
        npartitions=1,
    )

    obj = UpperCase(columns=list("DEF")).fit(X)
    X_expected = pd.DataFrame(
        [
            [
                0.0,
                0.0,
                0.0,
                "Q",
                "W",
                "nan",
            ],
            [
                0.0,
                0.0,
                0.0,
                "QQ",
                "WW",
                None,
            ],
            [0.0, 0.0, 0.0, "QQQ", "WWW", ""],
        ],
        columns=list("ABCDEF"),
    )
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))
