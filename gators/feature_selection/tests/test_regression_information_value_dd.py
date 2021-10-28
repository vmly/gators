# License: Apache-2.0
import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning.discretizer import Discretizer
from gators.feature_selection.regression_information_value import (
    RegressionInformationValue,
)


@pytest.fixture
def data():
    k = 3
    n_bins = 4
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
                "B": [1, 1, 0, 1, 0, 0],
                "C": ["a", "b", "b", "b", "a", "a"],
                "D": [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
                "F": [1, 2, 3, 1, 2, 4],
            }
        ),
        npartitions=1,
    )
    X_expected = X[["A", "B", "C"]].compute().copy()
    y = dd.from_pandas(
        pd.Series([11.56, 9.57, 33.33, 87.6, 0.01, -65.0], name="TARGET"), npartitions=1
    )
    discretizer = Discretizer(n_bins=n_bins)
    obj = RegressionInformationValue(k=k, discretizer=discretizer).fit(X, y)
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
