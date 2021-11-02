# License: Apache-2.0
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning.discretizer import Discretizer
from gators.feature_selection.multiclass_information_value import (
    MultiClassInformationValue,
)


@pytest.fixture
def data():
    k = 3
    n_bins = 4
    X = pd.DataFrame(
        {
            "A": [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "b", "b", "a", "a"],
            "D": [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
            "F": [1, 2, 3, 1, 2, 4],
        }
    )
    X_expected = X[["A", "B", "C"]].copy()
    y = pd.Series([1, 1, 2, 2, 0, 0], name="TARGET")
    discretizer = Discretizer(n_bins=n_bins)
    obj = MultiClassInformationValue(k=k, discretizer=discretizer).fit(X, y)
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_init():
    with pytest.raises(TypeError):
        _ = MultiClassInformationValue(k="a", discretizer=3)
    with pytest.raises(TypeError):
        _ = MultiClassInformationValue(k=2, discretizer="a")
