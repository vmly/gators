# License: Apache-2.0
import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning.discretizer import Discretizer
from gators.feature_selection.multiclass_information_value import (
    MultiClassInformationValue,
)

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    k = 3
    n_bins = 4
    X = ks.DataFrame(
        {
            "A": [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "b", "b", "a", "a"],
            "D": [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
            "F": [1, 2, 3, 1, 2, 4],
        }
    )
    X_expected = X[["A", "B", "C"]].to_pandas().copy()
    y = ks.Series([1, 1, 2, 2, 0, 0], name="TARGET")
    discretizer = Discretizer(n_bins=n_bins)
    obj = MultiClassInformationValue(k=k, discretizer=discretizer).fit(X, y)
    return obj, X, X_expected


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    X_new = X_new.to_pandas()
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_init():
    with pytest.raises(TypeError):
        _ = MultiClassInformationValue(k="a", discretizer=3)
    with pytest.raises(TypeError):
        _ = MultiClassInformationValue(k=2, discretizer="a")