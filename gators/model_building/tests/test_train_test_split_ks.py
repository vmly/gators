# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from gators.model_building.train_test_split import TrainTestSplit


@pytest.fixture()
def data_ordered():
    X = pd.DataFrame(np.arange(40).reshape(8, 5), columns=list("ABCDE"))
    y_name = "TARGET"
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1], name=y_name)
    test_ratio = 0.5
    obj = TrainTestSplit(test_ratio=test_ratio, strategy="ordered")
    X_train_expected = pd.DataFrame(
        {
            "A": [ 0,5,10,15],
            "B": [ 1,6,11,16],
            "C": [ 2,7,12,17],
            "D": [ 3,8,13,18],
            "E": [ 4,9,14,19],
        })
    X_test_expected = pd.DataFrame(
        {
            "A": {20, 25, 6: 30, 7: 35],
            "B": {21, 26, 6: 31, 7: 36],
            "C": {22, 27, 6: 32, 7: 37],
            "D": {23, 28, 6: 33, 7: 38],
            "E": {24, 29, 6: 34, 7: 39],
        })
    y_train_expected = pd.Series([ 0,1,2,0], name=y_name)
    y_test_expected = pd.Series({1, 2, 6: 0, 7: 1], name=y_name)
    return (
        obj,
        X,
        y,
        X_train_expected,
        X_test_expected,
        y_train_expected,
        y_test_expected,
    )


@pytest.fixture()
def data_random():
    X = pd.DataFrame(np.arange(40).reshape(8, 5), columns=list("ABCDE"))
    y_name = "TARGET"
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1], name=y_name)
    test_ratio = 0.5
    obj = TrainTestSplit(test_ratio=test_ratio, strategy="random", random_state=0)
    X_train_expected = pd.DataFrame(
        {
            "A": [ 0,15, 20, 25],
            "B": [ 1,16, 21, 26],
            "C": [ 2,17, 22, 27],
            "D": [ 3,18, 23, 28],
            "E": [ 4,19, 24, 29],
        })
    X_test_expected = pd.DataFrame(
        {
            "A": {6: 30,10,5, 7: 35],
            "B": {6: 31,11,6, 7: 36],
            "C": {6: 32,12,7, 7: 37],
            "D": {6: 33,13,8, 7: 38],
            "E": {6: 34,14,9, 7: 39],
        })
    y_train_expected = pd.Series([ 0,0, 1, 2], name=y_name)
    y_test_expected = pd.Series({6: 0,2,1, 7: 1], name=y_name)
    return (
        obj,
        X,
        y,
        X_train_expected,
        X_test_expected,
        y_train_expected,
        y_test_expected,
    )


@pytest.fixture()
def data_stratified():
    X = pd.DataFrame(np.arange(40).reshape(8, 5), columns=list("ABCDE"))
    y_name = "TARGET"
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1], name=y_name)
    test_ratio = 0.5
    obj = TrainTestSplit(test_ratio=test_ratio, strategy="stratified", random_state=0)

    X_train_expected = pd.DataFrame(
        {
            "A": [ 0,5,10],
            "B": [ 1,6,11],
            "C": [ 2,7,12],
            "D": [ 3,8,13],
            "E": [ 4,9,14],
        })
    X_test_expected = pd.DataFrame(
        {
            "A": {6: 30,15, 7: 35, 20, 25],
            "B": {6: 31,16, 7: 36, 21, 26],
            "C": {6: 32,17, 7: 37, 22, 27],
            "D": {6: 33,18, 7: 38, 23, 28],
            "E": {6: 34,19, 7: 39, 24, 29],
        })
    y_train_expected = pd.Series([ 0,1,2], name=y_name)
    y_test_expected = pd.Series({6: 0,0, 7: 1, 1, 2], name=y_name)
    return (
        obj,
        X,
        y,
        X_train_expected,
        X_test_expected,
        y_train_expected,
        y_test_expected,
    )


@pytest.fixture()
def data_ordered_ks():
    X = ks.DataFrame(np.arange(40).reshape(8, 5), columns=list("ABCDE"))
    y_name = "TARGET"
    y = ks.Series([0, 1, 2, 0, 1, 2, 0, 1], name=y_name)
    test_ratio = 0.5
    obj = TrainTestSplit(test_ratio=test_ratio, strategy="ordered")
    return obj, X, y


@pytest.fixture()
def data_random_ks():
    X = ks.DataFrame(np.arange(40).reshape(8, 5), columns=list("ABCDE"))
    y_name = "TARGET"
    y = ks.Series([0, 1, 2, 0, 1, 2, 0, 1], name=y_name)
    test_ratio = 0.5
    obj = TrainTestSplit(test_ratio=test_ratio, strategy="random", random_state=0)
    return obj, X, y


@pytest.fixture()
def data_stratified_ks():
    X = ks.DataFrame(np.arange(40).reshape(8, 5), columns=list("ABCDE"))
    y_name = "TARGET"
    y = ks.Series([0, 1, 2, 0, 1, 2, 0, 1], name=y_name)
    test_ratio = 0.5
    obj = TrainTestSplit(test_ratio=test_ratio, strategy="stratified", random_state=0)
    return obj, X, y


def test_ordered(data_ordered):
    (
        obj,
        X,
        y,
        X_train_expected,
        X_test_expected,
        y_train_expected,
        y_test_expected,
    ) = data_ordered
    X_train, X_test, y_train, y_test = obj.transform(X, y)
    assert_frame_equal(X_train, X_train_expected)
    assert_frame_equal(X_test, X_test_expected)
    assert_series_equal(y_train, y_train_expected)
    assert_series_equal(y_test, y_test_expected)


@pytest.mark.koalas
def test_ordered_ks(data_ordered_ks):
    obj, X, y = data_ordered_ks
    X_train, X_test, y_train, y_test = obj.transform(X, y)
    X_new = pd.concat([X_train.to_pandas(), X_test.to_pandas()])
    y_new = pd.concat([y_train.to_pandas(), y_test.to_pandas()])
    assert_frame_equal(X.to_pandas(), X_new.sort_index())
    assert_series_equal(y.to_pandas(), y_new.sort_index())


def test_random(data_random):
    (
        obj,
        X,
        y,
        X_train_expected,
        X_test_expected,
        y_train_expected,
        y_test_expected,
    ) = data_random
    X_train, X_test, y_train, y_test = obj.transform(X, y)
    assert_frame_equal(X_train, X_train_expected)
    assert_frame_equal(X_test, X_test_expected)
    assert_series_equal(y_train, y_train_expected)
    assert_series_equal(y_test, y_test_expected)


@pytest.mark.koalas
def test_random_ks(data_random_ks):
    obj, X, y = data_random_ks
    X_train, X_test, y_train, y_test = obj.transform(X, y)
    X_new = pd.concat([X_train.to_pandas(), X_test.to_pandas()])
    y_new = pd.concat([y_train.to_pandas(), y_test.to_pandas()])
    assert_frame_equal(X.to_pandas().drop("index", axis=1), X_new.sort_index())
    assert_series_equal(y.to_pandas(), y_new.sort_index())


def test_stratified(data_stratified):
    (
        obj,
        X,
        y,
        X_train_expected,
        X_test_expected,
        y_train_expected,
        y_test_expected,
    ) = data_stratified
    X_train, X_test, y_train, y_test = obj.transform(X, y)
    assert_frame_equal(X_train, X_train_expected)
    assert_frame_equal(X_test, X_test_expected)
    assert_series_equal(y_train, y_train_expected)
    assert_series_equal(y_test, y_test_expected)


@pytest.mark.koalas
def test_stratified_ks(data_stratified_ks):
    obj, X, y = data_stratified_ks
    X_train, X_test, y_train, y_test = obj.transform(X, y)
    X_new = pd.concat([X_train.to_pandas(), X_test.to_pandas()])
    y_new = pd.concat([y_train.to_pandas(), y_test.to_pandas()])
    assert_frame_equal(X.to_pandas().drop("index", axis=1), X_new.sort_index())
    assert_series_equal(y.to_pandas(), y_new.sort_index())


def test_input():
    with pytest.raises(TypeError):
        _ = TrainTestSplit(test_ratio="q", random_state="q", strategy="q")
    with pytest.raises(TypeError):
        _ = TrainTestSplit(test_ratio="q", random_state="q", strategy="q")
    with pytest.raises(TypeError):
        _ = TrainTestSplit(test_ratio=0.1, random_state="q", strategy="q")
    with pytest.raises(TypeError):
        _ = TrainTestSplit(test_ratio=0.1, random_state=0, strategy=0)
    with pytest.raises(ValueError):
        _ = TrainTestSplit(test_ratio=0.1, random_state=0, strategy="q")
