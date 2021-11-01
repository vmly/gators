# Licence Apache-2.0
from gators.feature_generation_dt.ordinal_day_of_month import OrdinalDayOfMonth
import pytest
import pandas as pd
import databricks.koalas as ks
ks.set_option('compute.default_index_type', 'distributed-sequence')


def test_init_pd():
    X = pd.DataFrame({'A': [0], 'B': [0]})
    obj = OrdinalDayOfMonth(columns=['A', 'B'])
    with pytest.raises(TypeError):
        _ = obj.fit(X)


@pytest.mark.koalas
def test_init_ks():
    X = ks.DataFrame({'A': [0], 'B': [0]})
    obj = OrdinalDayOfMonth(columns=['A', 'B'])
    with pytest.raises(TypeError):
        _ = obj.fit(X)
